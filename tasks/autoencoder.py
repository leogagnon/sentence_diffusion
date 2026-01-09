from functools import partial
import math
import os
import lightning as L
from omegaconf import OmegaConf
from dataclasses import dataclass
from typing import Any, List, Optional
from peft import LoraConfig
import torch
import random
from torch.utils.data import DataLoader, random_split
from torch.utils.data.dataset import Subset
from transformers import AutoTokenizer
from transformers.models.auto.modeling_auto import AutoModelForCausalLM
from peft import get_peft_model
from hydra.utils import instantiate
from model.encoder import EncoderConfig, EncoderModel
from model.decoder import DecoderConfig, DecoderModel
import os
import wandb
import hydra
from lightning.pytorch.utilities.rank_zero import rank_zero_info, rank_zero_only
from tqdm import tqdm
from mauve import compute_mauve, get_features_from_input
from transformers import get_constant_schedule_with_warmup
from data.datasets import DATA_SEED
import einx
import os
import torch
from torch.utils.data import Sampler
from typing import Iterator, Optional
from tasks.utils import *
from data.datasets import WikipediaDataset, FineWebDataset, get_dataloader, InfoLabel
import torch.nn as nn
import numpy as np


@dataclass
class AETaskConfig:
    lr: float
    batch_size: int
    encoder: EncoderConfig
    decoder: DecoderConfig
    dataset: dict
    val_size: int
    lr_warmup_steps: int = 1500
    denoising: bool = True
    delta_ent: float = 0.0
    reg_warmup_steps: int = 0
    delta_margin: float = 0.0
    reg_type: str = "none"
    sem_noise: float = 0.0
    sem_noise_warmup_steps: int = 0
    prefix_length: int = 0
    suffix_length: int = 128

    name: Optional[str] = None


class AETask(L.LightningModule):
    """
    Train a SEM encoder with a denoising Auto-Encoder task
    """

    def __init__(self, cfg: Optional[AETaskConfig] = None, **kwargs):
        super().__init__()

        if cfg == None:
            cfg = OmegaConf.to_object(
                OmegaConf.merge(
                    OmegaConf.create(AETaskConfig),
                    OmegaConf.create(kwargs),
                )
            )

        # Load encoder and decoder and make sure they are trainable
        self.decoder = DecoderModel(cfg.decoder).train().requires_grad_(True)
        cfg.encoder.latent_dim = self.decoder.dim
        self.encoder = EncoderModel(cfg.encoder).train().requires_grad_(True)

        # Setup dataset
        self.dataset = hydra.utils.instantiate(cfg.dataset)
        self.dataset: WikipediaDataset | FineWebDataset

        # This is with a fixed seed to make sure validation set never changes
        self.train_indices, self.val_indices = self.dataset.get_train_val_indices(
            val_size=cfg.val_size
        )

        self.cfg = cfg

        self.save_hyperparameters(
            OmegaConf.to_container(OmegaConf.structured(cfg)), logger=False
        )

    def compile(self):
        self.encoder.compile()
        self.decoder.compile()

    def setup(self, **kwargs):
        """Setup the data"""
        self.train_data = Subset(self.dataset, indices=self.train_indices)
        self.val_data = Subset(self.dataset, indices=self.val_indices)

    def train_dataloader(self):
        return get_dataloader(
            self.train_data,
            batch_size=self.cfg.batch_size,
            prefix_length=self.cfg.prefix_length,
            suffix_length=self.cfg.suffix_length,
            enc_tokenizer=self.encoder.tokenizer,
            dec_tokenizer=self.decoder.tokenizer,
            num_workers=int(os.environ["TORCH_NUM_WORKERS"]),
            encoder_mode="suffix",
            encoder_noise=self.cfg.denoising,
        )

    def val_dataloader(self):
        return get_dataloader(
            self.val_data,
            batch_size=self.cfg.batch_size,
            prefix_length=self.cfg.prefix_length,
            suffix_length=self.cfg.suffix_length,
            enc_tokenizer=self.encoder.tokenizer,
            dec_tokenizer=self.decoder.tokenizer,
            num_workers=int(os.environ["TORCH_NUM_WORKERS"]),
            encoder_mode="suffix",
            encoder_noise=self.cfg.denoising,
        )

    def configure_optimizers(self):
        no_decay = ["bias", "norm"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in self.named_parameters()
                    if not any(nd in n.lower() for nd in no_decay)
                ],
                "weight_decay": 0.01,
            },
            {
                "params": [
                    p
                    for n, p in self.named_parameters()
                    if any(nd in n.lower() for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=self.cfg.lr)
        if self.cfg.lr_warmup_steps > 0:
            scheduler = get_constant_schedule_with_warmup(
                optimizer, num_warmup_steps=self.cfg.lr_warmup_steps
            )
            scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}

            return [optimizer], [scheduler]
        else:
            return optimizer

    def training_step(self, batch, batch_idx):

        z, sem_out = self.encoder(
            batch["input_ids_enc"],
            batch["attention_mask_enc"],
            return_count=hasattr(self, "sem_usage_ema"),
            noise=cosine_warmup_get_value(
                step=self.global_step,
                max_value=self.cfg.sem_noise,
                warmup_steps=self.cfg.sem_noise_warmup_steps,
                exp=2,
            ),
        )

        # Compute decoder likelihood of input_ids (no need for attention mask cuz causal)
        logits = self.decoder(input_ids=batch["input_ids_dec"], z=z)

        info_mask_dec = batch["info_mask_dec"][:, 1:]
        targets = batch["input_ids_dec"][:, 1:].contiguous()
        logits = logits[:, :-1].contiguous()
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1),
            ignore_index=self.decoder.tokenizer.pad_token_id,
            reduction="none",
        )
        loss = loss.view_as(targets)

        loss = loss[info_mask_dec == InfoLabel.SUFFIX.value].mean()
        self.log(
            "train/loss",
            loss,
            on_epoch=False,
            on_step=True,
            sync_dist=True,
        )

        reg = None
        if (self.cfg.delta_ent > 0.0) and (self.cfg.reg_type == "ent"):
            ent, m_ent = sem_entropy(sem_out["probs"])
            reg = self.cfg.delta_ent * (ent - m_ent)
        elif (self.cfg.delta_margin > 0.0) and (self.cfg.reg_type == "margin"):
            reg = sem_margin(sem_out["probs"], delta=self.cfg.delta_margin)

        if reg is not None:
            delta = cosine_warmup_get_value(
                self.global_step,
                max_value=1.0,
                warmup_steps=self.cfg.reg_warmup_steps,
            )
            loss = loss + delta * reg

        if hasattr(self, "sem_usage_ema"):
            self.sem_usage_ema.update(sem_out["usage_count"], batch_size=z.shape[0])

        return loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):

        soft_z, sem_out = self.encoder(
            batch["input_ids_enc"],
            batch["attention_mask_enc"],
            return_count=True,
            noise=0.0,
        )

        ent, m_ent = sem_entropy(sem_out["probs"])
        self.log(
            "val/sem_entropy",
            ent.item(),
            on_epoch=True,
            sync_dist=True,
        )
        self.log(
            "val/sem_marginal_entropy",
            m_ent.item(),
            on_epoch=True,
            sync_dist=True,
        )

        self.log(
            "val/dead_words",
            torch.sum(sem_out["usage_count"] == 0).item() / len(sem_out["usage_count"]),
            on_epoch=True,
            sync_dist=True,
        )

        self.log(
            "val/latent_norm",
            soft_z.norm(p=2, dim=-1).mean().detach().item(),
            on_epoch=True,
            sync_dist=True,
        )

        hard_z, sem_out = self.encoder(
            batch["input_ids_enc"],
            batch["attention_mask_enc"],
            return_count=True,
            noise=0.0,
            temp=1e-4,
        )

        for latent, latent_type in zip([hard_z, soft_z], ["hard", "soft"]):
            logits = self.decoder(input_ids=batch["input_ids_dec"], z=latent)

            info_mask_dec = batch["info_mask_dec"][:, 1:]
            targets = batch["input_ids_dec"][:, 1:].contiguous()
            logits = logits[:, :-1].contiguous()
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=self.decoder.tokenizer.pad_token_id,
                reduction="none",
            )
            loss = loss.view_as(targets)

            suffix_loss = loss[info_mask_dec == InfoLabel.SUFFIX.value].mean()
            self.log(
                f"val/loss_{latent_type}", suffix_loss, on_epoch=True, sync_dist=True
            )

        # Maybe log some generations
        if (batch_idx == 0) and (rank_zero_only.rank == 0) and (hard_z != None):
            # Log generation from clean samples
            table_clean = wandb.Table(columns=["Original", "Reconstructed"])
            for original, reconstructed in zip(
                batch["input_str"][:5],
                self.decoder.tokenizer.batch_decode(
                    self.decoder.generate(
                        z=hard_z[:5], max_length=self.cfg.suffix_length
                    ),
                    skip_special_tokens=True,
                ),
            ):
                table_clean.add_data(original, reconstructed)
            wandb.log({"val/samples": table_clean})
