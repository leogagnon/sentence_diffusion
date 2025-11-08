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
from data.datasets import WikipediaDataset, FineWebDataset


@dataclass
class AETaskConfig:
    lr: float
    batch_size: int
    encoder: EncoderConfig
    decoder: DecoderConfig
    dataset: dict
    val_size: int
    lr_warmup_steps: int = 1500
    sub_p: float = 0.3
    delta_ent: float = 0.0
    delta_ent_warmup_steps: int = 0

    name: Optional[str] = None


class AETask(L.LightningModule):
    """
    Autoencoder Task.
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
        self.encoder = EncoderModel(cfg.encoder).train().requires_grad_(True)
        cfg.decoder.input_dim = self.encoder.latent_dim
        self.decoder = DecoderModel(cfg.decoder).train().requires_grad_(True)

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

    def random_substitution(self, input_ids):
        """
        Randomly subtitute token with some random word with some probablity
        """
        input_ids = input_ids.clone()

        probability = torch.full_like(
            input_ids, fill_value=self.cfg.sub_p, dtype=torch.float32
        )
        masked_indices = torch.bernoulli(probability).bool()
        vocab_size = len(self.encoder.tokenizer) - len(
            self.encoder.tokenizer.all_special_ids
        )
        random_words = torch.randint_like(input_ids, low=0, high=vocab_size)

        # Don't sub the first token (language token in SONAR)
        masked_indices[:, 0] = False

        input_ids[masked_indices] = random_words[masked_indices]

        return input_ids

    def compile(self):
        self.encoder.compile()
        self.decoder.compile()

    def setup(self, **kwargs):
        """Setup the data"""
        self.train_data = Subset(self.dataset, indices=self.train_indices)
        self.val_data = Subset(self.dataset, indices=self.val_indices)

    def train_dataloader(self):
        # We use a random, infinite sampler WITH replacement for convenience
        return DataLoader(
            self.train_data,
            batch_sampler=InfiniteDistributedUniformSampler(
                n=len(self.train_data), batch_size=self.cfg.batch_size
            ),
            collate_fn=self.get_collate_fn(),
        )

    def val_dataloader(self):
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            sampler = torch.utils.data.DistributedSampler(self.val_data, shuffle=False)
        else:
            sampler = torch.utils.data.SequentialSampler(self.val_data)

        return DataLoader(
            self.val_data,
            batch_size=self.cfg.batch_size,
            sampler=sampler,
            collate_fn=self.get_collate_fn(),
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

    def get_collate_fn(self):
        dec_tokenizer = self.decoder.tokenizer
        max_length = self.dataset.cfg.max_length

        def fn(batch):

            input_str = batch["input_str"]

            # Compute input_ids of the decoder if not already there
            # If is there we still need to add BOS
            if "input_ids" not in batch:
                input_ids_dec = dec_tokenizer.batch_encode_plus(
                    batch["input_str"],
                    truncation=True,
                    padding=True,
                    max_length=max_length,
                    return_tensors="pt",
                    add_special_tokens=True,
                    return_attention_mask=False,
                )["input_ids"]
            else:
                input_ids_dec = [
                    [dec_tokenizer.bos_token_id] + seq for seq in batch["input_ids"]
                ]
                input_ids_dec = dec_tokenizer.pad(
                    {"input_ids": input_ids_dec},
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                    max_length=max_length,
                    return_attention_mask=False,
                )["input_ids"]

            # Compute input_ids of encoder
            batch_enc = self.encoder.tokenizer.batch_encode_plus(
                input_str,
                truncation=True,
                padding="max_length",
                max_length=max_length,
                return_tensors="pt",
            )

            return {
                "input_ids_dec": input_ids_dec,
                "input_ids_enc": batch_enc["input_ids"],
                "attention_mask_enc": batch_enc["attention_mask"].bool(),
            }

        return fn

    def training_step(self, batch, batch_idx):

        z, dlc_probs = self.encoder(
            self.random_substitution(batch["input_ids_enc"]),
            batch["attention_mask_enc"],
        )

        # Compute decoder likelihood of input_ids (no need for attention mask cuz causal)
        logits = self.decoder(input_ids=batch["input_ids_dec"], z=z)

        # Compute loss
        logits = logits[:, :-1].contiguous()
        targets = batch["input_ids_dec"][:, 1:].contiguous()
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1),
            ignore_index=self.decoder.tokenizer.pad_token_id,
        )

        self.log(
            "train/loss",
            loss,
            on_epoch=False,
            on_step=True,
            sync_dist=True,
        )

        if self.cfg.delta_ent > 0.0:
            ent, m_ent = sem_entropy(dlc_probs)

            delta = cosine_warmup_get_value(
                self.global_step,
                max_value=self.cfg.delta_ent,
                warmup_steps=self.cfg.delta_ent_warmup_steps,
            )

            loss = loss + delta * (ent - m_ent)

        return loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):

        z, dlc_probs = self.encoder(
            batch["input_ids_enc"], batch["attention_mask_enc"], step=self.global_step
        )

        ent, m_ent = sem_entropy(dlc_probs)

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
            "val/latent_norm",
            z.norm(p=2, dim=-1).mean().detach().item(),
            on_epoch=True,
            sync_dist=True,
        )

        # Compute clean reconstruction loss
        logits = self.decoder(input_ids=batch["input_ids_dec"], z=z)

        logits = logits[:, :-1].contiguous()
        targets = batch["input_ids_dec"][:, 1:].contiguous()
        recon_loss = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1),
            ignore_index=self.decoder.tokenizer.pad_token_id,
        )
        self.log("val/loss", recon_loss, on_epoch=True, sync_dist=True)

        # Maybe log some generations
        if (batch_idx == 0) and (rank_zero_only.rank == 0) and (z != None):
            # Log generation from clean samples
            table_clean = wandb.Table(columns=["Original", "Reconstructed"])
            for original, reconstructed in zip(
                batch["input_str"][:5],
                self.decoder.tokenizer.batch_decode(
                    self.decoder.generate(
                        z=z[:5], max_length=self.dataset.cfg.max_length
                    ),
                    skip_special_tokens=True,
                ),
            ):
                table_clean.add_data(original, reconstructed)
            wandb.log({"val/samples": table_clean})
