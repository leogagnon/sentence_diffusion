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
from data.datasets import StoriesDatasetConfig, StoriesDataset
from hydra.utils import instantiate
from model.encoder import EncoderConfig, EncoderModel
from model.decoder import DecoderConfig, DecoderModel
import os
import wandb
import hydra
from lightning.pytorch.utilities.rank_zero import rank_zero_info, rank_zero_only
from tasks.autoencoder import AETask, InfiniteDistributedUniformSampler, compute_entropy
from data.datasets import WikipediaDataset
from tqdm import tqdm
from mauve import compute_mauve, get_features_from_input
import einx
from transformers import get_constant_schedule_with_warmup
from data.datasets import DATA_SEED, InfoLabel
from copy import deepcopy
import torch.nn.functional as F


def cosine_sim_mat(x):
    # x: [B, D], y: [B, D]
    x_ = F.normalize(x, p=2, dim=-1)
    return x_ @ x_.T


@dataclass
class DCSETaskConfig:
    lr: float
    batch_size: int
    encoder: EncoderConfig
    teacher: EncoderConfig
    dataset: dict
    val_size: int
    lr_warmup: bool = False
    temp: float = 0.01

    name: Optional[str] = None


class DCSETask(L.LightningModule):
    """
    Autoencoder Task.
    """

    def __init__(self, cfg: Optional[DCSETaskConfig] = None, **kwargs):
        super().__init__()

        if cfg == None:
            cfg = OmegaConf.to_object(
                OmegaConf.merge(
                    OmegaConf.create(DCSETaskConfig),
                    OmegaConf.create(kwargs),
                )
            )

        # Load encoder
        self.encoder = EncoderModel(cfg.encoder).train().requires_grad_(True)
        self.teacher = EncoderModel(cfg.teacher).eval().requires_grad_(False)

        # Setup dataset
        self.dataset = hydra.utils.instantiate(cfg.dataset)

        # This is with a fixed seed to make sure validation set never changes
        indices = torch.randperm(
            len(self.dataset),
            generator=torch.Generator().manual_seed(DATA_SEED),
        )
        self.train_indices = indices[: -cfg.val_size]
        self.val_indices = indices[-cfg.val_size :]

        self.cfg = cfg

        self.save_hyperparameters(
            OmegaConf.to_container(OmegaConf.structured(cfg)), logger=False
        )

    def train(self, mode=True):
        # Make sure encoder stays in eval mode
        super().train(mode)
        if hasattr(self, "encoder"):
            self.teacher.eval()
        return self

    def compile(self):
        self.encoder.compile()

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
            collate_fn=self.dataset.get_collate_and_tokenize_fn(
                conditional=False,
                enc_tokenizer=self.encoder.tokenizer,
                teacher_tokenizer=self.teacher.tokenizer,
            ),
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
            collate_fn=self.dataset.get_collate_and_tokenize_fn(
                conditional=False,
                enc_tokenizer=self.encoder.tokenizer,
                teacher_tokenizer=self.teacher.tokenizer,
            ),
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
        if self.cfg.lr_warmup:
            scheduler = get_constant_schedule_with_warmup(
                optimizer, num_warmup_steps=2000
            )
            scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}

            return [optimizer], [scheduler]
        else:
            return optimizer

    def training_step(self, batch, batch_idx):

        z_enc, dlc_probs = self.encoder(
            batch["input_ids_enc"],
            batch["attention_mask_enc"],
            step=self.global_step,
        )

        with torch.no_grad():
            z_teacher = self.teacher(
                batch["input_ids_teacher"],
                batch["attention_mask_teacher"],
                step=self.global_step,
            )

        # We put this in float32 autocast to avoid numerical issues with softmax
        with torch.autocast(device_type="cuda", dtype=torch.float32):
            enc_sim = cosine_sim_mat(z_enc.squeeze(1))
            teacher_sim = cosine_sim_mat(z_teacher.squeeze(1))

            # Remove the diagonal elements (self-similarity)
            N = enc_sim.size(0)
            offdiag_mask = ~torch.eye(N, dtype=bool)
            enc_sim = enc_sim[offdiag_mask].reshape(N, N - 1)
            teacher_sim = teacher_sim[offdiag_mask].reshape(N, N - 1)

            enc_log_p = torch.log_softmax(enc_sim / self.cfg.temp, dim=1)
            teacher_p = torch.softmax(teacher_sim / self.cfg.temp, dim=-1)

            # Loss is cross-entropy between teacher and student similarity distributions
            loss = -(teacher_p * enc_log_p).nansum(-1).mean()

        self.log(
            "train/loss",
            loss,
            on_epoch=False,
            on_step=True,
            sync_dist=True,
        )

        return loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):

        z_enc, dlc_probs = self.encoder(
            batch["input_ids_enc"],
            batch["attention_mask_enc"],
            step=self.global_step,
        )

        with torch.no_grad():
            z_teacher = self.teacher(
                batch["input_ids_teacher"],
                batch["attention_mask_teacher"],
                step=self.global_step,
            )

        with torch.autocast(device_type="cuda", dtype=torch.float32):
            enc_sim = cosine_sim_mat(z_enc.squeeze(1))
            teacher_sim = cosine_sim_mat(z_teacher.squeeze(1))

            # Remove the diagonal elements (self-similarity)
            N = enc_sim.size(0)
            offdiag_mask = ~torch.eye(N, dtype=bool)
            enc_sim = enc_sim[offdiag_mask].reshape(N, N - 1)
            teacher_sim = teacher_sim[offdiag_mask].reshape(N, N - 1)

            enc_log_p = torch.log_softmax(enc_sim / self.cfg.temp, dim=1)
            teacher_p = torch.softmax(teacher_sim / self.cfg.temp, dim=-1)

            # Loss is cross-entropy between teacher and student similarity distributions
            loss = -(teacher_p * enc_log_p).nansum(-1).mean()

        self.log("val/loss", loss, on_epoch=True, sync_dist=True)

        if isinstance(dlc_probs, list):
            # Flatten levels l1 = p(x_0), l2 = p(x_0, x_1), ...
            levels = [einx.rearrange("b L N V -> b L (N V)", l) for l in dlc_probs]
            ent = sum(
                [
                    compute_entropy(
                        level,
                        normalized=True,
                    ).mean()
                    for level in levels
                ]
            ) / len(levels)
            m_ent = sum(
                [
                    compute_entropy(level.mean(0), normalized=True).mean()
                    for level in levels
                ]
            ) / len(levels)
        else:
            ent = compute_entropy(dlc_probs, normalized=True).mean()
            m_ent = compute_entropy(dlc_probs.mean(0), normalized=True).mean()

        self.log(
            "val/sem_entropy",
            ent.detach().item(),
            on_epoch=True,
            sync_dist=True,
        )
        self.log(
            "val/sem_marginal_entropy",
            m_ent.detach().item(),
            on_epoch=True,
            sync_dist=True,
        )
