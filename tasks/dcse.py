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
from tasks.autoencoder import AETask, InfiniteDistributedUniformSampler, compute_entropy
from data.datasets import WikipediaDataset
from tqdm import tqdm
from mauve import compute_mauve, get_features_from_input
import einx
from transformers import get_constant_schedule_with_warmup
from data.datasets import DATA_SEED
from copy import deepcopy
import torch.nn.functional as F
from tasks.utils import *


@dataclass
class DCSETaskConfig:
    lr: float
    batch_size: int
    encoder: EncoderConfig
    teacher: EncoderConfig
    dataset: dict
    val_size: int
    lr_warmup_steps: float = 1500
    temp: float = 0.01
    delta_ent: float = 0.5
    delta_ent_warmup_steps: Optional[int] = None

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
        self.train_indices, self.val_indices = self.dataset.get_train_val_indices(
            val_size=cfg.val_size
        )

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
        self.teacher.compile()

    def setup(self, **kwargs):
        """Setup the data"""
        self.train_data = Subset(self.dataset, indices=self.train_indices)
        self.val_data = Subset(self.dataset, indices=self.val_indices)

    def get_collate_fn(self):
        # Just a big buffer, should never reach that
        max_length = 150

        def fn(batch):

            input_str = batch["input_str"]

            # Compute the input ids for the encoder
            batch_enc = self.encoder.tokenizer.batch_encode_plus(
                input_str,
                padding=True,
                truncation=True,
                add_special_tokens=True,
                max_length=max_length,
                return_tensors="pt",
            )

            if self.teacher.cfg.prompt is not None:
                input_str = [self.teacher.cfg.prompt + s for s in input_str]

            # Compute the input ids for the teacher
            batch_teacher = self.teacher.tokenizer.batch_encode_plus(
                input_str,
                truncation=True,
                padding=True,
                max_length=max_length,
                return_tensors="pt",
            )

            return {
                "input_ids_enc": batch_enc["input_ids"],
                "attention_mask_enc": batch_enc["attention_mask"].bool(),
                "input_ids_teacher": batch_teacher["input_ids"],
                "attention_mask_teacher": batch_teacher["attention_mask"].bool(),
            }

        return fn

    def train_dataloader(self):

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
                    for n, p in self.encoder.named_parameters()
                    if not any(nd in n.lower() for nd in no_decay)
                ],
                "weight_decay": 0.01,
            },
            {
                "params": [
                    p
                    for n, p in self.encoder.named_parameters()
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

    @torch.autocast(device_type="cuda", dtype=torch.float32)
    def dcse_loss(self, z_student, z_teacher):

        bs = z_student.size(0)

        # Compute cosine similarity matrices
        sim_student = torch.nn.functional.cosine_similarity(
            z_student[:, None], z_student[None], dim=-1
        )
        sim_teacher = torch.nn.functional.cosine_similarity(
            z_teacher[:, None], z_teacher[None], dim=-1
        )

        # Remove the diagonal elements (self-similarity)
        offdiag_mask = ~torch.eye(bs, dtype=bool)
        sim_student = sim_student[offdiag_mask].reshape(bs, bs - 1)
        sim_teacher = sim_teacher[offdiag_mask].reshape(bs, bs - 1)

        # Compute cross entropy
        loss = torch.nansum(
            -(
                torch.softmax(sim_teacher / self.cfg.temp, dim=-1)
                * torch.log_softmax(sim_student / self.cfg.temp, dim=-1)
            ),
            dim=-1,
        )

        return loss.mean()

    def training_step(self, batch, batch_idx):

        z_student, dlc_probs = self.encoder(
            batch["input_ids_enc"],
            batch["attention_mask_enc"],
        )

        with torch.no_grad():
            z_teacher = self.teacher(
                batch["input_ids_teacher"],
                batch["attention_mask_teacher"],
            )

        loss = self.dcse_loss(z_student, z_teacher)

        self.log(
            "train/loss",
            loss,
            on_epoch=False,
            on_step=True,
            sync_dist=True,
        )

        if self.cfg.delta_ent_warmup_steps is not None:
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

        z_student, dlc_probs = self.encoder(
            batch["input_ids_enc"],
            batch["attention_mask_enc"],
        )

        z_teacher = self.teacher(
            batch["input_ids_teacher"],
            batch["attention_mask_teacher"],
        )

        loss = self.dcse_loss(z_student, z_teacher)

        self.log("val/loss", loss, on_epoch=True, sync_dist=True)

        ent, m_ent = sem_entropy(dlc_probs)

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
