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
from data.datasets import WikipediaDataset, FineWebDataset
from tqdm import tqdm
from mauve import compute_mauve, get_features_from_input
import einx
from transformers import get_constant_schedule_with_warmup
from data.datasets import DATA_SEED
from copy import deepcopy
import torch.nn.functional as F
from tasks.utils import *
import torch.distributed as dist
import torch.nn as nn


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
    delta_ent: float = 0.0
    delta_margin: float = 0.0
    reg_warmup_steps: int = 10000
    ce_loss: bool = True
    sem_reset: bool = False
    sem_reset_schedule: Optional[List[int]] = None
    sem_reset_threshold: float = 1e-4
    reg_type: str = "none"

    name: Optional[str] = None


class DCSETask(L.LightningModule):
    """
    Train a SEM encoder by Distilling a pretrained Constrative Sentence Embedding (DCSE)
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
        self.dataset: WikipediaDataset | FineWebDataset

        # This is with a fixed seed to make sure validation set never changes
        self.train_indices, self.val_indices = self.dataset.get_train_val_indices(
            val_size=cfg.val_size
        )

        if cfg.sem_reset:
            self.sem_usage_ema = SEMUsageTracker()

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
        max_length = self.dataset.max_length

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

            # Because some teacher sentence embedding use instructions
            if self.teacher.cfg.prompt is not None:
                input_str_teacher = [self.teacher.cfg.prompt + s for s in input_str]
            else:
                input_str_teacher = input_str

            # Compute the input ids for the teacher
            batch_teacher = self.teacher.tokenizer.batch_encode_plus(
                input_str_teacher,
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

    
    def dcse_loss(self, z_student, z_teacher):

        with torch.autocast(device_type="cuda", enabled=False):

            bs = z_student.size(0)
            
            # Make sure everything is in float32 for stability
            z_student = z_student.to(dtype=torch.float32)
            z_teacher = z_teacher.to(dtype=torch.float32)

            # Compute cosine similarity matrices of teacher/student
            sim_student = torch.nn.functional.cosine_similarity(
                z_student[:, None], z_student[None], dim=-1
            )
            sim_teacher = torch.nn.functional.cosine_similarity(
                z_teacher[:, None], z_teacher[None], dim=-1,
            )

            if self.cfg.ce_loss:
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
            else:
                torch.triu((sim_student - sim_teacher) ** 2)

            return loss.mean()

    def training_step(self, batch, batch_idx):

        z_student, sem_out = self.encoder(
            batch["input_ids_enc"],
            batch["attention_mask_enc"],
            return_count=hasattr(self, "sem_usage_ema"),
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

        reg = None
        if (self.cfg.delta_ent > 0.0) and (self.cfg.reg_type == "ent"):
            ent, m_ent = sem_entropy(sem_out["probs"])
            reg = self.cfg.delta_ent * (ent - 0.5*m_ent)
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
            self.sem_usage_ema.update(
                sem_out["usage_count"], batch_size=z_student.shape[0]
            )

        return loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):

        z_student, sem_out = self.encoder(
            batch["input_ids_enc"], batch["attention_mask_enc"], return_count=True
        )

        z_teacher = self.teacher(
            batch["input_ids_teacher"],
            batch["attention_mask_teacher"],
        )

        loss = self.dcse_loss(z_student, z_teacher)

        self.log("val/loss", loss, on_epoch=True, sync_dist=True)

        ent, m_ent = sem_entropy(sem_out["probs"])

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

        self.log(
            "val/dead_words",
            torch.sum(sem_out["usage_count"] == 0).item() / len(sem_out["usage_count"]),
            on_epoch=True,
            sync_dist=True,
        )

        self.log(
            "val/latent_norm",
            z_student.norm(p=2, dim=-1).mean().detach().item(),
            on_epoch=True,
            sync_dist=True,
        )

    def on_before_zero_grad(self, optimizer):
        if (
            self.cfg.sem_reset
            and (self.global_step >= self.cfg.sem_reset_schedule[0])
            and (self.global_step <= self.cfg.sem_reset_schedule[1])
            and (self.global_step % self.cfg.sem_reset_schedule[-1] == 0)
        ):
            self.trainer.strategy.barrier()

            with torch.no_grad():
                if rank_zero_only.rank == 0:

                    # Compute new weights (with dead vertices reset)
                    bound = 1 / (self.encoder.sem.cfg.input_dim**0.5)
                    dead_mask = self.sem_usage_ema.usage < self.cfg.sem_reset_threshold

                    self.encoder.sem.proj_in.weight[dead_mask] = (
                        self.encoder.sem.proj_in.weight[dead_mask].uniform_(
                            -bound, bound
                        )
                    )

                    self.encoder.sem.proj_out.weight[:, dead_mask] = (
                        self.encoder.sem.proj_out.weight[:, dead_mask].uniform_(
                            -bound, bound
                        )
                    )

                for param in self.encoder.sem.parameters():
                    self.trainer.strategy.broadcast(param.data, src=0)
