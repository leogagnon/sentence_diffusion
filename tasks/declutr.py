import random
from dataclasses import dataclass, field
from typing import Optional

import einx
import lightning as L
import torch
from lightning.pytorch.utilities.rank_zero import rank_zero_only
from torch.utils.data.dataset import Subset
from transformers import get_cosine_schedule_with_warmup
from omegaconf import OmegaConf

from data import DeCLUTRIterable, LanguageDataset, LanguageDatasetConfig
from model.encoder import EncoderConfig, EncoderModel
from tasks.utils import *
import torch.distributed as dist
from pytorch_metric_learning.losses import NTXentLoss


@dataclass
class SEMResetConfig:
    enabled: bool = False
    simplex_mode: bool = False
    threshold: float = 1e-4
    start: int = 5000
    end: int = 15000
    interval: int = 1000


@dataclass
class DeCLUTRTaskConfig:
    lr: float
    lr_warmup_steps: int
    batch_size: int
    encoder: EncoderConfig
    dataset: LanguageDatasetConfig
    min_span_length: int = 16
    max_span_length: int = 64
    num_anchors: int = 2
    num_positives: int = 2
    loss_temp: float = 0.05
    sem_noise: float = 0.0
    sem_reset_config: SEMResetConfig = field(default_factory=SEMResetConfig)

    name: Optional[str] = None


class DeCLUTRTask(L.LightningModule):
    """
    Train a SEM encoder with the DeCLUTR objective
    """

    def __init__(self, cfg: Optional[DeCLUTRTaskConfig] = None, **kwargs):
        super().__init__()

        if cfg == None:
            cfg = OmegaConf.to_object(
                OmegaConf.merge(
                    OmegaConf.create(DeCLUTRTaskConfig),
                    OmegaConf.create(kwargs),
                )
            )

        self.encoder = EncoderModel(cfg.encoder)

        if cfg.encoder.sem is not None:   
            self.sem_usage_ema = SEMUsageTracker()

        self.loss_fn = NTXentLoss(temperature=cfg.loss_temp)

        self.cfg = cfg

        self.save_hyperparameters(
            OmegaConf.to_container(OmegaConf.structured(cfg)), logger=False
        )

    def compute_loss(self, anchors, positives):
        # Format inputs for NTXentLoss (all embeddings, matching labels for positive pairs)
        embeddings = torch.cat((anchors, positives), dim=0)
        labels = torch.arange(
            anchors.size(0), device=anchors.device, dtype=torch.long
        ).repeat(2)

        # Compute NTXent loss
        loss = self.loss_fn(embeddings, labels)

        return loss

    def compile(self):
        if self.encoder is not None:
            self.encoder.compile()

    def setup(self, **kwargs):

        self.dataset = LanguageDataset(self.cfg.dataset)

        self.train_data = Subset(self.dataset, indices=self.dataset.train_indices)
        self.val_data = Subset(self.dataset, indices=self.dataset.val_indices)

    def configure_optimizers(self):
        no_decay = ["bias", "norm"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in self.encoder.named_parameters()
                    if not any(nd in n.lower() for nd in no_decay)
                ],
                "weight_decay": 0.1,
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
            scheduler = get_cosine_schedule_with_warmup(
                optimizer, num_warmup_steps=self.cfg.lr_warmup_steps, num_training_steps=20000
            )
            scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}

            return [optimizer], [scheduler]
        else:
            return optimizer

    def train_dataloader(self):
        return DeCLUTRIterable.get_dataloader(
            dataset=self.train_data,
            tokenizer=self.encoder.tokenizer,
            batch_size=self.cfg.batch_size,
            min_span_length=self.cfg.min_span_length,
            max_span_length=self.cfg.max_span_length,
            num_anchors=self.cfg.num_anchors,
            num_positives=self.cfg.num_positives,
            seed=random.randint(
                0, 100000
            ),  # Dataset should be different if restarted,,
        )

    def val_dataloader(self):
        return DeCLUTRIterable.get_dataloader(
            dataset=self.val_data,
            tokenizer=self.encoder.tokenizer,
            batch_size=self.cfg.batch_size,
            min_span_length=self.cfg.min_span_length,
            max_span_length=self.cfg.max_span_length,
            num_anchors=self.cfg.num_anchors,
            num_positives=self.cfg.num_positives,
            seed=32,
        )

    def training_step(self, batch, batch_idx):

        z_anchors, sem_out = self.encoder(
            input_ids=batch["anchor_ids"],
            attention_mask=(
                batch["anchor_ids"] != self.encoder.tokenizer.pad_token_id
            ).long(),
            return_count=True,
            noise=self.cfg.sem_noise
        )
        z_positives, _ = self.encoder(
            input_ids=batch["positive_ids"],
            attention_mask=(
                batch["positive_ids"] != self.encoder.tokenizer.pad_token_id
            ).long(),
            noise=self.cfg.sem_noise
        )
        # Group positives from the same anchor together and average their embeddings
        z_positives = einx.rearrange(
            "(b p) d -> b p d", z_positives, p=self.cfg.num_positives
        ).mean(dim=1)

        # If distributed, gather all representations
        if self.trainer.num_devices > 1:
            z_anchors = einx.rearrange(
                "w b d -> (w b) d", self.all_gather(z_anchors, sync_grads=True)
            )
            z_positives = einx.rearrange(
                "w b d -> (w b) d", self.all_gather(z_positives, sync_grads=True)
            )

        loss = self.compute_loss(z_anchors, z_positives)
        self.log("train/loss", loss, on_step=True, on_epoch=False, sync_dist=True)

        if self.cfg.encoder.sem is not None:
            self.sem_usage_ema.update(sem_out["usage_count"], batch_size=z_anchors.shape[0])

        return loss

    def validation_step(self, batch, batch_idx):
        
        temps = [1e-4, None] if self.cfg.encoder.sem is not None else [None]
        names = ["hard", "soft"] if self.cfg.encoder.sem is not None else ["soft"]
        for sem_temp, label in zip(temps, names):
            z_anchors, sem_out = self.encoder(
                input_ids=batch["anchor_ids"],
                attention_mask=(
                    batch["anchor_ids"] != self.encoder.tokenizer.pad_token_id
                ).long(),
                temp=sem_temp,
            )
            if "probs" in sem_out.keys():
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

            z_positives, _ = self.encoder(
                input_ids=batch["positive_ids"],
                attention_mask=(
                    batch["positive_ids"] != self.encoder.tokenizer.pad_token_id
                ).long(),
                temp=sem_temp,
            )
            z_positives = einx.rearrange(
                "(b p) d -> b p d", z_positives, p=self.cfg.num_positives
            ).mean(dim=1)

            if self.trainer.num_devices > 1:
                z_anchors = einx.rearrange(
                    "w b d -> (w b) d", self.all_gather(z_anchors, sync_grads=False)
                )
                z_positives = einx.rearrange(
                    "w b d -> (w b) d", self.all_gather(z_positives, sync_grads=False)
                )

            # NTXent loss
            loss = self.compute_loss(z_anchors, z_positives)
            self.log(
                f"val/loss_{label}", loss, on_step=False, on_epoch=True, sync_dist=True
            )

        if (batch_idx == 0) and (rank_zero_only.rank == 0):

            # Log % of dead words/simplices
            if self.cfg.encoder.sem is not None:
                is_dead = self.sem_usage_ema.usage < self.cfg.sem_reset_config.threshold
                dead_words_ratio = torch.sum(is_dead).item() / is_dead.numel()
                dead_simplices = torch.sum((~is_dead).sum(1) == 1).item()
                dead_words_per_simplex = torch.sum(is_dead, dim=1).float().mean().item()

                self.log(
                    "val/dead_words_ratio",
                    dead_words_ratio,
                    on_epoch=True,
                    sync_dist=False,
                    rank_zero_only=True,
                )
                self.log(
                    "val/dead_simplices",
                    dead_simplices,
                    on_epoch=True,
                    sync_dist=False,
                    rank_zero_only=True,
                )
                self.log(
                    "val/dead_words_per_simplex",
                    dead_words_per_simplex,
                    on_epoch=True,
                    sync_dist=False,
                    rank_zero_only=True,
                )
