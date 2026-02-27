import random
from dataclasses import dataclass, field
from typing import List, Optional

import einx
import lightning as L
import torch
import torch.nn.functional as F
from lightning.pytorch.utilities.rank_zero import rank_zero_only
from torch.utils.data.dataset import Subset
from transformers import get_cosine_schedule_with_warmup
from omegaconf import OmegaConf

from data import SimCSEIterable, LanguageDataset, LanguageDatasetConfig
from model.encoder import EncoderConfig, EncoderModel
from tasks.utils import eval_mteb, sem_entropy, SEMUsageTracker
from pytorch_metric_learning.losses import NTXentLoss


@dataclass
class SimCSETaskConfig:
    lr: float
    lr_warmup_steps: int
    batch_size: int
    encoder: EncoderConfig
    dataset: LanguageDatasetConfig
    min_span_length: int = 32
    max_span_length: int = 384
    loss_temp: float = 0.05
    sem_noise: float = 0.0
    dropout: float = 0.15
    mteb_tasks: Optional[List[str]] = None
    mteb_batch_size: int = 256
    mteb_limit: Optional[int] = None

    name: Optional[str] = None


class SimCSETask(L.LightningModule):
    """
    Train an Encoder with the SimCSE objective:
    encode each sentence twice with different dropout masks to form positive pairs,
    use in-batch negatives with NT-Xent loss.
    """

    def __init__(self, cfg: Optional[SimCSETaskConfig] = None, **kwargs):
        super().__init__()

        if cfg is None:
            cfg = OmegaConf.to_object(
                OmegaConf.merge(
                    OmegaConf.create(SimCSETaskConfig),
                    OmegaConf.create(kwargs),
                )
            )

        self.encoder = EncoderModel(cfg.encoder)

        # Override all dropout layers in the backbone with the configured rate
        import torch.nn as nn
        for module in self.encoder.transformer.modules():
            if isinstance(module, nn.Dropout):
                module.p = cfg.dropout

        if cfg.encoder.sem is not None:
            self.sem_usage_ema = SEMUsageTracker()

        self.loss_fn = NTXentLoss(temperature=cfg.loss_temp)

        self.cfg = cfg

        self.save_hyperparameters(
            OmegaConf.to_container(OmegaConf.structured(cfg)), logger=False
        )

    def _encode(self, input_ids, attention_mask, **kwargs):
        return self.encoder(input_ids=input_ids, attention_mask=attention_mask, **kwargs)

    def _compute_loss(self, z1, z2):
        embeddings = torch.cat([z1, z2], dim=0)
        labels = torch.arange(z1.size(0), device=z1.device, dtype=torch.long).repeat(2)
        return self.loss_fn(embeddings, labels)

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
                optimizer,
                num_warmup_steps=self.cfg.lr_warmup_steps,
                num_training_steps=20000,
            )
            scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
            return [optimizer], [scheduler]
        else:
            return optimizer

    def _get_dataloader(self, dataset, seed):
        return SimCSEIterable.get_dataloader(
            dataset=dataset,
            tokenizer=self.encoder.tokenizer,
            batch_size=self.cfg.batch_size,
            min_span_length=self.cfg.min_span_length,
            max_span_length=self.cfg.max_span_length,
            seed=seed,
        )

    def train_dataloader(self):
        return self._get_dataloader(self.train_data, seed=random.randint(0, 100000))

    def val_dataloader(self):
        return self._get_dataloader(self.val_data, seed=42)

    def training_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        local_batch_size = input_ids.shape[0]

        # Encode the same batch twice — dropout creates different views (SimCSE)
        z1, sem_out = self._encode(
            input_ids,
            attention_mask,
            return_count=(self.cfg.encoder.sem is not None),
            noise=self.cfg.sem_noise,
        )
        z2, _ = self._encode(
            input_ids,
            attention_mask,
            noise=self.cfg.sem_noise,
        )

        if self.trainer.num_devices > 1:
            z1 = einx.rearrange("w b d -> (w b) d", self.all_gather(z1, sync_grads=True))
            z2 = einx.rearrange("w b d -> (w b) d", self.all_gather(z2, sync_grads=True))

        loss = self._compute_loss(z1, z2)
        self.log("train/loss", loss, on_step=True, on_epoch=False, sync_dist=True)

        if self.cfg.encoder.sem is not None:
            self.sem_usage_ema.update(sem_out["usage_count"], batch_size=local_batch_size)

        return loss

    def validation_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]

        # Keep encoder in train mode so dropout is active (otherwise z1 == z2)
        self.encoder.train()

        temps = [1e-4, None] if self.cfg.encoder.sem is not None else [None]
        names = ["hard", "soft"] if self.cfg.encoder.sem is not None else ["soft"]

        for idx, (sem_temp, label) in enumerate(zip(temps, names)):
            z1, sem_out = self._encode(
                input_ids,
                attention_mask,
                temp=sem_temp,
                return_count=(idx == 0 and self.cfg.encoder.sem is not None),
            )
            z2, _ = self._encode(input_ids, attention_mask, temp=sem_temp)

            if "probs" in sem_out:
                ent, m_ent = sem_entropy(sem_out["probs"])
                self.log("val/sem_entropy", ent.item(), on_epoch=True, sync_dist=True)
                self.log("val/sem_marginal_entropy", m_ent.item(), on_epoch=True, sync_dist=True)

            if self.trainer.num_devices > 1:
                z1 = einx.rearrange("w b d -> (w b) d", self.all_gather(z1, sync_grads=False))
                z2 = einx.rearrange("w b d -> (w b) d", self.all_gather(z2, sync_grads=False))

            loss = self._compute_loss(z1, z2)
            self.log(f"val/loss_{label}", loss, on_step=False, on_epoch=True, sync_dist=True)

        if (batch_idx == 0) and (rank_zero_only.rank == 0):
            if self.cfg.encoder.sem is not None:
                is_dead = self.sem_usage_ema.usage < 1e-4
                self.log("val/dead_words_ratio", torch.sum(is_dead).item() / is_dead.numel(),
                         on_epoch=True, sync_dist=False, rank_zero_only=True)

            if self.cfg.mteb_tasks:
                mteb_scores = eval_mteb(
                    encoder=self.encoder,
                    tasks=self.cfg.mteb_tasks,
                    batch_size=self.cfg.mteb_batch_size,
                    limit=self.cfg.mteb_limit,
                    device=str(self.device),
                )
                by_mode: dict[str, list[float]] = {}
                for key, score in mteb_scores.items():
                    mode = key.split("/")[-1]
                    by_mode.setdefault(mode, []).append(score)
                for mode, scores in by_mode.items():
                    self.log(f"mteb/mean/{mode}", sum(scores) / len(scores),
                             on_epoch=True, sync_dist=False, rank_zero_only=True)