import random
from dataclasses import dataclass, field
from typing import Optional

import einx
import lightning as L
import torch
import torch.nn.functional as F
from torch.utils.data.dataset import Subset
from transformers import get_cosine_schedule_with_warmup
from omegaconf import OmegaConf
from pytorch_metric_learning.losses import NTXentLoss

from data import (
    ContrastiveMixtureIterable,
    DeCLUTRIterable,
    LanguageDataset,
    LanguageDatasetConfig,
    PairDatasetIterable,
    SameLabelPairIterable,
    load_anli_pairs,
    load_ibm_argq_groups,
    load_paws_pairs,
    load_vitaminc_pairs,
    load_yelp_polarity_groups,
)
from model.encoder import EncoderConfig, EncoderModel
from tasks.utils import SEMUsageTracker, sem_entropy


@dataclass
class E5MixtureTaskConfig:
    lr: float
    lr_warmup_steps: int
    batch_size: int
    encoder: EncoderConfig
    declutr_dataset: LanguageDatasetConfig

    # DeCLUTR (FineWeb) span parameters
    declutr_min_span_length: int = 32
    declutr_max_span_length: int = 512
    declutr_num_anchors: int = 2
    declutr_num_positives: int = 2

    # Max token length for pair datasets (VitaminC, ANLI, PAWS, Yelp, IBM ArgQ)
    max_length: int = 512

    loss_temp: float = 0.01
    sem_noise: float = 0.0

    # Seed for the mixture selector (shared across ranks — do NOT rank-mix)
    dataset_seed: int = 42

    name: Optional[str] = None


class E5MixtureTask(L.LightningModule):
    """
    Multi-task contrastive training with a uniform mixture of:
      - DeCLUTR on FineWeb (span-level positives)
      - VitaminC  (claim / supporting evidence)
      - ANLI      (premise / entailed hypothesis)
      - PAWS      (paraphrase pairs)
      - Yelp Polarity (same-sentiment text pairs)
      - IBM ArgQ  (same-topic argument pairs)

    Each batch is drawn entirely from one dataset, chosen uniformly at random.
    All DDP ranks agree on the dataset choice via a shared deterministic seed
    (the selector RNG is NOT mixed with rank/worker id).
    """

    def __init__(self, cfg: Optional[E5MixtureTaskConfig] = None, **kwargs):
        super().__init__()

        if cfg is None:
            cfg = OmegaConf.to_object(
                OmegaConf.merge(
                    OmegaConf.create(E5MixtureTaskConfig),
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
        embeddings = torch.cat((anchors, positives), dim=0)
        labels = torch.arange(
            anchors.size(0), device=anchors.device, dtype=torch.long
        ).repeat(2)
        return self.loss_fn(embeddings, labels)

    def setup(self, stage=None):
        # ----- FineWeb (DeCLUTR) -----
        fineweb = LanguageDataset(self.cfg.declutr_dataset)
        self.train_fineweb = Subset(fineweb, indices=fineweb.train_indices)
        self.val_fineweb = Subset(fineweb, indices=fineweb.val_indices)

        # ----- Pair datasets -----
        vitaminc_anchors, vitaminc_positives = load_vitaminc_pairs()
        anli_anchors, anli_positives = load_anli_pairs()
        paws_anchors, paws_positives = load_paws_pairs()
        yelp_groups = load_yelp_polarity_groups()
        ibm_groups = load_ibm_argq_groups()

        tok = self.encoder.tokenizer
        seed = self.cfg.dataset_seed

        self.train_iterables = [
            # 1. DeCLUTR on FineWeb
            DeCLUTRIterable(
                self.train_fineweb,
                tokenizer=tok,
                min_span_length=self.cfg.declutr_min_span_length,
                max_span_length=self.cfg.declutr_max_span_length,
                num_anchors=self.cfg.declutr_num_anchors,
                num_positives=self.cfg.declutr_num_positives,
                adjacent_positives=False,
                masked_anchors=False,
                seed=seed,
            ),
            # 2. VitaminC
            PairDatasetIterable(
                vitaminc_anchors,
                vitaminc_positives,
                tokenizer=tok,
                max_length=self.cfg.max_length,
                seed=seed + 1,
            ),
            # 3. ANLI
            PairDatasetIterable(
                anli_anchors,
                anli_positives,
                tokenizer=tok,
                max_length=self.cfg.max_length,
                seed=seed + 2,
            ),
            # 4. PAWS
            PairDatasetIterable(
                paws_anchors,
                paws_positives,
                tokenizer=tok,
                max_length=self.cfg.max_length,
                seed=seed + 3,
            ),
            # 5. Yelp Polarity (same-label pairs)
            SameLabelPairIterable(
                yelp_groups,
                tokenizer=tok,
                max_length=self.cfg.max_length,
                seed=seed + 4,
            ),
            # 6. IBM ArgQ (same-topic pairs)
            SameLabelPairIterable(
                ibm_groups,
                tokenizer=tok,
                max_length=self.cfg.max_length,
                seed=seed + 5,
            ),
        ]

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

    def train_dataloader(self):
        return ContrastiveMixtureIterable.get_dataloader(
            iterables=self.train_iterables,
            batch_size=self.cfg.batch_size,
            tokenizer=self.encoder.tokenizer,
            seed=self.cfg.dataset_seed,
        )

    def val_dataloader(self):
        return DeCLUTRIterable.get_dataloader(
            dataset=self.val_fineweb,
            tokenizer=self.encoder.tokenizer,
            batch_size=self.cfg.batch_size,
            min_span_length=self.cfg.declutr_min_span_length,
            max_span_length=self.cfg.declutr_max_span_length,
            num_anchors=self.cfg.declutr_num_anchors,
            num_positives=self.cfg.declutr_num_positives,
            adjacent_positives=False,
            masked_anchors=False,
            seed=42,
        )

    def training_step(self, batch, batch_idx):
        z_anchors, sem_out = self.encoder(
            input_ids=batch["anchor_ids"],
            attention_mask=(
                batch["anchor_ids"] != self.encoder.tokenizer.pad_token_id
            ).long(),
            return_count=True,
            noise=self.cfg.sem_noise,
        )
        z_positives, _ = self.encoder(
            input_ids=batch["positive_ids"],
            attention_mask=(
                batch["positive_ids"] != self.encoder.tokenizer.pad_token_id
            ).long(),
            noise=self.cfg.sem_noise,
        )

        # Average positives per anchor (num_positives may be 1 for pair datasets)
        num_positives = z_positives.size(0) // z_anchors.size(0)
        if num_positives > 1:
            z_positives = einx.rearrange(
                "(b p) d -> b p d", z_positives, p=num_positives
            ).mean(dim=1)

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
            self.sem_usage_ema.update(sem_out["usage_count"], batch_size=batch["anchor_ids"].shape[0])

        return loss

    def validation_step(self, batch, batch_idx):
        temps = [1e-4, None] if self.cfg.encoder.sem is not None else [None]
        names = ["hard", "soft"] if self.cfg.encoder.sem is not None else ["soft"]

        for idx, (sem_temp, label) in enumerate(zip(temps, names)):
            z_anchors, sem_out = self.encoder(
                input_ids=batch["anchor_ids"],
                attention_mask=(
                    batch["anchor_ids"] != self.encoder.tokenizer.pad_token_id
                ).long(),
                temp=sem_temp,
            )
            if "probs" in sem_out:
                ent, m_ent = sem_entropy(sem_out["probs"])
                self.log("val/sem_entropy", ent.item(), on_epoch=True, sync_dist=True)
                self.log("val/sem_marginal_entropy", m_ent.item(), on_epoch=True, sync_dist=True)

            z_positives, _ = self.encoder(
                input_ids=batch["positive_ids"],
                attention_mask=(
                    batch["positive_ids"] != self.encoder.tokenizer.pad_token_id
                ).long(),
                temp=sem_temp,
            )
            num_positives = z_positives.size(0) // z_anchors.size(0)
            if num_positives > 1:
                z_positives = einx.rearrange(
                    "(b p) d -> b p d", z_positives, p=num_positives
                ).mean(dim=1)

            if self.trainer.num_devices > 1:
                z_anchors = einx.rearrange(
                    "w b d -> (w b) d", self.all_gather(z_anchors, sync_grads=False)
                )
                z_positives = einx.rearrange(
                    "w b d -> (w b) d", self.all_gather(z_positives, sync_grads=False)
                )

            loss = self.compute_loss(z_anchors, z_positives)
            self.log(f"val/loss_{label}", loss, on_step=False, on_epoch=True, sync_dist=True)
