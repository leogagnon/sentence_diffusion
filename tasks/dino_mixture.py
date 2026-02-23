from dataclasses import dataclass
from typing import Optional

import einx
import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning.pytorch.utilities.rank_zero import rank_zero_only
from torch.utils.data.dataset import Subset
from transformers import get_cosine_schedule_with_warmup
from omegaconf import OmegaConf

from tasks.utils import SEMUsageTracker, sem_entropy


# ---------------------------------------------------------------------------
# DINOv2-style projection head
# ---------------------------------------------------------------------------

def _build_mlp(nlayers, in_dim, bottleneck_dim, hidden_dim):
    if nlayers == 1:
        return nn.Linear(in_dim, bottleneck_dim, bias=True)
    layers = [nn.Linear(in_dim, hidden_dim, bias=True), nn.GELU()]
    for _ in range(nlayers - 2):
        layers += [nn.Linear(hidden_dim, hidden_dim, bias=True), nn.GELU()]
    layers.append(nn.Linear(hidden_dim, bottleneck_dim, bias=True))
    return nn.Sequential(*layers)


class DINOHead(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=2048, bottleneck_dim=256, nlayers=3):
        super().__init__()
        self.mlp = _build_mlp(nlayers, in_dim, bottleneck_dim, hidden_dim)
        self.last_layer = nn.utils.weight_norm(
            nn.Linear(bottleneck_dim, out_dim, bias=False)
        )
        self.last_layer.weight_g.data.fill_(1)
        self.last_layer.weight_g.requires_grad = False
        self._init_weights()

    def _init_weights(self):
        for m in self.mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    @torch.autocast("cuda", enabled=False)
    def forward(self, x):
        x = x.float()
        x = self.mlp(x)
        x = F.normalize(x, dim=-1, eps=1e-12)
        return self.last_layer(x)

from data import (
    DeCLUTRIterable,
    LanguageDataset,
    LanguageDatasetConfig,
    MixedItemIterable,
    PairDatasetIterable,
    SameLabelPairIterable,
    load_anli_pairs,
    load_ibm_argq_groups,
    load_paws_pairs,
    load_vitaminc_pairs,
    load_yelp_polarity_groups,
)
from model.encoder import EncoderConfig, EncoderModel


@dataclass
class DINOMixtureTaskConfig:
    lr: float
    lr_warmup_steps: int
    batch_size: int
    encoder: EncoderConfig
    declutr_dataset: LanguageDatasetConfig

    # DeCLUTR span parameters (num_anchors=1, num_positives=1 fixed)
    declutr_min_span_length: int = 128
    declutr_max_span_length: int = 256

    # Max token length for pair datasets
    max_length: int = 128

    # DINO hyperparameters
    num_prototypes: int = 4096
    head_hidden_dim: int = 2048
    head_bottleneck_dim: int = 256
    head_nlayers: int = 3
    student_temp: float = 0.1
    teacher_temp: float = 0.04
    center_momentum: float = 0.9
    teacher_momentum: float = 0.996

    sem_noise: float = 0.0

    dataset_seed: int = 42
    name: Optional[str] = None


class DINOMixtureTask(L.LightningModule):
    """
    Multi-task self-distillation (DINO) over a uniform mixture of:
      - DeCLUTR on FineWeb  (adjacent span pairs, num_anchors=1 num_positives=1)
      - VitaminC, ANLI, PAWS (explicit entailment/paraphrase pairs)
      - Yelp Polarity, IBM ArgQ (same-label pairs)

    DINO's loss is pairwise — H(teacher(positive), student(anchor)) — with no
    explicit in-batch negative comparison.  Collapse is prevented by centering +
    sharpening + EMA teacher, all of which are dataset-agnostic.  This allows
    fully heterogeneous batches without the DDP synchronization overhead required
    by InfoNCE.
    """

    def __init__(self, cfg: Optional[DINOMixtureTaskConfig] = None, **kwargs):
        super().__init__()

        if cfg is None:
            cfg = OmegaConf.to_object(
                OmegaConf.merge(
                    OmegaConf.create(DINOMixtureTaskConfig),
                    OmegaConf.create(kwargs),
                )
            )

        self.cfg = cfg

        if cfg.encoder.sem is not None:
            self.sem_usage_ema = SEMUsageTracker()

        # Student
        self.encoder = EncoderModel(cfg.encoder)
        self.proto_head = DINOHead(
            in_dim=cfg.encoder.latent_dim,
            out_dim=cfg.num_prototypes,
            hidden_dim=cfg.head_hidden_dim,
            bottleneck_dim=cfg.head_bottleneck_dim,
            nlayers=cfg.head_nlayers,
        )

        # Teacher (EMA of student — weights copied in on_fit_start)
        self.teacher_encoder = EncoderModel(cfg.encoder)
        self.teacher_proto_head = DINOHead(
            in_dim=cfg.encoder.latent_dim,
            out_dim=cfg.num_prototypes,
            hidden_dim=cfg.head_hidden_dim,
            bottleneck_dim=cfg.head_bottleneck_dim,
            nlayers=cfg.head_nlayers,
        )
        for p in self.teacher_encoder.parameters():
            p.requires_grad = False
        for p in self.teacher_proto_head.parameters():
            p.requires_grad = False

        # Centering buffer (EMA of teacher batch means, prevents collapse)
        self.register_buffer("center", torch.zeros(cfg.num_prototypes))

        self.save_hyperparameters(
            OmegaConf.to_container(OmegaConf.structured(cfg)), logger=False
        )

    # ------------------------------------------------------------------
    # Teacher initialisation and EMA update
    # ------------------------------------------------------------------

    def on_fit_start(self):
        # Both encoders load the same pretrained backbone weights, but the
        # randomly-initialised proto_heads start different — sync them here.
        with torch.no_grad():
            for p_s, p_t in zip(self.encoder.parameters(), self.teacher_encoder.parameters()):
                p_t.data.copy_(p_s.data)
            for p_s, p_t in zip(self.proto_head.parameters(), self.teacher_proto_head.parameters()):
                p_t.data.copy_(p_s.data)

    def on_train_batch_end(self, outputs, batch, batch_idx):
        # EMA: teacher ← m * teacher + (1 - m) * student
        m = self.cfg.teacher_momentum
        with torch.no_grad():
            for p_s, p_t in zip(self.encoder.parameters(), self.teacher_encoder.parameters()):
                p_t.data.mul_(m).add_(p_s.data * (1 - m))
            for p_s, p_t in zip(self.proto_head.parameters(), self.teacher_proto_head.parameters()):
                p_t.data.mul_(m).add_(p_s.data * (1 - m))

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------

    def dino_loss(self, student_logits, teacher_logits):
        student_log_probs = F.log_softmax(student_logits / self.cfg.student_temp, dim=-1)
        teacher_probs = F.softmax(
            (teacher_logits - self.center) / self.cfg.teacher_temp, dim=-1
        )
        return -torch.sum(teacher_probs * student_log_probs, dim=-1).mean()

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------

    def setup(self, stage=None):
        fineweb = LanguageDataset(self.cfg.declutr_dataset)
        self.train_fineweb = Subset(fineweb, indices=fineweb.train_indices)
        self.val_fineweb = Subset(fineweb, indices=fineweb.val_indices)

        vitaminc_anchors, vitaminc_positives = load_vitaminc_pairs()
        anli_anchors, anli_positives = load_anli_pairs()
        paws_anchors, paws_positives = load_paws_pairs()
        yelp_groups = load_yelp_polarity_groups()
        ibm_groups = load_ibm_argq_groups()

        tok = self.encoder.tokenizer
        seed = self.cfg.dataset_seed

        self.train_iterables = [
            # 1. DeCLUTR on FineWeb (1 anchor, 1 positive per item)
            DeCLUTRIterable(
                self.train_fineweb,
                tokenizer=tok,
                min_span_length=self.cfg.declutr_min_span_length,
                max_span_length=self.cfg.declutr_max_span_length,
                num_anchors=1,
                num_positives=1,
                adjacent_positives=True,
                masked_anchors=False,
                seed=seed,
            ),
            # 2. VitaminC
            PairDatasetIterable(
                vitaminc_anchors, vitaminc_positives,
                tokenizer=tok, max_length=self.cfg.max_length, seed=seed + 1,
            ),
            # 3. ANLI
            PairDatasetIterable(
                anli_anchors, anli_positives,
                tokenizer=tok, max_length=self.cfg.max_length, seed=seed + 2,
            ),
            # 4. PAWS
            PairDatasetIterable(
                paws_anchors, paws_positives,
                tokenizer=tok, max_length=self.cfg.max_length, seed=seed + 3,
            ),
            # 5. Yelp Polarity
            SameLabelPairIterable(
                yelp_groups, tokenizer=tok, max_length=self.cfg.max_length, seed=seed + 4,
            ),
            # 6. IBM ArgQ
            SameLabelPairIterable(
                ibm_groups, tokenizer=tok, max_length=self.cfg.max_length, seed=seed + 5,
            ),
        ]

    def configure_optimizers(self):
        no_decay = ["bias", "norm"]
        all_params = (
            list(self.encoder.named_parameters())
            + list(self.proto_head.named_parameters())
        )
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in all_params if not any(nd in n.lower() for nd in no_decay)],
                "weight_decay": 0.1,
            },
            {
                "params": [p for n, p in all_params if any(nd in n.lower() for nd in no_decay)],
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
        return MixedItemIterable.get_dataloader(
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
            num_anchors=1,
            num_positives=1,
            adjacent_positives=False,
            masked_anchors=False,
            seed=42,
        )

    # ------------------------------------------------------------------
    # Train / val steps
    # ------------------------------------------------------------------

    def training_step(self, batch, batch_idx):
        anchor_ids = batch["anchor_ids"]
        positive_ids = batch["positive_ids"]
        anchor_mask = (anchor_ids != self.encoder.tokenizer.pad_token_id).long()
        positive_mask = (positive_ids != self.encoder.tokenizer.pad_token_id).long()

        # Student sees BOTH views
        z_s_anc, sem_out = self.encoder(
            input_ids=anchor_ids,
            attention_mask=anchor_mask,
            return_count=True,
            noise=self.cfg.sem_noise,
        )
        z_s_pos, _ = self.encoder(
            input_ids=positive_ids,
            attention_mask=positive_mask,
            noise=self.cfg.sem_noise,
        )
        s_logits_anc = self.proto_head(z_s_anc)
        s_logits_pos = self.proto_head(z_s_pos)

        # Teacher sees BOTH views (no gradient)
        with torch.no_grad():
            z_t_anc, _ = self.teacher_encoder(
                input_ids=anchor_ids, attention_mask=anchor_mask
            )
            z_t_pos, _ = self.teacher_encoder(
                input_ids=positive_ids, attention_mask=positive_mask
            )
            t_logits_anc = self.teacher_proto_head(z_t_anc)
            t_logits_pos = self.teacher_proto_head(z_t_pos)

        # Cross-view loss only (exclude same-view to avoid trivial solution)
        loss = (self.dino_loss(s_logits_anc, t_logits_pos) + self.dino_loss(s_logits_pos, t_logits_anc)) / 2
        self.log("train/loss", loss, on_step=True, on_epoch=False, sync_dist=True)

        if self.cfg.encoder.sem is not None:
            self.sem_usage_ema.update(
                sem_out["usage_count"], batch_size=anchor_ids.shape[0]
            )

        # Update centering buffer with ALL teacher logits
        with torch.no_grad():
            all_teacher_logits = torch.cat([t_logits_anc, t_logits_pos], dim=0)
            if self.trainer.world_size > 1:
                all_teacher = einx.rearrange(
                    "w b k -> (w b) k", self.all_gather(all_teacher_logits)
                )
                batch_center = all_teacher.mean(0)
            else:
                batch_center = all_teacher_logits.mean(0)
            self.center.mul_(self.cfg.center_momentum).add_(
                batch_center * (1 - self.cfg.center_momentum)
            )

        return loss

    def validation_step(self, batch, batch_idx):
        anchor_ids = batch["anchor_ids"]
        positive_ids = batch["positive_ids"]
        anchor_mask = (anchor_ids != self.encoder.tokenizer.pad_token_id).long()
        positive_mask = (positive_ids != self.encoder.tokenizer.pad_token_id).long()

        temps = [1e-4, None] if self.cfg.encoder.sem is not None else [None]
        labels = ["hard", "soft"] if self.cfg.encoder.sem is not None else ["soft"]

        for sem_temp, label in zip(temps, labels):
            z_student, sem_out = self.encoder(
                input_ids=anchor_ids, attention_mask=anchor_mask, temp=sem_temp
            )
            student_logits = self.proto_head(z_student)

            with torch.no_grad():
                z_teacher, _ = self.teacher_encoder(
                    input_ids=positive_ids, attention_mask=positive_mask, temp=sem_temp
                )
                teacher_logits = self.teacher_proto_head(z_teacher)

            if self.cfg.encoder.sem is not None and "probs" in sem_out:
                ent, m_ent = sem_entropy(sem_out["probs"])
                self.log("val/sem_entropy", ent.item(), on_epoch=True, sync_dist=True)
                self.log(
                    "val/sem_marginal_entropy", m_ent.item(), on_epoch=True, sync_dist=True
                )

            loss = self.dino_loss(student_logits, teacher_logits)
            self.log(
                f"val/loss_{label}", loss, on_step=False, on_epoch=True, sync_dist=True
            )

        if (batch_idx == 0) and (rank_zero_only.rank == 0):
            if self.cfg.encoder.sem is not None and self.sem_usage_ema.usage is not None:
                is_dead = self.sem_usage_ema.usage < 1e-4
                dead_words_ratio = torch.sum(is_dead).item() / is_dead.numel()
                dead_simplices = torch.sum((~is_dead).sum(1) == 1).item()
                dead_words_per_simplex = torch.sum(is_dead, dim=1).float().mean().item()
                self.log(
                    "val/dead_words_ratio", dead_words_ratio,
                    on_epoch=True, sync_dist=False, rank_zero_only=True,
                )
                self.log(
                    "val/dead_simplices", dead_simplices,
                    on_epoch=True, sync_dist=False, rank_zero_only=True,
                )
                self.log(
                    "val/dead_words_per_simplex", dead_words_per_simplex,
                    on_epoch=True, sync_dist=False, rank_zero_only=True,
                )
