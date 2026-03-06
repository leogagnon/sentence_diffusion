import random
from dataclasses import dataclass
from typing import List, Optional

import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning.pytorch.utilities.rank_zero import rank_zero_only
from transformers import get_cosine_schedule_with_warmup
from omegaconf import OmegaConf

from data import DistillPairDatasetIterable, load_snli_pairs, load_mnli_pairs
from model.encoder import EncoderConfig, EncoderModel
from tasks.utils import eval_mteb, sem_entropy, SEMUsageTracker


@dataclass
class DistillSimCSETaskConfig:
    lr: float
    lr_warmup_steps: int
    batch_size: int
    encoder: EncoderConfig
    teacher_model_name: str = "jinaai/jina-embeddings-v5-text-small-text-matching"
    sem_noise: float = 0.0
    max_length: int = 256
    teacher_max_length: int = 256
    mteb_tasks: Optional[List[str]] = None
    mteb_batch_size: int = 256
    mteb_limit: Optional[int] = None
    name: Optional[str] = None


class DistillSimCSETask(L.LightningModule):
    """
    Phase-1 distillation: student encoder trained to match frozen Jina teacher
    embeddings via per-sample cosine alignment on SNLI+MNLI entailment pairs.

    Loss: mean(1 - cos_sim(ψ(z_s_anchor), z_t_anchor))
        + mean(1 - cos_sim(ψ(z_s_positive), z_t_positive))

    ψ is a trainable nn.Linear(student_dim, teacher_dim).
    No in-batch negatives — no all_gather required.
    Optional STE for the SEM head (controlled by encoder.sem.use_ste).
    """

    def __init__(self, cfg: Optional[DistillSimCSETaskConfig] = None, **kwargs):
        super().__init__()

        if cfg is None:
            cfg = OmegaConf.to_object(
                OmegaConf.merge(
                    OmegaConf.create(DistillSimCSETaskConfig),
                    OmegaConf.create(kwargs),
                )
            )

        # Student encoder (backbone + optional SEM + out_proj)
        self.encoder = EncoderModel(cfg.encoder)

        if cfg.encoder.sem is not None:
            self.sem_usage_ema = SEMUsageTracker()

        # Teacher encoder (frozen Jina model, last-token pooling, no out_proj)
        teacher_cfg = EncoderConfig(
            model_name=cfg.teacher_model_name,
            train_backbone=False,
            last_token_pooling=True,
            no_out_proj=True,
            latent_length=1,
        )
        self.teacher_encoder = EncoderModel(teacher_cfg)
        for p in self.teacher_encoder.parameters():
            p.requires_grad = False

        # Distillation projection: student_dim → teacher_dim
        # cfg.encoder.latent_dim is set correctly by EncoderModel.__init__
        student_dim = cfg.encoder.latent_dim
        teacher_dim = self.teacher_encoder.cfg.latent_dim
        self.proj = nn.Linear(student_dim, teacher_dim, bias=False)

        self._teacher_mteb_done = False

        self.cfg = cfg

        self.save_hyperparameters(
            OmegaConf.to_container(OmegaConf.structured(cfg)), logger=False
        )

    def compile(self):
        if self.encoder is not None:
            self.encoder.compile()

    def setup(self, stage=None):
        train_a, train_p = load_snli_pairs("train")
        mnli_a, mnli_p = load_mnli_pairs("train")
        self.train_pairs = (train_a + mnli_a, train_p + mnli_p)

        val_a, val_p = load_snli_pairs("validation")
        mnli_val_a, mnli_val_p = load_mnli_pairs("validation_matched")
        self.val_pairs = (val_a + mnli_val_a, val_p + mnli_val_p)

    def configure_optimizers(self):
        no_decay = ["bias", "norm"]
        all_params = (
            list(self.encoder.named_parameters())
            + list(self.proj.named_parameters())
        )
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in all_params
                    if not any(nd in n.lower() for nd in no_decay)
                ],
                "weight_decay": 0.1,
            },
            {
                "params": [
                    p for n, p in all_params
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

    def _get_dataloader(self, anchors, positives, seed):
        return DistillPairDatasetIterable.get_dataloader(
            anchors=anchors,
            positives=positives,
            student_tokenizer=self.encoder.tokenizer,
            teacher_tokenizer=self.teacher_encoder.tokenizer,
            batch_size=self.cfg.batch_size,
            student_max_length=self.cfg.max_length,
            teacher_max_length=self.cfg.teacher_max_length,
            seed=seed,
        )

    def train_dataloader(self):
        return self._get_dataloader(*self.train_pairs, seed=random.randint(0, 100000))

    def val_dataloader(self):
        return self._get_dataloader(*self.val_pairs, seed=42)

    @staticmethod
    def _distill_loss(z_student_proj: torch.Tensor, z_teacher: torch.Tensor) -> torch.Tensor:
        """Mean (1 - cosine_similarity) over the batch."""
        return 1.0 - F.cosine_similarity(z_student_proj, z_teacher, dim=-1).mean()

    def training_step(self, batch, batch_idx):
        anchor_ids     = batch["anchor_ids"]
        positive_ids   = batch["positive_ids"]
        t_anchor_ids   = batch["teacher_anchor_ids"]
        t_positive_ids = batch["teacher_positive_ids"]

        anchor_mask   = (anchor_ids   != self.encoder.tokenizer.pad_token_id).long()
        positive_mask = (positive_ids != self.encoder.tokenizer.pad_token_id).long()
        t_anchor_mask   = (t_anchor_ids   != self.teacher_encoder.tokenizer.pad_token_id).long()
        t_positive_mask = (t_positive_ids != self.teacher_encoder.tokenizer.pad_token_id).long()

        # Student forward (STE active if configured and encoder is in train mode)
        z_s_anc, sem_out = self.encoder(
            input_ids=anchor_ids,
            attention_mask=anchor_mask,
            return_count=(self.cfg.encoder.sem is not None),
            noise=self.cfg.sem_noise,
        )
        z_s_pos, _ = self.encoder(
            input_ids=positive_ids,
            attention_mask=positive_mask,
            noise=self.cfg.sem_noise,
        )

        # Teacher forward (frozen, no grad)
        with torch.no_grad():
            z_t_anc, _ = self.teacher_encoder(
                input_ids=t_anchor_ids,
                attention_mask=t_anchor_mask,
            )
            z_t_pos, _ = self.teacher_encoder(
                input_ids=t_positive_ids,
                attention_mask=t_positive_mask,
            )

        # Distillation loss (cast to float32 for numerical stability)
        loss = (
            self._distill_loss(self.proj(z_s_anc.float()), z_t_anc.float())
            + self._distill_loss(self.proj(z_s_pos.float()), z_t_pos.float())
        )

        self.log("train/loss", loss, on_step=True, on_epoch=False, sync_dist=True)

        if self.cfg.encoder.sem is not None:
            self.sem_usage_ema.update(sem_out["usage_count"], batch_size=anchor_ids.shape[0])

        return loss

    def validation_step(self, batch, batch_idx):
        anchor_ids     = batch["anchor_ids"]
        positive_ids   = batch["positive_ids"]
        t_anchor_ids   = batch["teacher_anchor_ids"]
        t_positive_ids = batch["teacher_positive_ids"]

        t_anchor_mask   = (t_anchor_ids   != self.teacher_encoder.tokenizer.pad_token_id).long()
        t_positive_mask = (t_positive_ids != self.teacher_encoder.tokenizer.pad_token_id).long()

        # Teacher is temp-independent — run once
        with torch.no_grad():
            z_t_anc, _ = self.teacher_encoder(
                input_ids=t_anchor_ids,
                attention_mask=t_anchor_mask,
            )
            z_t_pos, _ = self.teacher_encoder(
                input_ids=t_positive_ids,
                attention_mask=t_positive_mask,
            )
        z_t_anc = z_t_anc.float()
        z_t_pos = z_t_pos.float()

        temps = [1e-4, None] if self.cfg.encoder.sem is not None else [None]
        names = ["hard", "soft"] if self.cfg.encoder.sem is not None else ["soft"]

        for idx, (sem_temp, label) in enumerate(zip(temps, names)):
            anchor_mask   = (anchor_ids   != self.encoder.tokenizer.pad_token_id).long()
            positive_mask = (positive_ids != self.encoder.tokenizer.pad_token_id).long()

            z_s_anc, sem_out = self.encoder(
                input_ids=anchor_ids,
                attention_mask=anchor_mask,
                temp=sem_temp,
                return_count=(idx == 0 and self.cfg.encoder.sem is not None),
            )
            z_s_pos, _ = self.encoder(
                input_ids=positive_ids,
                attention_mask=positive_mask,
                temp=sem_temp,
            )

            if "probs" in sem_out:
                ent, m_ent = sem_entropy(sem_out["probs"])
                self.log("val/sem_entropy", ent.item(), on_epoch=True, sync_dist=True)
                self.log("val/sem_marginal_entropy", m_ent.item(), on_epoch=True, sync_dist=True)

                # Correlation between pairwise SEM hamming distance and teacher cosine similarity
                # Only for flat SEM (probs is a Tensor, not a list as in DLC)
                if idx == 0 and isinstance(sem_out["probs"], torch.Tensor):
                    codes = sem_out["probs"].argmax(-1)  # (B, L)
                    hamming = (codes.unsqueeze(0) != codes.unsqueeze(1)).float().mean(-1)  # (B, B)
                    z_t_norm = F.normalize(z_t_anc, dim=-1)
                    cos_sim = z_t_norm @ z_t_norm.T  # (B, B)
                    B = codes.shape[0]
                    mask = torch.triu(torch.ones(B, B, device=codes.device, dtype=torch.bool), diagonal=1)
                    corr = torch.corrcoef(torch.stack([hamming[mask], 1.0 - cos_sim[mask]]))[0, 1]
                    self.log("val/hamming_cos_corr", corr, on_step=False, on_epoch=True, sync_dist=True)

            loss = (
                self._distill_loss(self.proj(z_s_anc.float()), z_t_anc)
                + self._distill_loss(self.proj(z_s_pos.float()), z_t_pos)
            )
            self.log(f"val/loss_{label}", loss, on_step=False, on_epoch=True, sync_dist=True)

        if (batch_idx == 0) and (rank_zero_only.rank == 0):
            if self.cfg.encoder.sem is not None and self.sem_usage_ema.usage is not None:
                is_dead = self.sem_usage_ema.usage < 1e-4
                self.log(
                    "val/dead_words_ratio",
                    torch.sum(is_dead).item() / is_dead.numel(),
                    on_epoch=True, sync_dist=False, rank_zero_only=True,
                )

            if self.cfg.mteb_tasks:
                if not self._teacher_mteb_done:
                    teacher_mteb_scores = eval_mteb(
                        encoder=self.teacher_encoder,
                        tasks=self.cfg.mteb_tasks,
                        batch_size=self.cfg.mteb_batch_size,
                        limit=self.cfg.mteb_limit,
                        device=str(self.device),
                    )
                    teacher_by_mode: dict[str, list[float]] = {}
                    for key, score in teacher_mteb_scores.items():
                        mode = key.split("/")[-1]
                        teacher_by_mode.setdefault(mode, []).append(score)
                    for mode, scores in teacher_by_mode.items():
                        self.log(
                            f"mteb_teacher/mean/{mode}", sum(scores) / len(scores),
                            on_epoch=True, sync_dist=False, rank_zero_only=True,
                        )
                    self._teacher_mteb_done = True

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
                    self.log(
                        f"mteb/mean/{mode}", sum(scores) / len(scores),
                        on_epoch=True, sync_dist=False, rank_zero_only=True,
                    )
