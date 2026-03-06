import random
from dataclasses import dataclass
from typing import List, Optional

import einx
import lightning as L
import torch
import torch.nn.functional as F
from lightning.pytorch.utilities.rank_zero import rank_zero_only
from transformers import get_cosine_schedule_with_warmup
from omegaconf import OmegaConf

from data import PairDatasetIterable, load_snli_pairs, load_mnli_pairs
from model.encoder import EncoderConfig, EncoderModel
from tasks.utils import eval_mteb, sem_entropy, SEMUsageTracker
from pytorch_metric_learning.losses import NTXentLoss


@dataclass
class SupervisedSimCSETaskConfig:
    lr: float
    lr_warmup_steps: int
    batch_size: int
    encoder: EncoderConfig
    loss_temp: float = 0.05
    sem_noise: float = 0.0
    max_length: int = 128
    mteb_tasks: Optional[List[str]] = None
    mteb_batch_size: int = 256
    mteb_limit: Optional[int] = None
    lr_sem: Optional[float] = None  # if set, SEM parameters use this lr instead of `lr`

    seed: int = 42
    name: Optional[str] = None


class SupervisedSimCSETask(L.LightningModule):
    """
    Supervised SimCSE: contrastive training using NLI entailment pairs
    (SNLI + MNLI) as positives, with in-batch negatives and NT-Xent loss.
    No dropout augmentation, no hard negatives.
    """

    def __init__(self, cfg: Optional[SupervisedSimCSETaskConfig] = None, **kwargs):
        super().__init__()

        if cfg is None:
            cfg = OmegaConf.to_object(
                OmegaConf.merge(
                    OmegaConf.create(SupervisedSimCSETaskConfig),
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
        embeddings = torch.cat([anchors, positives], dim=0)
        labels = torch.arange(anchors.size(0), device=anchors.device, dtype=torch.long).repeat(2)
        with torch.autocast(device_type="cuda", enabled=False):
            loss = self.loss_fn(embeddings.float(), labels)
        return loss

    def compile(self):
        if self.encoder is not None:
            self.encoder.compile()

    def setup(self, **kwargs):
        train_a, train_p = load_snli_pairs("train")
        mnli_a, mnli_p = load_mnli_pairs("train")
        self.train_pairs = (train_a + mnli_a, train_p + mnli_p)

        val_a, val_p = load_snli_pairs("validation")
        mnli_val_a, mnli_val_p = load_mnli_pairs("validation_matched")
        self.val_pairs = (val_a + mnli_val_a, val_p + mnli_val_p)

    def configure_optimizers(self):
        no_decay = ["bias", "norm"]
        use_sem_lr = self.cfg.lr_sem is not None and self.cfg.encoder.sem is not None

        def is_sem(name):
            return name.startswith("sem.")

        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in self.encoder.named_parameters()
                    if not is_sem(n) and not any(nd in n.lower() for nd in no_decay)
                ],
                "lr": self.cfg.lr,
                "weight_decay": 0.1,
            },
            {
                "params": [
                    p for n, p in self.encoder.named_parameters()
                    if not is_sem(n) and any(nd in n.lower() for nd in no_decay)
                ],
                "lr": self.cfg.lr,
                "weight_decay": 0.0,
            },
        ]
        if use_sem_lr:
            optimizer_grouped_parameters += [
                {
                    "params": [
                        p for n, p in self.encoder.named_parameters()
                        if is_sem(n) and not any(nd in n.lower() for nd in no_decay)
                    ],
                    "lr": self.cfg.lr_sem,
                    "weight_decay": 0.1,
                },
                {
                    "params": [
                        p for n, p in self.encoder.named_parameters()
                        if is_sem(n) and any(nd in n.lower() for nd in no_decay)
                    ],
                    "lr": self.cfg.lr_sem,
                    "weight_decay": 0.0,
                },
            ]
        else:
            # Merge SEM params into the main groups (existing behavior)
            optimizer_grouped_parameters[0]["params"] += [
                p for n, p in self.encoder.named_parameters()
                if is_sem(n) and not any(nd in n.lower() for nd in no_decay)
            ]
            optimizer_grouped_parameters[1]["params"] += [
                p for n, p in self.encoder.named_parameters()
                if is_sem(n) and any(nd in n.lower() for nd in no_decay)
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

    @staticmethod
    def _dedup_mask(anchor_ids):
        """Return a bool mask keeping only the first occurrence of each unique anchor row."""
        _, inv = torch.unique(anchor_ids, dim=0, sorted=False, return_inverse=True)
        perm = torch.arange(anchor_ids.size(0), device=anchor_ids.device)
        first = perm.new_full((inv.max().item() + 1,), anchor_ids.size(0))
        first.scatter_reduce_(0, inv, perm, reduce="amin")
        keep = torch.zeros(anchor_ids.size(0), dtype=torch.bool, device=anchor_ids.device)
        keep[first] = True
        return keep

    def _get_dataloader(self, anchors, positives, seed, replacement=True):
        return PairDatasetIterable.get_dataloader(
            anchors=anchors,
            positives=positives,
            tokenizer=self.encoder.tokenizer,
            batch_size=self.cfg.batch_size,
            max_length=self.cfg.max_length,
            seed=seed,
            replacement=replacement,
        )

    def train_dataloader(self):
        return self._get_dataloader(*self.train_pairs, seed=self.cfg.seed, replacement=False)

    def val_dataloader(self):
        return self._get_dataloader(*self.val_pairs, seed=42)

    def training_step(self, batch, batch_idx):
        anchor_ids, positive_ids = batch["anchor_ids"], batch["positive_ids"]

        z_anchors, sem_out = self.encoder(
            input_ids=anchor_ids,
            attention_mask=(anchor_ids != self.encoder.tokenizer.pad_token_id).long(),
            return_count=(self.cfg.encoder.sem is not None),
            noise=self.cfg.sem_noise,
        )
        z_positives, _ = self.encoder(
            input_ids=positive_ids,
            attention_mask=(positive_ids != self.encoder.tokenizer.pad_token_id).long(),
            noise=self.cfg.sem_noise,
        )

        if self.trainer.world_size > 1:
            z_anchors = einx.rearrange("w b d -> (w b) d", self.all_gather(z_anchors, sync_grads=True))
            z_positives = einx.rearrange("w b d -> (w b) d", self.all_gather(z_positives, sync_grads=True))
            pad = self.cfg.max_length - anchor_ids.shape[1]
            anchor_ids_padded = torch.nn.functional.pad(anchor_ids, (0, pad), value=self.encoder.tokenizer.pad_token_id)
            anchor_ids_all = einx.rearrange("w b s -> (w b) s", self.all_gather(anchor_ids_padded))
        else:
            anchor_ids_all = anchor_ids

        keep = self._dedup_mask(anchor_ids_all)
        loss = self.compute_loss(z_anchors[keep], z_positives[keep])
        self.log("train/loss", loss, on_step=True, on_epoch=False, sync_dist=True)

        if self.cfg.encoder.sem is not None:
            self.sem_usage_ema.update(sem_out["usage_count"], batch_size=anchor_ids.shape[0])

        return loss

    def validation_step(self, batch, batch_idx):
        anchor_ids, positive_ids = batch["anchor_ids"], batch["positive_ids"]
        if self.trainer.world_size > 1:
            pad = self.cfg.max_length - anchor_ids.shape[1]
            anchor_ids_padded = torch.nn.functional.pad(anchor_ids, (0, pad), value=self.encoder.tokenizer.pad_token_id)
            anchor_ids_all = einx.rearrange("w b s -> (w b) s", self.all_gather(anchor_ids_padded))
        else:
            anchor_ids_all = anchor_ids
        keep = self._dedup_mask(anchor_ids_all)

        temps = [1e-4, None] if self.cfg.encoder.sem is not None else [None]
        names = ["hard", "soft"] if self.cfg.encoder.sem is not None else ["soft"]

        _hard_codes = None      # (B, L) int — set during hard pass
        _soft_z_anchors = None  # (B, D) float — set during soft pass

        for idx, (sem_temp, label) in enumerate(zip(temps, names)):
            z_anchors, sem_out = self.encoder(
                input_ids=anchor_ids,
                attention_mask=(anchor_ids != self.encoder.tokenizer.pad_token_id).long(),
                temp=sem_temp,
                return_count=(idx == 0 and self.cfg.encoder.sem is not None),
            )
            z_positives, _ = self.encoder(
                input_ids=positive_ids,
                attention_mask=(positive_ids != self.encoder.tokenizer.pad_token_id).long(),
                temp=sem_temp,
            )

            if "probs" in sem_out:
                ent, m_ent = sem_entropy(sem_out["probs"])
                self.log("val/sem_entropy", ent.item(), on_epoch=True, sync_dist=True)
                self.log("val/sem_marginal_entropy", m_ent.item(), on_epoch=True, sync_dist=True)
                if idx == 0 and isinstance(sem_out["probs"], torch.Tensor):
                    _hard_codes = sem_out["probs"].argmax(-1)  # (B, L)
                    if self.trainer.world_size > 1:
                        _hard_codes = einx.rearrange("w b l -> (w b) l", self.all_gather(_hard_codes))

            if self.trainer.world_size > 1:
                z_anchors = einx.rearrange("w b d -> (w b) d", self.all_gather(z_anchors, sync_grads=False))
                z_positives = einx.rearrange("w b d -> (w b) d", self.all_gather(z_positives, sync_grads=False))

            if sem_temp is None:
                _soft_z_anchors = z_anchors.float()

            loss = self.compute_loss(z_anchors[keep], z_positives[keep])
            self.log(f"val/loss_{label}", loss, on_step=False, on_epoch=True, sync_dist=True)

        if _hard_codes is not None and _soft_z_anchors is not None:
            hamming = (_hard_codes.unsqueeze(0) != _hard_codes.unsqueeze(1)).float().mean(-1)  # (B, B)
            cos_sim = F.normalize(_soft_z_anchors, dim=-1) @ F.normalize(_soft_z_anchors, dim=-1).T  # (B, B)
            B = _hard_codes.shape[0]
            mask = torch.triu(torch.ones(B, B, device=_hard_codes.device, dtype=torch.bool), diagonal=1)
            corr = torch.corrcoef(torch.stack([hamming[mask], 1.0 - cos_sim[mask]]))[0, 1]
            self.log("val/hamming_cos_corr", corr, on_step=False, on_epoch=True, sync_dist=True)

        if (batch_idx == 0) and (rank_zero_only.rank == 0):
            if (self.cfg.encoder.sem is not None) and (self.sem_usage_ema.usage != None):
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
