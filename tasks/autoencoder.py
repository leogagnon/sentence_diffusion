import os
import random
from dataclasses import dataclass
from typing import Optional

import lightning as L
import torch
import torch.nn as nn
import wandb
from einops import rearrange, repeat
from lightning.pytorch.utilities.rank_zero import rank_zero_only
from omegaconf import OmegaConf
from torch.utils.data.dataset import Subset
from transformers import get_cosine_schedule_with_warmup

from data import InfoLabel, LanguageDataset, LanguageDatasetConfig, PrefixSuffixIterable
from model.decoder import DecoderConfig, DecoderModel
from model.encoder import EncoderConfig, EncoderModel
from tasks.utils import *
from model.diffusion_continuous import *

from functools import partial


@dataclass
class AETaskConfig:
    lr: float
    batch_size: int
    decoder: DecoderConfig
    dataset: LanguageDatasetConfig
    encoder: Optional[EncoderConfig] = None
    pretrained_ae_id: Optional[str] = None
    pretrained_declutr_id: Optional[str] = None
    finetune_encoder: bool = False
    encoder_lr: Optional[float] = None  # defaults to lr if not set
    lr_warmup_steps: int = 1500
    prefix_length: int = 0
    suffix_length: int = 128
    context_length: int = 0
    # Noise applied to latent during training
    prior_logsnr_max: float = 5.0  # defines t_min (least-noisy end); higher = less noise
    noise_schedule: bool = False   # if True, sample t ~ Uniform[t_min, 1.0] each step
    train_schedule: str = "cosine"
    schedule_scale: float = 1.0
    # Soft thought projection (maps latent -> soft prompt tokens for decoder)
    soft_prompt_len: int = 16
    noise_cond: bool = False       # if True, condition soft_thought_enc on the noise level
    # Dropout regularization on latent conditioning
    latent_feature_dropout_prob: float = 0.1
    latent_conditioning_dropout_prob: float = 0.0
    name: Optional[str] = None


class AETask(L.LightningModule):
    """
    Stage-1 pretraining: train the decoder to reconstruct text from a noisy latent.

    The encoder is frozen (with a trainable adapter). Gaussian noise is added to the
    latent during training at a fixed SNR level (controlled by prior_logsnr_max), so
    the decoder learns to extract information despite the noise level that will later
    be used by the diffusion model prior.
    """

    def __init__(self, cfg: Optional[AETaskConfig] = None, **kwargs):
        super().__init__()

        if cfg is None:
            cfg = OmegaConf.to_object(
                OmegaConf.merge(
                    OmegaConf.create(AETaskConfig),
                    OmegaConf.create(kwargs),
                )
            )

        self.train_schedule = partial(
            time_to_alpha,
            alpha_schedule=get_sampling_schedule(cfg.train_schedule),
            scale=cfg.schedule_scale,
        )

        # --- Load encoder ---
        if cfg.pretrained_ae_id is not None:
            assert cfg.encoder is None, "Cannot specify both pretrained_ae_id and encoder"
            from tasks.autoencoder import AETask as _AETask
            task = _AETask.load_from_checkpoint(
                os.path.join(os.environ["LOG_DIR"], "checkpoints/", cfg.pretrained_ae_id, "last.ckpt"),
                strict=False,
                map_location=torch.device("cpu"),
            )
            cfg.dataset = task.cfg.dataset
            cfg.prefix_length = task.cfg.prefix_length
            cfg.suffix_length = task.cfg.suffix_length
            cfg.context_length = task.cfg.context_length
            self.encoder = task.encoder

        elif cfg.pretrained_declutr_id is not None:
            assert cfg.encoder is None, "Cannot specify both pretrained_declutr_id and encoder"
            from tasks.declutr import DeCLUTRTask
            task = DeCLUTRTask.load_from_checkpoint(
                os.path.join(os.environ["LOG_DIR"], "checkpoints/", cfg.pretrained_declutr_id, "last.ckpt"),
                strict=False,
                map_location=torch.device("cpu"),
            )
            cfg.dataset = task.cfg.dataset
            assert cfg.prefix_length is not None
            assert cfg.suffix_length is not None
            assert cfg.context_length is not None
            self.encoder = task.encoder

        elif cfg.encoder is not None:
            assert cfg.prefix_length is not None
            assert cfg.suffix_length is not None
            assert cfg.context_length is not None
            self.encoder = EncoderModel(cfg.encoder)

        else:
            raise ValueError("Must specify one of: encoder, pretrained_ae_id, pretrained_declutr_id")

        if cfg.finetune_encoder:
            self.encoder.train().requires_grad_(True)
        else:
            self.encoder.eval().requires_grad_(False)

        # --- Decoder ---
        self.decoder = DecoderModel(cfg.decoder).train().requires_grad_(True)

        # --- Soft thought projection (latent -> soft prompt tokens for decoder) ---
        pre_proj_dim = 96
        self.soft_thought_proj = nn.Sequential(
            nn.Linear(self.encoder.backbone_dim, cfg.soft_prompt_len * pre_proj_dim, bias=False),
            Rearrange("b (l d) -> b l d", l=cfg.soft_prompt_len, d=pre_proj_dim),
            nn.Linear(pre_proj_dim, self.decoder.latent_dim, bias=False),
        )
        self.soft_thought_enc = AttentionLayers(
            causal=False,
            dim=self.decoder.latent_dim,
            depth=3,
            heads=8,
            attn_dropout=0.0,
            ff_dropout=0.0,
            rel_pos_bias=False,
            ff_glu=True,
            ff_swish=True,
            use_adaptive_rmsnorm=cfg.noise_cond,
            use_adaptive_layerscale=cfg.noise_cond,
            dim_condition=self.decoder.latent_dim if cfg.noise_cond else None,
            adaptive_condition_mlp_expansion=4 if cfg.noise_cond else None,
            adaptive_condition_mlp=cfg.noise_cond,
        )
        if cfg.noise_cond:
            self.noise_emb = ScaledSinusoidalEmbedding(self.decoder.latent_dim)
        if cfg.soft_prompt_len > 0:
            self.null_soft_thought = nn.Parameter(
                torch.zeros(1, cfg.soft_prompt_len, self.decoder.latent_dim)
            )
            nn.init.normal_(self.null_soft_thought, std=0.02)
        else:
            self.null_soft_thought = None

        self.encoder_mode = "context" if cfg.context_length > 0 else "suffix"
        self._t_min_cache: Optional[float] = None

        self.cfg = cfg
        self.save_hyperparameters(OmegaConf.to_container(OmegaConf.structured(cfg)), logger=False)

    def train(self, mode: bool = True):
        super().train(mode)
        if not self.cfg.finetune_encoder:
            self.encoder.eval()
        return self

    def setup(self, **kwargs):
        self.dataset = LanguageDataset(self.cfg.dataset)
        self.train_data = Subset(self.dataset, indices=self.dataset.train_indices)
        self.val_data = Subset(self.dataset, indices=self.dataset.val_indices)

    def train_dataloader(self):
        return PrefixSuffixIterable.get_dataloader(
            self.train_data,
            batch_size=self.cfg.batch_size,
            prefix_length=self.cfg.prefix_length,
            suffix_length=self.cfg.suffix_length,
            context_length=self.cfg.context_length,
            enc_tok=self.encoder.tokenizer,
            dec_tok=self.decoder.tokenizer,
            encoder_mode=self.encoder_mode,
            encoder_noise=False,
            seed=random.randint(0, 100000),
            num_dlc_ph=self.cfg.soft_prompt_len,
        )

    def val_dataloader(self):
        return PrefixSuffixIterable.get_dataloader(
            self.val_data,
            batch_size=self.cfg.batch_size,
            prefix_length=self.cfg.prefix_length,
            suffix_length=self.cfg.suffix_length,
            context_length=self.cfg.context_length,
            enc_tok=self.encoder.tokenizer,
            dec_tok=self.decoder.tokenizer,
            encoder_mode=self.encoder_mode,
            encoder_noise=False,
            seed=42,
            num_dlc_ph=self.cfg.soft_prompt_len,
        )

    def configure_optimizers(self):
        no_decay = ["bias", "norm"]
        encoder_lr = self.cfg.encoder_lr if self.cfg.encoder_lr is not None else self.cfg.lr

        def param_groups(named_params, lr):
            named_params = [(n, p) for n, p in named_params if p.requires_grad]
            return [
                {"params": [p for n, p in named_params if not any(nd in n.lower() for nd in no_decay)], "weight_decay": 0.01, "lr": lr},
                {"params": [p for n, p in named_params if     any(nd in n.lower() for nd in no_decay)], "weight_decay": 0.0,  "lr": lr},
            ]

        optimizer_grouped_parameters = param_groups(
            [(n, p) for n, p in self.named_parameters() if not n.startswith("encoder.")],
            self.cfg.lr,
        )
        if self.cfg.finetune_encoder:
            optimizer_grouped_parameters += param_groups(
                self.encoder.named_parameters(),
                encoder_lr,
            )

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

    def _get_t_min(self):
        """Binary search for t s.t. train_schedule(t) == sigmoid(prior_logsnr_max). Cached."""
        if self._t_min_cache is not None:
            return self._t_min_cache
        target_alpha = torch.sigmoid(torch.tensor(self.cfg.prior_logsnr_max)).item()
        lo, hi = 0.0, 1.0
        for _ in range(24):
            mid = 0.5 * (lo + hi)
            if self.train_schedule(torch.tensor([mid])).item() >= target_alpha:
                lo = mid
            else:
                hi = mid
        self._t_min_cache = 0.5 * (lo + hi)
        return self._t_min_cache

    def _get_prior_alpha(self, batch_size, device, dtype=None):
        """Fixed alpha at t_min — used for validation and when noise_schedule=False."""
        alpha = torch.sigmoid(
            torch.tensor(self.cfg.prior_logsnr_max, device=device, dtype=torch.float32)
        ).expand(batch_size)
        if dtype is not None:
            alpha = alpha.to(dtype)
        return alpha

    def _sample_alpha(self, batch_size, device, dtype=None):
        """Sample alpha uniformly over [t_min, 1.0] when noise_schedule=True."""
        t_min = self._get_t_min()
        t = t_min + (1.0 - t_min) * torch.rand(batch_size, device=device, dtype=torch.float32)
        alpha = self.train_schedule(t)
        if dtype is not None:
            alpha = alpha.to(dtype)
        return alpha

    def _encode_latent(self, batch):
        with torch.set_grad_enabled(self.cfg.finetune_encoder and self.training):
            z = self.encoder(
                batch["input_ids_enc"],
                attention_mask=batch["attention_mask_enc"],
                only_backbone=True,
            )
        return z

    def _get_input_embeds(self, batch, z, alpha_1d=None):
        """
        alpha_1d: per-sample alpha values [B]. If provided, noise is added at that level
                  and (if noise_cond=True) the soft thought encoder is conditioned on it.
                  Pass None for a clean (no-noise) forward pass.
        """
        input_ids = batch["input_ids_dec"]
        input_embeds = self.decoder.backbone.get_input_embeddings()(input_ids)

        # Add Gaussian noise
        if alpha_1d is not None:
            alpha = right_pad_dims_to(z, alpha_1d)
            z = alpha.sqrt() * z + (1 - alpha).sqrt() * torch.randn_like(z)

        # Feature-level dropout
        if self.training and self.cfg.latent_feature_dropout_prob > 0:
            p = self.cfg.latent_feature_dropout_prob
            mask_shape = (z.shape[0], *([1] * (z.ndim - 2)), z.shape[-1])
            feature_keep_mask = (torch.rand(mask_shape, device=z.device) >= p).to(z.dtype)
            z = z * feature_keep_mask

        # Project latent to soft thought tokens
        if z.ndim == 3 and z.shape[1] == 1:
            z = z[:, 0]
        soft_thought = self.soft_thought_proj(z).to(input_embeds.dtype)

        # Noise conditioning on soft thought encoder
        noise_embd = None
        if self.cfg.noise_cond:
            if alpha_1d is not None:
                a = alpha_1d.view(alpha_1d.shape[0], -1)[:, 0]
            else:
                # Clean latent (alpha=1) — use ones as a "no noise" signal
                a = torch.ones(z.shape[0], device=z.device, dtype=z.dtype)
            noise_embd = self.noise_emb(rearrange(a * 1000, "b -> b 1"))
            if noise_embd.ndim == 2:
                noise_embd = rearrange(noise_embd, "b d -> b 1 d")
        soft_thought = self.soft_thought_enc(soft_thought, condition=noise_embd)

        # Conditioning dropout (classifier-free style)
        if self.training and self.cfg.latent_conditioning_dropout_prob > 0:
            p = self.cfg.latent_conditioning_dropout_prob
            dropout_mask = torch.rand((soft_thought.shape[0],), device=soft_thought.device) < p
            if dropout_mask.any():
                null = (
                    repeat(self.null_soft_thought, "1 l d -> b l d", b=soft_thought.shape[0]).to(soft_thought.dtype)
                    if self.null_soft_thought is not None
                    else torch.zeros_like(soft_thought)
                )
                soft_thought = torch.where(rearrange(dropout_mask, "b -> b 1 1"), null, soft_thought)

        # Fill DLC placeholder tokens with soft thoughts
        dlc_ph_mask = batch["info_mask_dec"] == InfoLabel.DLC.value
        input_embeds[dlc_ph_mask] = soft_thought.view(-1, self.decoder.latent_dim)

        return input_embeds

    def _suffix_loss(self, logits, batch):
        info_mask_dec = batch["info_mask_dec"][:, 1:]
        targets = batch["input_ids_dec"][:, 1:].contiguous()
        logits = logits[:, :-1].contiguous()
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1),
            reduction="none",
        )
        loss = loss.view_as(targets)
        return loss[info_mask_dec == InfoLabel.SUFFIX.value].mean()

    def training_step(self, batch, batch_idx):
        z = self._encode_latent(batch)
        if self.cfg.noise_schedule:
            alpha_1d = self._sample_alpha(z.size(0), z.device, z.dtype)
        else:
            alpha_1d = self._get_prior_alpha(z.size(0), z.device, z.dtype)
        input_embeds = self._get_input_embeds(batch, z, alpha_1d=alpha_1d)
        logits = self.decoder(input_embeds=input_embeds)
        loss = self._suffix_loss(logits, batch)
        self.log("train/loss", loss.item(), on_epoch=False, on_step=True, sync_dist=True)
        return loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        z = self._encode_latent(batch)
        prior_alpha = self._get_prior_alpha(z.size(0), z.device, z.dtype)

        # Loss at prior noise level (t_min) — comparable across runs regardless of noise_schedule
        input_embeds_noisy = self._get_input_embeds(batch, z, alpha_1d=prior_alpha)
        logits_noisy = self.decoder(input_embeds=input_embeds_noisy)
        noisy_loss = self._suffix_loss(logits_noisy, batch)
        self.log("val/loss_noisy", noisy_loss.item(), on_epoch=True, sync_dist=True)

        # Loss without noise (upper bound on decoder quality with clean latent)
        input_embeds_clean = self._get_input_embeds(batch, z, alpha_1d=None)
        logits_clean = self.decoder(input_embeds=input_embeds_clean)
        clean_loss = self._suffix_loss(logits_clean, batch)
        self.log("val/loss_clean", clean_loss.item(), on_epoch=True, sync_dist=True)

        if (batch_idx == 0) and (rank_zero_only.rank == 0):
            prefix_str = batch["prefix_str"][:5]
            true_suffix_str = batch["suffix_str"][:5]

            gen_batch = {k: v[:5] for k, v in batch.items()}
            input_embeds_gen = self._get_input_embeds(gen_batch, z[:5], alpha_1d=prior_alpha[:5])

            generated_suffix_str = self.decoder.tokenizer.batch_decode(
                self.decoder.generate(
                    prefix_embeds=input_embeds_gen[:, : -self.cfg.suffix_length],
                    max_length=self.cfg.suffix_length,
                ),
                skip_special_tokens=True,
            )

            table = wandb.Table(columns=["Prefix", "True Suffix", "Generated Suffix"])
            for i in range(5):
                table.add_data(prefix_str[i], true_suffix_str[i], generated_suffix_str[i])
            wandb.log({"val/samples": table})
            del table
