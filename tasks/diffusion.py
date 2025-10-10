import os
import random
from collections import namedtuple
from dataclasses import dataclass
from functools import partial, singledispatchmethod
from typing import *
import math

import hydra
import lightning as L
from mauve.compute_mauve import get_features_from_input, compute_mauve
import numpy as np
import torch
import torch.nn.functional as F
import wandb
from einops import rearrange, reduce, repeat
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from torchmetrics.functional import kl_divergence
from tqdm import tqdm
from transformers.activations import ACT2FN

from model.gaussian_diffusion import *
from tasks.autoencoder import AETask
import torch.nn as nn
from data.stories import StoriesDatasetConfig, StoriesDataset
from torch.optim.swa_utils import AveragedModel, get_ema_avg_fn
from lightning.pytorch.utilities.rank_zero import rank_zero_info


@dataclass
class GaussianDiffusionTaskConfig:
    model: DiTConfig
    batch_size: int
    lr: float
    pretrained_ae_id: str
    name: Optional[str] = None
    data_seed: int = 42

    loss: str = "l2"
    sampling_timesteps: int = 50
    train_schedule: str = "cosine"
    sampling_schedule: Optional[str] = None
    diffusion_objective: str = "pred_v"
    schedule_scale: float = 1.0
    sampler: str = "ddpm"
    normalize_latent: bool = False
    validation_mauve: bool = False
    max_generation_length: int = 150


class GaussianDiffusionTask(L.LightningModule):
    """
    Trains a latent diffusion transformer on the latent space of a pretrained autoencoder (ae_id).
    """

    def __init__(
        self, cfg: Optional[GaussianDiffusionTaskConfig] = None, **kwargs
    ) -> None:
        super().__init__()

        if cfg == None:
            cfg = OmegaConf.to_object(
                OmegaConf.merge(
                    OmegaConf.create(GaussianDiffusionTaskConfig),
                    OmegaConf.create(kwargs),
                )
            )

        # Init noise schedules
        self.train_schedule = partial(
            time_to_alpha,
            alpha_schedule=get_sampling_schedule(cfg.train_schedule),
            scale=cfg.schedule_scale,
        )
        if cfg.sampling_schedule != None:
            self.sampling_schedule = partial(
                time_to_alpha,
                alpha_schedule=get_sampling_schedule(cfg.sampling_schedule),
                scale=cfg.schedule_scale,
            )
        else:
            self.sampling_schedule = self.train_schedule

        # Extract encoder, decoder and dataset from pretrained autoencoder
        ae_task = AETask.load_from_checkpoint(
            os.path.join(
                os.environ["LATENT_CONTROL_CKPT_DIR"],
                cfg.pretrained_ae_id,
                "last.ckpt",
            ),
            strict=False,
        )
        self.train_indices = ae_task.train_indices
        self.val_indices = ae_task.val_indices
        self.dataset = ae_task.dataset

        # Load and process encoder (eval, no gradients, bfloat16, flash attention)
        self.encoder = ae_task.encoder.to(torch.bfloat16).eval().requires_grad_(False)
        try:
            # Try to use flash attention if available
            self.encoder.transformer.auto_model.set_attn_implementation(
                "flash_attention_2"
            )
        except:
            pass

        # Load and process decoder (merge adapter, eval, no gradients, bfloat16, flash attention)
        self.decoder = ae_task.decoder.eval().requires_grad_(False)
        self.decoder.backbone = self.decoder.backbone.merge_and_unload()
        self.decoder = self.decoder.to(torch.bfloat16)
        self.decoder.backbone.set_attn_implementation("flash_attention_2")

        # Init diffusion model (and EMA model)
        cfg.model.latent_dim = self.encoder.latent_dim
        cfg.model.latent_len = self.encoder.latent_len
        self.model = DiT(cfg.model)
        self.ema_model = (
            AveragedModel(self.model, avg_fn=get_ema_avg_fn())
            .eval()
            .requires_grad_(False)
        )

        # Init latent normalization if needed
        if cfg.normalize_latent:
            self.register_buffer(
                "latent_mean",
                torch.zeros(size=(cfg.model.latent_len, cfg.model.latent_dim)).float(),
            )
            self.latent_mean: torch.FloatTensor
            self.register_buffer(
                "latent_scale",
                torch.ones(size=(cfg.model.latent_len, cfg.model.latent_dim)).float(),
            )
            self.latent_scale: torch.FloatTensor

        self.cfg = cfg
        # Important for checkpoints
        self.save_hyperparameters(
            OmegaConf.to_container(OmegaConf.structured(cfg)), logger=False
        )

    def train(self, mode: bool = True):
        super().train(mode)
        # Make sure encoder and decoder and EMA are always in eval mode
        self.encoder.eval()
        self.decoder.eval()
        self.ema_model.eval()

    def setup(self, stage: Optional[str] = None):
        self.train_data = Subset(self.dataset, indices=self.train_indices)
        self.val_data = Subset(self.dataset, indices=self.val_indices)

    def normalize_latent(self, x_start):
        eps = 1e-5

        return (x_start - self.latent_mean) / (self.latent_scale).clamp(min=eps)

    def unnormalize_latent(self, x_start):
        eps = 1e-5

        return x_start * (self.latent_scale.clamp(min=eps)) + self.latent_mean

    def configure_optimizers(self):
        return torch.optim.AdamW(self.model.parameters(), lr=self.cfg.lr)

    def train_dataloader(self):
        return DataLoader(
            self.train_data,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            collate_fn=self.dataset.get_collate_and_tokenize_fn(
                enc_tokenizer=self.encoder.tokenizer,
            ),
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_data,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            collate_fn=self.dataset.get_collate_and_tokenize_fn(
                enc_tokenizer=self.encoder.tokenizer,
            ),
        )

    def on_fit_start(self):
        # Ensure decoder is on CPU to save GPU memory
        self.decoder = self.decoder.cpu()
        self.ema_model.module = self.ema_model.module.cpu()

        # Compute latent mean and scale if needed (on 20000 samples)
        if self.cfg.normalize_latent:
            with torch.no_grad():
                latent_samples = []
                for batch in tqdm(
                    DataLoader(
                        Subset(
                            self.train_data,
                            torch.randperm(
                                len(self.train_data),
                                generator=torch.Generator().manual_seed(
                                    self.cfg.data_seed
                                ),
                            )[:20000],
                        ),
                        batch_size=self.cfg.batch_size,
                        collate_fn=self.dataset.get_collate_and_tokenize_fn(
                            enc_tokenizer=self.encoder.tokenizer,
                        ),
                        shuffle=False,
                        generator=torch.Generator().manual_seed(self.cfg.data_seed),
                    ),
                    desc="Computing latent mean and scale on 20000 training samples...",
                ):
                    latent_samples.append(
                        self.encoder(
                            batch["input_ids_enc"].cuda(),
                            attention_mask=batch["attention_mask_enc"].cuda(),
                        )
                    )

                latent_samples = torch.cat(latent_samples, dim=0)
                self.latent_mean = torch.mean(latent_samples, dim=0)
                self.latent_scale = torch.std(
                    latent_samples - self.latent_mean, unbiased=False, dim=0
                )

                print("Latent mean and scale computed.")

    def training_step(self, batch, batch_idx=None):
        assert self.encoder.training == False
        assert self.model.training == True

        # Compute latents
        with torch.no_grad():
            latent = self.encoder(
                batch["input_ids_enc"],
                attention_mask=batch["attention_mask_enc"],
            )
            if self.cfg.normalize_latent:
                latent = self.normalize_latent(latent)

        loss = compute_diffusion_loss(
            self.model,
            latent,
            schedule=self.train_schedule,
            diffusion_objective=self.cfg.diffusion_objective,
            loss_name=self.cfg.loss,
        )

        self.log(
            "train/loss",
            loss.detach().cpu().numpy().item(),
            prog_bar=True,
            add_dataloader_idx=False,
            batch_size=latent.shape[0],
            on_step=True,
            on_epoch=False,
            sync_dist=True,
        )

        return loss
    
    @torch.no_grad()
    def validation_step(self, batch, batch_idx=None):
    
        # Compute latents
        latent = self.encoder(
            batch["input_ids_enc"],
            attention_mask=batch["attention_mask_enc"],
        )
        if self.cfg.normalize_latent:
            latent = self.normalize_latent(latent)

        loss = compute_diffusion_loss(
            self.ema_model.module,
            latent,
            schedule=self.train_schedule,
            diffusion_objective=self.cfg.diffusion_objective,
            loss_name=self.cfg.loss,
        )

        self.log(
            "val/loss",
            loss.cpu().numpy().item(),
            prog_bar=True,
            add_dataloader_idx=False,
            batch_size=latent.shape[0],
            on_epoch=True,
            on_step=False,
            sync_dist=True,
        )

        return loss

    def on_train_batch_end(self, outputs, batch, batch_idx):
        # Update EMA model every step
        if batch_idx % self.trainer.accumulate_grad_batches == 0:
            self.ema_model.update_parameters(self.model)

    def on_validation_epoch_start(self):
        self.ema_model = self.ema_model.cuda()
        self.model = self.model.cpu()

    def on_validation_epoch_end(self):
        if self.cfg.validation_mauve:
            mauve_score = self.get_mauve_score()
            self.log(
                "val/mauve",
                mauve_score,
                prog_bar=True,
                add_dataloader_idx=False,
                on_epoch=True,
                on_step=False,
                sync_dist=True,
            )
        self.ema_model = self.ema_model.cpu()
        self.model = self.model.cuda()
