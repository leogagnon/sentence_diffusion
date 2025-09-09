import os
import random
from collections import namedtuple
from dataclasses import dataclass
from functools import partial, singledispatchmethod
from typing import *
import math

import hydra
import lightning as L
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


@dataclass
class DSMDiffusionConfig:
    model: DiTConfig
    dataset: dict
    batch_size: int
    val_split: float
    lr: float
    ae_id: str
    name: Optional[str] = None

    loss: str = "l2"
    sampling_timesteps: int = 50
    train_schedule: str = "cosine"
    sampling_schedule: Optional[str] = None
    diffusion_objective: str = "pred_v"
    schedule_scale: float = 1.0
    sampler: str = "ddpm"
    normalize_latent: bool = False


class DSMDiffusion(L.LightningModule):
    """
    Trains a diffusion model.
    """

    def __init__(self, cfg: Optional[DSMDiffusionConfig] = None, **kwargs) -> None:
        super().__init__()

        if cfg == None:
            cfg = OmegaConf.to_object(
                OmegaConf.merge(
                    OmegaConf.create(DSMDiffusionConfig),
                    OmegaConf.create(kwargs),
                )
            )

        # Setup diffusion stuff
        self.model = DiT(cfg.model)

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

        if cfg.normalize_latent:
            # Buffers for latent mean and scale values
            self.register_buffer("latent_mean", torch.tensor(0).to(torch.float32))
            self.latent_mean: torch.FloatTensor
            self.register_buffer("latent_scale", torch.tensor(1).to(torch.float32))
            self.latent_scale: torch.FloatTensor

        # Load the AutoEncoder Task
        ae_task = AETask.load_from_checkpoint(
            os.path.join(
                os.environ["LATENT_CONTROL_CKPT_DIR"],
                cfg.ae_id,
                "last.ckpt",
            ),
            strict=False,
        )
        ae_task.setup()

        self.encoder = ae_task.encoder.eval().requires_grad_(False)
        self.decoder = ae_task.decoder.eval().requires_grad_(False).cpu()
        self.train_data = ae_task.train_data
        self.val_data = ae_task.val_data

        self.cfg = cfg
        # Important for checkpoints
        self.save_hyperparameters(
            OmegaConf.to_container(OmegaConf.structured(cfg)), logger=False
        )

    @property
    def loss_fn(self):
        if self.cfg.loss == "l1":
            return F.l1_loss
        elif self.cfg.loss == "l2":
            return F.mse_loss
        elif self.cfg.loss == "smooth_l1":
            return F.smooth_l1_loss
        else:
            raise ValueError(f"invalid loss type {self.cfg.loss}")

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
            collate_fn=lambda x: x,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_data,
            batch_size=self.cfg.batch_size,
            collate_fn=lambda x: x,
            shuffle=False,
        )

    def training_step(self, batch, batch_idx=None):
        
        # Compute latents
        with torch.no_grad():
            latent = self.encoder(batch["input_ids_enc"])

        if self.cfg.normalize_latent:
            latent_ = rearrange(latent, "b s d -> (b s) d")
            self.latent_mean = torch.mean(latent_, dim=0)
            self.latent_scale = torch.std(latent_ - self.latent_mean, unbiased=False)
            latent = self.normalize_latent(latent)

        loss = compute_diffusion_loss(
            self.model,
            latent,
            schedule=self.train_schedule,
            diffusion_objective=self.cfg.diffusion_objective,
            loss_fn=self.loss_fn,
            cond=batch["cond_tokens"],
            cond_input_ids=batch["cond_input_ids"],
            cond_ignore_mask=batch["cond_ignore_mask"],
        )

        self.log(
            "train/loss",
            loss.detach().cpu().numpy().item(),
            prog_bar=True,
            add_dataloader_idx=False,
            batch_size=latent.shape[0],
        )

        return loss
