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
from ema_pytorch import EMA


@dataclass
class GaussianDiffusionTaskConfig:
    model: DiTConfig
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


class GaussianDiffusionTask(L.LightningModule):
    """
    Trains a latent diffusion transformer on the latent space of a pretrained autoencoder (ae_id).
    """

    def __init__(self, cfg: Optional[GaussianDiffusionTaskConfig] = None, **kwargs) -> None:
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
                cfg.ae_id,
                "last.ckpt",
            ),
            strict=False,
        )
        ae_task.setup()

        self.encoder = ae_task.encoder.eval().requires_grad_(False)
        self.decoder = ae_task.decoder.eval().requires_grad_(False)
        self.train_data = ae_task.train_data
        self.val_data = ae_task.val_data

        # Init latent normalization if needed
        if cfg.normalize_latent:
            self.register_buffer(
                "latent_mean", torch.zeros(size=self.encoder.latent_shape).float()
            )
            self.latent_mean: torch.FloatTensor
            self.register_buffer(
                "latent_scale", torch.ones(size=self.encoder.latent_shape).float()
            )
            self.latent_scale: torch.FloatTensor

        # Init diffusion model
        cfg.model.latent_shape = self.encoder.latent_shape
        self.model = DiT(cfg.model)

        self.ema = EMA(self.model, beta=0.995, update_every=10, power=3/4).cpu()

        self.cfg = cfg
        # Important for checkpoints
        self.save_hyperparameters(
            OmegaConf.to_container(OmegaConf.structured(cfg)), logger=False
        )

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

    def on_fit_start(self):
        # Ensure decoder is on CPU to save GPU memory
        self.decoder = self.decoder.cpu()

        # Compute latent mean and scale if needed (on 10000 samples)
        if self.cfg.normalize_latent:
            with torch.no_grad():
                latent_samples = []
                for batch in tqdm(
                    DataLoader(
                        Subset(
                            self.train_data,
                            torch.randperm(len(self.train_data))[:10000],
                        ),
                        batch_size=self.cfg.batch_size,
                        collate_fn=lambda x: x,
                        shuffle=False,
                    ),
                    desc="Computing latent mean and scale on 10000 training samples...",
                ):
                    latent_samples.append(self.encoder(batch["input_ids_enc"].cuda(), attention_mask=batch["attention_mask_enc"].cuda()))

                latent_samples = torch.cat(latent_samples, dim=0)
                self.latent_mean = torch.mean(latent_samples, dim=0)
                self.latent_scale = torch.std(
                    latent_samples - self.latent_mean, unbiased=False
                )

                print("Latent mean and scale computed.")
                self.ema.ema_model.latent_mean = self.latent_mean
                self.ema.ema_model.latent_scale = self.latent_scale

    def sample(self, xd):
        pass

    def training_step(self, batch, batch_idx=None):

        # Compute latents
        with torch.no_grad():
            assert self.encoder.training == False
            latent = self.encoder(batch["input_ids_enc"], attention_mask=batch["attention_mask_enc"])
            if self.cfg.normalize_latent:
                latent = self.normalize_latent(latent)

        loss = compute_diffusion_loss(
            self.model,
            latent,
            schedule=self.train_schedule,
            diffusion_objective=self.cfg.diffusion_objective,
            loss_name=self.cfg.loss
        )

        self.log(
            "train/loss",
            loss.detach().cpu().numpy().item(),
            prog_bar=True,
            add_dataloader_idx=False,
            batch_size=latent.shape[0],
            on_step=True,
            on_epoch=False
        )

        return loss
    
    def validation_step(self, batch, batch_idx=None):

        # Compute latents
        with torch.no_grad():
            assert self.encoder.training == False
            latent = self.encoder(batch["input_ids_enc"], attention_mask=batch["attention_mask_enc"])
            if self.cfg.normalize_latent:
                latent = self.normalize_latent(latent)

        loss = compute_diffusion_loss(
            self.model,
            latent,
            schedule=self.train_schedule,
            diffusion_objective=self.cfg.diffusion_objective,
            loss_name=self.cfg.loss
        )

        self.log(
            "val/loss",
            loss.detach().cpu().numpy().item(),
            prog_bar=True,
            add_dataloader_idx=False,
            batch_size=latent.shape[0],
            on_epoch=True,
            on_step=False
        )

        return loss
