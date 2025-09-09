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

from model.diffusion_transformer import DiT, DiTConfig
from tasks.autoencoder import AETask
import torch.nn as nn

ModelPrediction = namedtuple("ModelPrediction", ["pred_noise", "pred_x_start", "pred_v"])

@dataclass
class DSMDiffusionConfig:
    model: DiTConfig
    dataset: dict
    batch_size: int
    val_split: float
    lr: float
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
    Trains a diffusion model with a Denoising Score Matching (DSM, https://arxiv.org/pdf/2101.09258) loss, i.e. maximum likelihood.
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

        assert cfg.sampler in {
            "ddim",
            "ddpm",
            "dpmpp",
        }, "sampler must be one of ddim, ddpm, dpmpp"

        assert cfg.diffusion_objective in {
            "pred_noise",
            "pred_x0",
            "pred_v",
        }, "objective must be one of pred_noise, pred_x0, pred_v"

        if cfg.train_schedule == "simple_linear":
            alpha_schedule = simple_linear_schedule
        elif cfg.train_schedule == "beta_linear":
            alpha_schedule = beta_linear_schedule
        elif cfg.train_schedule == "cosine":
            alpha_schedule = cosine_schedule
        elif cfg.train_schedule == "sigmoid":
            alpha_schedule = sigmoid_schedule
        else:
            raise ValueError(f"invalid noise schedule {cfg.train_schedule}")

        self.train_schedule = partial(
            time_to_alpha, alpha_schedule=alpha_schedule, scale=cfg.schedule_scale
        )

        if cfg.sampling_schedule is None:
            sampling_alpha_schedule = None
        elif cfg.sampling_schedule == "simple_linear":
            sampling_alpha_schedule = simple_linear_schedule
        elif cfg.sampling_schedule == "beta_linear":
            sampling_alpha_schedule = beta_linear_schedule
        elif cfg.sampling_schedule == "cosine":
            sampling_alpha_schedule = cosine_schedule
        elif cfg.sampling_schedule == "sigmoid":
            sampling_alpha_schedule = sigmoid_schedule
        else:
            raise ValueError(f"invalid sampling schedule {cfg.sampling_schedule}")

        if exists(sampling_alpha_schedule):
            self.sampling_schedule = partial(
                time_to_alpha,
                alpha_schedule=sampling_alpha_schedule,
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

        # Setup dataset and freeze it (since it contains models)
        self.dataset = hydra.utils.instantiate(cfg.dataset)
        self.dataset.requires_grad_(False)
        self.train_data, self.val_data = random_split(
            self.dataset, [1 - cfg.val_split, cfg.val_split]
        )

        AETask.load_from_checkpoint(
            os.path.join(
                os.environ["LATENT_CONTROL_CKPT_DIR"],
                cfg.pretrained_id,
                "last.ckpt",
            ),
            strict=False,
        )

        self.model = DiT(cfg.model)

        self.cfg = cfg
        # Important for checkpoints
        self.save_hyperparameters(
            OmegaConf.to_container(OmegaConf.structured(cfg)), logger=False
        )

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

    def compute_diffusion_loss(
        self,
        latent,
        class_id=None,
        cond=None,
        cond_input_ids=None,
        cond_ignore_mask=None,
    ):
        # NOTE: Important to flip the <ignore_mask> to a <don't_ignore_mask>
        cond_mask = None
        if cond_ignore_mask != None:
            cond_mask = torch.logical_not(cond_ignore_mask)

        bs, l, d = (*latent.shape,)
        device = latent.device

        times = torch.zeros((bs,), device=device).float().uniform_(0, 1.0)
        noise = torch.randn_like(latent)

        alpha = self.train_schedule(times)
        alpha = right_pad_dims_to(latent, alpha)

        z_t = alpha.sqrt() * latent + (1 - alpha).sqrt() * noise

        # Sample unconditionally with some probability
        if self.model.cfg.seq_conditional and (
            random.random() < self.model.cfg.seq_unconditional_prob
        ):
            cond = None
            cond_input_ids = None
            cond_mask = None

        if (
            self.model.cfg.class_conditional
            and self.model.cfg.class_unconditional_prob > 0
        ):
            assert exists(class_id)
            class_unconditional_mask = self.model.class_unconditional_bernoulli.sample(
                class_id.shape
            ).bool()
            class_id[class_unconditional_mask] = self.model.cfg.num_classes

        self_cond = None

        if self.model.cfg.self_condition and (
            random.random() < self.model.cfg.train_prob_self_cond
        ):
            with torch.no_grad():
                model_output = self.diffusion_model_predictions(
                    z_t,
                    times,
                    class_id=class_id,
                    cond=cond,
                    cond_mask=cond_mask,
                )
                self_cond = model_output.pred_x_start.detach()

        # predict and take gradient step

        predictions = self.diffusion_model_predictions(
            z_t,
            times,
            x_self_cond=self_cond,
            class_id=class_id,
            cond=cond,
            cond_input_ids=cond_input_ids,
            cond_mask=cond_mask,
        )

        if self.cfg.diffusion_objective == "pred_x0":
            target = latent
            pred = predictions.pred_x_start
        elif self.cfg.diffusion_objective == "pred_noise":
            target = noise
            pred = predictions.pred_noise
        elif self.cfg.diffusion_objective == "pred_v":
            target = alpha.sqrt() * noise - (1 - alpha).sqrt() * latent
            assert exists(predictions.pred_v)
            pred = predictions.pred_v

        loss = self.loss_fn(pred, target, reduction="none")
        loss = rearrange(
            [reduce(loss[i], "l d -> 1", "mean") for i in range(latent.shape[0])],
            "b 1 -> b 1",
        )

        return loss.mean()

    def training_step(self, batch, batch_idx=None):

        latent = batch["latent"]
        if self.cfg.normalize_latent:
            latent_ = rearrange(latent, "b s d -> (b s) d")
            self.latent_mean = torch.mean(latent_, dim=0)
            self.latent_scale = torch.std(latent_ - self.latent_mean, unbiased=False)
            latent = self.normalize_latent(latent)

        loss = self.compute_diffusion_loss(
            latent,
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
