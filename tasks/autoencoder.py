from functools import partial
import math
import os
import lightning as L
from omegaconf import OmegaConf
from dataclasses import dataclass
from typing import Any, List, Optional
from peft import LoraConfig
import torch
import random
import numpy as np
from torch.utils.data import DataLoader, random_split
from torch.utils.data.dataset import Subset
from transformers import AutoTokenizer
from transformers.models.auto.modeling_auto import AutoModelForCausalLM
from peft import get_peft_model
from model.gaussian_diffusion import DiTConfig, DiT, right_pad_dims_to
from data.stories import StoriesDatasetConfig, StoriesDataset
from hydra.utils import instantiate
from model.encoder import EncoderConfig, EncoderModel
from model.decoder import DecoderConfig, DecoderModel
import evaluate
import os
import wandb
import hydra
from model.gaussian_diffusion import time_to_alpha, cosine_schedule
from lightning.pytorch.utilities.rank_zero import rank_zero_info, rank_zero_only


def reparameterize(mean, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return eps.mul(std).add_(mean)


@dataclass
class AETaskConfig:
    lr: float
    batch_size: int
    decoder: DecoderConfig
    max_generation_length: int
    dataset: dict
    val_size: int
    encoder: Optional[dict] = None

    data_seed: int = 42

    z_noise_type: str = "none"  # fixed, schedule, none
    z_noise_alpha: float = 0.999
    z_dropout_p: float = 0.0
    kl_beta: float = 1e-5

    name: Optional[str] = None
    encoder_prompt: Optional[str] = None


class AETask(L.LightningModule):
    """
    Autoencoder Task.
    """

    def __init__(self, cfg: Optional[AETaskConfig] = None, **kwargs):
        super().__init__()

        if cfg == None:
            cfg = OmegaConf.to_object(
                OmegaConf.merge(
                    OmegaConf.create(AETaskConfig),
                    OmegaConf.create(kwargs),
                )
            )

        # Load encoder and decoder
        if cfg.encoder != None:
            self.encoder = hydra.utils.instantiate(cfg.encoder)
            cfg.decoder.input_dim = self.encoder.latent_dim

        if cfg.z_noise_type == "variational":
            assert (
                self.encoder.cfg.variational == True
            ), "Encoder must be variational if z_noise_type is variational"

        self.decoder = DecoderModel(cfg.decoder)

        # Setup dataset
        self.dataset = hydra.utils.instantiate(cfg.dataset)

        indices = torch.randperm(
            len(self.dataset),
            generator=torch.Generator().manual_seed(cfg.data_seed),
        )
        self.train_indices = indices[: -cfg.val_size]
        self.val_indices = indices[-cfg.val_size :]

        self.cfg = cfg

        self.save_hyperparameters(
            OmegaConf.to_container(OmegaConf.structured(cfg)), logger=False
        )

    def sample_alpha(self, z):
        if self.cfg.z_noise_type == "fixed":
            return torch.full(
                size=(z.size(0),), fill_value=self.cfg.z_noise_alpha, device=z.device
            )
        elif self.cfg.z_noise_type == "schedule":
            t = torch.zeros((z.size(0),), device=z.device).float().uniform_(0, 1.0)
            return time_to_alpha(t=t, alpha_schedule=cosine_schedule, scale=3.0)
        else:
            raise ValueError(f"Unknown z_noise_type {self.cfg.z_noise_type}")

    def setup(self, **kwargs):
        """Setup the data"""
        self.train_data = Subset(self.dataset, indices=self.train_indices)
        self.val_data = Subset(self.dataset, indices=self.val_indices)

    def train_dataloader(self):
        return DataLoader(
            self.train_data,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            collate_fn=self.dataset.get_collate_and_tokenize_fn(
                enc_tokenizer=(
                    self.encoder.tokenizer if self.cfg.encoder != None else None
                ),
                dec_tokenizer=self.decoder.tokenizer,
                prompt=self.cfg.encoder_prompt,
            ),
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_data,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            collate_fn=self.dataset.get_collate_and_tokenize_fn(
                enc_tokenizer=(
                    self.encoder.tokenizer if self.cfg.encoder != None else None
                ),
                dec_tokenizer=self.decoder.tokenizer,
                prompt=self.cfg.encoder_prompt,
            ),
        )

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.cfg.lr)
        return optimizer

    def training_step(self, batch, batch_idx):

        # If there is an encoder, compute z and potentially add noise
        z = None
        alpha = None
        if self.cfg.encoder != None:
            z = self.encoder(batch["input_ids_enc"], batch["attention_mask_enc"])

            if self.cfg.z_noise_type == "variational":
                # Variational AE style reparameterization
                mean, logvar = z
                z = reparameterize(mean, logvar)
            elif self.cfg.z_noise_type != "none":
                # Add scale-invariant noise to z
                # Keeps norm of z roughly the same and SNR controlled by alpha
                alpha = self.sample_alpha(z)
                alpha = right_pad_dims_to(z, alpha)

                z_flat = z.view(z.shape[0], -1)
                norm = z_flat.norm(dim=1, keepdim=True).detach()
                w = (norm / (z_flat.shape[1] ** 0.5)).clamp_min(1e-6)
                w = right_pad_dims_to(z, w)

                z = alpha.sqrt() * z + (1 - alpha).sqrt() * w * torch.randn_like(z)

            # Apply z dropout
            if self.cfg.z_dropout_p > 0.0:
                mask = torch.rand_like(z) < self.cfg.z_dropout_p
                z = z.masked_fill(mask, 0.0)

        # Compute decoder likelihood of input_ids
        logits = self.decoder(input_ids=batch["input_ids_dec"], z=z, alpha=alpha)

        # Compute loss
        targets = batch["input_ids_dec"].masked_fill(
            batch["attention_mask_dec"] == 0, -100
        )
        logits = logits[:, :-1].contiguous()
        targets = targets[:, 1:].contiguous()
        recon_loss = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1),
            ignore_index=-100,
        )

        self.log(
            "train/reconstruction_loss",
            recon_loss,
            on_epoch=False,
            on_step=True,
            sync_dist=True,
        )

        # If variational, add KL loss
        if self.cfg.z_noise_type == "variational":
            kl_loss = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
            recon_loss += self.cfg.kl_beta * kl_loss
            self.log(
                "train/KL_loss",
                kl_loss,
                on_epoch=False,
                on_step=True,
                sync_dist=True,
            )

        return recon_loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):

        # If there is an encoder, compute z and potentially add noise
        z = None
        if self.cfg.encoder != None:
            z = self.encoder(batch["input_ids_enc"], batch["attention_mask_enc"])

            self.log(
                "val/latent_norm",
                z.norm(p=2, dim=-1).mean().detach().item(),
                on_epoch=True,
                sync_dist=True,
            )

        # Compute clean reconstruction loss
        logits = self.decoder(batch["input_ids_dec"], z)
        targets = batch["input_ids_dec"].masked_fill(
            batch["attention_mask_dec"] == 0, -100
        )
        logits = logits[:, :-1].contiguous()
        targets = targets[:, 1:].contiguous()
        recon_loss = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1),
            ignore_index=-100,
        )
        self.log("val/reconstruction_loss", recon_loss, on_epoch=True, sync_dist=True)

        # Maybe log some generations
        if (batch_idx == 0) and (rank_zero_only.rank == 0) and (z != None):
            # Log generations from interpolated samples
            group_indices = torch.randperm(z.shape[0]).chunk(2)
            z_groups = [z[indices] for indices in group_indices]
            z_interp = 0.5 * z_groups[0] + 0.5 * z_groups[1]
            gen_interp_ids = self.decoder.generate(
                z=z_interp, max_length=self.cfg.max_generation_length
            )
            table = wandb.Table(columns=["S1", "S2", "Interpolated"])
            for s1, s2, s_interp in zip(
                [batch["input_str"][i] for i in group_indices[0]][:10],
                [batch["input_str"][i] for i in group_indices[1]][:10],
                self.decoder.tokenizer.batch_decode(
                    gen_interp_ids[:10], skip_special_tokens=True
                ),
            ):
                table.add_data(s1, s2, s_interp)
            wandb.log({"val/interp_samples": table})

            # Log generation from clean samples
            table_clean = wandb.Table(columns=["Original", "Reconstructed"])
            for original, reconstructed in zip(
                batch["input_str"][:10],
                self.decoder.tokenizer.batch_decode(
                    self.decoder.generate(
                        z=z[:10], max_length=self.cfg.max_generation_length
                    ),
                    skip_special_tokens=True,
                ),
            ):
                table_clean.add_data(original, reconstructed)
            wandb.log({"val/clean_samples": table_clean})
