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
from model.gaussian_diffusion import DiTConfig, DiT
from data.stories import StoriesDatasetConfig, StoriesDataset
from hydra.utils import instantiate
from model.encoder import EncoderConfig, EncoderModel
from model.decoder import DecoderConfig, DecoderModel
import evaluate
import os
import wandb
from tasks.finetune import FinetuneTask
import hydra


@dataclass
class AETaskConfig:
    lr: float
    train_batch_size: int
    val_batch_size: int
    encoder: dict
    decoder: DecoderConfig
    max_generation_length: int
    dataset: StoriesDatasetConfig
    val_size: int
    z_noise_alpha: float
    z_dropout_p: float
    has_noising: bool = True
    name: Optional[str] = None


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
        self.encoder = hydra.utils.instantiate(cfg.encoder)
        cfg.decoder.prompt_generator_cfg.input_dim = self.encoder.latent_dim
        self.decoder = DecoderModel(cfg.decoder)
        
        for mod in self.decoder.modules():
            if isinstance(mod, torch.nn.Dropout):
                mod.p = 0.0

        # Setup dataset
        self.dataset = StoriesDataset(
            cfg.dataset,
            dec_tokenizer=self.decoder.tokenizer,
            enc_tokenizer=self.encoder.tokenizer,
        )

        indices = torch.randperm(len(self.dataset))
        self.register_buffer("train_indices", indices[:-4096])
        self.register_buffer("val_indices", indices[-4096:])

        self.cfg = cfg

        self.save_hyperparameters(
            OmegaConf.to_container(OmegaConf.structured(cfg)), logger=False
        )

    def setup(self, **kwargs):
        """Setup the data"""
        self.train_data = Subset(self.dataset, indices=self.train_indices)
        self.val_data = Subset(
            self.dataset, indices=self.val_indices[: self.cfg.val_size]
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_data,
            batch_size=self.cfg.train_batch_size,
            shuffle=True,
            collate_fn=lambda x: x,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_data,
            batch_size=self.cfg.val_batch_size,
            shuffle=False,
            collate_fn=lambda x: x,
        )

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.cfg.lr)
        return optimizer

    def training_step(self, batch, batch_idx):

        assert self.encoder.training == True

        z = self.encoder(batch["input_ids_enc"], batch["attention_mask_enc"])

        # Add noise and dropout to latent code
        # This helps make the decoder more robust
        if self.cfg.has_noising:
            if self.cfg.z_noise_alpha < 1.0:
                z = self.cfg.z_noise_alpha * z + (
                    1 - math.sqrt(self.cfg.z_noise_alpha)
                ) * torch.randn_like(z)

            if self.cfg.z_dropout_p > 0.0:
                mask = torch.rand_like(z) < self.cfg.z_dropout_p
                z = z.masked_fill(mask, 0.0)

        # Get embeddings of input_ids
        logits = self.decoder(batch["input_ids_dec"], z)

        # Ignore padding tokens
        targets = batch["input_ids_dec"].masked_fill(
            batch["attention_mask_dec"] == 0, -1
        )

        logits = logits[:, :-1].contiguous()
        targets = targets[:, 1:].contiguous()

        # Apply cross-entropy loss
        recon_loss = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1),
            ignore_index=-1,
        )

        self.log("train/reconstruction_loss", recon_loss, on_epoch=False, on_step=True)

        return recon_loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):

        assert self.encoder.training == False

        # Encode input_ids
        z = self.encoder(batch["input_ids_enc"], batch["attention_mask_enc"])

        # Reconstruction loss of clean sample
        logits = self.decoder(batch["input_ids_dec"], z)
        targets = batch["input_ids_dec"].masked_fill(
            batch["attention_mask_dec"] == 0, -1
        )
        logits = logits[:, :-1].contiguous()
        targets = targets[:, 1:].contiguous()
        recon_loss = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1),
            ignore_index=-1,
        )
        self.log("val/reconstruction_loss", recon_loss, on_epoch=True)

        # Reconstruction loss of noised sample
        z_noised = 0.90 * z + (1 - math.sqrt(0.90)) * torch.randn_like(z)
        logits = self.decoder(batch["input_ids_dec"], z_noised)
        logits = logits[:, :-1].contiguous()
        recon_loss_noised = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1),
            ignore_index=-1,
        )
        self.log("val/reconstruction_loss_noised", recon_loss_noised, on_epoch=True)        

        # Log some generation for the first batch
        if batch_idx == 0:
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

            # Log generation from noised samples
            table_noised = wandb.Table(columns=["Original", "Noise+Reconstructed"])
            for original, reconstructed in zip(
                batch["input_str"][:10],
                self.decoder.tokenizer.batch_decode(
                    self.decoder.generate(
                        z=z_noised[:10], max_length=self.cfg.max_generation_length
                    ),
                    skip_special_tokens=True,
                ),
            ):
                table_noised.add_data(original, reconstructed)
            wandb.log({"val/noised_samples": table_noised})
