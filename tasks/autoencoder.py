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


@dataclass
class AETaskConfig:
    lr: float
    train_batch_size: int
    val_batch_size: int
    encoder: EncoderConfig
    decoder: DecoderConfig
    max_generation_length: int
    dataset: StoriesDatasetConfig
    val_size: int
    name: Optional[str] = None


class AETask(L.LightningModule):
    """
    Autoencoder Task. Combines an encoder and a decoder model to form an autoencoder.
    Trains the autoencoder to reconstruct the input text.
    Supports variational autoencoding.
    Evaluates using
        - Perplexity of interpolated samples in the latent space
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

        # Load decoder and encoder
        self.decoder = DecoderModel(cfg.decoder)
        cfg.encoder.out_proj_dim = self.decoder.backbone.config.hidden_size
        self.encoder = EncoderModel(cfg.encoder)
        for mod in self.decoder.modules():
            if isinstance(mod, torch.nn.Dropout):
                mod.p = 0.0

        # Setup dataset
        self.dataset = StoriesDataset(
            cfg.dataset,
            dec_tokenizer=self.decoder.tokenizer,
        )

        indices = torch.randperm(len(self.dataset))
        self.register_buffer("train_indices", indices[: -4096])
        self.register_buffer("val_indices", indices[-4096 :])

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
        optimizer = torch.optim.Adamax(self.parameters(), lr=self.cfg.lr)
        return optimizer

    def training_step(self, batch, batch_idx):

        z = self.encoder(batch["input_str"])

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

        # Encode input_ids
        z = self.encoder(batch["input_str"])

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
        z_noised = z + (0.7 * torch.randn_like(z))
        logits = self.decoder(batch["input_ids_dec"], z_noised)
        logits = logits[:, :-1].contiguous()
        recon_loss_noised = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1),
            ignore_index=-1,
        )
        self.log("val/reconstruction_loss_noised", recon_loss_noised, on_epoch=True)

        # Interpolation evaluation
        group_indices = torch.randperm(z.shape[0]).chunk(2)
        z_groups = [z[indices] for indices in group_indices]
        z_interp = 0.5 * z_groups[0] + 0.5 * z_groups[1]
        gen_interp_ids = self.decoder.generate(
            z=z_interp, max_length=self.cfg.max_generation_length
        )
        with self.decoder.backbone.disable_adapter():
            # Add BOS and EOS tokens
            bos = torch.full_like(
                gen_interp_ids[:, [0]],
                self.decoder.tokenizer.bos_token_id,
            )
            eos = torch.full_like(
                gen_interp_ids[:, [0]],
                self.decoder.tokenizer.eos_token_id,
            )
            gen_interp_ids_ = torch.cat([bos, gen_interp_ids, eos], dim=1)
            # Evaluate perplexity of interpolated samples with pre-trained decoder
            mask = gen_interp_ids_ != self.decoder.tokenizer.pad_token_id
            labels = gen_interp_ids_.masked_fill(~mask, -100)
            ppl_interp = torch.exp(
                self.decoder.backbone(
                    input_ids=gen_interp_ids_,
                    labels=labels,
                    attention_mask=mask,
                ).loss
            )
        self.log("val/ppl_interp", ppl_interp.item(), on_epoch=True)

        # Log some generation for the first batch
        if batch_idx == 0:
            # Log generations from interpolated samples
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
