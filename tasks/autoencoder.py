from functools import partial
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
from transformers import AutoTokenizer
from transformers.models.auto.modeling_auto import AutoModelForCausalLM
from peft import get_peft_model
from model.diffusion_transformer import DiTConfig, DiT
from data.stories import StoriesDatasetConfig, StoriesDataset
from hydra.utils import instantiate
from model.encoder import EncoderConfig, EncoderModel
from model.decoder import DecoderConfig, DecoderModel
import evaluate
import os
import wandb


@dataclass
class AETaskConfig:
    lr: float
    train_batch_size: int
    val_batch_size: int
    encoder: EncoderConfig
    decoder: DecoderConfig
    variational: bool
    input_sub_p: float
    kl_beta: float
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
        - BLEU score of reconstructed clean text
        - BLEU score of reconstructed corrupted text (with random token substitutions in the input)
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

        # Load decoder
        self.decoder = DecoderModel(cfg.decoder)

        # Load encoder
        self.encoder = EncoderModel(cfg.encoder)

        self.dataset = StoriesDataset(
            cfg.dataset,
            enc_tokenizer=self.encoder.tokenizer,
            dec_tokenizer=self.decoder.tokenizer,
        )
        self.train_data, self.val_data = random_split(
            self.dataset,
            [
                len(self.dataset) - cfg.val_size,
                cfg.val_size,
            ],
        )

        self.bleu = evaluate.load("bleu", experiment_id=os.urandom(15).hex()) # To avoid cache conflicts

        # Make sure there is no dropout in the decoder
        for mod in self.decoder.modules():
            if isinstance(mod, torch.nn.Dropout):
                mod.p = 0.0

        # Initialize variational/output projections
        in_dim = self.encoder.backbone.config.hidden_size
        out_dim = self.decoder.backbone.config.hidden_size
        if cfg.variational:
            self.fc_mean = torch.nn.Linear(in_dim, out_dim)
            self.fc_log_var = torch.nn.Linear(in_dim, out_dim)
        else:
            self.out_proj = torch.nn.Linear(
                in_dim,
                out_dim,
            )

        self.cfg = cfg
        
        self.save_hyperparameters(
            OmegaConf.to_container(OmegaConf.structured(cfg)), logger=False
        )

    def random_substitution(self, inputs, p=None):
        inputs = inputs.clone()
        probability = torch.full(
            inputs.shape,
            p if p is not None else self.cfg.input_sub_p,
            dtype=torch.float32,
            device=inputs.device,
        )

        masked_indices = torch.bernoulli(probability).bool()
        random_words = torch.randint(
            len(self.encoder.tokenizer),
            inputs.shape,
            dtype=torch.long,
            device=inputs.device,
        )
        inputs[masked_indices] = random_words[masked_indices]

        return inputs

    def reparameterize(self, mean, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mean)

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

        loss = 0.0

        input_ids_enc = batch["input_ids_enc"]
        if self.cfg.input_sub_p > 0:
            input_ids_enc = self.random_substitution(input_ids_enc)

        z = self.encoder(input_ids_enc, attention_mask=batch["attention_mask_enc"])

        if self.cfg.variational:
            mean = self.fc_mean(z)
            log_var = self.fc_log_var(z)
            z = self.reparameterize(mean, log_var)

            KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
            loss += self.cfg.kl_beta * KLD
            wandb.log({"train/KLD": KLD})
        else:
            z = self.out_proj(z)

        # Get embeddings of input_ids
        logits = self.decoder(
            batch["input_ids_dec"], z, attention_mask=batch["attention_mask_dec"]
        )

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
        loss += recon_loss

        wandb.log({"train/reconstruction_loss": recon_loss})

        return loss

    def validation_step(self, batch, batch_idx):

        # Encode clean and corrupted inputs
        loss = 0.0

        input_ids_enc_clean = batch["input_ids_enc"]
        input_ids_enc_corrupted = self.random_substitution(input_ids_enc_clean, p=0.3)

        z_clean = self.encoder(
            input_ids_enc_clean, attention_mask=batch["attention_mask_enc"]
        )
        z_corrupted = self.encoder(
            input_ids_enc_corrupted, attention_mask=batch["attention_mask_enc"]
        )

        if self.cfg.variational:
            mean_clean = self.fc_mean(z_clean)
            log_val_clean = self.fc_log_var(z_clean)
            KLD =  -0.5 * torch.sum(1 + log_val_clean - mean_clean.pow(2) - log_val_clean.exp())
            self.log("val/KLD", KLD, on_epoch=True)
            loss += self.cfg.kl_beta * KLD

            # Note: use mean for evaluation
            z_clean = mean_clean
            z_corrupted = self.fc_mean(z_corrupted)
        else:
            z_clean = self.out_proj(z_clean)
            z_corrupted = self.out_proj(z_corrupted)

        # Evaluate loss for clean inputs
        logits = self.decoder(
            batch["input_ids_dec"], z_clean, attention_mask=batch["attention_mask_dec"]
        )
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
        loss += recon_loss

        # Decode from clean and corrupted latents and evaluate BLEU
        gen_clean, gen_corrupted = [
            self.decoder.tokenizer.batch_decode(
                self.decoder.generate_from(z, max_length=self.cfg.max_generation_length),
                skip_special_tokens=True
            )
            for z in (z_clean, z_corrupted)
        ]

        if batch_idx == 0:
            # Log a table of clean and reconstructed sentences to W&B
            table = wandb.Table(columns=["Clean", "Reconstructed"])
            for clean, reconstructed in zip(batch["input_str"][:10], gen_clean[:10]):
                table.add_data(clean, reconstructed)
            wandb.log({"val/clean_samples": table})

        bleu_clean = self.bleu.compute(
            predictions=gen_clean, references=batch["input_str"]
        )["bleu"]
        bleu_corrupted = self.bleu.compute(
            predictions=gen_corrupted, references=batch["input_str"]
        )["bleu"]

        self.log("val/bleu_clean", bleu_clean, on_epoch=True)
        self.log("val/bleu_corrupted", bleu_corrupted, on_epoch=True)

        # Interpolate between pairs of clean latents and evaluate perplexity
        group_indices = torch.randperm(input_ids_enc_clean.shape[0]).chunk(2)
        z_groups = [z_clean[indices] for indices in group_indices]
        z_interp = 0.5 * z_groups[0] + 0.5 * z_groups[1]
        gen_interp_ids = self.decoder.generate_from(
            z_interp, max_length=self.cfg.max_generation_length
        )
        if batch_idx == 0:
            # Log a table of clean and reconstructed sentences to W&B
            table = wandb.Table(columns=["S1", "S2", "Interpolated"])
            for s1, s2, s_interp in zip(
                [batch["input_str"][i] for i in group_indices[0]][:10],
                [batch["input_str"][i] for i in group_indices[1]][:10],
                self.decoder.tokenizer.batch_decode(gen_interp_ids[:10], skip_special_tokens=True),
            ):
                table.add_data(s1, s2, s_interp)
            wandb.log({"val/interp_samples": table})

        with self.decoder.backbone.disable_adapter():
            # Disable LoRA layers for perplexity eval
            ppl_interp = torch.exp(
                self.decoder.backbone(
                    input_ids=gen_interp_ids, labels=gen_interp_ids
                ).loss
            )

        self.log("val/ppl_interp", ppl_interp.item(), on_epoch=True)
