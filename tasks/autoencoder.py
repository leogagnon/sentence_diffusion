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
import wandb


@dataclass
class AETaskConfig:
    lr: float
    batch_size: int
    encoder: EncoderConfig
    decoder: DecoderConfig
    variational: bool
    input_sub_p: float
    kl_beta: float
    dataset: StoriesDatasetConfig


class AETask(L.LightningModule):
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
                int(len(self.dataset) * 0.9),
                len(self.dataset) - int(len(self.dataset) * 0.9),
            ],
        )

        self.bleu = evaluate.load("bleu")

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

    def random_substitution(self, inputs):

        probability = torch.full(
            inputs.shape,
            self.cfg.input_sub_p,
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
            batch_size=self.cfg.batch_size,
            shuffle=True,
            collate_fn=lambda x: x,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_data,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            collate_fn=lambda x: x,
        )

    def configure_optimizers(self):
        optimizer = torch.optim.Adamax(self.parameters(), lr=self.cfg.lr)
        return optimizer

    def training_step(self, batch, batch_idx):

        if self.cfg.input_sub_p > 0:
            input_ids_enc = self.random_substitution(batch["input_ids_enc"])

        z = self.encoder(input_ids_enc, attention_mask=batch["attention_mask_enc"])

        if self.cfg.variational:
            mean = self.fc_mean(z)
            log_var = self.fc_log_var(z)
            z = self.reparameterize(mean, log_var)

        # Get embeddings of input_ids
        logits = self.decoder(
            batch["input_ids_dec"], z, attention_mask=batch["attention_mask_dec"]
        )

        # Ignore padding tokens
        targets = batch["input_ids_dec"].masked_fill(
            batch["attention_mask_dec"] == 0, -1
        )

        logits = logits[:,:-1].contiguous()
        targets = targets[:,1:].contiguous()

        # Apply cross-entropy loss
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1),
            ignore_index=-1,
        )

        if self.cfg.variational:
            KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
            loss += self.cfg.kl_beta * KLD

        return loss

    def validation_step(self, batch, batch_idx):

        input_ids_enc_clean = batch["input_ids_enc"]
        input_ids_enc_corrupted = self.random_substitution(input_ids_enc_clean)

        z_clean = self.encoder(
            input_ids_enc_clean, attention_mask=batch["attention_mask_enc"]
        )
        z_corrupted = self.encoder(
            input_ids_enc_corrupted, attention_mask=batch["attention_mask_enc"]
        )
        if self.cfg.variational:
            mean_clean = self.fc_mean(z_clean)
            log_var_clean = self.fc_log_var(z_clean)
            z_clean = self.reparameterize(mean_clean, log_var_clean)

            mean_corrupted = self.fc_mean(z_corrupted)
            log_var_corrupted = self.fc_log_var(z_corrupted)
            z_corrupted = self.reparameterize(mean_corrupted, log_var_corrupted)

        # Get embeddings of input_ids
        logits = self.decoder(
            batch["input_ids_dec"], z_clean, attention_mask=batch["attention_mask_dec"]
        )
        targets = batch["input_ids_dec"].masked_fill(
            batch["attention_mask_dec"] == 0, -1
        )
        logits = logits[:,:-1].contiguous()
        targets = targets[:,1:].contiguous()
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1),
            ignore_index=-1,
        )

        if self.cfg.variational:
            KLD = -0.5 * torch.sum(
                1 + log_var_clean - mean_clean.pow(2) - log_var_clean.exp()
            )
            loss += self.cfg.kl_beta * KLD

        wandb.log({"loss": loss})
        wandb.log({"KLD": KLD})

        gen_clean, gen_corrupted = [
            self.decoder.tokenizer.batch_decode(
                self.decoder.generate_from(z, max_length=60)
            )
            for z in (z_clean, z_corrupted)
        ]

        bleu_clean = self.bleu.compute(
            predictions=gen_clean, references=batch["input_str"]
        )['bleu']
        bleu_corrupted = self.bleu.compute(
            predictions=gen_corrupted, references=batch["input_str"]
        )['bleu']

        z_groups = [
            z_clean[indices]
            for indices in torch.randperm(input_ids_enc_clean.shape[0]).chunk(2)
        ]
        z_interp = 0.5 * z_groups[0] + 0.5 * z_groups[1]
        gen_interp_ids = self.decoder.generate_from(z_interp, max_length=60)

        with self.decoder.backbone.disable_adapter():
            # Disable LoRA layers for perplexity eval
            ppl_interp = torch.exp(self.decoder.backbone(input_ids=gen_interp_ids, labels=gen_interp_ids).loss)

        wandb.log({"bleu_clean": bleu_clean})
        wandb.log({"bleu_corrupted": bleu_corrupted})
        wandb.log({"ppl_interp": ppl_interp.item()})
