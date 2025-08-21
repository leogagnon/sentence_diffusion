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
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from transformers.models.auto.modeling_auto import AutoModelForCausalLM
from peft import get_peft_model
from model.diffusion_transformer import DiTConfig, DiT
from data.stories.main import StoriesDatasetConfig, StoriesDataset
from hydra.utils import instantiate
from model.encoder import EncoderConfig, EncoderModel
from model.decoder import DecoderConfig, DecoderModel


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

        # Load tokenizers
        enc_tokenizer = AutoTokenizer.from_pretrained(
            cfg.encoder.name, add_bos_token=False
        )
        dec_tokenizer = AutoTokenizer.from_pretrained(
            cfg.decoder.name, add_bos_token=False
        )
        # Make sure decoder has padding token (i.e. GPT2 doesn't)
        if dec_tokenizer.pad_token_id is None:
            dec_tokenizer.add_special_tokens({"pad_token": "[PAD]"})
            self.decoder.backbone.resize_token_embeddings(len(dec_tokenizer))
        self.dec_bos_token = dec_tokenizer.bos_token_id
        self.enc_vocab_size = len(enc_tokenizer)
        self.dataset = StoriesDataset(
            cfg.dataset, tokenizers={"enc": enc_tokenizer, "dec": dec_tokenizer}
        )

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
            self.enc_vocab_size, inputs.shape, dtype=torch.long, device=inputs.device
        )
        inputs[masked_indices] = random_words[masked_indices]

        return inputs

    def reparameterize(self, mean, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mean)

    def train_dataloader(self):
        return DataLoader(
            self.dataset,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            collate_fn=lambda x: x,
        )

    def configure_optimizers(self):
        optimizer = torch.optim.Adamax(self.parameters(), lr=self.cfg.lr)
        return optimizer

    def training_step(self, batch, batch_idx):

        if self.cfg.input_sub_p > 0:
            input_ids_enc = self.random_substitution(batch["input_ids_enc"])

        z = self.encoder(input_ids_enc, attention_mask=batch["padding_mask_enc"])

        if self.cfg.variational:
            mean = self.fc_mean(z)
            log_var = self.fc_log_var(z)
            z = self.reparameterize(mean, log_var)

        # Get embeddings of input_ids
        logits = self.decoder(
            batch["input_ids_dec"], z, attention_mask=batch["padding_mask_dec"]
        )

        # Ignore padding tokens
        targets = batch["input_ids_dec"].masked_fill(batch["padding_mask_dec"], -1)

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
