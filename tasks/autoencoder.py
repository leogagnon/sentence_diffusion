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


@dataclass
class DecoderTaskConfig:
    lr: float
    batch_size: int
    lm_name: str
    encoder: dict
    lora_config: dict
    dataset: StoriesDatasetConfig


class AETask(L.LightningModule):
    def __init__(self, cfg: Optional[DecoderTaskConfig] = None, **kwargs):
        super().__init__()

        if cfg == None:
            cfg = OmegaConf.to_object(
                OmegaConf.merge(
                    OmegaConf.create(DecoderTaskConfig),
                    OmegaConf.create(kwargs),
                )
            )

        self.cfg = cfg

        # Load language model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained(cfg.lm_name, add_bos_token=False)
        self.decoder_lm = get_peft_model(
            AutoModelForCausalLM.from_pretrained(cfg.lm_name, device_map="auto"),
            LoraConfig(**cfg.lora_config),
        )
        self.bos_token = self.tokenizer.bos_token_id

        self.encoder = instantiate(cfg.encoder)

        # Load prompt generator (use the model's dimension as <dim>)
        if "VariationalEncoder" in cfg.encoder["target"]:
            cfg.encoder["out_dim"] = self.decoder_lm.config.hidden_size

        # Remove dropout
        for mod in self.decoder_lm.modules():
            if isinstance(mod, torch.nn.Dropout):
                mod.p = 0.0

        self.dataset = StoriesDataset(cfg.dataset, tokenizer)

    def train_dataloader(self):
        return DataLoader(
            self.dataset,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            collate_fn=lambda x: x,
        )

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.cfg.lr)
        return optimizer

    def training_step(self, batch, batch_idx):

        loss = 0

        if "VariationalEncoder" in self.cfg.encoder["target"]:
            z, enc_loss = self.encoder(batch["sentences"])
            loss += enc_loss
        else:
            z = self.encoder(batch["input_ids_enc"])

        # Get embeddings of input_ids
        input_ids_dec = torch.cat(
            [
                torch.full_like(
                    batch["input_ids"][:, [0]],
                    self.bos_token,
                ),
                batch["input_ids_dec"],
            ],
            dim=1,
        )
        tokens = self.decoder_lm.get_input_embeddings()(input_ids_dec)
        tokens = torch.cat([z, tokens], dim=1)

        # Forward pass
        output = self.decoder_lm(inputs_embeds=tokens)
        logits = output.logits[:, z.shape[1] : -1, :]
        logits = logits.contiguous()

        # Apply cross-entropy loss
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)), batch["input_ids"].view(-1)
        )

        if enc_loss is not None:
            loss += enc_loss
            
        return loss
