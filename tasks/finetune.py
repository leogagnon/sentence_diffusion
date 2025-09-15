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


@dataclass
class FinetuneTaskConfig:
    lr: float
    batch_size: int
    decoder: DecoderConfig
    max_generation_length: int
    dataset: StoriesDatasetConfig
    val_size: int

class FinetuneTask(L.LightningModule):
    """
    Finetunes a pretrained decoder (e.g. GPT) on the stories dataset.
    """

    def __init__(self, cfg: Optional[FinetuneTaskConfig] = None, **kwargs) -> None:
        super().__init__()

        if cfg == None:
            cfg = OmegaConf.to_object(
                OmegaConf.merge(
                    OmegaConf.create(FinetuneTaskConfig),
                    OmegaConf.create(kwargs),
                )
            )

        self.decoder = DecoderModel(cfg.decoder)

        self.dataset = StoriesDataset(
            cfg.dataset,
            dec_tokenizer=self.decoder.tokenizer,
        )
        # Randomly choose split into train and val
        indices = torch.randperm(len(self.dataset))
        self.register_buffer("train_indices", indices[: -cfg.val_size])
        self.register_buffer("val_indices", indices[-cfg.val_size :])

        self.cfg = cfg
        # Important for checkpoints
        self.save_hyperparameters(
            OmegaConf.to_container(OmegaConf.structured(cfg)), logger=False
        )

    def setup(self, **kwargs):
        """Setup the data"""
        self.train_data = Subset(self.dataset, indices=self.train_indices)
        self.val_data = Subset(self.dataset, indices=self.val_indices)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.decoder.parameters(), lr=self.cfg.lr)

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

        logits = self.decoder(batch["input_ids_dec"])
        targets = batch["input_ids_dec"].masked_fill(
            batch["attention_mask_dec"] == 0, -100
        )
        logits = logits[:, :-1].contiguous()
        targets = targets[:, 1:].contiguous()
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1),
            ignore_index=-100,
        )

        self.log(
            "train/loss",
            loss.detach().cpu().numpy().item(),
            prog_bar=True,
            add_dataloader_idx=False,
            batch_size=logits.shape[0],
            on_step=True,
            on_epoch=False
        )

        return loss

    def validation_step(self, batch, batch_idx=None):

        logits = self.decoder(batch["input_ids_dec"])
        targets = batch["input_ids_dec"].masked_fill(
            batch["attention_mask_dec"] == 0, -100
        )
        logits = logits[:, :-1].contiguous()
        targets = targets[:, 1:].contiguous()
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1),
            ignore_index=-100,
        )

        self.log(
            "val/loss",
            loss.detach().cpu().numpy().item(),
            prog_bar=True,
            add_dataloader_idx=False,
            batch_size=logits.shape[0],
            on_step=False,
            on_epoch=True
        )

        return loss