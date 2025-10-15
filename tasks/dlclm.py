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
import os
import wandb
import hydra
from model.gaussian_diffusion import time_to_alpha, cosine_schedule
from lightning.pytorch.utilities.rank_zero import rank_zero_info, rank_zero_only
from tasks.autoencoder import AETask
from data.wiki import WikipediaDataset


@dataclass
class DLCLMTaskConfig:
    lr: float
    batch_size: int
    pretrained_ae_id: str

    data_seed: int = 42

    name: Optional[str] = None


class DLCLMTask(L.LightningModule):
    """
    Modify a language decoder to sample from p(z)p(x|z)
    where z is a DLC of a pretrained autoencoder.
    """

    def __init__(self, cfg: Optional[DLCLMTaskConfig] = None, **kwargs):
        super().__init__()

        if cfg == None:
            cfg = OmegaConf.to_object(
                OmegaConf.merge(
                    OmegaConf.create(DLCLMTaskConfig),
                    OmegaConf.create(kwargs),
                )
            )

        # Extract encoder, decoder and dataset from pretrained autoencoder
        ae_task = AETask.load_from_checkpoint(
            os.path.join(
                os.environ["LATENT_CONTROL_CKPT_DIR"],
                cfg.pretrained_ae_id,
                "last.ckpt",
            ),
            strict=False,
        )
        self.train_indices = ae_task.train_indices
        self.val_indices = ae_task.val_indices
        self.dataset = ae_task.dataset
        self.dataset: WikipediaDataset

        # Load and process encoder (eval, no gradients, bfloat16, flash attention)
        assert ae_task.encoder.cfg.sem_cfg != None, "DLCLM requires a SEM encoder"
        self.encoder = ae_task.encoder.to(torch.bfloat16).eval().requires_grad_(False)
        try:
            # Try to use flash attention if available
            self.encoder.transformer.auto_model.set_attn_implementation(
                "flash_attention_2"
            )
        except:
            pass

        # We finetune the decoder from the AE training (by adding tokens for the SEM tokens)
        # We merge the initial LoRA adapter and create a new one for the DLC finetuning
        self.decoder = ae_task.decoder
        self.decoder.backbone = self.decoder.backbone.merge_and_unload()
        self.decoder.backbone.resize_token_embeddings(
            len(ae_task.decoder.tokenizer) + ae_task.encoder.cfg.sem_cfg.V
        )
        self.decoder = self.decoder.train().requires_grad_(True)
        self.decoder.backbone = get_peft_model(
            self.decoder.backbone,
            LoraConfig(**self.decoder.cfg.lora_cfg),
        )

        self.cfg = cfg

        self.save_hyperparameters(
            OmegaConf.to_container(OmegaConf.structured(cfg)), logger=False
        )

    def setup(self, stage: Optional[str] = None):
        self.train_data = Subset(self.dataset, indices=self.train_indices)
        self.val_data = Subset(self.dataset, indices=self.val_indices)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.decoder.parameters(), lr=self.cfg.lr)

    def train_dataloader(self):
        return DataLoader(
            self.train_data,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            collate_fn=self.dataset.get_collate_and_tokenize_fn(
                enc_tokenizer=self.encoder.tokenizer,
                dec_tokenizer=self.decoder.tokenizer,
            ),
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_data,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            collate_fn=self.dataset.get_collate_and_tokenize_fn(
                enc_tokenizer=self.encoder.tokenizer,
                dec_tokenizer=self.decoder.tokenizer,
            ),
        )

    def training_step(self, batch, batch_idx):

        dlc = self.encoder(
            batch["input_ids_enc"], batch["attention_mask_enc"], return_sem=True
        ).argmax(-1)
        dlc += len(self.decoder.tokenizer)  # Shift SEM tokens to new token indices

        # Prepend BOS + DLC to decoder input ids
        input_ids_dec = torch.cat(
            [
                torch.full_like(
                    dlc[:, [0]],
                    fill_value=self.decoder.tokenizer.bos_token_id,
                ),
                dlc,
                batch["input_ids_dec"],
            ],
            dim=1,
        )

        # Compute loss
        targets = input_ids_dec.clone()[:, 1:].contiguous()
        logits = self.decoder(input_ids=input_ids_dec)[:, :-1].contiguous()
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1),
            ignore_index=self.decoder.tokenizer.pad_token_id,
            reduction="none",
        )
        loss = loss.view_as(targets)

        # p(z)p(x|z)
        full_loss = loss.mean()
        self.log(
            "train/full_loss",
            full_loss,
            on_epoch=False,
            on_step=True,
            sync_dist=True,
        )

        # p(x|z) only, more like the reconstruction loss
        # -2 because the length of input_ids_dec includes the BOS and EOS tokens
        self.log(
            "train/cond_loss",
            loss[:, -(batch["input_ids_dec"].size(1) - 1) :].mean(),
            on_epoch=False,
            on_step=True,
            sync_dist=True,
        )

        return full_loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        dlc = self.encoder(
            batch["input_ids_enc"], batch["attention_mask_enc"], return_sem=True
        ).argmax(-1)
        dlc += len(self.decoder.tokenizer)  # Shift SEM tokens to new token indices

        # Prepend BOS + DLC to decoder input ids
        input_ids_dec = torch.cat(
            [
                torch.full_like(
                    dlc[:, [0]],
                    fill_value=self.decoder.tokenizer.bos_token_id,
                ),
                dlc,
                batch["input_ids_dec"],
            ],
            dim=1,
        )

        # Compute loss
        targets = input_ids_dec.clone()[:, 1:].contiguous()
        logits = self.decoder(input_ids=input_ids_dec)[:, :-1].contiguous()
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1),
            ignore_index=self.decoder.tokenizer.pad_token_id,
            reduction="none",
        )
        loss = loss.view_as(targets)

        # p(z)p(x|z)
        full_loss = loss.mean()
        self.log(
            "val/full_loss",
            full_loss,
            on_epoch=False,
            on_step=True,
            sync_dist=True,
        )

        # p(x|z) only, more like the reconstruction loss
        # -2 because the length of input_ids_dec includes the BOS and EOS tokens
        self.log(
            "val/cond_loss",
            loss[:, -(batch["input_ids_dec"].size(1) - 1) :].mean(),
            on_epoch=False,
            on_step=True,
            sync_dist=True,
        )
