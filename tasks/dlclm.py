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
from data.stories import StoriesDatasetConfig, StoriesDataset
from hydra.utils import instantiate
from model.encoder import EncoderConfig, EncoderModel
from model.decoder import DecoderConfig, DecoderModel
import os
import wandb
import hydra
from lightning.pytorch.utilities.rank_zero import rank_zero_info, rank_zero_only
from tasks.autoencoder import AETask, InfiniteDistributedUniformSampler
from data.wiki import WikipediaDataset
from tqdm import tqdm
from mauve import compute_mauve, get_features_from_input
import einx
from transformers import get_constant_schedule_with_warmup
from data.wiki import DATA_SEED

@dataclass
class DLCLMTaskConfig:
    lr: float
    lr_warmup: bool
    batch_size: int
    pretrained_ae_id: str
    pure_ar_baseline: bool = False
    
    # If not giving pretrained_ae_id
    decoder: Optional[DecoderConfig] = None

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
                os.environ["LOG_DIR"],
                "checkpoints/",
                cfg.pretrained_ae_id,
                "last.ckpt",
            ),
            strict=True,
            map_location=torch.device("cpu"),
        )
        self.train_indices = ae_task.train_indices
        self.val_indices = ae_task.val_indices
        self.dataset = ae_task.dataset
        self.dataset: WikipediaDataset

        # If decoder is not given, start from the DAE's decoder
        if cfg.decoder is None:
            self.decoder = ae_task.decoder
            if self.decoder.cfg.lora_cfg != None:
                self.decoder.backbone = self.decoder.backbone.merge_and_unload()
            del self.decoder.prompt_generator
            self.decoder = self.decoder.train().requires_grad_(True)
        else:
            rank_zero_info("Starting from a fresh decoder!")
            self.decoder = DecoderModel(cfg.decoder).train().requires_grad_(True)

        if not cfg.pure_ar_baseline:
            assert ae_task.encoder.cfg.sem is not None

            # Load encoder and freeze it
            self.encoder = ae_task.encoder.eval().requires_grad_(False)

            # Turn on DLC mode on decoder
            self.decoder.is_dlc = True
            # Add <|think|> token
            self.decoder.tokenizer.add_special_tokens(
                {"additional_special_tokens": ["<|think|>"]}
            )
            self.decoder.tokenizer.think_token_id = (
                self.decoder.tokenizer.convert_tokens_to_ids("<|think|>")
            )
            # Add DLC tokens
            self.decoder.backbone.resize_token_embeddings(
                len(self.decoder.tokenizer) + self.encoder.sem.cfg.V, mean_resizing=True
            )
            # Just making sure
            self.decoder.requires_grad_(True)

        # To not eval MAUVE every validation step
        self.val_epoch_counter = 0

        self.cfg = cfg

        self.save_hyperparameters(
            OmegaConf.to_container(OmegaConf.structured(cfg)), logger=False
        )

    def compile(self):
        if hasattr(self, 'encoder'):
            self.encoder.compile()
        self.decoder.compile()

    def train(self, mode=True):
        # Make sure encoder stays in eval mode
        super().train(mode)
        if hasattr(self, 'encoder'):
            self.encoder.eval()
        return self

    def setup(self, stage: Optional[str] = None):
        self.train_data = Subset(self.dataset, indices=self.train_indices)
        self.val_data = Subset(self.dataset, indices=self.val_indices)

    def configure_optimizers(self):
        no_decay = ["bias", "norm"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in self.decoder.named_parameters()
                    if not any(nd in n.lower() for nd in no_decay)
                ],
                "weight_decay": 0.01,
            },
            {
                "params": [
                    p
                    for n, p in self.decoder.named_parameters()
                    if any(nd in n.lower() for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=self.cfg.lr)
        if self.cfg.lr_warmup:
            scheduler = get_constant_schedule_with_warmup(
                optimizer, num_warmup_steps=2000
            )
            scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}

            return [optimizer], [scheduler]
        else:
            return optimizer

    def train_dataloader(self):
        return DataLoader(
            self.train_data,
            batch_sampler=InfiniteDistributedUniformSampler(
                n=len(self.train_data), batch_size=self.cfg.batch_size
            ),
            collate_fn=self.dataset.get_collate_and_tokenize_fn(
                conditional=False,
                enc_tokenizer=(
                    self.encoder.tokenizer if hasattr(self, "encoder") else None
                ),
                dec_tokenizer=self.decoder.tokenizer,
            ),
        )

    def val_dataloader(self):
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            sampler = torch.utils.data.DistributedSampler(self.val_data, shuffle=False)
        else:
            sampler = torch.utils.data.SequentialSampler(self.val_data)
        return DataLoader(
            self.val_data,
            batch_size=self.cfg.batch_size,
            sampler=sampler,
            collate_fn=self.dataset.get_collate_and_tokenize_fn(
                conditional=False,
                enc_tokenizer=(
                    self.encoder.tokenizer if hasattr(self, "encoder") else None
                ),
                dec_tokenizer=self.decoder.tokenizer,
            ),
        )

    def training_step(self, batch, batch_idx):
        with torch.no_grad():
            input_ids_dec = [batch["cont_ids_dec"]]
            if hasattr(self, "encoder"):
                # Compute DLC. IMPORTANT : Shift SEM tokens to new token indices
                dlc_ids = self.encoder(
                    batch["cont_ids_enc"], batch["cont_mask_enc"], return_dlc=True
                ) + len(self.decoder.tokenizer)
                think_token = torch.full(
                    size=(dlc_ids.shape[0], 1),
                    fill_value=self.decoder.tokenizer.think_token_id,
                    dtype=dlc_ids.dtype,
                    device=dlc_ids.device,
                )
                # <|think|> DLC <|bos|> x <|eos|>
                input_ids_dec = [think_token, dlc_ids] + input_ids_dec
            input_ids_dec = torch.cat(input_ids_dec, dim=1)

        # Maybe add prompt before the <|think|> token
        logits = self.decoder(input_ids=input_ids_dec)
        targets = input_ids_dec[:, 1:].contiguous()
        logits = logits[:, :-1].contiguous()
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1),
            ignore_index=self.decoder.tokenizer.pad_token_id,
            reduction="none",
        )
        loss = loss.view_as(targets)
        full_loss = loss.mean()

        # p(z)p(x|z)
        self.log(
            "train/full_loss",
            full_loss,
            on_epoch=False,
            on_step=True,
            sync_dist=True,
        )

        # p(x|z) only, more like the reconstruction loss
        cont_len = batch["cont_ids_dec"].size(1) - 1  # Do not include BOS token
        self.log(
            "train/cond_loss",
            loss[:, -cont_len:].mean(),
            on_epoch=False,
            on_step=True,
            sync_dist=True,
        )

        return full_loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            input_ids_dec = [batch["cont_ids_dec"]]
            if hasattr(self, "encoder"):
                # Compute DLC. IMPORTANT : Shift SEM tokens to new token indices
                dlc_ids = self.encoder(
                    batch["cont_ids_enc"], batch["cont_mask_enc"], return_dlc=True
                ) + len(self.decoder.tokenizer)
                think_token = torch.full(
                    size=(dlc_ids.shape[0], 1),
                    fill_value=self.decoder.tokenizer.think_token_id,
                    dtype=dlc_ids.dtype,
                    device=dlc_ids.device,
                )
                # <|think|> DLC <|bos|> x <|eos|>
                input_ids_dec = [think_token, dlc_ids] + input_ids_dec
            input_ids_dec = torch.cat(input_ids_dec, dim=1)

        # Maybe add prompt before the <|think|> token
        logits = self.decoder(input_ids=input_ids_dec)
        targets = input_ids_dec[:, 1:].contiguous()
        logits = logits[:, :-1].contiguous()
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1),
            ignore_index=self.decoder.tokenizer.pad_token_id,
            reduction="none",
        )
        loss = loss.view_as(targets)
        full_loss = loss.mean()

        # p(z)p(x|z)
        full_loss = loss.mean()
        self.log(
            "val/full_loss",
            full_loss,
            on_epoch=True,
            on_step=False,
            sync_dist=True,
        )

        # p(x|z) only, more like the reconstruction loss
        # -2 because the length of input_ids_dec includes the BOS and EOS tokens
        cont_len = batch["cont_ids_dec"].size(1) - 1  # Do not include BOS token
        self.log(
            "train/cond_loss",
            loss[:, -cont_len:].mean(),
            on_epoch=False,
            on_step=True,
            sync_dist=True,
        )

        if (batch_idx == 0) and (rank_zero_only.rank == 0):
            # Log reconstruction samples
            table_clean = wandb.Table(columns=["Original", "Reconstructed"])
            for original, reconstructed in zip(
                batch["input_str"][:5],
                self.decoder.tokenizer.batch_decode(
                    self.decoder.generate(
                        dlc=dlc_ids[:5] if hasattr(self, 'encoder') else None,
                        batch_size=5,
                        max_length=self.dataset.cfg.max_length,
                    ),
                    skip_special_tokens=True,
                ),
            ):
                table_clean.add_data(original, reconstructed)
            wandb.log({"val/clean_samples": table_clean})
            del table_clean

            # Log MAUVE score every 5 validation steps (takes a few minutes)
            if self.val_epoch_counter % 5 == 0:
                mauve = self.eval_mauve()
                wandb.log({"val/MAUVE": mauve})

    def eval_mauve(self, seed: int = 1337):
        # Load reference features (5120 sequences pre-featurized with GPT2-large)
        ref_feats = torch.load("mauve_eval_feats.pt", map_location=torch.device("cpu"))

        # Generate unconditionally
        with torch.inference_mode():
            gen_text = []
            for _ in tqdm(
                range(len(ref_feats) // 128), desc=f"Generating paragraphs..."
            ):
                gen_text.extend(
                    self.decoder.tokenizer.batch_decode(
                        self.decoder.generate(
                            max_length=self.dataset.cfg.max_length,
                            batch_size=128,
                            dlc_len=self.encoder.sem.dlc_len if hasattr(self, 'encoder') else None,
                        ),
                        skip_special_tokens=True,
                    )
                )

        # Featurize generated text
        gen_feats = get_features_from_input(
            None, None, gen_text, "gpt2-large", 150, 0, "generated paragraphs", 64
        )

        # Comput MAUVE against reference features (remove AMP since it caused errors that I don't wanna debug)
        with torch.autocast(device_type="cuda", enabled=False):
            mauve = compute_mauve(
                p_features=gen_feats,
                q_features=ref_feats,
                max_text_length=self.dataset.cfg.max_length,
                batch_size=64,
                device_id=0,
                featurize_model_name="gpt2-large",
                seed=seed,
            )

        # Avoid CUDA memory leaks
        torch.cuda.empty_cache()

        return mauve.mauve
