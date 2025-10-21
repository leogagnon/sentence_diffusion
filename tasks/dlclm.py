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
from tasks.autoencoder import AETask
from data.wiki import WikipediaDataset
from tqdm import tqdm
from mauve import compute_mauve, get_features_from_input
import einx


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
                os.environ["LOG_DIR"],
                "checkpoints/",
                cfg.pretrained_ae_id,
                "last.ckpt",
            ),
            strict=True,
            map_location="cpu",
        )
        self.train_indices = ae_task.train_indices
        self.val_indices = ae_task.val_indices
        self.dataset = ae_task.dataset
        self.dataset: WikipediaDataset

        if ae_task.encoder.cfg.sem_cfg != None:
            V = ae_task.encoder.cfg.sem_cfg.V
        elif ae_task.encoder.cfg.hsem_cfg != None:
            V = ae_task.encoder.cfg.hsem_cfg.V
        else:
            assert False, "DLCLM requires a SEM encoder"
        self.encoder = ae_task.encoder.eval().requires_grad_(False)

        # We finetune the decoder from the AE training, without the prompt generator, with new token embeddings
        # We merge the initial LoRA adapter and create a new one for the DLC finetuning
        self.decoder = ae_task.decoder
        if self.decoder.cfg.lora_cfg != None:
            self.decoder.backbone = self.decoder.backbone.merge_and_unload()
        self.decoder.backbone.resize_token_embeddings(
            len(ae_task.decoder.tokenizer) + V
        )
        self.decoder = self.decoder.train().requires_grad_(True)
        del self.decoder.prompt_generator

        self.val_epoch_counter = 0

        self.cfg = cfg

        self.save_hyperparameters(
            OmegaConf.to_container(OmegaConf.structured(cfg)), logger=False
        )

    def train(self, mode=True):
        # Make sure encoder stays in eval mode
        super().train(mode)
        self.encoder.eval()
        return self

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
            drop_last=True,
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
        with torch.no_grad():
            dlc_logits = self.encoder(
                batch["input_ids_enc"], batch["attention_mask_enc"], return_sem=True
            )
            if self.encoder.cfg.sem_cfg != None:
                dlc = dlc_logits.argmax(-1)
            elif self.encoder.cfg.hsem_cfg != None:
                V = dlc_logits[0].shape[-1]
                dlc = [dlc_logits[0].argmax(-1).squeeze()]
                for i in range(len(dlc_logits) - 1):
                    level = torch.gather(
                        input=dlc_logits[1],
                        dim=2,
                        index=einx.rearrange("b l -> b l n v", dlc[i], n=1, v=V),
                    )
                    dlc.append(torch.argmax(level, dim=-1).squeeze())
                dlc = torch.cat(dlc, dim=1)

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
        dlc_logits = self.encoder(
                batch["input_ids_enc"], batch["attention_mask_enc"], return_sem=True
            )
        if self.encoder.cfg.sem_cfg != None:
            dlc = dlc_logits.argmax(-1)
        elif self.encoder.cfg.hsem_cfg != None:
            V = dlc_logits[0].shape[-1]
            dlc = [dlc_logits[0].argmax(-1).squeeze()]
            for i in range(len(dlc_logits) - 1):
                level = torch.gather(
                    input=dlc_logits[1],
                    dim=2,
                    index=einx.rearrange("b l -> b l n v", dlc[i], n=1, v=V),
                )
                dlc.append(torch.argmax(level, dim=-1).squeeze())
            dlc = torch.cat(dlc, dim=1)
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
            on_epoch=True,
            on_step=False,
            sync_dist=True,
        )

        # p(x|z) only, more like the reconstruction loss
        # -2 because the length of input_ids_dec includes the BOS and EOS tokens
        self.log(
            "val/cond_loss",
            loss[:, -(batch["input_ids_dec"].size(1) - 1) :].mean(),
            on_epoch=True,
            on_step=False,
            sync_dist=True,
        )

        if (batch_idx == 0) and (rank_zero_only.rank == 0):
            # Log reconstruction samples
            table_clean = wandb.Table(columns=["Original", "Reconstructed"])
            for original, reconstructed in zip(
                batch["input_str"][:5],
                self.decoder.tokenizer.batch_decode(
                    self.decoder.generate(
                        prompt=input_ids_dec[:5, : (dlc.shape[1] + 1)],
                        max_length=self.dataset.cfg.max_length,
                    ),
                    skip_special_tokens=True,
                ),
            ):
                table_clean.add_data(original, reconstructed)
            wandb.log({"val/clean_samples": table_clean})
            del table_clean

            # Log MAUVE score every 5 validation steps
            if self.val_epoch_counter % 5 == 0:
                mauve = self.eval_mauve()
                wandb.log({"val/MAUVE": mauve})

    def eval_mauve(self, seed: int = 1337):
        # Load reference features
        ref_feats = torch.load("mauve_eval_feats.pt")

        # Generate unconditionally
        with torch.inference_mode():
            if self.encoder.cfg.sem_cfg != None:
                dlc_len = self.encoder.cfg.sem_cfg.L
            else: 
                dlc_len = self.encoder.cfg.hsem_cfg.L * self.encoder.cfg.hsem_cfg.D
            gen_text = []
            for _ in tqdm(
                range(len(ref_feats) // 128), desc=f"Generating paragraphs..."
            ):
                gen_text.extend(
                    self.decoder.tokenizer.batch_decode(
                        self.decoder.generate(
                            max_length=150,
                            batch_size=128,
                            generate_prompt=True,
                            dlc_len=dlc_len,
                        ),
                        skip_special_tokens=True,
                    )
                )

        # Featurize for MAUVE
        gen_feats = get_features_from_input(
            None, None, gen_text, "gpt2-large", 150, 0, "generated paragraphs", 64
        )

        # Comput MAUVE
        with torch.autocast(device_type="cuda", enabled=False):
            mauve = compute_mauve(
                p_features=gen_feats,
                q_features=ref_feats,
                max_text_length=150,
                batch_size=64,
                device_id=0,
                featurize_model_name="gpt2-large",
                seed=seed,
            )

        torch.cuda.empty_cache()

        return mauve.mauve
