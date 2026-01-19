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
from torch.utils.data import DataLoader, random_split
from torch.utils.data.dataset import Subset
from transformers import AutoTokenizer
from transformers.models.auto.modeling_auto import AutoModelForCausalLM
from peft import get_peft_model
from hydra.utils import instantiate
from model.encoder import EncoderConfig, EncoderModel
from model.decoder import DecoderConfig, DecoderModel
import os
import wandb
import hydra
from lightning.pytorch.utilities.rank_zero import rank_zero_info, rank_zero_only
from tqdm import tqdm
from mauve import compute_mauve, get_features_from_input
from transformers import get_constant_schedule_with_warmup
import einx
import os
import torch
from torch.utils.data import Sampler
from typing import Iterator, Optional
from tasks.utils import *
from data import WikipediaDataset, FineWebDataset, get_dataloader, InfoLabel
import torch.nn as nn
from dataclasses import dataclass, field


@dataclass
class SEMResetConfig:
    enabled: bool = False
    simplex_mode: bool = False
    threshold: float = 1e-4
    start: int = 5000
    end: int = 15000
    interval: int = 1000


@dataclass
class AETaskConfig:
    lr: float
    batch_size: int
    encoder: EncoderConfig
    decoder: DecoderConfig
    dataset: str
    lr_warmup_steps: int = 1500
    denoising: bool = True
    sem_noise: float = 0.0
    sem_noise_warmup_steps: int = 0
    prefix_length: int = 0
    suffix_length: int = 128
    context_length: int = 0
    encoder_mode: str = "suffix"
    sem_reset_config: SEMResetConfig = field(default_factory=SEMResetConfig)

    name: Optional[str] = None


class AETask(L.LightningModule):
    """
    Train a SEM encoder with a denoising Auto-Encoder task
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
        assert (
            cfg.decoder.cross_attention == True
        ), "AE decoder must use cross-attention"
        self.decoder = DecoderModel(cfg.decoder).train().requires_grad_(True)
        cfg.encoder.latent_dim = self.decoder.latent_dim
        self.encoder = EncoderModel(cfg.encoder)

        self.sem_usage_ema = SEMUsageTracker()

        self.cfg = cfg

        self.save_hyperparameters(
            OmegaConf.to_container(OmegaConf.structured(cfg)), logger=False
        )

    def compile(self):
        self.encoder.compile()
        self.decoder.compile()

    def setup(self, **kwargs):

        if self.cfg.dataset == "wikipedia":
            self.dataset = WikipediaDataset()
        elif self.cfg.dataset == "fineweb":
            self.dataset = FineWebDataset()
        else:
            raise ValueError(f"Unknown dataset {self.cfg.dataset}")

        self.train_data = Subset(self.dataset, indices=self.dataset.train_indices)
        self.val_data = Subset(self.dataset, indices=self.dataset.val_indices)

    def train_dataloader(self):
        return get_dataloader(
            self.train_data,
            batch_size=self.cfg.batch_size,
            prefix_length=self.cfg.prefix_length,
            suffix_length=self.cfg.suffix_length,
            context_length=self.cfg.context_length,
            enc_tokenizer=self.encoder.tokenizer,
            dec_tokenizer=self.decoder.tokenizer,
            encoder_mode=self.cfg.encoder_mode,
            encoder_noise=self.cfg.denoising,
            seed=random.randint(0, 100000),  # Dataset should be different if restarted
        )

    def val_dataloader(self):
        return get_dataloader(
            self.val_data,
            batch_size=self.cfg.batch_size,
            prefix_length=self.cfg.prefix_length,
            suffix_length=self.cfg.suffix_length,
            enc_tokenizer=self.encoder.tokenizer,
            context_length=self.cfg.context_length,
            dec_tokenizer=self.decoder.tokenizer,
            encoder_mode=self.cfg.encoder_mode,
            encoder_noise=False,  # No noise at validation
            seed=42,  # Always the same validation set for consistency
        )

    def configure_optimizers(self):
        no_decay = ["bias", "norm"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in self.named_parameters()
                    if not any(nd in n.lower() for nd in no_decay)
                ],
                "weight_decay": 0.01,
            },
            {
                "params": [
                    p
                    for n, p in self.named_parameters()
                    if any(nd in n.lower() for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=self.cfg.lr)
        if self.cfg.lr_warmup_steps > 0:
            scheduler = get_constant_schedule_with_warmup(
                optimizer, num_warmup_steps=self.cfg.lr_warmup_steps
            )
            scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}

            return [optimizer], [scheduler]
        else:
            return optimizer

    def training_step(self, batch, batch_idx):

        # Encode input_ids to get z
        z, sem_out = self.encoder(
            batch["input_ids_enc"],
            batch["attention_mask_enc"],
            return_count=True,
            noise=cosine_warmup_get_value(
                step=self.global_step,
                max_value=self.cfg.sem_noise,
                warmup_steps=self.cfg.sem_noise_warmup_steps,
                exp=2,
            ),
        )

        # Compute decoder likelihood of input_ids given z
        logits = self.decoder(input_ids=batch["input_ids_dec"], z=z)

        # Compute loss on suffix only
        info_mask_dec = batch["info_mask_dec"][:, 1:]
        targets = batch["input_ids_dec"][:, 1:].contiguous()
        logits = logits[:, :-1].contiguous()
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1),
            ignore_index=self.decoder.tokenizer.pad_token_id,
            reduction="none",
        )
        loss = loss.view_as(targets)
        loss = loss[info_mask_dec == InfoLabel.SUFFIX.value].mean()
        self.log(
            "train/loss",
            loss.item(),
            on_epoch=False,
            on_step=True,
            sync_dist=True,
        )

        self.sem_usage_ema.update(sem_out["usage_count"], batch_size=z.shape[0])

        return loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):

        soft_z, sem_out = self.encoder(
            batch["input_ids_enc"],
            batch["attention_mask_enc"],
            noise=0.0,
        )

        # Log entropy, marginal entropy and dead words fraction
        if "probs" in sem_out.keys():
            ent, m_ent = sem_entropy(sem_out["probs"])
            self.log(
                "val/sem_entropy",
                ent.item(),
                on_epoch=True,
                sync_dist=True,
            )
            self.log(
                "val/sem_marginal_entropy",
                m_ent.item(),
                on_epoch=True,
                sync_dist=True,
            )

        # Decode with hard latents
        hard_z, _ = self.encoder(
            batch["input_ids_enc"],
            batch["attention_mask_enc"],
            noise=0.0,
            temp=1e-4,
        )

        for latent, latent_type in zip([hard_z, soft_z], ["hard", "soft"]):
            logits = self.decoder(input_ids=batch["input_ids_dec"], z=latent)

            info_mask_dec = batch["info_mask_dec"][:, 1:]
            targets = batch["input_ids_dec"][:, 1:].contiguous()
            logits = logits[:, :-1].contiguous()
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=self.decoder.tokenizer.pad_token_id,
                reduction="none",
            )
            loss = loss.view_as(targets)

            suffix_loss = loss[info_mask_dec == InfoLabel.SUFFIX.value].mean()
            self.log(
                f"val/loss_{latent_type}",
                suffix_loss.item(),
                on_epoch=True,
                sync_dist=True,
            )
        if (batch_idx == 0) and (rank_zero_only.rank == 0):

            # Log % of dead words/simplices
            is_dead = self.sem_usage_ema.usage < self.cfg.sem_reset_config.threshold
            dead_words_ratio = torch.sum(is_dead).item() / is_dead.numel()
            dead_simplices = torch.sum((~is_dead).sum(1) == 1).item()
            dead_words_per_simplex = torch.sum(is_dead).sum(1).mean().item()

            self.log(
                "val/dead_words_ratio",
                dead_words_ratio,
                on_epoch=True,
                sync_dist=False,
            )
            self.log(
                "val/dead_simplices",
                dead_simplices,
                on_epoch=True,
                sync_dist=False,
            )
            self.log(
                "val/dead_words_per_simplex",
                dead_words_per_simplex,
                on_epoch=True,
                sync_dist=False,
            )

            # Reconstruct a few (5) samples from hard SEMs
            prefix_str = batch["prefix_str"][:5]
            prefix_ids = self.decoder.tokenizer.batch_encode_plus(prefix_str)[
                "input_ids"
            ]
            true_suffix_str = batch["suffix_str"][:5]
            generated_suffix_str = self.decoder.tokenizer.batch_decode(
                self.decoder.generate(
                    z=hard_z[:5], max_length=self.cfg.suffix_length, prefix=prefix_ids
                ),
                skip_special_tokens=True,
            )

            # Log to wandb
            table = wandb.Table(columns=["Prefix", "True Suffix", "Generated Suffix"])
            for i in range(5):
                table.add_data(
                    prefix_str[i], true_suffix_str[i], generated_suffix_str[i]
                )
            wandb.log({"val/samples": table})
            del table

    def on_before_zero_grad(self, optimizer):
        if (
            self.cfg.sem_reset_config.enabled
            and (self.global_step >= self.cfg.sem_reset_config.start)
            and (self.global_step <= self.cfg.sem_reset_config.end)
            and (self.global_step % self.cfg.sem_reset_config.interval == 0)
        ):
            rank_zero_info("Entered reset")
            self.trainer.strategy.barrier()

            with torch.no_grad():
                if rank_zero_only.rank == 0:
                    
                    # Compute what to reset
                    is_dead = (
                        self.sem_usage_ema.usage < self.cfg.sem_reset_config.threshold
                    )
                    if self.cfg.sem_reset_config.simplex_mode:
                        dead_simplices = (~is_dead).sum(1) == 1
                        reset_mask = einx.rearrange(
                            "L -> (L V)", dead_simplices, V=self.encoder.sem.cfg.V
                        )
                    else:
                        reset_mask = einx.rearrange(
                            "L V -> (L V)", is_dead
                        )

                    # Reset input/output proj associated with whatever's dead
                    if reset_mask.sum() > 0:
                        bound = 1 / (self.encoder.sem.cfg.input_dim**0.5)
                        self.encoder.sem.proj_in.weight[reset_mask] = (
                            self.encoder.sem.proj_in.weight[reset_mask].uniform_(
                                -bound, bound
                            )
                        )

                        self.encoder.out_proj.weight[:, reset_mask] = (
                            self.encoder.out_proj.weight[:, reset_mask].uniform_(
                                -bound, bound
                            )
                        )
                        rank_zero_info(
                            f"SEM Reset applied at step {self.global_step} to {torch.sum(reset_mask).item()} dead SEM words!"
                        )

                for param in self.encoder.sem.proj_in.parameters():
                    self.trainer.strategy.broadcast(param.data, src=0)

                for param in self.encoder.out_proj.parameters():
                    self.trainer.strategy.broadcast(param.data, src=0)
