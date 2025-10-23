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
from tqdm import tqdm
from mauve import compute_mauve, get_features_from_input
from transformers import get_constant_schedule_with_warmup
from data.wiki import DATA_SEED

import os
import torch
from torch.utils.data import Sampler
from typing import Iterator, Optional


class InfiniteDistributedUniformSampler(Sampler[int]):
    """
    Infinite, per-rank independent uniform sampling *with replacement*.

    - No epoch notion (no set_epoch).
    - Works in single-process and DDP.
    - Yields indices forever → use max_steps / manual break in training loop.
    - Deterministic if `seed` is set; otherwise non-deterministic.

    Args:
        dataset: map-style dataset (needs __len__).
        seed: optional base seed; if None, a random 64-bit seed is used.
        chunk_size: draw this many indices per RNG call (perf tweak).
    """

    def __init__(
        self,
        n: int,
        batch_size: int,
        seed: Optional[int] = None,
        chunk_size: int = 4096,
    ):

        self.n = n
        self.batch_size = batch_size

        # Rank/world
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            self.rank = torch.distributed.get_rank()
        else:
            self.rank = 0

        # Base seed (None → random)
        if seed is None:
            seed = int.from_bytes(os.urandom(8), "little", signed=False)

        # Mix in rank to get independent streams per process
        mixed = (int(seed) ^ (0x9E3779B97F4A7C15 * (self.rank + 1))) & ((1 << 63) - 1)

        self._g = torch.Generator()
        self._g.manual_seed(mixed)
        self._chunk = int(chunk_size)

    def __len__(self) -> int:
        # Sentinel for frameworks that read len(); loader never actually exhausts.
        return 2**31 - 1

    def __iter__(self) -> Iterator[int]:
        while True:
            # Vectorized draw, then yield scalars
            idx = torch.randint(
                0, self.n, (self._chunk, self.batch_size), generator=self._g
            )
            # Yield as Python lists (fast path in DataLoader)
            for row in idx:
                yield row.tolist()


@dataclass
class AETaskConfig:
    lr: float
    batch_size: int
    decoder: DecoderConfig
    dataset: dict
    val_size: int
    encoder: Optional[EncoderConfig] = None
    lr_warmup: bool = False
    sub_p: float = 0.3

    name: Optional[str] = None


class AETask(L.LightningModule):
    """
    Autoencoder Task.
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
        if cfg.encoder != None:
            self.encoder = EncoderModel(cfg.encoder).train().requires_grad_(True)
            cfg.decoder.input_dim = self.encoder.latent_dim

        self.decoder = DecoderModel(cfg.decoder).train().requires_grad_(True)

        # Setup dataset
        self.dataset = hydra.utils.instantiate(cfg.dataset)

        # This is with a fixed seed to make sure validation set never changes
        indices = torch.randperm(
            len(self.dataset),
            generator=torch.Generator().manual_seed(DATA_SEED),
        )
        self.train_indices = indices[: -cfg.val_size]
        self.val_indices = indices[-cfg.val_size :]

        self.cfg = cfg

        self.save_hyperparameters(
            OmegaConf.to_container(OmegaConf.structured(cfg)), logger=False
        )

    def random_substitution(self, input_ids):
        # It is important that this function is not in-place
        input_ids = input_ids.clone()

        probability = torch.full_like(
            input_ids, fill_value=self.cfg.sub_p, dtype=torch.float32
        )
        masked_indices = torch.bernoulli(probability).bool()
        vocab_size = len(self.encoder.tokenizer) - len(
            self.encoder.tokenizer.all_special_ids
        )
        random_words = torch.randint_like(input_ids, low=0, high=vocab_size)

        input_ids[masked_indices] = random_words[masked_indices]

        return input_ids

    def compile(self):
        if self.cfg.encoder != None:
            self.encoder.compile()
        self.decoder.compile()

    def setup(self, **kwargs):
        """Setup the data"""
        self.train_data = Subset(self.dataset, indices=self.train_indices)
        self.val_data = Subset(self.dataset, indices=self.val_indices)

    def train_dataloader(self):
        return DataLoader(
            self.train_data,
            batch_sampler=InfiniteDistributedUniformSampler(
                n=len(self.train_data), batch_size=self.cfg.batch_size
            ),
            collate_fn=self.dataset.get_collate_and_tokenize_fn(
                conditional=False,
                enc_tokenizer=(
                    self.encoder.tokenizer if self.cfg.encoder != None else None
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
                    self.encoder.tokenizer if self.cfg.encoder != None else None
                ),
                dec_tokenizer=self.decoder.tokenizer,
            ),
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
        if self.cfg.lr_warmup:
            scheduler = get_constant_schedule_with_warmup(
                optimizer, num_warmup_steps=2000
            )
            scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}

            return [optimizer], [scheduler]
        else:
            return optimizer

    def training_step(self, batch, batch_idx):
        # If there is an encoder, compute z
        z = None
        if self.cfg.encoder != None:
            z = self.encoder(
                self.random_substitution(batch["cont_ids_enc"]), batch["cont_mask_enc"]
            )

        # Compute decoder likelihood of input_ids (no need for attention mask cuz causal)
        logits = self.decoder(input_ids=batch["cont_ids_dec"], z=z)

        # Compute loss
        targets = batch["cont_ids_dec"].masked_fill(batch["cont_mask_dec"] == 0, -100)
        logits = logits[:, :-1].contiguous()
        targets = targets[:, 1:].contiguous()
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1),
            ignore_index=-100,
        )

        self.log(
            "train/loss",
            loss,
            on_epoch=False,
            on_step=True,
            sync_dist=True,
        )

        return loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):

        # If there is an encoder, compute z and potentially add noise
        z = None
        if self.cfg.encoder != None:
            z = self.encoder(batch["cont_ids_enc"], batch["cont_mask_enc"])

            self.log(
                "val/latent_norm",
                z.norm(p=2, dim=-1).mean().detach().item(),
                on_epoch=True,
                sync_dist=True,
            )

        # Compute clean reconstruction loss
        logits = self.decoder(batch["cont_ids_dec"], z)
        targets = batch["cont_ids_dec"].masked_fill(batch["cont_mask_dec"] == 0, -100)
        logits = logits[:, :-1].contiguous()
        targets = targets[:, 1:].contiguous()
        recon_loss = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1),
            ignore_index=-100,
        )
        self.log("val/loss", recon_loss, on_epoch=True, sync_dist=True)

        # Maybe log some generations
        if (batch_idx == 0) and (rank_zero_only.rank == 0) and (z != None):
            # Log generation from clean samples
            table_clean = wandb.Table(columns=["Original", "Reconstructed"])
            for original, reconstructed in zip(
                batch["input_str"][:5],
                self.decoder.tokenizer.batch_decode(
                    self.decoder.generate(
                        z=z[:5], max_length=self.dataset.cfg.max_length
                    ),
                    skip_special_tokens=True,
                ),
            ):
                table_clean.add_data(original, reconstructed)
            wandb.log({"val/samples": table_clean})