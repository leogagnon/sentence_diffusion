from functools import partial
import math
import os
import lightning as L
from omegaconf import OmegaConf
from dataclasses import dataclass
from typing import Any, List, Optional
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
from tasks.autoencoder import AETask, InfiniteDistributedUniformSampler
from data.datasets import WikipediaDataset
from tqdm import tqdm
from mauve import compute_mauve, get_features_from_input
import einx
from transformers import get_constant_schedule_with_warmup
from tasks.dcse import DCSETask, DCSETaskConfig
from enum import Enum


class InfoLabel(Enum):
    CONT = 0
    DLC = 1
    PAD = 2
    PROMPT = 3


@dataclass
class DLCLMTaskConfig:
    lr: float
    batch_size: int
    lr_warmup_steps: int = 1500
    pretrained_ae_id: Optional[str] = None
    pretrained_dcse_id: Optional[str] = None
    conditional: bool = False
    min_cond_split: int = 32
    seq_len: int = 128

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

        # Load pre-training task
        self.ar_baseline = False
        if cfg.pretrained_ae_id is not None:
            pretraining_task = AETask.load_from_checkpoint(
                os.path.join(
                    os.environ["LOG_DIR"],
                    "checkpoints/",
                    cfg.pretrained_ae_id,
                    "last.ckpt",
                ),
                strict=False,
                map_location=torch.device("cpu"),
            )
        elif cfg.pretrained_dcse_id is not None:
            pretraining_task = DCSETask.load_from_checkpoint(
                os.path.join(
                    os.environ["LOG_DIR"],
                    "checkpoints/",
                    cfg.pretrained_dcse_id,
                    "last.ckpt",
                ),
                strict=False,
                map_location=torch.device("cpu"),
            )
        else:
            self.ar_baseline = True

        # Extract dataset
        self.train_indices = pretraining_task.train_indices
        self.val_indices = pretraining_task.val_indices
        self.dataset = pretraining_task.dataset

        # Change the max length of the sequences to fit the task config
        if cfg.conditional:
            self.dataset.cfg.length_interval = [cfg.seq_len, cfg.seq_len]

        # Create/extract decoder
        if cfg.decoder is None:
            assert self.cfg.pretrained_ae_id is not None
            self.decoder = pretraining_task.decoder
            del self.decoder.prompt_generator
            self.decoder = self.decoder.train().requires_grad_(True)
        else:
            self.decoder = DecoderModel(cfg.decoder).train().requires_grad_(True)

        # Extract encoder and setup decoder vocab
        if not self.ar_baseline:
            assert pretraining_task.encoder.cfg.sem is not None

            # Load encoder and freeze it
            self.encoder = pretraining_task.encoder.eval().requires_grad_(False)

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
        if hasattr(self, "encoder"):
            self.encoder.compile()
        self.decoder.compile()

    def train(self, mode=True):
        # Make sure encoder stays in eval mode
        super().train(mode)
        if hasattr(self, "encoder"):
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
        if self.cfg.lr_warmup_steps > 0:
            scheduler = get_constant_schedule_with_warmup(
                optimizer, num_warmup_steps=self.cfg.lr_warmup_steps
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
            collate_fn=self.get_collate_fn(),
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
            collate_fn=self.get_collate_fn(),
        )

    def get_collate_fn(self):
        dec_tokenizer = self.decoder.tokenizer
        max_length = self.dataset.max_length

        @torch.no_grad()
        def fn(batch):

            input_str = batch["input_str"]

            if "input_ids" not in batch:
                # Compute input_ids of the decoder if not already there
                # This adds <|bos|> and <|eos|>
                input_ids_dec = dec_tokenizer.batch_encode_plus(
                    input_str,
                    truncation=True,
                    max_length=max_length + 64,
                    add_special_tokens=True,
                    return_attention_mask=False,
                )["input_ids"]
            else:
                input_ids_dec = batch["input_ids"]

            # If runnign the AR baseline, simply return
            if self.ar_baseline:
                input_ids_dec = dec_tokenizer.pad(
                    {"input_ids": input_ids_dec},
                    padding=True,
                    return_tensors="pt",
                    return_attention_mask=False,
                )["input_ids"]
                info_mask_dec = torch.full_like(
                    input_ids_dec, fill_value=InfoLabel.CONT.value, dtype=torch.int32
                )
                info_mask_dec[input_ids_dec == dec_tokenizer.pad_token_id] = (
                    InfoLabel.PAD.value
                )
                return {"input_ids_dec": input_ids_dec, "info_mask_dec": info_mask_dec}

            # If conditional, split sequence in two : prompt, continuation
            # Else consider empty prompt and the whole sequence as continuation
            if self.cfg.conditional:
                prompt_len = torch.randint(
                    low=self.cfg.min_cond_split,
                    high=max_length - self.cfg.min_cond_split,
                    size=(len(input_ids_dec),),
                )
                prompt_dec = [
                    x[:l] + [dec_tokenizer.think_token_id]
                    for x, l in zip(input_ids_dec, prompt_len)
                ]
                continuation_dec = [
                    [dec_tokenizer.bos_token_id] + x[l : l + max_length]
                    for x, l in zip(input_ids_dec, prompt_len)
                ]
                continuation_str = dec_tokenizer.batch_decode(
                    continuation_dec, skip_special_tokens=True
                )
            else:
                prompt_dec = [
                    [dec_tokenizer.think_token_id] for _ in range(len(input_ids_dec))
                ]
                continuation_dec = input_ids_dec
                continuation_str = input_str

            # Compute DLC of continuation
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                continuation_enc = self.encoder.tokenizer.batch_encode_plus(
                    continuation_str,
                    truncation=True,
                    padding="max_length",
                    max_length=max_length,
                    return_tensors="pt",
                ).to("cuda")
                dlc_ids = self.encoder(
                    continuation_enc["input_ids"],
                    continuation_enc["attention_mask"].bool(),
                    return_dlc=True,
                )[1] + len(dec_tokenizer)

            # Build the final input_ids
            input_ids_dec = [
                prompt + dlc + continuation
                for (prompt, dlc, continuation) in zip(
                    prompt_dec, dlc_ids.tolist(), continuation_dec
                )
            ]
            input_ids_dec = dec_tokenizer.pad(
                {"input_ids": input_ids_dec},
                padding="longest",
                return_tensors="pt",
                return_attention_mask=False,
            )["input_ids"]

            # Build the info mask
            info_mask_dec = torch.full_like(
                input_ids_dec, fill_value=InfoLabel.CONT.value, dtype=torch.int32
            )
            info_mask_dec[input_ids_dec == dec_tokenizer.pad_token_id] = (
                InfoLabel.PAD.value
            )
            for i in range(len(info_mask_dec)):
                # prompt <|think|> ==> PROMPT
                info_mask_dec[i, : len(prompt_dec[i]) + 1] = InfoLabel.PROMPT.value
                # DLC ==> DLC
                info_mask_dec[
                    i,
                    (len(prompt_dec[i])) : (len(prompt_dec[i]) + dlc_ids.shape[1]) :,
                ] = InfoLabel.DLC.value
                # Rest is CONT

            return {
                "input_ids_dec": input_ids_dec,
                "info_mask_dec": info_mask_dec,
                "input_str": input_str,
            }

        return fn

    def training_step(self, batch, batch_idx):

        logits = self.decoder(input_ids=batch["input_ids_dec"])

        # Compute loss (no mask/reduce, we do it after)
        info_mask_dec = batch["info_mask_dec"][:, 1:]
        targets = batch["input_ids_dec"][:, 1:].contiguous()
        logits = logits[:, :-1].contiguous()
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1),
            reduction="none",
        )
        loss = loss.view_as(targets)
        full_loss = loss[
            (info_mask_dec == InfoLabel.DLC.value)
            + (info_mask_dec == InfoLabel.CONT.value)
        ].mean()

        # p(z)p(x|z)
        self.log(
            "train/full_loss",
            full_loss,
            on_epoch=False,
            on_step=True,
            sync_dist=True,
        )

        # p(x|z) only, more like the reconstruction loss
        if not self.ar_baseline:
            cond_loss = loss[info_mask_dec == InfoLabel.CONT.value].mean()
            self.log(
                "train/cond_loss",
                cond_loss,
                on_epoch=False,
                on_step=True,
                sync_dist=True,
            )

        return full_loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):

        logits = self.decoder(input_ids=batch["input_ids_dec"])

        # Compute loss (no mask/reduce, we do it after)
        info_mask_dec = batch["info_mask_dec"][:, 1:]
        targets = batch["input_ids_dec"][:, 1:].contiguous()
        logits = logits[:, :-1].contiguous()
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1),
            reduction="none",
        )
        loss = loss.view_as(targets)
        full_loss = loss[
            (info_mask_dec == InfoLabel.DLC.value)
            + (info_mask_dec == InfoLabel.CONT.value)
        ].mean()

        # p(z)p(x|z)
        self.log(
            "val/full_loss",
            full_loss,
            on_epoch=True,
            on_step=False,
            sync_dist=True,
        )

        if not self.ar_baseline:
            # p(x|z) only, more like the reconstruction loss
            cond_loss = loss[info_mask_dec == InfoLabel.CONT.value].mean()
            self.log(
                "val/cond_loss",
                cond_loss,
                on_epoch=True,
                on_step=False,
                sync_dist=True,
            )

        if (batch_idx == 0) and (rank_zero_only.rank == 0):
            # Log MAUVE score every 5 validation steps (takes a few minutes)
            if (self.val_epoch_counter % 5 == 0) and not self.cfg.conditional:
                mauve = self.eval_mauve()
                wandb.log({"val/MAUVE": mauve})
                torch.cuda.empty_cache()

            if not self.ar_baseline:

                # Gather prompt <|think|> DLC
                # prompt may be empty, in which case it's just DLC
                prompt = [
                    batch["input_ids_dec"][i][
                        (batch["info_mask_dec"][i] == InfoLabel.DLC.value)
                        + (batch["info_mask_dec"][i] == InfoLabel.PROMPT.value)
                    ].tolist()
                    for i in range(5)
                ]

                reconstruction = self.decoder.tokenizer.batch_decode(
                    self.decoder.generate(
                        prompt=prompt,
                        max_length=self.dataset.max_length,
                    ),
                    skip_special_tokens=True,
                )

                # Put in table and log
                table_clean = wandb.Table(columns=["Original", "Reconstructed"])
                for original, reconstructed in zip(
                    batch["input_str"][:5],
                    reconstruction,
                ):
                    table_clean.add_data(original, reconstructed)
                wandb.log({"val/clean_samples": table_clean})
                del table_clean

    def eval_mauve(
        self,
        seed: int = 1337,
        gen_kwargs: Optional[dict] = None,
        gen_kwargs_dlc: Optional[dict] = None,
    ):
        # Load reference features (5120 sequences pre-featurized with GPT2-large)
        ref_feats = torch.load(
            "MAUVE_eval_feats_wiki.pt", map_location=torch.device("cpu")
        )

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
                            dlc_len=(
                                self.encoder.sem.dlc_len
                                if hasattr(self, "encoder")
                                else None
                            ),
                            gen_kwargs=gen_kwargs,
                            gen_kwargs_dlc=gen_kwargs_dlc,
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
