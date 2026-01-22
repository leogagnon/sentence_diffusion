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
from tasks.autoencoder import AETask
from data import WikipediaDataset, FineWebDataset, InfoLabel
from tqdm import tqdm
from mauve import compute_mauve, get_features_from_input
import einx
from transformers import (
    get_constant_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
)
from enum import Enum
from tasks.utils import *
from typing import Tuple, List, ClassVar
from dataclasses import field
from data import get_dataloader


@dataclass
class DLCARTaskConfig:
    lr: float
    lr_warmup_steps: int
    batch_size: int
    decoder: DecoderConfig
    dataset: str
    eval_gen_ppl: bool
    DLC_dropout: float = 0.0

    pretrained_ae_id: Optional[str] = None

    prefix_length: Optional[int] = None
    suffix_length: Optional[int] = None
    context_length: Optional[int] = None
    encoder_mode: Optional[str] = None

    name: Optional[str] = None


class DLCARTask(L.LightningModule):
    """
    Finetune a Language Model to use DLCs (DLC-LM)
    """

    def __init__(self, cfg: Optional[DLCARTaskConfig] = None, **kwargs):
        super().__init__()

        if cfg == None:
            cfg = OmegaConf.to_object(
                OmegaConf.merge(
                    OmegaConf.create(DLCARTaskConfig),
                    OmegaConf.create(kwargs),
                )
            )

        # Get encoder from pretrained AE if specified
        # Else this will be a baseline LM without DLCs
        self.encoder = None
        if cfg.pretrained_ae_id is not None:

            ae_task = AETask.load_from_checkpoint(
                os.path.join(
                    os.environ["LOG_DIR"],
                    "checkpoints/",
                    cfg.pretrained_ae_id,
                    "last.ckpt",
                ),
                strict=False,
                map_location=torch.device("cpu"),
            )

            # Sync configs
            assert ae_task.cfg.dataset == cfg.dataset
            cfg.prefix_length = ae_task.cfg.prefix_length
            cfg.suffix_length = ae_task.cfg.suffix_length
            cfg.context_length = ae_task.cfg.context_length
            cfg.encoder_mode = ae_task.cfg.encoder_mode

            self.encoder = ae_task.encoder.eval().requires_grad_(False)

            # Set DLC params in decoder config
            cfg.decoder.dlc_vocab_size = self.encoder.sem.cfg.V
            cfg.decoder.dlc_len = self.encoder.sem.dlc_len
        else:
            assert cfg.prefix_length is not None
            assert cfg.suffix_length is not None
            cfg.encoder_mode = "none"
            cfg.context_length = 0

        # Create decoder  
        self.decoder = DecoderModel(cfg.decoder).train().requires_grad_(True)

        # Initialize Generative PPL eval model
        if cfg.eval_gen_ppl:
            # NOTE: Putting a module in a list avoids Lightning auto-moving it to device
            # We keep it on CPU until needed to save GPU memory
            self.ppl_model = [
                AutoModelForCausalLM.from_pretrained(
                    "meta-llama/Llama-3.2-3B", torch_dtype=torch.bfloat16
                ).cpu()
            ]
            self.ppl_tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B")
            self.ppl_tok.pad_token = self.ppl_tok.eos_token

        # To not eval MAUVE every validation step
        self.val_epoch_counter = 0

        self.cfg = cfg

        self.save_hyperparameters(
            OmegaConf.to_container(OmegaConf.structured(cfg)), logger=False
        )

    def compile(self):
        if self.encoder is not None:
            self.encoder.compile()
        self.decoder.compile()

    def train(self, mode=True):
        # Make sure encoder stays in eval mode
        super().train(mode)
        if self.encoder is not None:
            self.encoder.eval()
        return self

    def setup(self, **kwargs):

        if self.cfg.dataset == "wikipedia":
            self.dataset = WikipediaDataset()
        elif self.cfg.dataset == "fineweb":
            self.dataset = FineWebDataset()
        else:
            raise ValueError(f"Unknown dataset {self.cfg.dataset}")

        self.train_data = Subset(self.dataset, indices=self.dataset.train_indices)
        self.val_data = Subset(self.dataset, indices=self.dataset.val_indices)

    def configure_optimizers(self):
        no_decay = ["bias", "norm"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in self.decoder.named_parameters()
                    if not any(nd in n.lower() for nd in no_decay)
                ],
                "weight_decay": 0.05,
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
        return get_dataloader(
            self.train_data,
            batch_size=self.cfg.batch_size,
            prefix_length=self.cfg.prefix_length,
            suffix_length=self.cfg.suffix_length,
            context_length=self.cfg.context_length,
            enc_tokenizer=self.encoder.tokenizer if self.encoder is not None else None,
            dec_tokenizer=self.decoder.tokenizer,
            encoder_mode=self.cfg.encoder_mode,
            encoder_noise=False,  # No noise during DLC-LM finetuning
            seed=random.randint(0, 100000),  # Dataset should be different if restarted,
            num_dlc_ph=self.encoder.sem.dlc_len if self.encoder is not None else 0,
        )

    def val_dataloader(self):
        return get_dataloader(
            self.val_data,
            batch_size=self.cfg.batch_size,
            prefix_length=self.cfg.prefix_length,
            suffix_length=self.cfg.suffix_length,
            enc_tokenizer=self.encoder.tokenizer if self.encoder is not None else None,
            context_length=self.cfg.context_length,
            dec_tokenizer=self.decoder.tokenizer,
            encoder_mode=self.cfg.encoder_mode,
            encoder_noise=False,
            seed=42,  # Always the same validation set for consistency,
            num_dlc_ph=self.encoder.sem.dlc_len if self.encoder is not None else 0,
        )

    @torch.inference_mode()
    def _fill_DLCs_in_batch(self, batch):
        """
        Use the encoder to compute DLCs for the batch and fill them in the input_ids_dec placeholders
        """
        sem_out = self.encoder(
            batch["input_ids_enc"],
            batch["attention_mask_enc"],
            return_dlc=True,
        )[1]
        dlc_ids = sem_out["dlc"] + len(self.decoder.tokenizer)
        batch["input_ids_dec"][batch["info_mask_dec"] == InfoLabel.DLC.value] = (
            dlc_ids.view(-1)
        )
        return batch

    def _create_dlc_dropout_mask(self, batch) -> torch.Tensor:
        """
        Create attention mask with probabilistic blocking of DLC to suffix attention
        according to DLC_dropout probability."""
        bs, seq_len = batch["input_ids_dec"].shape
        device = batch["input_ids_dec"].device
        # base causal mask (prevent attending to future tokens)
        causal = torch.triu(
            torch.ones((1, 1, seq_len, seq_len), device=device, dtype=torch.bool),
            diagonal=1,
        )
        attention_mask = torch.zeros(
            (bs, 1, seq_len, seq_len), dtype=torch.float32, device=device
        )
        attention_mask = attention_mask.masked_fill(causal, float("-inf"))

        # probabilistic block: suffix queries cannot attend DLC keys (per batch) with prob DLC_dropout
        # also cannot attend to the SECOND think token (SPECIAL token)
        think_tokens = batch["info_mask_dec"] == InfoLabel.SPECIAL.value
        # Second occurrence is where cumsum == 2 and special_mask is True
        second_think_token = (
            torch.cumsum(think_tokens.int(), dim=1) == 2
        ) & think_tokens
        DLC_mask = (
            (batch["info_mask_dec"] == InfoLabel.DLC.value) + second_think_token
        ).view(bs, 1, 1, seq_len)
        suffix_mask = (batch["info_mask_dec"] == InfoLabel.SUFFIX.value).view(
            bs, 1, seq_len, 1
        )
        drop = (torch.rand(bs, device=device) < self.cfg.DLC_dropout).view(bs, 1, 1, 1)
        block = drop * suffix_mask * DLC_mask  # (bs, 1, seq_len, seq_len)
        attention_mask = attention_mask.masked_fill(block, float("-inf"))

        return attention_mask

    def training_step(self, batch, batch_idx):

        # Fill DLCs in the batch
        if self.encoder is not None:
            batch = self._fill_DLCs_in_batch(batch)

        attention_mask = (
            self._create_dlc_dropout_mask(batch) if self.cfg.DLC_dropout > 0 else None
        )

        logits = self.decoder(
            input_ids=batch["input_ids_dec"], attention_mask=attention_mask
        )

        # Compute loss
        info_mask_dec = batch["info_mask_dec"][:, 1:]
        targets = batch["input_ids_dec"][:, 1:].contiguous()
        logits = logits[:, :-1].contiguous()
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1),
            reduction="none",
        )
        loss = loss.view_as(targets)

        # p(z)
        if self.encoder is not None:
            dlc_loss = loss[info_mask_dec == InfoLabel.DLC.value].mean()
            self.log(
                "train/dlc_loss",
                dlc_loss,
                on_epoch=False,
                on_step=True,
                sync_dist=True,
            )

        # p(x|z)
        cond_loss = loss[(info_mask_dec == InfoLabel.SUFFIX.value)].mean()
        self.log(
            "train/suffix_loss",
            cond_loss,
            on_epoch=False,
            on_step=True,
            sync_dist=True,
        )

        # Train on all non-PAD tokens (including PREFIX, SUFFIX, DLC and SPECIAL tokens)
        full_loss = loss[(info_mask_dec != InfoLabel.PAD.value)].mean()
        self.log(
            "train/full_loss",
            full_loss,
            on_epoch=False,
            on_step=True,
            sync_dist=True,
        )

        return full_loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):

        # Fill DLCs in the batch
        if self.encoder is not None:
            batch = self._fill_DLCs_in_batch(batch)

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

        # p(z)p(x|z)
        if self.encoder is not None:
            dlc_loss = loss[info_mask_dec == InfoLabel.DLC.value].mean()
            self.log(
                "val/dlc_loss",
                dlc_loss,
                on_epoch=True,
                on_step=False,
                sync_dist=True,
            )

        # p(x|z) only, more like the reconstruction loss
        cond_loss = loss[(info_mask_dec == InfoLabel.SUFFIX.value)].mean()
        self.log(
            "val/suffix_loss",
            cond_loss,
            on_epoch=True,
            on_step=False,
            sync_dist=True,
        )

        if batch_idx == 0:

            # Evaluate generative perplexity
            if self.cfg.eval_gen_ppl:

                # Move generative PPL eval model to GPU
                ppl_model = self.ppl_model[0]
                ppl_model.to(batch["input_ids_dec"].device)

                # Gather prompt and DLC from input_ids
                prefix = [
                    batch["input_ids_dec"][i][
                        (batch["info_mask_dec"][i] == InfoLabel.PREFIX.value)
                    ].tolist()
                    for i in range(len(batch["input_ids_dec"]))
                ]

                # Generate for two temperatures
                for temp in [0.1, 1.0]:
                    gen_suffix_str = self.decoder.tokenizer.batch_decode(
                        self.decoder.generate(
                            prefix=prefix,
                            max_length=self.cfg.suffix_length,
                            gen_kwargs_dlc={"temperature": temp},
                            gen_kwargs={"temperature": temp},
                        ),
                        skip_special_tokens=True,
                    )

                    # Tokenize the full sequences with perplexity model
                    ppl_batch = self.ppl_tok.batch_encode_plus(
                        [p + c for p, c in zip(batch["prefix_str"], gen_suffix_str)],
                        padding=True,
                        return_tensors="pt",
                        return_offsets_mapping=True,
                        add_special_tokens=False,
                        return_attention_mask=False,
                    ).to(device=batch["input_ids_dec"].device)

                    # Compute token index where continuation starts in the new indices
                    split_idx = split_index_from_offsets(
                        ppl_batch["offset_mapping"],
                        [len(p) for p in batch["prefix_str"]],
                    )

                    # Compute conditional perplexity of suffix given prefix
                    # I.e. ignore prompt and padding tokens in loss (set to -100)
                    labels = ppl_batch["input_ids"].clone()
                    labels[labels == self.ppl_tok.pad_token_id] = -100
                    for i in range(len(labels)):
                        labels[i, : split_idx[i]] = -100
                    ppl = torch.exp(
                        ppl_model(input_ids=ppl_batch["input_ids"], labels=labels).loss
                    )
                    self.log(
                        f"val/gen_ppl_t={temp}",
                        ppl.item(),
                        on_epoch=True,
                        on_step=False,
                        sync_dist=True,
                    )

                # Move ppl model back to CPU
                ppl_model.cpu()
                torch.cuda.empty_cache()

            # Log some reconstructed samples
            if (rank_zero_only.rank == 0) and (self.encoder is not None):

                prefix_str = batch["prefix_str"][:5]

                true_suffix_str = batch["suffix_str"][:5]

                # Generate from ground truth DLC
                dlc = [
                    batch["input_ids_dec"][i][
                        (batch["info_mask_dec"][i] == InfoLabel.DLC.value)
                    ].tolist()
                    for i in range(5)
                ]
                dlc = None if len(dlc[0]) == 0 else dlc
                gen_suffix_str = self.decoder.tokenizer.batch_decode(
                    self.decoder.generate(
                        prefix=prefix[:5],
                        dlc=dlc,
                        max_length=self.cfg.suffix_length,
                    ),
                    skip_special_tokens=True,
                )

                table = wandb.Table(
                    columns=["Prefix", "True Suffix", "Generated Suffix"]
                )
                for i in range(5):
                    table.add_data(prefix_str[i], true_suffix_str[i], gen_suffix_str[i])
                wandb.log({"val/samples": table})
                del table
