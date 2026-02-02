import os
import random
from dataclasses import dataclass
from typing import Optional

import lightning as L
import torch
import wandb
from lightning.pytorch.utilities.rank_zero import rank_zero_only
from omegaconf import OmegaConf
from torch.utils.data.dataset import Subset
from transformers import AutoTokenizer, get_constant_schedule_with_warmup
from transformers.models.auto.modeling_auto import AutoModelForCausalLM

from data import InfoLabel, LanguageDataset, LanguageDatasetConfig, PrefixSuffixIterable
from model.diffusion import DiTConfig, DiTModel
from tasks.declutr import DeCLUTRTask
from tasks.autoencoder import AETask
from tasks.utils import *


@dataclass
class DLCMDTaskConfig:
    lr: float
    lr_warmup_steps: int
    batch_size: int
    dit: DiTConfig

    eval_gen_ppl: bool
    ancestral: bool = False
    val_batch_size: Optional[int] = None

    dataset: Optional[LanguageDatasetConfig] = None
    pretrained_declutr_id: Optional[str] = None
    pretrained_ae_id: Optional[str] = None

    prefix_length: Optional[int] = None
    suffix_length: Optional[int] = None
    context_length: Optional[int] = None

    name: Optional[str] = None


class DLCMDTask(L.LightningModule):
    """
    Finetune a Masked Diffusion Language Model to use DLCs (DLC-LM)
    """

    def __init__(self, cfg: Optional[DLCMDTaskConfig] = None, **kwargs):
        super().__init__()

        if cfg == None:
            cfg = OmegaConf.to_object(
                OmegaConf.merge(
                    OmegaConf.create(DLCMDTaskConfig),
                    OmegaConf.create(kwargs),
                )
            )

        # Get encoder from pretrained AE if specified
        # Else this will be a baseline LM without DLCs
        self.encoder = None
        if cfg.pretrained_ae_id is not None:

            task = AETask.load_from_checkpoint(
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
            cfg.dataset = task.cfg.dataset
            cfg.prefix_length = task.cfg.prefix_length
            cfg.suffix_length = task.cfg.suffix_length
            cfg.context_length = task.cfg.context_length

            self.encoder = task.encoder.eval().requires_grad_(False)

            # Set DLC params in decoder config
            cfg.dit.dlc_vocab_size = self.encoder.sem.cfg.V
            cfg.dit.dlc_len = self.encoder.sem.dlc_len

        elif cfg.pretrained_declutr_id is not None:

            task = DeCLUTRTask.load_from_checkpoint(
                os.path.join(
                    os.environ["LOG_DIR"],
                    "checkpoints/",
                    cfg.pretrained_declutr_id,
                    "last.ckpt",
                ),
                strict=False,
                map_location=torch.device("cpu"),
            )

            # Sync configs
            cfg.dataset = task.cfg.dataset
            assert cfg.prefix_length is not None
            assert cfg.suffix_length is not None
            assert cfg.context_length is not None

            self.encoder = task.encoder.eval().requires_grad_(False).to(torch.bfloat16)

            cfg.dit.dlc_vocab_size = self.encoder.sem.cfg.V
            cfg.dit.dlc_len = self.encoder.sem.dlc_len
        else:
            assert cfg.prefix_length is not None
            assert cfg.suffix_length is not None
            cfg.context_length = 0

        # Create decoder
        self.dit = DiTModel(cfg.dit).train().requires_grad_(True)

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

        self.encoder_mode = "none"
        if self.encoder is not None:
            if cfg.context_length > 0:
                self.encoder_mode = "context"
            else:
                self.encoder_mode = "suffix"

        # To not eval MAUVE every validation step
        self.val_epoch_counter = 0

        self.cfg = cfg

        self.save_hyperparameters(
            OmegaConf.to_container(OmegaConf.structured(cfg)), logger=False
        )

    def compile(self):
        if self.encoder is not None:
            self.encoder.compile()
        self.dit.compile()

    def train(self, mode=True):
        # Make sure encoder stays in eval mode
        super().train(mode)
        if self.encoder is not None:
            self.encoder.eval()
        return self

    def setup(self, **kwargs):

        self.dataset = LanguageDataset(self.cfg.dataset)

        self.train_data = Subset(self.dataset, indices=self.dataset.train_indices)
        self.val_data = Subset(self.dataset, indices=self.dataset.val_indices)

    def configure_optimizers(self):
        no_decay = ["bias", "norm"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in self.dit.named_parameters()
                    if not any(nd in n.lower() for nd in no_decay)
                ],
                "weight_decay": 0.05,
            },
            {
                "params": [
                    p
                    for n, p in self.dit.named_parameters()
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
        return PrefixSuffixIterable.get_dataloader(
            self.train_data,
            batch_size=self.cfg.batch_size,
            prefix_length=self.cfg.prefix_length,
            suffix_length=self.cfg.suffix_length,
            context_length=self.cfg.context_length,
            enc_tok=self.encoder.tokenizer if self.encoder is not None else None,
            dec_tok=self.dit.tokenizer,
            encoder_mode=self.encoder_mode,
            encoder_noise=False,  # No noise during DLC-LM finetuning
            seed=random.randint(0, 100000),  # Dataset should be different if restarted,
            num_dlc_ph=self.encoder.sem.dlc_len if self.encoder is not None else 0,
        )

    def val_dataloader(self):
        return PrefixSuffixIterable.get_dataloader(
            self.val_data,
            batch_size=(
                self.cfg.val_batch_size
                if self.cfg.val_batch_size is not None
                else self.cfg.batch_size
            ),
            prefix_length=self.cfg.prefix_length,
            suffix_length=self.cfg.suffix_length,
            enc_tok=self.encoder.tokenizer if self.encoder is not None else None,
            context_length=self.cfg.context_length,
            dec_tok=self.dit.tokenizer,
            encoder_mode=self.encoder_mode,
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
        dlc_ids = sem_out["dlc"] + len(self.dit.tokenizer)
        batch["input_ids_dec"][batch["info_mask_dec"] == InfoLabel.DLC.value] = (
            dlc_ids.view(-1)
        )
        return batch

    def training_step(self, batch, batch_idx):

        # Fill DLCs in the batch
        if self.encoder is not None:
            batch = self._fill_DLCs_in_batch(batch)

        # Everything is conditional on the prefix and special tokens (never mask those)
        cond_mask = (batch["info_mask_dec"] == InfoLabel.PREFIX.value) + (
            batch["info_mask_dec"] == InfoLabel.SPECIAL.value
        )

        if self.cfg.ancestral:
            # Mask the suffix and don't compute loss on it (this trains the DLC sampling)
            attention_mask_0 = (batch["info_mask_dec"] != InfoLabel.PAD.value) * (
                batch["info_mask_dec"] != InfoLabel.SUFFIX.value
            )
            input_ids_0 = batch["input_ids_dec"].clone()
            input_ids_0[batch["info_mask_dec"] == InfoLabel.SUFFIX.value] = (
                self.dit.tokenizer.mask_token_id
            )
            cond_mask_0 = cond_mask

            loss_0 = self.dit.compute_loss(
                input_ids_0,
                attention_mask=attention_mask_0,
                cond_mask=cond_mask_0,
            )

            # Only compute loss on the suffix sampling part
            attention_mask_1 = batch["info_mask_dec"] == InfoLabel.SUFFIX.value
            input_ids_1 = batch["input_ids_dec"]
            cond_mask_1 = cond_mask + (batch["info_mask_dec"] == InfoLabel.DLC.value)
            loss_1 = self.dit.compute_loss(
                input_ids_1,
                attention_mask=attention_mask_1,
                cond_mask=cond_mask_1,
            )

            # Merge the losses
            loss = (loss_0 + loss_1) / 2
        else:
            attention_mask = batch["info_mask_dec"] != InfoLabel.PAD.value

            # Start with prefix and <think> tokens unmasked
            cond_mask = (batch["info_mask_dec"] == InfoLabel.PREFIX.value) + (
                batch["info_mask_dec"] == InfoLabel.SPECIAL.value
            )
            loss = self.dit.compute_loss(
                batch["input_ids_dec"],
                attention_mask=attention_mask,
                cond_mask=cond_mask,
            )

        self.log("train/loss", loss, on_step=True, on_epoch=False, sync_dist=True)

        return loss

    def eval_ppl(self, prefix_str, suffix_str, device):
        ppl_batch = self.ppl_tok.batch_encode_plus(
            [p + c for p, c in zip(prefix_str, suffix_str)],
            padding=True,
            return_tensors="pt",
            return_offsets_mapping=True,
            add_special_tokens=False,
            return_attention_mask=False,
        ).to(device)

        # Compute token index where continuation starts in the new indices
        split_idx = split_index_from_offsets(
            ppl_batch["offset_mapping"],
            [len(p) for p in prefix_str],
        )

        # Compute conditional perplexity of suffix given prefix
        # I.e. ignore prompt and padding tokens in loss (set to -100)
        # Move generative PPL eval model to GPU
        ppl_model = self.ppl_model[0].to(device)

        labels = ppl_batch["input_ids"].clone()
        labels[labels == self.ppl_tok.pad_token_id] = -100
        for i in range(len(labels)):
            labels[i, : split_idx[i]] = -100
        ppl = torch.exp(ppl_model(input_ids=ppl_batch["input_ids"], labels=labels).loss)

        del ppl_model
        torch.cuda.empty_cache()

        return ppl.item()

    @torch.amp.autocast("cuda", dtype=torch.float32)
    def validation_step(self, batch, batch_idx):

        # Fill DLCs in the batch
        if self.encoder is not None:
            batch = self._fill_DLCs_in_batch(batch)

        if batch_idx <= 1:
            # Evaluate generative perplexity
            if self.cfg.eval_gen_ppl:

                if not self.cfg.ancestral:
                    # p(DLC, suffix | prefix)
                    prior = batch["input_ids_dec"].clone()
                    prior[
                        (batch["info_mask_dec"] == InfoLabel.SUFFIX.value)
                        + (batch["info_mask_dec"] == InfoLabel.DLC.value)
                    ] = self.dit.tokenizer.mask_token_id
                    gen_suffix_str_joint = self.dit.tokenizer.batch_decode(
                        self.dit.sample(
                            prior=prior,
                        )[:, -self.cfg.suffix_length :]
                    )
                    ppl_joint = self.eval_ppl(
                        batch["prefix_str"],
                        gen_suffix_str_joint,
                        device=batch["input_ids_dec"].device,
                    )
                    self.log(
                        "val/gen_ppl",
                        ppl_joint,
                        on_epoch=True,
                        on_step=False,
                        sync_dist=True,
                    )

                if self.encoder is not None:

                    # p(DLC | prefix) * p(suffix | prefix, DLC)
                    prior = batch["input_ids_dec"].clone()
                    prior[
                        (batch["info_mask_dec"] == InfoLabel.SUFFIX.value)
                        + (batch["info_mask_dec"] == InfoLabel.DLC.value)
                    ] = self.dit.tokenizer.mask_token_id
                    frozen_mask = batch["info_mask_dec"] != InfoLabel.DLC.value
                    prior = self.dit.sample(
                        prior=prior,
                        frozen_mask=frozen_mask,
                    )
                    gen_suffix_str_ancestral = self.dit.tokenizer.batch_decode(
                        self.dit.sample(
                            prior=prior, num_steps=self.dit.cfg.sampling_steps // 2
                        )[:, -self.cfg.suffix_length :]
                    )
                    ppl_ancestral = self.eval_ppl(
                        batch["prefix_str"],
                        gen_suffix_str_ancestral,
                        device=batch["input_ids_dec"].device,
                    )
                    self.log(
                        "val/gen_ppl_ancestral",
                        ppl_ancestral,
                        on_epoch=True,
                        on_step=False,
                        sync_dist=True,
                    )

                    if (rank_zero_only.rank == 0) and (wandb.run is not None):

                        prefix_str = batch["prefix_str"][:5]

                        true_suffix_str = batch["suffix_str"][:5]

                        # Generate conditional on the prefix AND the DLCs
                        prior = batch["input_ids_dec"][:5].clone()
                        prior[
                            (batch["info_mask_dec"][:5] == InfoLabel.SUFFIX.value)
                        ] = self.dit.tokenizer.mask_token_id
                        gen_suffix_str = self.dit.tokenizer.batch_decode(
                            self.dit.sample(prior=prior)[:, -self.cfg.suffix_length :]
                        )

                        table = wandb.Table(
                            columns=["Prefix", "True Suffix", "Generated Suffix"]
                        )
                        for i in range(min(5, len(prefix_str))):
                            table.add_data(
                                prefix_str[i], true_suffix_str[i], gen_suffix_str[i]
                            )
                        wandb.log({"val/samples": table})
                        del table
