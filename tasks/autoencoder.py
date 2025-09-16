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
from model.gaussian_diffusion import DiTConfig, DiT
from data.stories import StoriesDatasetConfig, StoriesDataset
from hydra.utils import instantiate
from model.encoder import EncoderConfig, EncoderModel
from model.decoder import DecoderConfig, DecoderModel
import evaluate
import os
import wandb
from tasks.finetune import FinetuneTask


@dataclass
class AETaskConfig:
    lr: float
    train_batch_size: int
    val_batch_size: int
    encoder: EncoderConfig
    pretrained_decoder_id: str
    input_sub_p: float
    kl_beta: float
    max_generation_length: int
    dataset: StoriesDatasetConfig
    val_size: int
    name: Optional[str] = None


class AETask(L.LightningModule):
    """
    Autoencoder Task. Combines an encoder and a decoder model to form an autoencoder.
    Trains the autoencoder to reconstruct the input text.
    Supports variational autoencoding.
    Evaluates using
        - BLEU score of reconstructed clean text
        - Perplexity of interpolated samples in the latent space
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

        # Load decoder
        finetune_task = FinetuneTask.load_from_checkpoint(
            os.path.join(
                os.environ["LATENT_CONTROL_CKPT_DIR"],
                cfg.pretrained_decoder_id,
                "last.ckpt",
            ),
            strict=False,
        )
        self.train_indices = finetune_task.train_indices
        self.val_indices = finetune_task.val_indices

        self.decoder: DecoderModel = finetune_task.decoder
        if self.decoder.cfg.lora_cfg is not None:
            # Merge pretraining LoRA weights and create new ones for AE training
            self.decoder.backbone = self.decoder.backbone.merge_and_unload()
            self.decoder.backbone = get_peft_model(
                self.decoder.backbone,
                LoraConfig(**self.decoder.cfg.lora_cfg),
            )

        # Load encoder (set the encoder output dimension to match the decoder input dimension)
        cfg.encoder.out_proj_dim = self.decoder.backbone.config.hidden_size
        self.encoder = EncoderModel(cfg.encoder)

        self.dataset = StoriesDataset(
            cfg.dataset,
            enc_tokenizer=self.encoder.tokenizer,
            dec_tokenizer=self.decoder.tokenizer,
        )

        self.bleu = evaluate.load(
            "bleu", experiment_id=os.urandom(15).hex()
        )  # Random experiment_id to avoid cache conflicts

        # Make sure there is no dropout in the decoder
        for mod in self.decoder.modules():
            if isinstance(mod, torch.nn.Dropout):
                mod.p = 0.0

        self.cfg = cfg

        self.save_hyperparameters(
            OmegaConf.to_container(OmegaConf.structured(cfg)), logger=False
        )

    def setup(self, **kwargs):
        """Setup the data"""
        self.train_data = Subset(self.dataset, indices=self.train_indices)
        self.val_data = Subset(
            self.dataset, indices=self.val_indices[: self.cfg.val_size]
        )

    def random_substitution(self, inputs, p=None):
        inputs = inputs.clone()
        probability = torch.full(
            inputs.shape,
            p if p is not None else self.cfg.input_sub_p,
            dtype=torch.float32,
            device=inputs.device,
        )

        masked_indices = torch.bernoulli(probability).bool()
        random_words = torch.randint(
            len(self.encoder.tokenizer),
            inputs.shape,
            dtype=torch.long,
            device=inputs.device,
        )
        inputs[masked_indices] = random_words[masked_indices]

        return inputs

    def reparameterize(self, mean, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mean)

    def train_dataloader(self):
        return DataLoader(
            self.train_data,
            batch_size=self.cfg.train_batch_size,
            shuffle=True,
            collate_fn=lambda x: x,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_data,
            batch_size=self.cfg.val_batch_size,
            shuffle=False,
            collate_fn=lambda x: x,
        )

    def configure_optimizers(self):
        optimizer = torch.optim.Adamax(self.parameters(), lr=self.cfg.lr)
        return optimizer

    def training_step(self, batch, batch_idx):

        loss = 0.0

        input_ids_enc = batch["input_ids_enc"]
        if self.cfg.input_sub_p > 0:
            input_ids_enc = self.random_substitution(input_ids_enc)

        z = self.encoder(input_ids_enc, attention_mask=batch["attention_mask_enc"])

        if self.encoder.cfg.variational:
            # If variational, reparameterize and compute KL loss
            mean, log_var = z
            z = self.reparameterize(mean, log_var)

            KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
            loss += self.cfg.kl_beta * KLD
            wandb.log({"train/KLD": KLD})

        # Get embeddings of input_ids
        logits = self.decoder(batch["input_ids_dec"], z)

        # Ignore padding tokens
        targets = batch["input_ids_dec"].masked_fill(
            batch["attention_mask_dec"] == 0, -1
        )

        logits = logits[:, :-1].contiguous()
        targets = targets[:, 1:].contiguous()

        # Apply cross-entropy loss
        recon_loss = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1),
            ignore_index=-1,
        )
        loss += recon_loss

        wandb.log({"train/reconstruction_loss": recon_loss})

        return loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        
        loss = 0.0

        # Encode input_ids
        input_ids_enc = batch["input_ids_enc"]
        z = self.encoder(input_ids_enc, attention_mask=batch["attention_mask_enc"])
        if self.encoder.cfg.variational:
            mean, log_var = z
            KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
            self.log("val/KLD", KLD, on_epoch=True)
            loss += self.cfg.kl_beta * KLD

            # Note: use mean for evaluation
            z = mean

        # Loss evaluation
        logits = self.decoder(batch["input_ids_dec"], z)
        targets = batch["input_ids_dec"].masked_fill(
            batch["attention_mask_dec"] == 0, -1
        )
        logits = logits[:, :-1].contiguous()
        targets = targets[:, 1:].contiguous()
        recon_loss = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1),
            ignore_index=-1,
        )
        self.log("val/reconstruction_loss", recon_loss, on_epoch=True)
        loss += recon_loss

        # Noise robustness evaluation
        z_noised = 0.7 * z + (1 - math.sqrt(0.7)) * torch.randn_like(z)
        generation = self.decoder.tokenizer.batch_decode(
            self.decoder.generate(z=z_noised, max_length=self.cfg.max_generation_length),
            skip_special_tokens=True,
        )
        bleu_noised = self.bleu.compute(
            predictions=generation, references=batch["input_str"]
        )["bleu"]
        self.log("val/bleu_noised", bleu_noised, on_epoch=True)
        if batch_idx == 0:
            table = wandb.Table(columns=["Original", "Noise+Reconstructed"])
            for original, reconstructed in zip(batch["input_str"][:10], generation[:10]):
                table.add_data(original, reconstructed)
            wandb.log({"val/noise_samples": table})

        # Interpolation evaluation
        group_indices = torch.randperm(input_ids_enc.shape[0]).chunk(2)
        z_groups = [z[indices] for indices in group_indices]
        z_interp = 0.5 * z_groups[0] + 0.5 * z_groups[1]
        gen_interp_ids = self.decoder.generate(
            z=z_interp, max_length=self.cfg.max_generation_length
        )
        with self.decoder.backbone.disable_adapter():
            # Evaluate perplexity of interpolated samples with pre-trained decoder
            mask = gen_interp_ids != self.decoder.tokenizer.pad_token_id
            labels = gen_interp_ids.masked_fill(~mask, -100)
            ppl_interp = torch.exp(
                self.decoder.backbone(
                    input_ids=gen_interp_ids,
                    labels=labels,
                    attention_mask=mask,
                ).loss
            )
        self.log("val/ppl_interp", ppl_interp.item(), on_epoch=True)
        if batch_idx == 0:
            table = wandb.Table(columns=["S1", "S2", "Interpolated"])
            for s1, s2, s_interp in zip(
                [batch["input_str"][i] for i in group_indices[0]][:10],
                [batch["input_str"][i] for i in group_indices[1]][:10],
                self.decoder.tokenizer.batch_decode(
                    gen_interp_ids[:10], skip_special_tokens=True
                ),
            ):
                table.add_data(s1, s2, s_interp)
            wandb.log({"val/interp_samples": table})