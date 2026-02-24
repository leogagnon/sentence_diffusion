import os
import random
from dataclasses import dataclass
from typing import List, Optional

import lightning as L
import numpy as np
import torch
import wandb
from mauve import get_features_from_input, compute_mauve
from lightning.pytorch.utilities.rank_zero import rank_zero_only
from omegaconf import OmegaConf
from torch.utils.data.dataset import Subset
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup
from transformers.models.auto.modeling_auto import AutoModelForCausalLM

from data import InfoLabel, LanguageDataset, LanguageDatasetConfig, PrefixSuffixIterable
from model.diffusion import DiTConfig, DiTModel
from tasks.declutr import DeCLUTRTask
from tasks.autoencoder import AETask
from tasks.dino_mixture import DINOMixtureTask
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

    eval_mauve: bool = True
    mauve_reference_features_path: Optional[str] =  "mauve_reference_features.npy"
    mauve_model_name: str = "gpt2-large"
    mauve_max_len: int = 150
    mauve_device_id: int = 0
    mauve_batch_size: int = 64

    dataset: Optional[LanguageDatasetConfig] = None
    pretrained_declutr_id: Optional[str] = None
    pretrained_ae_id: Optional[str] = None
    pretrained_dino_id: Optional[str] = None

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

            self.encoder = task.encoder.eval().requires_grad_(False)

            cfg.dit.dlc_vocab_size = self.encoder.sem.cfg.V
            cfg.dit.dlc_len = self.encoder.sem.dlc_len

        elif cfg.pretrained_dino_id is not None:

            task = DINOMixtureTask.load_from_checkpoint(
                os.path.join(
                    os.environ["LOG_DIR"],
                    "checkpoints/",
                    cfg.pretrained_dino_id,
                    "last.ckpt",
                ),
                strict=False,
                map_location=torch.device("cpu"),
            )

            # Sync dataset config (DINO uses declutr_dataset, not dataset)
            cfg.dataset = task.cfg.declutr_dataset
            assert cfg.prefix_length is not None
            assert cfg.suffix_length is not None
            assert cfg.context_length is not None

            self.encoder = task.encoder.eval().requires_grad_(False)

            cfg.dit.dlc_vocab_size = self.encoder.sem.cfg.V
            cfg.dit.dlc_len = self.encoder.sem.dlc_len

        else:
            assert cfg.prefix_length is not None
            assert cfg.suffix_length is not None
            assert cfg.ancestral == False
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

        # Load MAUVE reference features if needed
        self.val_generated_texts_sample = []
        self.val_generated_texts_true = []
        self.val_generated_texts_random = []
        if cfg.eval_mauve:
            assert cfg.mauve_reference_features_path is not None, \
                "mauve_reference_features_path must be provided when eval_mauve=True"
            print(f"Loading MAUVE reference features from {cfg.mauve_reference_features_path}")
            self.mauve_reference_features = np.load(cfg.mauve_reference_features_path)
            print(f"Loaded {self.mauve_reference_features.shape[0]} reference features")

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
            scheduler = get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=self.cfg.lr_warmup_steps,
                num_training_steps=20000,
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

    def compute_loss(self, batch) -> torch.Tensor:
        """Compute masked diffusion loss. Handles ancestral training mode."""
        # Compute loss on suffix + DLC, condition on everything else
        attention_mask = (batch["info_mask_dec"] == InfoLabel.SUFFIX.value) + (
            batch["info_mask_dec"] == InfoLabel.DLC.value
        )
        cond_mask = torch.logical_not(attention_mask)
        loss = self.dit.compute_loss(
            batch["input_ids_dec"],
            attention_mask=attention_mask,
            cond_mask=cond_mask,
        )

        if self.cfg.ancestral:
            # Mask the suffix, only compute loss on the DLC, condition on everything else
            input_ids_dlc = batch["input_ids_dec"].clone()
            input_ids_dlc[batch["info_mask_dec"] == InfoLabel.SUFFIX.value] = (
                self.dit.tokenizer.mask_token_id
            )
            attention_mask_dlc = batch["info_mask_dec"] == InfoLabel.DLC.value
            cond_mask_dlc = torch.logical_not(attention_mask_dlc)

            dlc_loss = self.dit.compute_loss(
                input_ids_dlc,
                attention_mask=attention_mask_dlc,
                cond_mask=cond_mask_dlc,
            )
            loss = 0.5 * loss + 0.5 * dlc_loss

        return loss

    @torch.no_grad()
    def generate_suffix(self, batch, latent_generation_mode: str = "sample") -> List[str]:
        """
        Generate suffixes for the given batch.

        Args:
            latent_generation_mode: How to generate the DLC latent:
                - "sample": Joint masked diffusion p(DLC, suffix | prefix)
                - "true": True DLC from encoder, then sample suffix (requires encoder)
                - "random": Random DLC token IDs, then sample suffix (requires encoder)
        """
        if latent_generation_mode == "sample":
            prior = batch["input_ids_dec"].clone()
            prior[
                (batch["info_mask_dec"] == InfoLabel.SUFFIX.value)
                + (batch["info_mask_dec"] == InfoLabel.DLC.value)
            ] = self.dit.tokenizer.mask_token_id
            return self.dit.tokenizer.batch_decode(
                self.dit.sample(prior=prior)[:, -self.cfg.suffix_length:]
            )

        elif latent_generation_mode == "true":
            # DLC positions already filled with true values by _fill_DLCs_in_batch
            prior = batch["input_ids_dec"].clone()
            prior[(batch["info_mask_dec"] == InfoLabel.SUFFIX.value)] = (
                self.dit.tokenizer.mask_token_id
            )
            return self.dit.tokenizer.batch_decode(
                self.dit.sample(prior=prior)[:, -self.cfg.suffix_length:]
            )

        elif latent_generation_mode == "random":
            prior = batch["input_ids_dec"].clone()
            prior[(batch["info_mask_dec"] == InfoLabel.SUFFIX.value)] = (
                self.dit.tokenizer.mask_token_id
            )
            # Fill DLC positions with random tokens from DLC vocabulary
            dlc_vocab_size = self.encoder.sem.cfg.V
            tok_offset = len(self.dit.tokenizer)
            bs = prior.shape[0]
            dlc_len = self.encoder.sem.dlc_len
            random_dlc = (
                torch.randint(0, dlc_vocab_size, (bs, dlc_len), device=prior.device)
                + tok_offset
            )
            prior[batch["info_mask_dec"] == InfoLabel.DLC.value] = random_dlc.view(-1)
            # Freeze everything except suffix (DLC + prefix are fixed)
            frozen_mask = batch["info_mask_dec"] != InfoLabel.SUFFIX.value
            return self.dit.tokenizer.batch_decode(
                self.dit.sample(prior=prior, frozen_mask=frozen_mask)[:, -self.cfg.suffix_length:]
            )

        else:
            raise ValueError(f"Unknown latent_generation_mode: {latent_generation_mode}")

    def training_step(self, batch, batch_idx):

        # Fill DLCs in the batch
        if self.encoder is not None:
            batch = self._fill_DLCs_in_batch(batch)

        loss = self.compute_loss(batch)

        self.log("train/loss", loss, on_step=True, on_epoch=False, sync_dist=True)

        return loss

    @torch.amp.autocast("cuda", enabled=False)
    def validation_step(self, batch, batch_idx):

        # Fill DLCs in the batch
        if self.encoder is not None:
            batch = self._fill_DLCs_in_batch(batch)

        loss = self.compute_loss(batch)
        self.log("val/loss", loss, on_epoch=True, on_step=False, sync_dist=True)

        if batch_idx <= 1:
            if self.cfg.eval_gen_ppl:

                if not self.cfg.ancestral:
                    gen_suffix_sample = self.generate_suffix(batch, "sample")
                    self.log(
                        "val/gen_ppl_sample",
                        eval_ppl(self.ppl_model[0], self.ppl_tok, batch["prefix_str"], gen_suffix_sample, batch["input_ids_dec"].device),
                        on_epoch=True,
                        on_step=False,
                        sync_dist=True,
                    )

                if self.encoder is not None:

                    gen_suffix_true = self.generate_suffix(batch, "true")
                    self.log(
                        "val/gen_ppl_true",
                        eval_ppl(self.ppl_model[0], self.ppl_tok, batch["prefix_str"], gen_suffix_true, batch["input_ids_dec"].device),
                        on_epoch=True,
                        on_step=False,
                        sync_dist=True,
                    )

                    gen_suffix_random = self.generate_suffix(batch, "random")
                    self.log(
                        "val/gen_ppl_random",
                        eval_ppl(self.ppl_model[0], self.ppl_tok, batch["prefix_str"], gen_suffix_random, batch["input_ids_dec"].device),
                        on_epoch=True,
                        on_step=False,
                        sync_dist=True,
                    )

                    # p(DLC | prefix) * p(suffix | prefix, DLC) via ancestral 2-stage sampling
                    prior = batch["input_ids_dec"].clone()
                    prior[
                        (batch["info_mask_dec"] == InfoLabel.SUFFIX.value)
                        + (batch["info_mask_dec"] == InfoLabel.DLC.value)
                    ] = self.dit.tokenizer.mask_token_id
                    prior = self.dit.sample(
                        prior=prior,
                        frozen_mask=batch["info_mask_dec"] != InfoLabel.DLC.value,
                    )
                    gen_suffix_str_ancestral = self.dit.tokenizer.batch_decode(
                        self.dit.sample(
                            prior=prior,
                            frozen_mask=torch.logical_not(
                                (batch["info_mask_dec"] == InfoLabel.DLC.value)
                                + (batch["info_mask_dec"] == InfoLabel.SUFFIX.value)
                            ),
                            num_steps=self.dit.cfg.sampling_steps // 2,
                        )[:, -self.cfg.suffix_length:]
                    )
                    self.log(
                        "val/gen_ppl_ancestral",
                        eval_ppl(self.ppl_model[0], self.ppl_tok, batch["prefix_str"], gen_suffix_str_ancestral, batch["input_ids_dec"].device),
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
                            self.dit.sample(prior=prior)[:, -self.cfg.suffix_length:]
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

        # MAUVE: accumulate samples across all batches (only when not in ancestral mode)
        if self.cfg.eval_mauve and not self.cfg.ancestral:
            gen_suffix_sample = self.generate_suffix(batch, "sample")
            for prefix_str, suffix_str in zip(batch["prefix_str"], gen_suffix_sample):
                self.val_generated_texts_sample.append(prefix_str + suffix_str)

            if self.encoder is not None:
                gen_suffix_true = self.generate_suffix(batch, "true")
                for prefix_str, suffix_str in zip(batch["prefix_str"], gen_suffix_true):
                    self.val_generated_texts_true.append(prefix_str + suffix_str)

                gen_suffix_random = self.generate_suffix(batch, "random")
                for prefix_str, suffix_str in zip(batch["prefix_str"], gen_suffix_random):
                    self.val_generated_texts_random.append(prefix_str + suffix_str)

    def _compute_and_log_mauve(self, generated_texts, mode):
        """Gather generated texts across ranks and compute MAUVE score for a given mode."""
        if len(generated_texts) == 0:
            return

        if self.trainer.num_devices > 1:
            import torch.distributed as dist
            if dist.is_initialized():
                gathered_texts = [None] * self.trainer.world_size
                dist.all_gather_object(gathered_texts, generated_texts)
                if rank_zero_only.rank == 0:
                    generated_texts = [
                        text for rank_texts in gathered_texts for text in rank_texts
                    ]
                    print(f"Gathered {len(generated_texts)} texts from {self.trainer.world_size} ranks")

        if rank_zero_only.rank == 0:
            print(f"Computing MAUVE ({mode}) with {len(generated_texts)} generated samples...")

            n_samples = len(generated_texts)
            reference_features_subset = self.mauve_reference_features[:n_samples]

            generated_features = get_features_from_input(
                features=None,
                tokenized_texts=None,
                texts=generated_texts,
                featurize_model_name=self.cfg.mauve_model_name,
                max_len=self.cfg.mauve_max_len,
                device_id=self.cfg.mauve_device_id,
                name="generated text",
                batch_size=self.cfg.mauve_batch_size,
                verbose=False,
            )

            mauve_result = compute_mauve(
                p_features=reference_features_subset,
                q_features=generated_features,
                verbose=False,
            )

            self.log(
                f"val/mauve_{mode}",
                mauve_result.mauve,
                on_epoch=True,
                rank_zero_only=True,
                sync_dist=False,
            )
            self.log(
                f"val/mauve_frontier_integral_{mode}",
                mauve_result.frontier_integral,
                on_epoch=True,
                rank_zero_only=True,
                sync_dist=False,
            )

            print(f"MAUVE ({mode}) score: {mauve_result.mauve:.4f}")
            print(f"Frontier integral ({mode}): {mauve_result.frontier_integral:.4f}")

    def on_validation_epoch_start(self):
        if self.cfg.eval_gen_ppl:
            self.ppl_model[0] = self.ppl_model[0].to(self.device)
        if self.cfg.eval_mauve:
            self.val_generated_texts_sample = []
            self.val_generated_texts_true = []
            self.val_generated_texts_random = []
        torch.cuda.empty_cache()

    def on_validation_epoch_end(self):
        if self.cfg.eval_mauve:
            self._compute_and_log_mauve(self.val_generated_texts_sample, "sample")
            if self.encoder is not None:
                self._compute_and_log_mauve(self.val_generated_texts_true, "true")
                self._compute_and_log_mauve(self.val_generated_texts_random, "random")

        if self.cfg.eval_gen_ppl:
            self.ppl_model[0] = self.ppl_model[0].cpu()
        torch.cuda.empty_cache()
