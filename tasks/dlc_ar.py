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
from model.decoder import DecoderConfig, DecoderModel
from tasks.autoencoder import AETask
from tasks.declutr import DeCLUTRTask
from tasks.dino_mixture import DINOMixtureTask
from tasks.utils import *


@dataclass
class DLCARTaskConfig:
    lr: float
    lr_warmup_steps: int
    batch_size: int
    decoder: DecoderConfig
    eval_gen_ppl: bool
    DLC_dropout: float = 0.0

    eval_mauve: bool = True
    mauve_reference_features_path: Optional[str] = "mauve_reference_features.npy"
    mauve_model_name: str = "gpt2-large"
    mauve_max_len: int = 150
    mauve_device_id: int = 0
    mauve_batch_size: int = 64

    dataset: Optional[LanguageDatasetConfig] = None
    pretrained_ae_id: Optional[str] = None
    pretrained_declutr_id: Optional[str] = None
    pretrained_dino_id: Optional[str] = None

    prefix_length: Optional[int] = None
    suffix_length: Optional[int] = None
    context_length: Optional[int] = None

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
            cfg.decoder.dlc_vocab_size = self.encoder.sem.cfg.V
            cfg.decoder.dlc_len = self.encoder.sem.dlc_len
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

            cfg.dataset = task.cfg.dataset
            assert cfg.prefix_length is not None
            assert cfg.suffix_length is not None
            assert cfg.context_length is not None

            self.encoder = task.encoder.eval().requires_grad_(False)

            cfg.decoder.dlc_vocab_size = self.encoder.sem.cfg.V
            cfg.decoder.dlc_len = self.encoder.sem.dlc_len

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

            # Sync configs (DINO uses declutr_dataset, not dataset)
            cfg.dataset = task.cfg.declutr_dataset
            assert cfg.prefix_length is not None
            assert cfg.suffix_length is not None
            assert cfg.context_length is not None

            self.encoder = task.encoder.eval().requires_grad_(False)

            cfg.decoder.dlc_vocab_size = self.encoder.sem.cfg.V
            cfg.decoder.dlc_len = self.encoder.sem.dlc_len

        else:
            assert cfg.dataset is not None
            assert cfg.prefix_length is not None
            assert cfg.suffix_length is not None
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
        self.decoder.compile()

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
            dec_tok=self.decoder.tokenizer,
            encoder_mode=self.encoder_mode,
            encoder_noise=False,  # No noise during DLC-LM finetuning
            seed=random.randint(0, 100000),  # Dataset should be different if restarted,
            num_dlc_ph=self.encoder.sem.dlc_len if self.encoder is not None else 0,
        )

    def val_dataloader(self):
        return PrefixSuffixIterable.get_dataloader(
            self.val_data,
            batch_size=self.cfg.batch_size,
            prefix_length=self.cfg.prefix_length,
            suffix_length=self.cfg.suffix_length,
            enc_tok=self.encoder.tokenizer if self.encoder is not None else None,
            context_length=self.cfg.context_length,
            dec_tok=self.decoder.tokenizer,
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

    def compute_loss(self, batch, attention_mask=None) -> dict:
        """Compute decoder losses. Returns dict with full_loss, suffix_loss, and dlc_loss."""
        logits = self.decoder(
            input_ids=batch["input_ids_dec"], attention_mask=attention_mask
        )

        info_mask_dec = batch["info_mask_dec"][:, 1:]
        targets = batch["input_ids_dec"][:, 1:].contiguous()
        logits = logits[:, :-1].contiguous()
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1),
            reduction="none",
        )
        loss = loss.view_as(targets)

        dlc_loss = None
        if self.encoder is not None:
            dlc_loss = loss[info_mask_dec == InfoLabel.DLC.value].mean()

        suffix_loss = loss[(info_mask_dec == InfoLabel.SUFFIX.value)].mean()
        full_loss = loss[(info_mask_dec != InfoLabel.PAD.value)].mean()

        return {"full_loss": full_loss, "suffix_loss": suffix_loss, "dlc_loss": dlc_loss}

    @torch.no_grad()
    def generate_suffix(self, batch, latent_generation_mode: str = "sample") -> List[str]:
        """
        Generate suffixes for the given batch.

        Args:
            latent_generation_mode: How to generate the DLC latent:
                - "sample": Sample DLCs autoregressively p(DLC|prefix), then p(suffix|prefix,DLC)
                - "true": Use true DLC from encoder (requires encoder)
                - "random": Use random DLC token IDs from DLC vocabulary (requires encoder)
        """
        prefix = [
            batch["input_ids_dec"][i][
                (batch["info_mask_dec"][i] == InfoLabel.PREFIX.value)
            ].tolist()
            for i in range(len(batch["input_ids_dec"]))
        ]

        if latent_generation_mode == "sample":
            return self.decoder.tokenizer.batch_decode(
                self.decoder.generate(
                    prefix=prefix,
                    max_length=self.cfg.suffix_length,
                    gen_kwargs_dlc={"temperature": 1.0},
                    gen_kwargs={"temperature": 1.0},
                ),
                skip_special_tokens=True,
            )

        elif latent_generation_mode == "true":
            dlc = [
                batch["input_ids_dec"][i][
                    (batch["info_mask_dec"][i] == InfoLabel.DLC.value)
                ].tolist()
                for i in range(len(batch["input_ids_dec"]))
            ]
            return self.decoder.tokenizer.batch_decode(
                self.decoder.generate(
                    prefix=prefix,
                    dlc=dlc,
                    max_length=self.cfg.suffix_length,
                ),
                skip_special_tokens=True,
            )

        elif latent_generation_mode == "random":
            bs = len(prefix)
            dlc_vocab_size = self.encoder.sem.cfg.V
            dlc_len = self.encoder.sem.dlc_len
            tok_offset = len(self.decoder.tokenizer)
            random_dlc = [
                (torch.randint(0, dlc_vocab_size, (dlc_len,)) + tok_offset).tolist()
                for _ in range(bs)
            ]
            return self.decoder.tokenizer.batch_decode(
                self.decoder.generate(
                    prefix=prefix,
                    dlc=random_dlc,
                    max_length=self.cfg.suffix_length,
                ),
                skip_special_tokens=True,
            )

        else:
            raise ValueError(f"Unknown latent_generation_mode: {latent_generation_mode}")

    def training_step(self, batch, batch_idx):

        # Fill DLCs in the batch
        if self.encoder is not None:
            batch = self._fill_DLCs_in_batch(batch)

        attention_mask = (
            self._create_dlc_dropout_mask(batch) if self.cfg.DLC_dropout > 0 else None
        )

        losses = self.compute_loss(batch, attention_mask=attention_mask)

        if losses["dlc_loss"] is not None:
            self.log(
                "train/dlc_loss",
                losses["dlc_loss"],
                on_epoch=False,
                on_step=True,
                sync_dist=True,
            )

        self.log(
            "train/suffix_loss",
            losses["suffix_loss"],
            on_epoch=False,
            on_step=True,
            sync_dist=True,
        )
        self.log(
            "train/full_loss",
            losses["full_loss"],
            on_epoch=False,
            on_step=True,
            sync_dist=True,
        )

        return losses["full_loss"]

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):

        # Fill DLCs in the batch
        if self.encoder is not None:
            batch = self._fill_DLCs_in_batch(batch)

        losses = self.compute_loss(batch)

        if losses["dlc_loss"] is not None:
            self.log(
                "val/dlc_loss",
                losses["dlc_loss"],
                on_epoch=True,
                on_step=False,
                sync_dist=True,
            )

        self.log(
            "val/suffix_loss",
            losses["suffix_loss"],
            on_epoch=True,
            on_step=False,
            sync_dist=True,
        )

        if batch_idx <= 1:

            if self.cfg.eval_gen_ppl:

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

                    if (rank_zero_only.rank == 0) and (wandb.run is not None):
                        table = wandb.Table(
                            columns=["Prefix", "True Suffix", "Generated Suffix"]
                        )
                        for i in range(5):
                            table.add_data(
                                batch["prefix_str"][i],
                                batch["suffix_str"][i],
                                gen_suffix_sample[i],
                            )
                        wandb.log({"val/samples": table})
                        del table

        # MAUVE: accumulate samples across all batches
        if self.cfg.eval_mauve:
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
