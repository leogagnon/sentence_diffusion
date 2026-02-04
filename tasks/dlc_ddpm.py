import os
import random
from dataclasses import dataclass
from typing import Optional
from functools import partial
from itertools import chain

import lightning as L
import torch
import wandb
from lightning.pytorch.utilities.rank_zero import rank_zero_only
from omegaconf import OmegaConf
from torch.utils.data.dataset import Subset
from transformers import AutoTokenizer, get_constant_schedule_with_warmup
from transformers.models.auto.modeling_auto import AutoModelForCausalLM

from data import InfoLabel, LanguageDataset, LanguageDatasetConfig, PrefixSuffixIterable
from tasks.declutr import DeCLUTRTask
from tasks.autoencoder import AETask
from tasks.utils import *
from model.decoder import DecoderModel, DecoderConfig
from model.diffusion_continuous import *


@dataclass
class GaussianDiffusionTaskConfig:
    model: DiTContinuousConfig
    batch_size: int
    lr: float
    lr_warmup_steps: int
    decoder: DecoderConfig
    diffusion_beta: float = 5.0
    noise_delta: float = 0.0
    eval_gen_ppl: bool = True

    pretrained_declutr_id: Optional[str] = None
    pretrained_ae_id: Optional[str] = None

    dataset: Optional[LanguageDatasetConfig] = None
    prefix_length: Optional[int] = None
    suffix_length: Optional[int] = None
    context_length: Optional[int] = None

    loss: str = "l2"
    sampling_timesteps: int = 50
    train_schedule: str = "cosine"
    sampling_schedule: Optional[str] = None
    diffusion_objective: str = "pred_v"
    schedule_scale: float = 1.0
    sampler: str = "ddpm"
    normalize_latent: bool = False

    name: Optional[str] = None


class GaussianDiffusionTask(L.LightningModule):
    """
    Trains a latent diffusion transformer on the latent space of a pretrained autoencoder (ae_id).
    """

    def __init__(
        self, cfg: Optional[GaussianDiffusionTaskConfig] = None, **kwargs
    ) -> None:
        super().__init__()

        if cfg == None:
            cfg = OmegaConf.to_object(
                OmegaConf.merge(
                    OmegaConf.create(GaussianDiffusionTaskConfig),
                    OmegaConf.create(kwargs),
                )
            )

        # Init noise schedules
        self.train_schedule = partial(
            time_to_alpha,
            alpha_schedule=get_sampling_schedule(cfg.train_schedule),
            scale=cfg.schedule_scale,
        )
        if cfg.sampling_schedule != None:
            self.sampling_schedule = partial(
                time_to_alpha,
                alpha_schedule=get_sampling_schedule(cfg.sampling_schedule),
                scale=cfg.schedule_scale,
            )
        else:
            self.sampling_schedule = self.train_schedule

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
            self.encoder.out_proj = nn.Identity()

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

            # Init encoder (and remove out proj; no longer needed)
            self.encoder = task.encoder.eval().requires_grad_(False)
            self.encoder.out_proj = nn.Identity()

        else:
            assert cfg.prefix_length is not None
            assert cfg.suffix_length is not None
            cfg.context_length = 0

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

        # Init decoder (with cross attention dim = encoder latent dim)
        cfg.decoder.condition_dim = self.encoder.backbone_dim
        self.decoder = DecoderModel(cfg.decoder).train().requires_grad_(True)

        # Init DiT (with seq conditional dim = decoder latent dim; latent dim = encoder latent dim)
        cfg.model.latent_dim = self.encoder.backbone_dim
        cfg.model.seq_conditional_dim = self.decoder.latent_dim
        self.model = DiTContinuous(cfg.model).train().requires_grad_(True)

        # Init latent normalization if needed
        if cfg.normalize_latent:
            self.register_buffer(
                "latent_mean",
                torch.zeros(size=(cfg.model.latent_dim,)).float(),
            )
            self.latent_mean: torch.FloatTensor
            self.register_buffer(
                "latent_scale",
                torch.ones(size=(cfg.model.latent_dim,)).float(),
            )
            self.latent_scale: torch.FloatTensor

        self.encoder_mode = "none"
        if self.encoder is not None:
            if cfg.context_length > 0:
                self.encoder_mode = "context"
            else:
                self.encoder_mode = "suffix"

        self.cfg = cfg
        # Important for checkpoints
        self.save_hyperparameters(
            OmegaConf.to_container(OmegaConf.structured(cfg)), logger=False
        )

    def train(self, mode: bool = True):
        super().train(mode)
        # Make sure encoder and decoder and EMA are always in eval mode
        self.encoder.eval()
        self.decoder.eval()

        return self

    def setup(self, **kwargs):

        self.dataset = LanguageDataset(self.cfg.dataset)

        self.train_data = Subset(self.dataset, indices=self.dataset.train_indices)
        self.val_data = Subset(self.dataset, indices=self.dataset.val_indices)

    def normalize_latent(self, x_start):
        eps = 1e-5

        return (x_start - self.latent_mean) / (self.latent_scale).clamp(min=eps)

    def unnormalize_latent(self, x_start):
        eps = 1e-5

        return x_start * (self.latent_scale.clamp(min=eps)) + self.latent_mean

    def configure_optimizers(self):
        trainable_params = chain(
            self.model.named_parameters(), self.decoder.named_parameters()
        )
        no_decay = ["bias", "norm"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in trainable_params
                    if not any(nd in n.lower() for nd in no_decay)
                ],
                "weight_decay": 0.01,
            },
            {
                "params": [
                    p
                    for n, p in trainable_params
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
            dec_tok=self.decoder.tokenizer,
            encoder_mode=self.encoder_mode,
            encoder_noise=False,  # No noise during DLC-LM finetuning
            seed=random.randint(0, 100000),  # Dataset should be different if restarted,
            num_dlc_ph=0,
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
            num_dlc_ph=0,
        )

    @torch.no_grad()
    def on_fit_start(self):

        # Compute latent mean and scale if needed (on 10000 samples, per rank)
        if self.cfg.normalize_latent:
            if rank_zero_only.rank == 0:
                print("Computing latent mean and scale...")

                dl_iter = iter(self.train_dataloader())
                latent_samples = []
                for _ in tqdm(range(10000 // self.cfg.batch_size)):
                    batch = next(dl_iter)
                    latent_samples.append(
                        self.encoder(
                            batch["input_ids_enc"].cuda(),
                            attention_mask=batch["attention_mask_enc"].cuda(),
                            only_backbone=True,
                        )
                    )

                latent_samples = torch.cat(latent_samples, dim=0)

                latent_mean = torch.mean(latent_samples, dim=0)
                latent_scale = torch.std(
                    latent_samples - latent_mean, unbiased=False, dim=0
                )

                del dl_iter
            else:
                latent_mean = torch.zeros(size=(self.cfg.model.latent_dim,)).float()
                latent_scale = torch.zeros(size=(self.cfg.model.latent_dim,)).float()

            if self.trainer.num_devices > 1:
                # We do it this way (compute stats on rank 0 only) to avoid all_gather memory issues
                latent_mean = self.trainer.strategy.broadcast(latent_mean, src_rank=0)
                latent_scale = self.trainer.strategy.broadcast(latent_scale, src_rank=0)
            
            self.latent_mean.copy_(latent_mean)
            self.latent_scale.copy_(latent_scale)

            print("Done!")

    @torch.no_grad()
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

    def training_step(self, batch, batch_idx=None):

        with torch.no_grad():
            # Compute latent with encoder
            z = self.encoder(
                batch["input_ids_enc"],
                attention_mask=batch["attention_mask_enc"],
                only_backbone=True,
            )
            if self.cfg.normalize_latent:
                z = self.normalize_latent(z)

        # Maybe add a little noise to the latent (interpolating between latents and standard normal)
        # Supposed to help mitigate sampling errors of the diffusion model
        if self.cfg.noise_delta > 0.0:
            noise = torch.randn_like(z)
            z = noise * self.cfg.noise_delta + z * math.sqrt(
                (1.0 - self.cfg.noise_delta**2)
            )

        # Compute suffix loss (conditionned on prefix + latent)
        logits, hidden_states = self.decoder(
            input_ids=batch["input_ids_dec"], z=z, return_hidden_states=True
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
        suffix_loss = loss[(info_mask_dec == InfoLabel.SUFFIX.value)].mean()
        self.log(
            "train/suffix_loss",
            suffix_loss,
            on_epoch=False,
            on_step=True,
            sync_dist=True,
        )

        # Compute diffusion loss (conditionned on prefix hidden states)
        diffusion_loss = compute_diffusion_loss(
            self.model,
            z,
            cond=hidden_states[:, : self.cfg.prefix_length],
            schedule=self.train_schedule,
            diffusion_objective=self.cfg.diffusion_objective,
            loss_name=self.cfg.loss,
        )
        self.log(
            "train/diffusion_loss",
            diffusion_loss,
            on_epoch=False,
            on_step=True,
            sync_dist=True,
        )

        full_loss = suffix_loss + self.cfg.diffusion_beta * diffusion_loss

        self.log(
            "train/loss",
            full_loss,
            prog_bar=True,
            add_dataloader_idx=False,
            batch_size=z.shape[0],
            on_step=True,
            on_epoch=False,
            sync_dist=True,
        )

        return full_loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx=None):

        # Compute latent with encoder
        z = self.encoder(
            batch["input_ids_enc"],
            attention_mask=batch["attention_mask_enc"],
            only_backbone=True,
        )
        if self.cfg.normalize_latent:
            z = self.normalize_latent(z)

        # Compute suffix loss (conditionned on prefix + latent)
        logits, hidden_states = self.decoder(
            input_ids=batch["input_ids_dec"], z=z, return_hidden_states=True
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
        suffix_loss = loss[(info_mask_dec == InfoLabel.SUFFIX.value)].mean()
        self.log(
            "val/suffix_loss",
            suffix_loss,
            on_epoch=False,
            on_step=True,
            sync_dist=True,
        )

        # Compute diffusion loss (conditionned on prefix hidden states)
        diffusion_loss = compute_diffusion_loss(
            self.model,
            z,
            cond=hidden_states[:, : self.cfg.prefix_length],
            schedule=self.train_schedule,
            diffusion_objective=self.cfg.diffusion_objective,
            loss_name=self.cfg.loss,
        )
        self.log(
            "val/diffusion_loss",
            diffusion_loss,
            on_epoch=False,
            on_step=True,
            sync_dist=True,
        )

        full_loss = suffix_loss + self.cfg.diffusion_beta * diffusion_loss

        self.log(
            "val/loss",
            full_loss,
            prog_bar=True,
            add_dataloader_idx=False,
            batch_size=z.shape[0],
            on_step=True,
            on_epoch=False,
            sync_dist=True,
        )

        # Generate samples and compute perplexity with LLaMA 3B
        if self.cfg.eval_gen_ppl and batch_idx <= 2:

            z_samples = sample(
                self.model,
                schedule=self.sampling_schedule,
                batch_size=z.shape[0],
                sampling_timesteps=self.cfg.sampling_timesteps,
                sampler=self.cfg.sampler,
                diffusion_objective=self.cfg.diffusion_objective,
                cond=hidden_states[:, : self.cfg.prefix_length],
            )

            if self.cfg.normalize_latent:
                z_samples = self.unnormalize_latent(z_samples)

            suffix_sample_str = self.decoder.tokenizer.batch_decode(
                self.decoder.generate(
                    max_length=self.cfg.suffix_length,
                    z=z_samples,
                    prefix=batch["input_ids_dec"][:, : self.cfg.prefix_length],
                ),
                skip_special_tokens=True,
            )

            ppl_samples = self.eval_ppl(
                batch["prefix_str"],
                suffix_sample_str,
                device=batch["input_ids_dec"].device,
            )
            self.log(
                "val/gen_ppl",
                ppl_samples,
                on_epoch=False,
                on_step=True,
                sync_dist=True,
            )
