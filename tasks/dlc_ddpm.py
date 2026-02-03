import os
import random
from dataclasses import dataclass
from typing import Optional
from functools import partial

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
    max_generation_length: int = 150


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

            # Sync configs
            cfg.dataset = task.cfg.dataset
            assert cfg.prefix_length is not None
            assert cfg.suffix_length is not None
            assert cfg.context_length is not None

            self.encoder = task.encoder.eval().requires_grad_(False)

            cfg.decoder.dlc_vocab_size = self.encoder.sem.cfg.V
            cfg.decoder.dlc_len = self.encoder.sem.dlc_len
        else:
            assert cfg.prefix_length is not None
            assert cfg.suffix_length is not None
            cfg.context_length = 0

        # Create decoder  
        self.decoder = DecoderModel(cfg.decoder).train().requires_grad_(True)

        # Init latent normalization if needed
        if cfg.normalize_latent:
            self.register_buffer(
                "latent_mean",
                torch.zeros(size=(cfg.model.latent_dim)).float(),
            )
            self.latent_mean: torch.FloatTensor
            self.register_buffer(
                "latent_scale",
                torch.ones(size=(cfg.model.latent_dim)).float(),
            )
            self.latent_scale: torch.FloatTensor

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

        # Create dit
        self.model = DiTContinuous(cfg.model).train().requires_grad_(True)

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
        no_decay = ["bias", "norm"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if not any(nd in n.lower() for nd in no_decay)
                ],
                "weight_decay": 0.01,
            },
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
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
            num_dlc_ph=0
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
            num_dlc_ph=0
        )

    @torch.no_grad()
    def on_fit_start(self):

        # Compute latent mean and scale if needed (on 10000 samples, per rank)
        if self.cfg.normalize_latent:
            print("Computing latent mean and scale...")

            dl_iter = iter(self.train_dataloader())
            count = 0
            latent_samples = []
            while count < 10000:
                batch = next(dl_iter)
                latent_samples.append(
                    self.encoder(
                        batch["input_ids_enc"].cuda(),
                        attention_mask=batch["attention_mask_enc"].cuda(),
                    )
                )
                count += batch["input_ids_enc"].shape[0]

            latent_samples = torch.cat(latent_samples, dim=0)
            
            latent_samples = einx.rearrange("w b d -> (w b) d", self.all_gather(latent_samples))

            self.latent_mean = torch.mean(latent_samples, dim=0)
            self.latent_scale = torch.std(
                latent_samples - self.latent_mean, unbiased=False, dim=0
            )

            del dl_iter

            print("Done!")

    def training_step(self, batch, batch_idx=None):

        # Compute latents
        with torch.no_grad():
            _, out = self.encoder(
                batch["input_ids_enc"],
                attention_mask=batch["attention_mask_enc"],
            )
            latent = out["latent"]
            if self.cfg.normalize_latent:
                latent = self.normalize_latent(latent)

        loss = compute_diffusion_loss(
            self.model,
            latent,
            schedule=self.train_schedule,
            diffusion_objective=self.cfg.diffusion_objective,
            loss_name=self.cfg.loss,
        )

        self.log(
            "train/loss",
            loss.detach().cpu().numpy().item(),
            prog_bar=True,
            add_dataloader_idx=False,
            batch_size=latent.shape[0],
            on_step=True,
            on_epoch=False,
            sync_dist=True,
        )

        return loss
    
    @torch.no_grad()
    def validation_step(self, batch, batch_idx=None):
    
        # Compute latents
        _, out = self.encoder(
            batch["input_ids_enc"],
            attention_mask=batch["attention_mask_enc"],
        )
        if self.cfg.normalize_latent:
            latent = self.normalize_latent(latent)

        loss = compute_diffusion_loss(
            self.model,
            latent,
            schedule=self.train_schedule,
            diffusion_objective=self.cfg.diffusion_objective,
            loss_name=self.cfg.loss,
        )

        self.log(
            "val/loss",
            loss.cpu().numpy().item(),
            prog_bar=True,
            add_dataloader_idx=False,
            batch_size=latent.shape[0],
            on_epoch=True,
            on_step=False,
            sync_dist=True,
        )

        return loss