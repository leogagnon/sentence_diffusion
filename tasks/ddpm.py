import os
import random
from dataclasses import dataclass
from typing import List, Optional
from functools import partial
from itertools import chain

import lightning as L
import numpy as np
import torch
import wandb
from lightning.pytorch.utilities.rank_zero import rank_zero_only
from omegaconf import OmegaConf
from torch.utils.data.dataset import Subset
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup
from transformers.models.auto.modeling_auto import AutoModelForCausalLM

from mauve import get_features_from_input, compute_mauve

from data import InfoLabel, LanguageDataset, LanguageDatasetConfig, PrefixSuffixIterable
from tasks.declutr import DeCLUTRTask
from tasks.autoencoder import AETask
from tasks.dino_mixture import DINOMixtureTask
from tasks.utils import *
from model.decoder import DecoderModel, DecoderConfig
from model.encoder import EncoderModel, EncoderConfig
from model.diffusion_continuous import *
from torch.optim.swa_utils import AveragedModel, get_ema_avg_fn


@dataclass
class GaussianDiffusionTaskConfig:
    batch_size: int
    lr: float
    lr_warmup_steps: int
    decoder: DecoderConfig
    encoder_lr: Optional[float] = None
    encoder: Optional[EncoderConfig] = None
    model: Optional[DiTContinuousConfig] = None
    use_ema: bool = False
    diffusion_beta: float = 5.0
    diffusion_beta_warmup_steps: Optional[int] = None
    suffix_loss_weight: float = 1.0
    eval_gen_ppl: bool = True
    soft_prompt_len: int = 16

    eval_mauve: bool = True
    mauve_reference_features_path: Optional[str] = "mauve_reference_features.npy"
    mauve_model_name: str = "gpt2-large"
    mauve_max_len: int = 150
    mauve_device_id: int = 0
    mauve_batch_size: int = 64

    pretrained_declutr_id: Optional[str] = None
    pretrained_ae_id: Optional[str] = None
    pretrained_dino_id: Optional[str] = None

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
    prior_t_min: float = 0.0
    prior_enforce_t_min: bool = True
    prior_logsnr_max: Optional[float] = 5.0
    normalize_latent: bool = False
    encoder_post_layernorm: bool = False
    encoder_post_l2norm: bool = False
    encoder_post_l2norm_eps: float = 1e-8
    latent_feature_dropout_prob: float = 0.1
    latent_conditioning_dropout_prob: float = 0.0
    latent_noise: bool = False
    soft_thought_noise_cond: bool = False
    finetune_encoder: bool = False
    encoder_adapter: bool = True
    encoder_adapter_hidden_dim: Optional[int] = None
    num_latent_for_precomputed_stats: int = 30000

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

        if cfg.pretrained_ae_id is not None:
            assert cfg.encoder is None, "Cannot specify both pretrained_ae_id and encoder in config"

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

            self.encoder = task.encoder
            self.encoder.out_proj = nn.Identity()

        elif cfg.pretrained_declutr_id is not None:
            assert cfg.encoder is None, "Cannot specify both pretrained_declutr_id and encoder in config"

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
            self.encoder = task.encoder
            self.encoder.out_proj = nn.Identity()

        elif cfg.pretrained_dino_id is not None:
            assert cfg.encoder is None, "Cannot specify both pretrained_dino_id and encoder in config"

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

            self.encoder = task.encoder
            self.encoder.out_proj = nn.Identity()

        elif cfg.encoder is not None:

            assert cfg.prefix_length is not None
            assert cfg.suffix_length is not None
            assert cfg.context_length is not None

            self.encoder = EncoderModel(cfg.encoder)

        else:
            assert cfg.prefix_length is not None
            assert cfg.suffix_length is not None
            cfg.context_length = 0
            self.encoder = None

        if self.encoder is not None:
            if cfg.finetune_encoder:
                self.encoder.train().requires_grad_(True)
            else:
                self.encoder.eval().requires_grad_(False)

        self.encoder_adapter = None
        if self.encoder is not None and cfg.encoder_adapter and not cfg.finetune_encoder:
            hidden_dim = (
                cfg.encoder_adapter_hidden_dim
                if cfg.encoder_adapter_hidden_dim is not None
                else self.encoder.backbone_dim * 4
            )
            self.encoder_adapter = nn.Sequential(
                nn.Linear(self.encoder.backbone_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, self.encoder.backbone_dim),
            )
            # Residual adapter starts as an exact identity mapping.
            nn.init.zeros_(self.encoder_adapter[-1].weight)
            nn.init.zeros_(self.encoder_adapter[-1].bias)

        self.encoder_post_layernorm = None
        if self.encoder is not None and cfg.encoder_post_layernorm:
            self.encoder_post_layernorm = nn.LayerNorm(
                self.encoder.backbone_dim,
                elementwise_affine=False,
            )

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

        # Init decoder (with cross attention dim = encoder latent dim)
        self.decoder = DecoderModel(cfg.decoder).train().requires_grad_(True)
        self.null_soft_thought = None

        # Init DiT (with seq conditional dim = decoder latent dim; latent dim = encoder latent dim)
        if cfg.model is not None:
            cfg.model.latent_dim = self.encoder.backbone_dim
            cfg.model.seq_conditional_dim = self.decoder.latent_dim
            self.model = DiTContinuous(cfg.model).train().requires_grad_(True)

            # Maybe init EMA model
            if cfg.use_ema:
                self.ema_model = [
                    AveragedModel(self.model, avg_fn=get_ema_avg_fn())
                    .eval()
                    .requires_grad_(False)
                ]

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

            # A small network which transforms the diffusion latent into a soft prompt for the decoder
            pre_proj_dim = 96
            self.soft_thought_proj = nn.Sequential(
                nn.Linear(
                    self.encoder.backbone_dim,
                    cfg.soft_prompt_len * pre_proj_dim,
                    bias=False,
                ),
                Rearrange("b (l d) -> b l d", l=cfg.soft_prompt_len, d=pre_proj_dim),
                nn.Linear(pre_proj_dim, self.decoder.latent_dim, bias=False),
            )
            self.soft_thought_enc = AttentionLayers(
                causal=False,
                dim=self.decoder.latent_dim,
                depth=3,
                heads=8,
                attn_dropout=0.0,
                ff_dropout=0.0,
                rel_pos_bias=False,
                ff_glu=True,
                ff_swish=True,
                # Noise conditioning stuff
                use_adaptive_rmsnorm=cfg.soft_thought_noise_cond,
                use_adaptive_layerscale=cfg.soft_thought_noise_cond,
                dim_condition=(
                    self.decoder.latent_dim if cfg.soft_thought_noise_cond else None
                ),
                adaptive_condition_mlp_expansion=(
                    4 if cfg.soft_thought_noise_cond else None
                ),
                adaptive_condition_mlp=cfg.soft_thought_noise_cond,
            )
            if cfg.soft_thought_noise_cond:
                self.soft_thought_noise_emb = ScaledSinusoidalEmbedding(
                    self.decoder.latent_dim
                )

            if cfg.soft_prompt_len > 0:
                self.null_soft_thought = nn.Parameter(
                    torch.zeros(1, cfg.soft_prompt_len, self.decoder.latent_dim)
                )
                nn.init.normal_(self.null_soft_thought, std=0.02)
        else:
            self.model = None

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
        # Make sure encoder is always in eval mode
        if self.encoder is not None and not self.cfg.finetune_encoder:
            self.encoder.eval()
        if self.cfg.use_ema:
            self.ema_model[0].eval()

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

    def _get_prior_alpha0(self, batch_size, device, dtype=None):
        if self.cfg.prior_logsnr_max is not None:
            alpha0 = torch.sigmoid(
                torch.tensor(self.cfg.prior_logsnr_max, device=device, dtype=torch.float32)
            )
            alpha0 = alpha0.expand(batch_size)
        else:
            t0 = torch.full(
                (batch_size,),
                self._get_effective_prior_t_min(),
                device=device,
                dtype=torch.float32,
            )
            alpha0 = self.train_schedule(t0)

        if dtype is not None:
            alpha0 = alpha0.to(dtype)

        return alpha0

    def _get_effective_prior_t_min(self):
        if not self.cfg.prior_enforce_t_min:
            return 0.0

        base_t_min = float(self.cfg.prior_t_min)
        if self.cfg.prior_logsnr_max is None:
            return base_t_min

        target_alpha = torch.sigmoid(torch.tensor(self.cfg.prior_logsnr_max)).item()
        lo, hi = 0.0, 1.0
        for _ in range(24):
            mid = 0.5 * (lo + hi)
            alpha_mid = self.train_schedule(torch.tensor([mid])).item()
            if alpha_mid >= target_alpha:
                lo = mid
            else:
                hi = mid

        return max(base_t_min, 0.5 * (lo + hi))

    def _get_current_diffusion_beta(self) -> float:
        base_beta = float(self.cfg.diffusion_beta)
        if (not self.training) or self.model is None:
            return base_beta

        warmup_steps = self.cfg.diffusion_beta_warmup_steps
        if warmup_steps is None:
            warmup_steps = self.cfg.lr_warmup_steps

        if warmup_steps is None or warmup_steps <= 0:
            return base_beta

        warmup_ratio = min(1.0, float(self.global_step) / float(warmup_steps))
        return base_beta * warmup_ratio

    def _encode_latent(self, batch):
        if self.cfg.finetune_encoder:
            z = self.encoder(
                batch["input_ids_enc"],
                attention_mask=batch["attention_mask_enc"],
                only_backbone=True,
            )
        else:
            with torch.no_grad():
                z = self.encoder(
                    batch["input_ids_enc"],
                    attention_mask=batch["attention_mask_enc"],
                    only_backbone=True,
                )

        if self.encoder_post_layernorm is not None:
            z = self.encoder_post_layernorm(z)

        if self.cfg.encoder_post_l2norm:
            z = torch.nn.functional.normalize(
                z,
                p=2,
                dim=-1,
                eps=self.cfg.encoder_post_l2norm_eps,
            )

        if self.cfg.normalize_latent:
            z = self.normalize_latent(z)

        if self.encoder_adapter is not None:
            z = z + self.encoder_adapter(z)

        return z

    def configure_optimizers(self):
        base_lr = self.cfg.lr
        encoder_lr = self.cfg.encoder_lr if self.cfg.encoder_lr is not None else base_lr
        no_decay = ["bias", "norm"]
        optimizer_grouped_parameters = []

        def add_param_groups(named_params, group_lr):
            named_params = [(n, p) for n, p in named_params if p.requires_grad]
            if len(named_params) == 0:
                return

            decay_params = [
                p
                for n, p in named_params
                if not any(nd in n.lower() for nd in no_decay)
            ]
            no_decay_params = [
                p for n, p in named_params if any(nd in n.lower() for nd in no_decay)
            ]

            if len(decay_params) > 0:
                optimizer_grouped_parameters.append(
                    {
                        "params": decay_params,
                        "weight_decay": 0.01,
                        "lr": group_lr,
                    }
                )
            if len(no_decay_params) > 0:
                optimizer_grouped_parameters.append(
                    {
                        "params": no_decay_params,
                        "weight_decay": 0.0,
                        "lr": group_lr,
                    }
                )

        if self.model is not None:
            add_param_groups(self.model.named_parameters(), base_lr)
        add_param_groups(self.decoder.named_parameters(), base_lr)

        if self.encoder_adapter is not None:
            add_param_groups(self.encoder_adapter.named_parameters(), base_lr)

        if self.cfg.finetune_encoder and self.encoder is not None:
            add_param_groups(self.encoder.named_parameters(), encoder_lr)

        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=base_lr)
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
            num_dlc_ph=self.cfg.soft_prompt_len,
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
            num_dlc_ph=self.cfg.soft_prompt_len,
        )

    @torch.no_grad()
    def on_fit_start(self):

        # Compute latent mean and scale if needed (on 10000 samples, per rank)
        if self.cfg.normalize_latent and (self.model is not None):
            if rank_zero_only.rank == 0:
                print("Computing latent mean and scale...")

                dl_iter = iter(self.train_dataloader())
                latent_samples = []
                for _ in tqdm(
                    range(
                        self.cfg.num_latent_for_precomputed_stats // self.cfg.batch_size
                    )
                ):
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

                latent_mean = latent_mean.cpu().float()
                latent_scale = latent_scale.cpu().float()

                del dl_iter
            else:
                latent_mean = (
                    torch.zeros(size=(self.cfg.model.latent_dim,)).cpu().float()
                )
                latent_scale = (
                    torch.zeros(size=(self.cfg.model.latent_dim,)).cpu().float()
                )

            if self.trainer.num_devices > 1:
                # We do it this way (compute stats on rank 0 only) to avoid all_gather memory issues
                latent_mean = self.trainer.strategy.broadcast(latent_mean, src=0)
                latent_scale = self.trainer.strategy.broadcast(latent_scale, src=0)

            self.latent_mean.copy_(latent_mean)
            self.latent_scale.copy_(latent_scale)

            self.latent_mean = self.latent_mean.to(self.device)
            self.latent_scale = self.latent_scale.to(self.device)

            print("Done!")

    def get_input_embeds(self, batch, z, alpha=None):
        """
        Given a batch and the corresponding diffusion latents z, compute the soft thoughts with self.soft_thought_gen
        and fill them in the placeholders of the input_ids (with the DLC tags)
        """
        # Embed the input_ids
        input_ids = batch["input_ids_dec"]
        input_embeds = self.decoder.backbone.get_input_embeddings()(input_ids)

        # Feature-level dropout directly on latent conditioning (before soft prompt projection).
        if self.training and self.cfg.latent_feature_dropout_prob > 0:
            p = self.cfg.latent_feature_dropout_prob
            if not (0.0 <= p <= 1.0):
                raise ValueError(f"latent_feature_dropout_prob must be in [0, 1], got {p}")
            if z.ndim < 2:
                raise ValueError(f"Expected latent tensor with ndim >= 2, got shape {tuple(z.shape)}")
            mask_shape = (z.shape[0], *([1] * (z.ndim - 2)), z.shape[-1])
            feature_keep_mask = (torch.rand(mask_shape, device=z.device) >= p).to(z.dtype)
            z = z * feature_keep_mask

        # Compute noise embedding
        if alpha is not None and self.cfg.soft_thought_noise_cond:
            if alpha.ndim > 1:
                alpha = alpha.view(alpha.shape[0], -1)[:, 0]
            noise_input = rearrange(alpha * 1000, "b -> b 1")
            noise_embd = self.soft_thought_noise_emb(noise_input)
            if noise_embd.ndim == 2:
                noise_embd = rearrange(noise_embd, "b d -> b 1 d")
            elif noise_embd.ndim != 3:
                raise ValueError(
                    f"Unexpected noise embedding shape: {tuple(noise_embd.shape)}"
                )
        else:
            noise_embd = None

        # soft_thought_proj expects [B, D] latents.
        if z.ndim == 3 and z.shape[1] == 1:
            z = z[:, 0]
        elif z.ndim != 2:
            raise ValueError(
                f"Expected latent shape [B, D] or [B, 1, D] before soft thought projection, got {tuple(z.shape)}"
            )

        # Compute soft thoughts
        soft_thought = self.soft_thought_proj(z).to(input_embeds.dtype)
        soft_thought = self.soft_thought_enc(soft_thought, condition=noise_embd)

        # Classifier-free style dropout on latent conditioning to discourage brittle reliance.
        if self.training and self.cfg.latent_conditioning_dropout_prob > 0:
            p = self.cfg.latent_conditioning_dropout_prob
            if not (0.0 <= p <= 1.0):
                raise ValueError(
                    f"latent_conditioning_dropout_prob must be in [0, 1], got {p}"
                )
            dropout_mask = (
                torch.rand((soft_thought.shape[0],), device=soft_thought.device) < p
            )
            if dropout_mask.any():
                null_soft_thought = (
                    repeat(
                        self.null_soft_thought,
                        "1 l d -> b l d",
                        b=soft_thought.shape[0],
                    ).to(soft_thought.dtype)
                    if self.null_soft_thought is not None
                    else torch.zeros_like(soft_thought)
                )
                soft_thought = torch.where(
                    rearrange(dropout_mask, "b -> b 1 1"),
                    null_soft_thought,
                    soft_thought,
                )

        # Replace the DLC ph tokens with the corresponding soft thoughts
        dlc_ph_mask = batch["info_mask_dec"] == InfoLabel.DLC.value
        input_embeds[dlc_ph_mask] = soft_thought.view(-1, self.decoder.latent_dim)

        return input_embeds

    def compute_loss(self, batch) -> dict:
        """
        Compute decoder (suffix) loss and diffusion loss.

        Uses self.training to determine:
                    - Training: optionally applies latent_noise and uses self.model for diffusion loss
          - Validation: uses alpha=0.95, uses EMA model (if use_ema) for diffusion loss

        Returns dict with full_loss, suffix_loss, diffusion_loss (None if no diffusion model).
        """
        if self.encoder is not None:
            z = self._encode_latent(batch)

            if self.cfg.latent_noise and self.training:
                alpha_1d = self._get_prior_alpha0(
                    batch_size=z.size(0), device=z.device, dtype=z.dtype
                )
                alpha = right_pad_dims_to(z, alpha_1d)
                z_noised = alpha.sqrt() * z + (1 - alpha).sqrt() * torch.randn_like(z)
                input_embeds = self.get_input_embeds(batch, z_noised, alpha_1d)
            else:
                input_embeds = self.get_input_embeds(
                    batch,
                    z,
                    alpha=(
                        self._get_prior_alpha0(
                            batch_size=z.size(0), device=z.device, dtype=z.dtype
                        )
                        if self.cfg.soft_thought_noise_cond
                        else None
                    ),
                )

            logits, hidden_states = self.decoder(
                input_embeds=input_embeds, return_hidden_states=True
            )
        else:
            z = None
            hidden_states = None
            logits = self.decoder(input_ids=batch["input_ids_dec"])

        # Compute decoder loss
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

        # Diffusion loss
        diffusion_loss = None
        if self.model is not None:
            diffusion_beta = self._get_current_diffusion_beta()
            diffusion_model = self.model if self.training else (
                self.ema_model[0].module if self.cfg.use_ema else self.model
            )
            diffusion_loss = compute_diffusion_loss(
                diffusion_model,
                z,
                cond=hidden_states[:, : self.cfg.prefix_length],
                schedule=self.train_schedule,
                diffusion_objective=self.cfg.diffusion_objective,
                loss_name=self.cfg.loss,
                t_min=self._get_effective_prior_t_min(),
            )
            full_loss = (
                self.cfg.suffix_loss_weight * suffix_loss
                + diffusion_beta * diffusion_loss
            )
        else:
            full_loss = self.cfg.suffix_loss_weight * suffix_loss

        return {
            "full_loss": full_loss,
            "suffix_loss": suffix_loss,
            "diffusion_loss": diffusion_loss,
            "z": z,
            "hidden_states": hidden_states,
        }

    @torch.no_grad()
    def generate_suffix(self, batch, latent_generation_mode: str = "sample") -> List[str]:
        """
        Generate suffixes for the given batch.

        Args:
            latent_generation_mode: How to generate the diffusion latent z:
                - "sample": z ~ diffusion model conditioned on prefix hidden states (requires encoder + model)
                - "true": z = encoder(suffix), no diffusion (requires encoder)
                - "random": z ~ N(0,1) prior (requires encoder for soft thought projection)
        """
        if self.model is None or self.encoder is None:
            # No diffusion model or encoder — fall back to AR generation from prefix token ids
            return self.decoder.tokenizer.batch_decode(
                self.decoder.generate(
                    prefix=batch["input_ids_dec"][:, : self.cfg.prefix_length],
                    max_length=self.cfg.suffix_length,
                ),
                skip_special_tokens=True,
            )

        val_model = self.ema_model[0].module if self.cfg.use_ema else self.model

        # Compute true z and hidden_states (needed for "sample" conditioning and "true" mode)
        z_true = self._encode_latent(batch)

        input_embeds_true = self.get_input_embeds(
            batch,
            z_true,
            alpha=(
                self._get_prior_alpha0(
                    batch_size=z_true.size(0), device=z_true.device, dtype=z_true.dtype
                )
                if self.cfg.soft_thought_noise_cond
                else None
            ),
        )
        _, hidden_states = self.decoder(
            input_embeds=input_embeds_true, return_hidden_states=True
        )

        if latent_generation_mode == "sample":
            z = sample(
                val_model,
                schedule=self.sampling_schedule,
                batch_size=batch["input_ids_dec"].shape[0],
                sampling_timesteps=self.cfg.sampling_timesteps,
                sampler=self.cfg.sampler,
                diffusion_objective=self.cfg.diffusion_objective,
                cond=hidden_states[:, : self.cfg.prefix_length],
            )

        elif latent_generation_mode == "true":
            z = z_true

        elif latent_generation_mode == "random":
            z = torch.randn_like(z_true)

        else:
            raise ValueError(f"Unknown latent_generation_mode: {latent_generation_mode}")

        input_embeds = self.get_input_embeds(
            batch,
            z,
            alpha=(
                self._get_prior_alpha0(
                    batch_size=z.size(0), device=z.device, dtype=z.dtype
                )
                if self.cfg.soft_thought_noise_cond
                else None
            ),
        )
        return self.decoder.tokenizer.batch_decode(
            self.decoder.generate(
                prefix_embeds=input_embeds[:, : -self.cfg.suffix_length],
                max_length=self.cfg.suffix_length,
            ),
            skip_special_tokens=True,
        )

    def training_step(self, batch, batch_idx=None):

        losses = self.compute_loss(batch)

        self.log(
            "train/suffix_loss",
            losses["suffix_loss"],
            on_epoch=False,
            on_step=True,
            sync_dist=True,
        )

        if losses["diffusion_loss"] is not None:
            self.log(
                "train/diffusion_beta",
                self._get_current_diffusion_beta(),
                on_epoch=False,
                on_step=True,
                sync_dist=True,
            )
            self.log(
                "train/diffusion_loss",
                losses["diffusion_loss"],
                on_epoch=False,
                on_step=True,
                sync_dist=True,
            )

        self.log(
            "train/loss",
            losses["full_loss"],
            prog_bar=True,
            add_dataloader_idx=False,
            batch_size=batch["input_ids_dec"].shape[0],
            on_step=True,
            on_epoch=False,
            sync_dist=True,
        )

        return losses["full_loss"]

    @torch.no_grad()
    def validation_step(self, batch, batch_idx=None):

        losses = self.compute_loss(batch)

        self.log(
            "val/suffix_loss",
            losses["suffix_loss"],
            on_epoch=True,
            on_step=False,
            sync_dist=True,
        )

        if losses["diffusion_loss"] is not None:
            self.log(
                "val/diffusion_loss",
                losses["diffusion_loss"],
                on_epoch=True,
                on_step=False,
                sync_dist=True,
            )

        self.log(
            "val/loss",
            losses["full_loss"],
            add_dataloader_idx=False,
            batch_size=batch["input_ids_dec"].shape[0],
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )

        # Generate samples and compute metrics (PPL and/or MAUVE)
        if self.cfg.eval_gen_ppl or self.cfg.eval_mauve:

            suffix_sample = self.generate_suffix(batch, "sample")

            if self.cfg.eval_mauve:
                for prefix_str, suffix_str in zip(batch["prefix_str"], suffix_sample):
                    self.val_generated_texts_sample.append(prefix_str + suffix_str)

            if self.cfg.eval_gen_ppl:
                self.log(
                    "val/gen_ppl_sample",
                    eval_ppl(self.ppl_model[0], self.ppl_tok, batch["prefix_str"], suffix_sample, batch["input_ids_dec"].device),
                    on_epoch=True,
                    on_step=False,
                    sync_dist=True,
                )

            if self.model is not None:
                suffix_true = self.generate_suffix(batch, "true")

                if self.cfg.eval_mauve:
                    for prefix_str, suffix_str in zip(batch["prefix_str"], suffix_true):
                        self.val_generated_texts_true.append(prefix_str + suffix_str)

                if self.cfg.eval_gen_ppl:
                    self.log(
                        "val/gen_ppl_true",
                        eval_ppl(self.ppl_model[0], self.ppl_tok, batch["prefix_str"], suffix_true, batch["input_ids_dec"].device),
                        on_epoch=True,
                        on_step=False,
                        sync_dist=True,
                    )

                suffix_random = self.generate_suffix(batch, "random")

                if self.cfg.eval_mauve:
                    for prefix_str, suffix_str in zip(batch["prefix_str"], suffix_random):
                        self.val_generated_texts_random.append(prefix_str + suffix_str)

                if self.cfg.eval_gen_ppl:
                    self.log(
                        "val/gen_ppl_random",
                        eval_ppl(self.ppl_model[0], self.ppl_tok, batch["prefix_str"], suffix_random, batch["input_ids_dec"].device),
                        on_epoch=True,
                        on_step=False,
                        sync_dist=True,
                    )

                if (rank_zero_only.rank == 0) and (wandb.run is not None) and (batch_idx == 0):
                    table = wandb.Table(
                        columns=["Prefix", "True Suffix", "Gen. Suffix (true z)", "Gen. Suffix (sampled z)"]
                    )
                    for i in range(5):
                        table.add_data(
                            batch["prefix_str"][i],
                            batch["suffix_str"][i],
                            suffix_true[i],
                            suffix_sample[i]
                        )
                    wandb.log({"val/samples": table})
                    del table

    def on_train_batch_end(self, outputs, batch, batch_idx):
        # Update EMA model every step
        if (
            (batch_idx % self.trainer.accumulate_grad_batches == 0)
            and self.cfg.use_ema
            and (rank_zero_only.rank == 0)
        ):
            self.ema_model[0].update_parameters(self.model)

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
        if self.cfg.use_ema:

            # Broadcast EMA model from rank 0 to all ranks (on CPU)
            if self.trainer.num_devices > 1:
                for param in self.ema_model[0].parameters():
                    self.trainer.strategy.broadcast(param.data, src=0)

                for buffer in self.ema_model[0].buffers():
                    self.trainer.strategy.broadcast(buffer.data, src=0)

            # Move EMA model to GPU for validation
            self.ema_model[0] = self.ema_model[0].to(self.device)

        self.ppl_model[0] = self.ppl_model[0].to(self.device)

        # Reset accumulated texts for MAUVE computation
        if self.cfg.eval_mauve:
            self.val_generated_texts_sample = []
            self.val_generated_texts_true = []
            self.val_generated_texts_random = []

        torch.cuda.empty_cache()

    def on_validation_epoch_end(self):
        if self.cfg.eval_mauve:
            self._compute_and_log_mauve(self.val_generated_texts_sample, "sample")
            if self.model is not None:
                self._compute_and_log_mauve(self.val_generated_texts_true, "true")
                self._compute_and_log_mauve(self.val_generated_texts_random, "random")

        # Move EMA model back to CPU after validation to save GPU memory
        if self.cfg.use_ema:
            self.ema_model[0] = self.ema_model[0].cpu()
        self.ppl_model[0] = self.ppl_model[0].cpu()
        torch.cuda.empty_cache()
