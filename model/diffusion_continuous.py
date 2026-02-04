"""Diffusion Transformer (DiT) and Gaussian Diffusion utilities to train and sample."""

import math
from abc import ABC, abstractmethod, abstractproperty
from collections import Counter, defaultdict, namedtuple
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import timedelta
from functools import partial
from multiprocessing import cpu_count
from pathlib import Path
import random
from typing import Callable, Iterable, Optional, Tuple, Union

import einops
import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange
from omegaconf import MISSING
from torch import einsum, nn
from torch.optim import AdamW
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers.modeling_outputs import BaseModelOutput
from transformers.models.bart.modeling_bart import BartForConditionalGeneration
from x_transformers.x_transformers import (
    AbsolutePositionalEmbedding,
    AttentionLayers,
    Encoder,
    ScaledSinusoidalEmbedding,
    init_zero_,
)
from torch.nn import Sequential


@dataclass
class DiTContinuousConfig:
    n_layers: int
    n_heads: int
    n_embd: int
    seq_len: int
    seq_conditional_dim: Optional[int] = None
    latent_dim: Optional[int] = None

    dropout: float = 0.0
    class_conditional: bool = False
    num_classes: int = 0
    cond_modulation: Optional[bool] = False
    seq_unconditional_prob: Optional[float] = 0.1
    class_unconditional_prob: Optional[float] = 0.1
    self_condition: Optional[bool] = False
    train_prob_self_cond: Optional[float] = 0.5


class DiTContinuous(nn.Module):
    """
    Diffusion transformer (DiT, https://arxiv.org/pdf/2212.09748) with adaptive layer norm zero (adaLN-Zero) conditionning.
    Super-charged with other tricks and add-ons (self-conditionning, sequence-conditioning, class-conditionning)
    """

    def __init__(self, cfg: DiTContinuousConfig):
        super().__init__()

        assert cfg.latent_dim is not None, "latent_dim must be specified"

        self.cfg = cfg

        # Sinusoidal time embedding
        self.time_mlp = ScaledSinusoidalEmbedding(self.cfg.n_embd)

        # Absolute positional embedding
        self.pos_emb = AbsolutePositionalEmbedding(self.cfg.n_embd, self.cfg.n_embd)

        # Actual DiT
        self.latent_encoder = AttentionLayers(
            causal=False,
            dim=self.cfg.n_embd,
            depth=cfg.n_layers,
            heads=cfg.n_heads,
            attn_dropout=cfg.dropout,
            ff_dropout=cfg.dropout,
            rel_pos_bias=False,
            ff_glu=True,
            ff_swish=True,
            cross_attend=True,
            # DiT adalnzero stuff
            use_adaptive_rmsnorm=True,
            use_adaptive_layerscale=True,
            dim_condition=self.cfg.n_embd,
            adaptive_condition_mlp_expansion=4,
            adaptive_condition_mlp=True,
        )

        # Class conditionning setup
        if cfg.class_conditional:
            assert cfg.num_classes > 0
            self.class_embedding = nn.Sequential(
                nn.Embedding(cfg.num_classes + 1, self.cfg.n_embd),
                nn.Linear(self.cfg.n_embd, self.cfg.n_embd),
            )
            self.class_unconditional_bernoulli = torch.distributions.Bernoulli(
                probs=cfg.class_unconditional_prob
            )

        # Sequence conditionning setup
        assert cfg.seq_conditional_dim != None
        self.null_embedding_cond = nn.Embedding(1, self.cfg.n_embd)
        self.cond_proj = nn.Linear(cfg.seq_conditional_dim, self.cfg.n_embd)

        if cfg.cond_modulation:
            self.adalnzero_null_embedding = nn.Embedding(1, self.cfg.n_embd)

        # Setup self-conditionning
        if cfg.self_condition:
            input_dim = cfg.latent_dim * 2
            self.init_self_cond = nn.Parameter(torch.randn(1, cfg.latent_dim))
            nn.init.normal_(self.init_self_cond, std=0.02)
        else:
            input_dim = cfg.latent_dim

        # Setup input/output projections
        # (input_dim -> [seq_len * pre_proj_dim] -> [seq_len * n_embd])
        assert cfg.latent_dim % 8 == 0, "Latent dimension must be divisible by 8"
        pre_proj_dim = 96
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, cfg.seq_len * pre_proj_dim, bias=False),
            Rearrange("b 1 (l d) -> b l d", l=cfg.seq_len, d=pre_proj_dim),
            nn.Linear(pre_proj_dim, cfg.n_embd, bias=False),
        )
        self.output_proj = nn.Sequential(
            nn.Linear(cfg.n_embd, pre_proj_dim, bias=False),
            Rearrange("b l d -> b 1 (l d)"),
            nn.Linear(cfg.seq_len * pre_proj_dim, input_dim, bias=False),
        )
        init_zero_(self.output_proj[2])

    def forward(
        self,
        x: torch.Tensor,
        time,
        cond,
        cond_mask=None,
        x_self_cond=None,
        class_id=None,
        cond_input_ids=None,
    ):

        # Process the latent (maybe concatenate self-conditionning; project to internal shape, add pos_emb)
        if self.cfg.self_condition:
            if x_self_cond != None:
                x = torch.cat((x, x_self_cond), dim=-1)
            else:
                repeated_x_self_cond = repeat(
                    self.init_self_cond, "1 d -> b l d", b=x.shape[0], l=x.shape[1]
                )
                x = torch.cat((x, repeated_x_self_cond), dim=-1)
        x = self.input_proj(x)
        x = x + self.pos_emb(x)

        # Process seq condtionning (use null if any, generate mask if any, project to internal dim)
        cond = (
            repeat(self.null_embedding_cond.weight, "1 d -> b 1 d", b=x.shape[0])
            if cond is None
            else self.cond_proj(cond)
        )
        cond_mask = (
            torch.ones(
                (cond.shape[0], cond.shape[1]),
                device=cond.device,
                dtype=torch.bool,
            )
            if cond_mask is None
            else cond_mask
        )

        # Build the AdaLN-Zero conditionning (time embedding + maybe seq conditionning)
        condition = rearrange(self.time_mlp(time[None] * 1000), "b d -> b 1 d")
        if self.cfg.cond_modulation:
            if cond is None:
                condition = condition + repeat(
                    self.adalnzero_null_embedding.weight,
                    "1 d -> b 1 d",
                    b=x.shape[0],
                )
            else:
                pooled_cond = torch.where(
                    repeat(cond_mask, "b l -> b l d", d=cond.shape[-1]),
                    cond,
                    0,
                )
                pooled_cond = pooled_cond / einops.repeat(
                    cond_mask.sum(1), "b -> b () ()"
                )
                pooled_cond = pooled_cond.sum(1, keepdims=True)
                condition = condition + pooled_cond

        # Pass through DiT
        x = self.latent_encoder(
            x, condition=condition, context=cond, context_mask=cond_mask
        )

        # Project back to latent dim
        x = self.output_proj(x)

        return x


################################################
########## Gaussian Diffusion utils ############
################################################

ModelPrediction = namedtuple(
    "ModelPrediction", ["pred_noise", "pred_x_start", "pred_v"]
)


def predict_start_from_noise(z_t, t, noise, schedule):
    alpha = schedule(t)
    alpha = right_pad_dims_to(z_t, alpha)

    return (z_t - (1 - alpha).sqrt() * noise) / alpha.sqrt().clamp(min=1e-8)


def predict_noise_from_start(z_t, t, x0, schedule):
    alpha = schedule(t)
    alpha = right_pad_dims_to(z_t, alpha)

    return (z_t - alpha.sqrt() * x0) / (1 - alpha).sqrt().clamp(min=1e-8)


def predict_start_from_v(z_t, t, v, schedule):
    alpha = schedule(t)
    alpha = right_pad_dims_to(z_t, alpha)

    x = alpha.sqrt() * z_t - (1 - alpha).sqrt() * v

    return x


def predict_noise_from_v(z_t, t, v, schedule):
    alpha = schedule(t)
    alpha = right_pad_dims_to(z_t, alpha)

    eps = (1 - alpha).sqrt() * z_t + alpha.sqrt() * v

    return eps


def predict_v_from_start_and_eps(z_t, t, x, noise, schedule):
    alpha = schedule(t)
    alpha = right_pad_dims_to(z_t, alpha)

    v = alpha.sqrt() * noise - x * (1 - alpha).sqrt()

    return v


def get_sampling_timesteps(
    batch, *, sampling_timesteps, device, dtype=None, invert=False
):
    times = torch.linspace(1.0, 0.0, sampling_timesteps + 1, device=device, dtype=dtype)
    if invert:
        times = times.flip(dims=(0,))
    times = repeat(times, "t -> b t", b=batch)
    times = torch.stack((times[:, :-1], times[:, 1:]), dim=0)
    times = times.unbind(dim=-1)
    return times


def diffusion_model_predictions(
    model: DiTContinuous,
    z_t,
    t,
    schedule,
    diffusion_objective,
    x_self_cond=None,
    class_id=None,
    cond=None,
    cond_input_ids=None,
    cond_mask=None,
    cls_free_guidance=1.0,
) -> ModelPrediction:
    time_cond = schedule(t)
    model_output = model(
        x=z_t,
        time=time_cond,
        x_self_cond=x_self_cond,
        class_id=class_id,
        cond=cond,
        cond_input_ids=cond_input_ids,
        cond_mask=cond_mask,
    )
    if cls_free_guidance != 1.0:
        if exists(class_id):
            unc_class_id = torch.full_like(class_id, fill_value=model.cfg.num_classes)
        else:
            unc_class_id = None
        unc_model_output = model(
            x=z_t,
            time=time_cond,
            x_self_cond=x_self_cond,
            class_id=unc_class_id,
            cond=None,
            cond_input_ids=None,
            cond_mask=None,
        )
        model_output = model_output * cls_free_guidance + unc_model_output * (
            1 - cls_free_guidance
        )

    pred_v = None
    if diffusion_objective == "pred_noise":
        pred_noise = model_output
        x_start = predict_start_from_noise(z_t, t, pred_noise, sampling=schedule)
    elif diffusion_objective == "pred_x0":
        x_start = model_output
        pred_noise = predict_noise_from_start(z_t, t, x_start, schedule)
        pred_v = predict_v_from_start_and_eps(z_t, t, x_start, pred_noise, schedule)
    elif diffusion_objective == "pred_v":
        pred_v = model_output
        x_start = predict_start_from_v(z_t, t, pred_v, schedule)
        pred_noise = predict_noise_from_v(z_t, t, pred_v, schedule)
    else:
        raise ValueError(f"invalid objective {diffusion_objective}")

    return ModelPrediction(pred_noise, x_start, pred_v)


@torch.no_grad()
def ddim_sample(
    model: DiTContinuous,
    shape,
    class_id,
    cond,
    cond_input_ids,
    cond_mask,
    schedule,
    sampling_timesteps,
    cls_free_guidance,
    diffusion_objective,
    invert=False,
    z_t=None,
):
    param = next(model.parameters())
    batch, device, dtype = shape[0], param.device, param.dtype

    time_pairs = get_sampling_timesteps(
        batch,
        sampling_timesteps=sampling_timesteps,
        device=device,
        invert=invert,
        dtype=dtype,
    )
    if invert:
        assert exists(z_t)

    if not exists(z_t):
        z_t = torch.randn(shape, device=device, dtype=dtype)

    x_start = None

    for time, time_next in time_pairs:
        # get predicted x0

        model_output = diffusion_model_predictions(
            model,
            z_t,
            time,
            class_id=class_id,
            x_self_cond=x_start,
            cond=cond,
            cond_input_ids=cond_input_ids,
            cond_mask=cond_mask,
            schedule=schedule,
            cls_free_guidance=cls_free_guidance,
            diffusion_objective=diffusion_objective,
        )
        # get alpha sigma of time and next time

        alpha = schedule(time)
        alpha_next = schedule(time_next)
        alpha, alpha_next = map(partial(right_pad_dims_to, z_t), (alpha, alpha_next))

        # # calculate x0 and noise

        x_start = model_output.pred_x_start

        eps = model_output.pred_noise

        if (not invert) and time_next[0] <= 0:
            z_t = x_start
            continue
        if invert and time_next[0] >= 1:
            z_t = eps
            continue

        # get noise

        z_t = x_start * alpha_next.sqrt() + eps * (1 - alpha_next).sqrt()
    return z_t


@torch.no_grad()
def ddpm_sample(
    model: DiTContinuous,
    shape,
    class_id,
    cond,
    cond_input_ids,
    cond_mask,
    schedule,
    sampling_timesteps,
    diffusion_objective,
    cls_free_guidance,
    invert=False,
    z_t=None,
):
    param = next(model.parameters())
    batch, device, dtype = shape[0], param.device, param.dtype

    time_pairs = get_sampling_timesteps(
        batch, sampling_timesteps=sampling_timesteps, device=device, dtype=dtype
    )

    if not exists(z_t):
        z_t = torch.randn(shape, device=device, dtype=dtype)

    x_start = None

    for time, time_next in time_pairs:
        # get predicted x0

        model_output = diffusion_model_predictions(
            model,
            z_t,
            time,
            class_id=class_id,
            x_self_cond=x_start,
            cond=cond,
            cond_input_ids=cond_input_ids,
            cond_mask=cond_mask,
            schedule=schedule,
            cls_free_guidance=cls_free_guidance,
            diffusion_objective=diffusion_objective,
        )
        # get alpha sigma of time and next time

        alpha = schedule(time)
        alpha_next = schedule(time_next)
        alpha, alpha_next = map(partial(right_pad_dims_to, z_t), (alpha, alpha_next))

        alpha_now = alpha / alpha_next

        # # calculate x0 and noise

        x_start = model_output.pred_x_start

        eps = model_output.pred_noise

        if time_next[0] <= 0:
            z_t = x_start
            continue

        # get noise

        noise = torch.randn_like(z_t)

        z_t = (
            1 / alpha_now.sqrt() * (z_t - (1 - alpha_now) / (1 - alpha).sqrt() * eps)
            + torch.sqrt(1 - alpha_now) * noise
        )
    return z_t


@torch.no_grad()
def dpmpp_sample(
    model: DiTContinuous,
    shape,
    class_id,
    cond,
    cond_input_ids,
    cond_mask,
    schedule,
    sampling_timesteps,
    cls_free_guidance,
    diffusion_objective,
    invert=False,
    z_t=None,
):
    param = next(model.parameters())
    batch, device, dtype = shape[0], param.device, param.dtype

    time_pairs = get_sampling_timesteps(
        batch, sampling_timesteps=sampling_timesteps, device=device, dtype=dtype
    )

    if not exists(z_t):
        z_t = torch.randn(shape, device=device, dtype=dtype)

    x_start = None
    old_pred_x = []
    old_hs = []

    for time, time_next in time_pairs:
        # get predicted x0

        model_output = diffusion_model_predictions(
            model,
            z_t,
            time,
            class_id=class_id,
            x_self_cond=x_start,
            cond=cond,
            cond_input_ids=cond_input_ids,
            cond_mask=cond_mask,
            schedule=schedule,
            cls_free_guidance=cls_free_guidance,
            diffusion_objective=diffusion_objective,
        )
        # get alpha sigma of time and next time

        alpha = schedule(time)
        alpha_next = schedule(time_next)
        alpha, alpha_next = map(partial(right_pad_dims_to, z_t), (alpha, alpha_next))
        sigma, sigma_next = 1 - alpha, 1 - alpha_next

        alpha_now = alpha / alpha_next

        lambda_now = (log(alpha) - log(1 - alpha)) / 2
        lambda_next = (log(alpha_next) - log(1 - alpha_next)) / 2
        h = lambda_next - lambda_now

        # calculate x0 and noise
        if time_next[0] <= 0:
            z_t = x_start
            continue

        x_start = model_output.pred_x_start

        phi_1 = torch.expm1(-h)
        if len(old_pred_x) < 2:
            denoised_x = x_start
        else:
            h = lambda_next - lambda_now
            h_0 = old_hs[-1]
            r0 = h_0 / h
            gamma = -1 / (2 * r0)
            denoised_x = (1 - gamma) * x_start + gamma * old_pred_x[-1]

        z_t = (
            sigma_next.sqrt() / sigma.sqrt()
        ) * z_t - alpha_next.sqrt() * phi_1 * denoised_x
    return z_t


@torch.no_grad()
def sample(
    model: DiTContinuous,
    schedule,
    batch_size,
    sampling_timesteps,
    sampler,
    diffusion_objective,
    cls_free_guidance=1.0,
    class_id=None,
    cond=None,
    cond_input_ids=None,
    cond_mask=None,
):

    if sampler == "ddim":
        sample_fn = ddim_sample
    elif sampler == "ddpm":
        sample_fn = ddpm_sample
    elif sampler == "dpmpp":
        sample_fn = dpmpp_sample
    else:
        raise ValueError(f"invalid sampler {sampler}")
    return sample_fn(
        model=model,
        shape=(batch_size, 1, model.cfg.latent_dim),
        class_id=class_id,
        cond=cond,
        cond_input_ids=cond_input_ids,
        cond_mask=cond_mask,
        schedule=schedule,
        sampling_timesteps=sampling_timesteps,
        cls_free_guidance=cls_free_guidance,
        diffusion_objective=diffusion_objective,
    )


def right_pad_dims_to(x, t):
    padding_dims = x.ndim - t.ndim
    if padding_dims <= 0:
        return t
    return t.view(*t.shape, *((1,) * padding_dims))


def simple_linear_schedule(t, clip_min=1e-9):
    return (1 - t).clamp(min=clip_min)


def beta_linear_schedule(t, clip_min=1e-9):
    return torch.exp(-1e-4 - 10 * (t**2)).clamp(min=clip_min, max=1.0)


def cosine_schedule(t, start=0, end=1, tau=1, clip_min=1e-9):
    power = 2 * tau
    v_start = math.cos(start * math.pi / 2) ** power
    v_end = math.cos(end * math.pi / 2) ** power
    output = torch.cos((t * (end - start) + start) * math.pi / 2) ** power
    output = (v_end - output) / (v_end - v_start)
    return output.clamp(min=clip_min)


def sigmoid_schedule(t, start=-3, end=3, tau=1, clamp_min=1e-9):
    v_start = torch.tensor(start / tau).sigmoid()
    v_end = torch.tensor(end / tau).sigmoid()
    gamma = (-((t * (end - start) + start) / tau).sigmoid() + v_end) / (v_end - v_start)
    return gamma.clamp_(min=clamp_min, max=1.0)


def log_snr_to_alpha(log_snr):
    alpha = torch.sigmoid(log_snr)
    return alpha


def alpha_to_shifted_log_snr(alpha, scale=1):
    return log((alpha / (1 - alpha))).clamp(min=-15, max=15) + 2 * np.log(scale).item()


def time_to_alpha(t, alpha_schedule, scale):
    alpha = alpha_schedule(t)
    shifted_log_snr = alpha_to_shifted_log_snr(alpha, scale=scale)
    return log_snr_to_alpha(shifted_log_snr)


def log(t, eps=1e-12):
    return torch.log(t.clamp(min=eps))


def exists(x):
    return x is not None


def get_sampling_schedule(name):
    if name is None:
        return None
    elif name == "simple_linear":
        return simple_linear_schedule
    elif name == "beta_linear":
        return beta_linear_schedule
    elif name == "cosine":
        return cosine_schedule
    elif name == "sigmoid":
        return sigmoid_schedule
    else:
        raise ValueError(f"invalid noise schedule {name}")


def loss_fn(name):
    if name == "l1":
        return F.l1_loss
    elif name == "l2":
        return F.mse_loss
    elif name == "smooth_l1":
        return F.smooth_l1_loss
    else:
        raise ValueError(f"invalid loss type {name}")


def compute_diffusion_loss(
    model: DiTContinuous,
    latent,
    schedule,
    diffusion_objective,
    loss_name,
    cond,
    cond_mask=None,
    class_id=None,
    cond_input_ids=None,
):
    bs = latent.shape[0]
    device = latent.device
    
    # Make sure latent has shape (B, 1, D)
    if latent.ndim == 2:
        latent = latent.unsqueeze(1)
    else:
        assert latent.ndim == 3 and latent.shape[1] == 1, "Latent must have shape (B, 1, D)"

    times = torch.zeros((bs,), device=device).float().uniform_(0, 1.0)
    noise = torch.randn_like(latent)

    alpha = schedule(times)
    alpha = right_pad_dims_to(latent, alpha)

    z_t = alpha.sqrt() * latent + (1 - alpha).sqrt() * noise

    # Sample unconditionally with some probability
    if random.random() < model.cfg.seq_unconditional_prob:
        cond = None
        cond_input_ids = None
        cond_mask = None

    if model.cfg.class_conditional and model.cfg.class_unconditional_prob > 0:
        assert exists(class_id)
        class_unconditional_mask = model.class_unconditional_bernoulli.sample(
            class_id.shape
        ).bool()
        class_id[class_unconditional_mask] = model.cfg.num_classes

    self_cond = None

    if model.cfg.self_condition and (random.random() < model.cfg.train_prob_self_cond):
        with torch.no_grad():
            model_output = diffusion_model_predictions(
                model,
                z_t,
                times,
                schedule,
                diffusion_objective,
                class_id=class_id,
                cond=cond,
                cond_mask=cond_mask,
            )
            self_cond = model_output.pred_x_start.detach()

    # predict and take gradient step

    predictions = diffusion_model_predictions(
        model,
        z_t,
        times,
        schedule,
        diffusion_objective,
        x_self_cond=self_cond,
        class_id=class_id,
        cond=cond,
        cond_input_ids=cond_input_ids,
        cond_mask=cond_mask,
    )

    if diffusion_objective == "pred_x0":
        target = latent
        pred = predictions.pred_x_start
    elif diffusion_objective == "pred_noise":
        target = noise
        pred = predictions.pred_noise
    elif diffusion_objective == "pred_v":
        target = alpha.sqrt() * noise - (1 - alpha).sqrt() * latent
        assert exists(predictions.pred_v)
        pred = predictions.pred_v

    loss = loss_fn(loss_name)(pred, target, reduction="none")
    loss = loss.mean(dim=-1).mean()  # first average over latent length

    return loss
