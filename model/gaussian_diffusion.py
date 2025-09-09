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
from PIL import Image
from torch import einsum, nn
from torch.optim import AdamW
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers.modeling_outputs import BaseModelOutput
from transformers.models.bart.modeling_bart import BartForConditionalGeneration
from x_transformers.x_transformers import (AbsolutePositionalEmbedding,
                                           Encoder, ScaledSinusoidalEmbedding,
                                           init_zero_)

@dataclass
class DiTConfig:
    n_layers: int
    n_heads: int
    dropout: float
    scale_shift: bool
    cond_encoder_kwargs: Optional[dict]
    latent_shape: Optional[Tuple[int]] = None
    n_embd: Optional[int] = None
    seq_conditional: Optional[bool] = False
    seq_conditional_dim: Optional[int] = None
    class_conditional: Optional[bool] = False
    num_classes: Optional[int] = 0
    cond_modulation: Optional[bool] = False

    # DDPM features
    seq_unconditional_prob: Optional[float] = 0.1
    class_unconditional_prob: Optional[float] = 0.1
    self_condition: Optional[bool] = False
    train_prob_self_cond: Optional[float] = 0.5


class DiT(nn.Module):
    """
    Diffusion transformer (DiT, https://arxiv.org/pdf/2212.09748) with adaptive layer norm zero (adaLN-Zero) conditionning.
    Super-charged with other tricks and add-ons (self-conditionning, sequence-conditioning, class-conditionning)
    Can be the backbone of a DSM or GFN diffusion model.
    """

    def __init__(self, cfg: DiTConfig):
        super().__init__()

        self.cfg = cfg

        assert isinstance(cfg.latent_shape, Iterable) and (len(cfg.latent_shape) == 2)
        if self.cfg.n_embd == None:
            self.cfg.n_embd = cfg.latent_shape[1]

        # Init model
        sinu_pos_emb = ScaledSinusoidalEmbedding(self.cfg.n_embd)
        fourier_dim = self.cfg.n_embd

        time_emb_dim = self.cfg.n_embd * 4
        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_emb_dim),
            nn.GELU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )
        self.time_pos_embed_mlp = nn.Sequential(
            nn.GELU(), nn.Linear(time_emb_dim, self.cfg.n_embd)
        )

        self.pos_emb = AbsolutePositionalEmbedding(self.cfg.n_embd, self.cfg.n_embd)

        self.latent_encoder = Encoder(
            dim=self.cfg.n_embd,
            depth=cfg.n_layers,
            heads=cfg.n_heads,
            attn_dropout=cfg.dropout,
            ff_dropout=cfg.dropout,
            rel_pos_bias=False,
            ff_glu=True,
            cross_attend=cfg.seq_conditional,
            # DiT scale-shift stuff
            use_adaptive_layernorm=cfg.scale_shift,
            use_adaptive_layerscale=cfg.scale_shift,
            dim_condition=time_emb_dim,
            adaptive_condition_mlp=cfg.scale_shift,
        )

        if cfg.class_conditional:
            assert (
                False
            ), "Careful, never tested the class conditional setting for real."
            assert cfg.num_classes > 0
            self.class_embedding = nn.Sequential(
                nn.Embedding(cfg.num_classes + 1, self.cfg.n_embd),
                nn.Linear(self.cfg.n_embd, time_emb_dim),
            )
            self.class_unconditional_bernoulli = torch.distributions.Bernoulli(
                probs=cfg.class_unconditional_prob
            )
        if cfg.seq_conditional:
            assert cfg.seq_conditional_dim != None
            self.null_embedding_cond = nn.Embedding(1, self.cfg.n_embd)
            self.cond_proj = nn.Linear(cfg.seq_conditional_dim, self.cfg.n_embd)

        if cfg.self_condition:
            self.input_proj = nn.Linear(cfg.latent_shape[1] * 2, self.cfg.n_embd)
            self.init_self_cond = nn.Parameter(torch.randn(1, cfg.latent_shape[1]))
            nn.init.normal_(self.init_self_cond, std=0.02)
        else:
            self.input_proj = nn.Linear(cfg.latent_shape[1], self.cfg.n_embd)

        self.norm = nn.LayerNorm(self.cfg.n_embd)
        self.output_proj = nn.Linear(
            self.cfg.n_embd,
            cfg.latent_shape[1],
        )

        if cfg.cond_encoder_kwargs != None:
            self.cond_encoder = Encoder(
                dim=cfg.seq_conditional_dim,
                depth=cfg.cond_encoder_kwargs["n_layers"],
                heads=cfg.cond_encoder_kwargs["n_heads"],
            )

        if cfg.cond_modulation:
            assert cfg.seq_conditional
            self.adalnzero_cond_proj = nn.Sequential(
                nn.Linear(cfg.seq_conditional_dim, time_emb_dim),
                nn.GELU(),
                nn.Linear(time_emb_dim, time_emb_dim),
            )
            self.adalnzero_null_embedding = nn.Embedding(1, time_emb_dim)

        init_zero_(self.output_proj)

    def forward(
        self,
        x: torch.Tensor,
        time,
        x_self_cond=None,
        class_id=None,
        cond=None,
        cond_input_ids=None,
        cond_mask=None,
    ):

        time_emb = self.time_mlp(time[None] * 1000)

        time_emb = rearrange(time_emb, "b d -> b 1 d")

        if self.cfg.class_conditional:
            assert class_id != None
            class_emb = self.class_embedding(class_id)
            class_emb = rearrange(class_emb, "b d -> b 1 d")
            time_emb = time_emb + class_emb

        pos_emb = self.pos_emb(x)

        if self.cfg.self_condition:
            if x_self_cond != None:
                x = torch.cat((x, x_self_cond), dim=-1)
            else:
                repeated_x_self_cond = repeat(
                    self.init_self_cond, "1 d -> b l d", b=x.shape[0], l=x.shape[1]
                )
                x = torch.cat((x, repeated_x_self_cond), dim=-1)

        x_input = self.input_proj(x)
        tx_input = x_input + pos_emb + self.time_pos_embed_mlp(time_emb)

        if self.cfg.seq_conditional:
            context, context_mask = [], []
            if (cond is None) & (cond_input_ids is None):
                # If the model is conditional but no conditionning is passed, give <null_embedding_cond>
                null_context = repeat(
                    self.null_embedding_cond.weight, "1 d -> b 1 d", b=x.shape[0]
                )
                context.append(null_context)
                context_mask.append(
                    torch.tensor(
                        [[True] for _ in range(x.shape[0])],
                        dtype=bool,
                        device=x.device,
                    )
                )

                if self.cfg.cond_modulation:
                    condition = time_emb + repeat(
                        self.adalnzero_null_embedding.weight,
                        "1 d -> b 1 d",
                        b=x.shape[0],
                    )
                else:
                    condition = time_emb

            else:
                # Maybe process the conditionning tokens 
                if self.cfg.cond_encoder_kwargs != None:
                    cond = self.cond_encoder(cond, mask=cond_mask)
                   
                context.append(self.cond_proj(cond))
                context_mask.append(cond_mask)

                # If conditionning the model on <cond> through adalnzero
                if self.cfg.cond_modulation:
                    pooled_cond = torch.where(
                        repeat(cond_mask, "b l -> b l d", d=cond.shape[-1]),
                        cond,
                        0,
                    )
                    pooled_cond = pooled_cond / einops.repeat(
                        cond_mask.sum(1), "b -> b () ()"
                    )
                    pooled_cond = pooled_cond.sum(1, keepdims=True)
                    condition = time_emb + self.adalnzero_cond_proj(pooled_cond)
                else:
                    condition = time_emb

            context = torch.cat(context, dim=1)
            context_mask = torch.cat(context_mask, dim=1)

            x = self.latent_encoder(
                tx_input,
                context=context,
                context_mask=context_mask,
                condition=condition,
            )
        else:
            x = self.latent_encoder(tx_input, condition=time_emb)

        x = self.norm(x)
        x = self.output_proj(x)

        return x


ModelPrediction = namedtuple("ModelPrediction", ["pred_noise", "pred_x_start", "pred_v"])

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

def get_sampling_timesteps(batch, *, sampling_timesteps, device, invert=False):
    times = torch.linspace(1.0, 0.0, sampling_timesteps + 1, device=device)
    if invert:
        times = times.flip(dims=(0,))
    times = repeat(times, "t -> b t", b=batch)
    times = torch.stack((times[:, :-1], times[:, 1:]), dim=0)
    times = times.unbind(dim=-1)
    return times

def diffusion_model_predictions(
    model : DiT,
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
        z_t,
        time_cond,
        x_self_cond,
        class_id=class_id,
        cond=cond,
        cond_input_ids=cond_input_ids,
        cond_mask=cond_mask,
    )
    if cls_free_guidance != 1.0:
        if exists(class_id):
            unc_class_id = torch.full_like(
                class_id, fill_value=model.cfg.num_classes
            )
        else:
            unc_class_id = None
        unc_model_output = model(
            z_t,
            time_cond,
            x_self_cond,
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
        x_start = predict_start_from_noise(
            z_t, t, pred_noise, sampling=schedule
        )
    elif diffusion_objective == "pred_x0":
        x_start = model_output
        pred_noise = predict_noise_from_start(
            z_t, t, x_start, schedule
        )
        pred_v = predict_v_from_start_and_eps(
            z_t, t, x_start, pred_noise, schedule
        )
    elif diffusion_objective == "pred_v":
        pred_v = model_output
        x_start = predict_start_from_v(z_t, t, pred_v, schedule)
        pred_noise = predict_noise_from_v(z_t, t, pred_v, schedule)
    else:
        raise ValueError(f"invalid objective {diffusion_objective}")

    return ModelPrediction(pred_noise, x_start, pred_v)

@torch.no_grad()
def ddim_sample(
    model: DiT,
    shape,
    class_id,
    cond,
    cond_input_ids,
    cond_mask,
    schedule,
    sampling_timesteps,
    cls_free_guidance,
    invert=False,
    z_t=None,
):
    batch, device = shape[0], next(model.parameters()).device

    time_pairs = get_sampling_timesteps(batch, sampling_timesteps=sampling_timesteps, device=device, invert=invert)
    if invert:
        assert exists(z_t)

    if not exists(z_t):
        z_t = torch.randn(shape, device=device)

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
        )
        # get alpha sigma of time and next time

        alpha = schedule(time)
        alpha_next = schedule(time_next)
        alpha, alpha_next = map(
            partial(right_pad_dims_to, z_t), (alpha, alpha_next)
        )

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
    model: DiT,
    shape,
    class_id,
    cond,
    cond_input_ids,
    cond_mask,
    schedule,
    sampling_timesteps,
    cls_free_guidance,
    invert=False,
    z_t=None,
):
    batch, device = shape[0], next(model.parameters()).device

    time_pairs = get_sampling_timesteps(batch, sampling_timesteps=sampling_timesteps, device=device)

    if not exists(z_t):
        z_t = torch.randn(shape, device=device)

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
        )
        # get alpha sigma of time and next time

        alpha = schedule(time)
        alpha_next = schedule(time_next)
        alpha, alpha_next = map(
            partial(right_pad_dims_to, z_t), (alpha, alpha_next)
        )

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
            1
            / alpha_now.sqrt()
            * (z_t - (1 - alpha_now) / (1 - alpha).sqrt() * eps)
            + torch.sqrt(1 - alpha_now) * noise
        )
    return z_t

@torch.no_grad()
def dpmpp_sample(
    model: DiT,
    shape,
    class_id,
    cond,
    cond_input_ids,
    cond_mask,
    schedule,
    sampling_timesteps,
    cls_free_guidance,
    invert=False,
    z_t=None,
):
    batch, device = shape[0], next(model.parameters()).device

    time_pairs = get_sampling_timesteps(batch, sampling_timesteps=sampling_timesteps, device=device)

    if not exists(z_t):
        z_t = torch.randn(shape, device=device)

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
        )
        # get alpha sigma of time and next time

        alpha = schedule(time)
        alpha_next = schedule(time_next)
        alpha, alpha_next = map(
            partial(right_pad_dims_to, z_t), (alpha, alpha_next)
        )
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
    model : DiT,
    schedule,
    batch_size,
    sampling_timesteps,
    sampler,
    cls_free_guidance,
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
        (batch_size,) + tuple(model.cfg.latent_shape),
        model,
        class_id,
        cond,
        cond_input_ids,
        cond_mask,
        schedule,
        sampling_timesteps,
        cls_free_guidance
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
    elif name  == "beta_linear":
        return beta_linear_schedule
    elif name == "cosine":
        return cosine_schedule
    elif name == "sigmoid":
        return sigmoid_schedule
    else:
        raise ValueError(f"invalid noise schedule {name}")

def compute_diffusion_loss(
    model: DiT,
    latent,
    schedule,
    diffusion_objective,
    loss_fn,
    class_id=None,
    cond=None,
    cond_input_ids=None,
    cond_ignore_mask=None,
):
    # NOTE: Important to flip the <ignore_mask> to a <don't_ignore_mask>
    cond_mask = None
    if cond_ignore_mask != None:
        cond_mask = torch.logical_not(cond_ignore_mask)

    bs, l, d = (*latent.shape,)
    device = latent.device

    times = torch.zeros((bs,), device=device).float().uniform_(0, 1.0)
    noise = torch.randn_like(latent)

    alpha = schedule(times)
    alpha = right_pad_dims_to(latent, alpha)

    z_t = alpha.sqrt() * latent + (1 - alpha).sqrt() * noise

    # Sample unconditionally with some probability
    if model.cfg.seq_conditional and (
        random.random() < model.cfg.seq_unconditional_prob
    ):
        cond = None
        cond_input_ids = None
        cond_mask = None

    if (
        model.cfg.class_conditional
        and model.cfg.class_unconditional_prob > 0
    ):
        assert exists(class_id)
        class_unconditional_mask = model.class_unconditional_bernoulli.sample(
            class_id.shape
        ).bool()
        class_id[class_unconditional_mask] = model.cfg.num_classes

    self_cond = None

    if model.cfg.self_condition and (
        random.random() < model.cfg.train_prob_self_cond
    ):
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

    loss = loss_fn(pred, target, reduction="none")
    loss = rearrange(
        [reduce(loss[i], "l d -> 1", "mean") for i in range(latent.shape[0])],
        "b 1 -> b 1",
    )

    return loss.mean()