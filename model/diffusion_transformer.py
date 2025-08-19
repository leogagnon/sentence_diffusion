import argparse
import copy
import csv
import json
import math
import os
import random
import timeit
from abc import ABC, abstractmethod, abstractproperty
from collections import Counter, defaultdict, namedtuple
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import timedelta
from functools import partial
from multiprocessing import cpu_count
from pathlib import Path
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
    Encoder,
    ScaledSinusoidalEmbedding,
    init_zero_,
)


@dataclass
class DiTConfig:
    n_layers: int
    n_heads: int
    dropout: float
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
    Super-charged with other tricks and add-ons (self-conditionning, sequence-conditioning, class-conditionning, langevin model)
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
            # Enables DiT adalnzero 
            use_adaptive_layernorm=True,
            use_adaptive_layerscale=True,
            adaptive_condition_mlp=True,
            dim_condition=time_emb_dim,
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
            assert cfg.seq_conditional

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
        cond=None,
        cond_mask=None,
    ):

        time_emb = self.time_mlp(time[None] * 1000)

        time_emb = rearrange(time_emb, "b d -> b 1 d")

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
            if cond is None:
                # If the model is conditional but no conditionning 
                # is passed, give <null_embedding_cond>
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
                    # If directly taking in <input_ids>
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
