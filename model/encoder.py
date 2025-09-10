from abc import ABC, abstractmethod
import os
from peft.mapping_func import get_peft_model
from peft.tuners.lora.config import LoraConfig
import torch.nn as nn
from dataclasses import dataclass
from typing import Optional
from sentence_transformers import SentenceTransformer
from transformers.models.auto.modeling_auto import AutoModelForCausalLM, AutoModel
from transformers.models.auto.tokenization_auto import AutoTokenizer
import torch
from x_transformers.x_transformers import AttentionLayers
import einx
import math
import random


@dataclass
class CompressorConfig:
    n_layers: int
    n_heads: int
    act_masking_p: float
    act_delta: float
    feat_masking_p: float


@dataclass
class EncoderConfig:
    name: str
    k: int
    variational: bool = False
    out_proj_dim: Optional[int] = None
    lora_cfg: Optional[dict] = None
    compressor_cfg: Optional[CompressorConfig] = None


class EncoderModel(nn.Module):
    def __init__(self, cfg: EncoderConfig):
        super().__init__()
        self.cfg = cfg

        # Init backbone
        self.backbone = AutoModel.from_pretrained(
            cfg.name,
            device_map="auto",
        )

        if self.cfg.lora_cfg != None:
            self.backbone = get_peft_model(
                self.backbone,
                LoraConfig(**self.cfg.lora_cfg),
            )
        else:
            self.backbone.requires_grad_(False)

        self.tokenizer = AutoTokenizer.from_pretrained(cfg.name)

        assert cfg.out_proj_dim != None
        if cfg.variational:
            self.out_proj = nn.Linear(self.backbone.config.hidden_size, cfg.out_proj_dim)
            self.out_proj_logvar = nn.Linear(self.backbone.config.hidden_size, cfg.out_proj_dim)
        else:
            self.out_proj = nn.Linear(self.backbone.config.hidden_size, cfg.out_proj_dim)

        if cfg.compressor_cfg is not None:
            self.compressor = AttentionLayers(
                dim=self.backbone.config.hidden_size,
                depth=cfg.compressor_cfg.n_layers,
                heads=cfg.compressor_cfg.n_heads,
                cross_attend=True,
                causal=False,
            )
            self.placeholders = nn.Parameter(
                torch.randn(cfg.k, self.backbone.config.hidden_size), requires_grad=True
            )

    @property
    def latent_shape(self):
        return (self.cfg.k, self.backbone.config.hidden_size)

    def forward(self, input_ids, attention_mask=None):

        # Run through backbone and get first k tokens
        z = self.backbone(input_ids, attention_mask=attention_mask)
        z = z["last_hidden_state"]

        if hasattr(self, "compressor"):
            if self.training:
                if random.random() < 0.5:
                    if self.cfg.compressor_cfg.act_masking_p > 0:
                        # Mask each token with probability act_masking_p
                        mask = (
                            torch.rand(size=(z.shape[0], z.shape[1]), device=z.device)
                            < self.cfg.compressor_cfg.act_masking_p
                        )
                        mask = einx.rearrange("b n -> b n 1", mask)
                        z = z.masked_fill(mask, 0.0)
                else:
                    if self.cfg.compressor_cfg.act_delta > 0:
                        noise = torch.randn(z.shape, device=z.device)
                        z = (
                            self.cfg.compressor_cfg.act_delta * z
                            + (1 - math.sqrt(self.cfg.compressor_cfg.act_delta)) * noise
                        )

            placeholders = self.placeholders.unsqueeze(0).expand(z.shape[0], -1, -1)
            z = self.compressor(placeholders, context=z, context_mask=attention_mask)

            if (self.cfg.compressor_cfg.feat_masking_p > 0) and self.training:
                # Mask each feature independently with probability feat_masking_p
                mask = (
                    torch.rand(size=z.shape, device=z.device)
                    < self.cfg.compressor_cfg.feat_masking_p
                )
                z = z.masked_fill(mask, 0.0)
        else:
            z = z[:, : self.cfg.k]

        if self.cfg.variational:
            return self.out_proj(z), self.out_proj_logvar(z)
        else:
            return self.out_proj(z)