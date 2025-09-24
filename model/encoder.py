from abc import ABC, abstractmethod
from functools import partial
import os
from peft.mapping_func import get_peft_model
from peft.tuners.lora.config import LoraConfig
from sentence_transformers.models import InputModule
import torch.nn as nn
from dataclasses import dataclass
from typing import Optional
from sentence_transformers import SentenceTransformer
from torch.nn import functional as F
from transformers.models.auto.modeling_auto import AutoModelForCausalLM, AutoModel
from transformers.models.auto.tokenization_auto import AutoTokenizer
import torch
from x_transformers.x_transformers import AttentionLayers, ScaledSinusoidalEmbedding
import einx
import math
import random
from transformers import T5EncoderModel, T5Tokenizer
from sentence_transformers import SentenceTransformer
from abc import ABC, abstractmethod
from contextlib import nullcontext
from torch.nn import Sequential


@dataclass
class EncoderConfig:
    name: str
    lora_cfg: Optional[dict] = None


class EncoderModel(ABC, nn.Module):
    @property
    @abstractmethod
    def latent_dim(self):
        pass

    @abstractmethod
    def forward(self, input_str):
        pass


class SentenceT5Encoder(EncoderModel):
    def __init__(self, cfg: Optional[EncoderConfig] = None, **kwargs):
        super().__init__()

        if cfg == None:
            cfg = EncoderConfig(**kwargs)
        assert cfg.lora_cfg == None, "LoRA not supported for SentenceT5Encoder"

        self.backbone = SentenceTransformer(
            cfg.name,
        ).requires_grad_(False)
        self.tokenizer = self.backbone.tokenizer

        self.cfg = cfg

    @property
    def latent_dim(self):
        return self.backbone.get_sentence_embedding_dimension()

    @torch.no_grad()
    def forward(self, input_ids, attention_mask=None):
        batch = {
            "input_ids": input_ids,
            "attention_mask": attention_mask
        }
        z = self.backbone.forward(batch)["sentence_embedding"]

        return z

@dataclass
class CompressorConfig:
    n_layers: int
    n_heads: int
    dim: int
    k: int = 1

@dataclass
class DAEEncoderConfig:
    name: str
    normalize: bool
    token_dropout_p: float
    compressor_cfg: Optional[CompressorConfig] = None
    lora_cfg: Optional[dict] = None

class DAEEncoder(EncoderModel):
    def __init__(self, cfg: Optional[DAEEncoderConfig] = None, **kwargs):
        super().__init__()
        if cfg == None:
            cfg = DAEEncoderConfig(**kwargs)
        self.model = AutoModelForCausalLM.from_pretrained(
            cfg.name, trust_remote_code=True
        )
        self.tokenizer = T5Tokenizer.from_pretrained(cfg.name)

        # Remove decoder and LM head, we only need the encoder + bottleneck
        del self.model.decoder, self.model.dec_emb, self.model.lm_head

        if cfg.compressor_cfg != None:
            del self.model.bottleneck
            self.compressor_proj = nn.Linear(
                self.model.encoder.config.d_model, cfg.compressor_cfg.dim, bias=False
            )
            self.compressor = AttentionLayers(
                dim=cfg.compressor_cfg.dim,
                depth=cfg.compressor_cfg.n_layers,
                heads=cfg.compressor_cfg.n_heads,
                cross_attend=True,
                causal=False,
            )
            self.placeholder = nn.Parameter(
                torch.randn(cfg.compressor_cfg.k, cfg.compressor_cfg.dim), requires_grad=True
            )


        if cfg.lora_cfg != None:
            self.model.encoder = get_peft_model(
                self.model.encoder,
                LoraConfig(**cfg.lora_cfg),
            )
        else:
            self.model.encoder.requires_grad_(False)
            self.model.encoder.eval()

        self.cfg = cfg

    @property
    def latent_dim(self):
        if self.cfg.compressor_cfg != None:
            return self.cfg.compressor_cfg.dim
        else:
            return self.model.bottleneck.out_proj.out_features

    def forward(self, input_ids, attention_mask=None):
        with torch.no_grad() if (self.cfg.lora_cfg is None) else nullcontext():
            
            # Pass through T5 encoder
            batch = {
                "input_ids": input_ids,
                "attention_mask": attention_mask
            }
            hidden_states = self.model.encoder(**batch).last_hidden_state
            attention_mask = batch["attention_mask"]

            # Apply token dropout
            if (self.cfg.token_dropout_p > 0.0) and self.training:
                mask = (
                    torch.rand_like(hidden_states[:, :, 0]) < self.cfg.token_dropout_p
                )
                mask = einx.rearrange("b n -> b n d", mask, d=hidden_states.shape[-1])
                hidden_states = hidden_states.masked_fill(mask, 0.0)

            if self.cfg.compressor_cfg != None:
                # Apply cross-attention bottleneck with learned placeholders as queries
                hidden_states = self.compressor_proj(hidden_states)
                placeholders = einx.rearrange("k d -> b k d", self.placeholder, b=hidden_states.shape[0])
                latent = self.compressor(placeholders, context=hidden_states, context_mask=attention_mask)
            else:
                # Apply MHA bottleneck (cross-attention with mean-pooled query)
                hidden_states = hidden_states.repeat(
                    attention_mask.shape[0] // hidden_states.shape[0], 1, 1
                ) 
                mask_expanded = (
                    attention_mask.to(dtype=hidden_states.dtype)
                    .unsqueeze(-1)
                    .expand(hidden_states.shape)
                )
                mean_pooled_embedding = torch.sum(
                    hidden_states * mask_expanded, 1
                ) / torch.clamp(mask_expanded.sum(1), min=1e-9)
                latent, attn_weights = self.model.bottleneck(
                    mean_pooled_embedding.unsqueeze(1),
                    hidden_states,
                    hidden_states,
                    need_weights=False,
                    attn_mask=attention_mask.to(dtype=hidden_states.dtype)
                    .unsqueeze(1)
                    .repeat_interleave(self.model.num_heads, dim=0),
                )

            # Optionally normalize latent code
            if self.cfg.normalize:
                latent = F.normalize(
                    latent, p=2, dim=2
                )
            else:
                latent = latent
        
            if latent.shape[1] == 1:
                latent = latent.squeeze(1)

            return latent