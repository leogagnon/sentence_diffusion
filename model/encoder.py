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
import sentence_transformers
from lightning.pytorch.utilities.rank_zero import rank_zero_info
from einops import rearrange


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

@dataclass
class STEncoderConfig:
    name: str
    normalize: bool
    lora_cfg: Optional[dict] = None

class STEncoder(EncoderModel):
    def __init__(self, cfg: Optional[STEncoderConfig] = None, **kwargs):
        super().__init__()

        if cfg == None:
            cfg = STEncoderConfig(**kwargs)

        backbone = sentence_transformers.SentenceTransformer(
            cfg.name,
        )
        self.transformer = backbone[0]
        self.pooling = backbone[1]
        self.normalization = backbone[2]

        assert isinstance(self.transformer, sentence_transformers.models.Transformer), "Expected Transformer as first module"
        assert isinstance(self.pooling, sentence_transformers.models.Pooling), "Expected Pooling as second module"
        assert isinstance(self.normalization, sentence_transformers.models.Normalize), "Expected Normalize as third module"

        self.tokenizer = backbone.tokenizer
        self._latent_dim = backbone.get_sentence_embedding_dimension()

        if cfg.lora_cfg == None:
            self.requires_grad_(False)
            self.transformer.auto_model = self.transformer.auto_model.to(torch.bfloat16)
            try:
                self.transformer.auto_model.set_attn_implementation(
                    "flash_attention_2"
                )
            except:
                rank_zero_info("Tried to use flash attention in encoder, but it is not available.")

        self.cfg = cfg

    @property
    def latent_dim(self):
        return self._latent_dim

    @torch.no_grad()
    def forward(self, input_ids, attention_mask=None, normalize: Optional[bool] = None):
        batch = {"input_ids": input_ids, "attention_mask": attention_mask}
        batch = self.transformer(batch) 
        batch = self.pooling(batch) 

        normalize = self.cfg.normalize if normalize is None else normalize
        if normalize:
            batch = self.normalization(batch)

        return batch["sentence_embedding"]


@dataclass
class SEMHeadConfig:
    L: int
    V: int
    temp: float
    input_dim: Optional[int] = None


class SEMHead(nn.Module):
    def __init__(self, cfg: SEMHeadConfig):
        super().__init__()
        assert cfg.input_dim is not None, "input_dim has to be set"
        self.proj_in = nn.Linear(cfg.input_dim, cfg.L * cfg.V)
        self.norm = nn.LayerNorm(cfg.L * cfg.V, eps=1e-6)
        self.proj_out = nn.Linear(cfg.L * cfg.V, cfg.input_dim)
        self.cfg = cfg

    def forward(self, x):
        # x: (B, D)
        x = self.proj_in(x)
        x = self.norm(x)
        x = einx.rearrange("b (l v) -> b l v", x, l=self.cfg.L, v=self.cfg.V)
        x = torch.softmax(x / self.cfg.temp, dim=-1)
        x = einx.rearrange("b l v -> b (l v)", x)
        x = self.proj_out(x)

        return x


@dataclass
class CompressorConfig:
    n_layers: int
    n_heads: int
    k: int = 1


@dataclass
class DAEEncoderConfig:
    name: str
    normalize: bool
    dropout_p: float
    out_dim: int
    input_dropout: bool = False
    compressor_cfg: Optional[CompressorConfig] = None
    lora_cfg: Optional[dict] = None
    sem_cfg: Optional[SEMHeadConfig] = None


class DAEEncoder(EncoderModel):
    def __init__(self, cfg: Optional[DAEEncoderConfig] = None, **kwargs):
        super().__init__()
        if cfg == None:
            cfg = DAEEncoderConfig(**kwargs)
        self.backbone = AutoModelForCausalLM.from_pretrained(
            cfg.name, trust_remote_code=True
        ).encoder
        self.tokenizer = T5Tokenizer.from_pretrained(cfg.name)

        self.out_proj = nn.Linear(self.backbone.config.d_model, cfg.out_dim, bias=False)

        if cfg.compressor_cfg != None:
            self.compressor = AttentionLayers(
                dim=self.backbone.config.d_model,
                depth=cfg.compressor_cfg.n_layers,
                heads=cfg.compressor_cfg.n_heads,
                cross_attend=True,
                causal=False,
            )
            self.placeholder_tokens = nn.Parameter(
                torch.randn(cfg.compressor_cfg.k, self.backbone.config.d_model)
            )

        if cfg.lora_cfg != None:
            self.backbone = get_peft_model(
                self.backbone,
                LoraConfig(**cfg.lora_cfg),
            )
        else:
            self.backbone.requires_grad_(False)
            self.backbone.eval()

        if cfg.compressor_cfg != None:
            if cfg.compressor_cfg.k > 1:
                assert (
                    cfg.normalize == False
                ), "Normalization not compatible with k>1 compressor for now"

        if cfg.sem_cfg != None:
            if cfg.compressor_cfg != None:
                assert (
                    cfg.compressor_cfg.k == 1
                ), "SEM only compatible with k=1 compressor for now"

            cfg.sem_cfg.input_dim = cfg.out_dim
            self.sem = SEMHead(cfg.sem_cfg)

        self.cfg = cfg

    @property
    def latent_dim(self):
        return self.cfg.out_dim

    def forward(self, input_ids, attention_mask=None):
        # Encode input text into hidden states
        with torch.no_grad() if (self.cfg.lora_cfg is None) else nullcontext():

            if self.cfg.input_dropout and (self.cfg.dropout_p > 0.0) and self.training:
                mask = (
                    torch.rand_like(input_ids, dtype=torch.float) < self.cfg.dropout_p
                )
                input_ids = input_ids.masked_fill(mask, self.tokenizer.unk_token_id)

            batch = {"input_ids": input_ids, "attention_mask": attention_mask}
            hidden_states = self.backbone(**batch).last_hidden_state
            attention_mask = batch["attention_mask"]

            # Apply token dropout
            if (self.cfg.input_dropout == False) & (
                self.cfg.dropout_p > 0.0
            ) and self.training:
                mask = torch.rand_like(hidden_states[:, :, 0]) < self.cfg.dropout_p
                mask = einx.rearrange("b n -> b n d", mask, d=hidden_states.shape[-1])
                hidden_states = hidden_states.masked_fill(mask, 0.0)

        if self.cfg.compressor_cfg != None:
            # Cross-attend with placeholder tokens
            ph = einx.rearrange(
                "k d -> b k d", self.placeholder_tokens, b=hidden_states.shape[0]
            )
            latent = self.compressor(
                ph,
                context=hidden_states,
                context_mask=attention_mask,
            )
        else:
            # ; or compute mean-pooled embedding
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
            latent = mean_pooled_embedding.unsqueeze(1)

        # Project to output dimension
        latent = self.out_proj(latent)

        if self.cfg.sem_cfg != None:
            assert self.cfg.normalize == False, "SEM not compatible with normalization"

            latent = self.sem(latent.squeeze(1)).unsqueeze(1)

        # Optionally normalize latent code
        if self.cfg.normalize:
            latent = F.normalize(latent, p=2, dim=2)
        else:
            latent = latent

        if latent.shape[1] == 1:
            latent = latent.squeeze(1)

        return latent
