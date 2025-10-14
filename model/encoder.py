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
from transformers.models.m2m_100.modeling_m2m_100 import M2M100Encoder
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
class SEMHeadConfig:
    L: int
    V: int
    temp: float
    D: Optional[int]
    input_dim: Optional[int] = None


class SEMHead(nn.Module):
    def __init__(self, cfg: SEMHeadConfig):
        super().__init__()
        assert cfg.input_dim is not None, "input_dim has to be set"
        self.proj_in = nn.Linear(cfg.input_dim, cfg.L * cfg.V)
        self.norm = nn.LayerNorm(cfg.L * cfg.V, eps=1e-6)
        self.proj_out = nn.Linear(cfg.L * cfg.V, cfg.input_dim)
        self.cfg = cfg

    def forward(self, x, return_sem=False):
        # x: (B, D)
        x = self.proj_in(x)
        x = self.norm(x)
        x = einx.rearrange("b (l v) -> b l v", x, l=self.cfg.L, v=self.cfg.V)
        x = torch.softmax(x / self.cfg.temp, dim=-1)
        if return_sem:
            return x
        x = einx.rearrange("b l v -> b (l v)", x)
        x = self.proj_out(x)

        return x

@dataclass
class HSEMHeadConfig:
    L: int
    V: int
    D: int
    temp: float
    input_dim: Optional[int] = None

class HSEMHead(nn.Module):
    def __init__(self, cfg: HSEMHeadConfig, n_levels: int):
        super().__init__()
        assert cfg.input_dim is not None, "input_dim has to be set"
        self.levels = nn.ModuleList()
        for _ in range(n_levels):
            self.levels.append(SEMHead(cfg))
        self.cfg = cfg
        self.n_levels = n_levels
        

    def forward(self, x):
        # x: (B, D)
        for level in self.levels:
            x = level(x) + x  # Residual connection
        return x
    
@dataclass
class CompressorConfig:
    n_layers: int
    n_heads: int
    k: int = 1


class SONARTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.auto_model = M2M100Encoder.from_pretrained("cointegrated/SONAR_200_text_encoder")
    
    def get_sentence_embedding_dimension(self):
        return self.auto_model.config.hidden_size
    
    def forward(self, features, **kwargs):
        token_embeddings = self.auto_model(**features).last_hidden_state
        features.update({'token_embeddings': token_embeddings})
        return features

def random_substitution(input_ids, sub_p, vocab_size):

        probability = torch.full_like(
            input_ids,
            fill_value=sub_p,
            dtype=torch.float32,
        )
        masked_indices = torch.bernoulli(probability).bool()
        random_words = torch.randint_like(input_ids, low=0, high=vocab_size, dtype=torch.int64)

        input_ids[masked_indices] = random_words[masked_indices]

        return input_ids


@dataclass
class EncoderConfig:
    name: str
    lora_cfg: Optional[dict] = None


class EncoderModel(ABC, nn.Module):
    @property
    @abstractmethod
    def latent_dim(self):
        pass

    @property
    @abstractmethod
    def latent_len(self):
        pass

    @abstractmethod
    def forward(self, input_str):
        pass

@dataclass
class STEncoderConfig:
    name: str
    normalize: bool
    lora_cfg: Optional[dict] = None
    dropout_p: float = 0.0
    compressor_cfg: Optional[CompressorConfig] = None
    sem_cfg: Optional[SEMHeadConfig] = None
    variational: bool = False 

class STEncoder(EncoderModel):
    def __init__(self, cfg: Optional[STEncoderConfig] = None, **kwargs):
        super().__init__()

        if cfg == None:
            cfg = STEncoderConfig(**kwargs)

        if 'SONAR' in cfg.name:
            self.transformer = SONARTransformer()
            self.pooling = sentence_transformers.models.Pooling(
                self.transformer.get_sentence_embedding_dimension(),
                pooling_mode="mean"
            )
            self.tokenizer = AutoTokenizer.from_pretrained("cointegrated/SONAR_200_text_encoder")
            self.tokenizer.src_lang = "eng_Latn"
            self._latent_dim = self.transformer.get_sentence_embedding_dimension()
        else:
            backbone = sentence_transformers.SentenceTransformer(
                            cfg.name,
                        )
        
            self.transformer = backbone[0]
            assert isinstance(self.transformer, sentence_transformers.models.Transformer), "Expected Transformer as first module"
            
            self.pooling = backbone[1]
            assert isinstance(self.pooling, sentence_transformers.models.Pooling), "Expected Pooling as second module"

            if cfg.normalize:
                self.normalization = backbone[2]
                assert isinstance(self.normalization, sentence_transformers.models.Normalize), "Expected Normalize as third module (as cfg.normalize is True)"

            self.tokenizer = backbone.tokenizer
            self._latent_dim = self.transformer.auto_model.get_input_embeddings().weight.shape[1]

        if cfg.compressor_cfg != None:
            # Will replace self.pooling
            self.compressor = AttentionLayers(
                    dim=self.latent_dim,
                    depth=cfg.compressor_cfg.n_layers,
                    heads=cfg.compressor_cfg.n_heads,
                    cross_attend=True,
                    causal=False,
                )
            self.placeholder_tokens = nn.Parameter(
                torch.randn(cfg.compressor_cfg.k, self.latent_dim)
            )

        if cfg.sem_cfg != None:
            assert cfg.compressor_cfg == None
            assert cfg.normalize == False

            cfg.sem_cfg.input_dim = self.latent_dim
            self.sem = SEMHead(cfg.sem_cfg)
        
        if cfg.variational:
            assert cfg.normalize == False, "Normalization not compatible with variational AE"
            assert cfg.compressor_cfg == None, "Compressor not compatible with variational AE"
            self.out_mean = nn.Linear(self.latent_dim, self.latent_dim)
            self.out_logvar = nn.Linear(self.latent_dim, self.latent_dim)

        if cfg.lora_cfg == None:
            # Make the transformer non-trainable but faster!
            self.transformer.auto_model = self.transformer.auto_model.to(torch.bfloat16)
            try:
                self.transformer.auto_model.set_attn_implementation(
                    "flash_attention_2"
                )
            except:
                rank_zero_info("Tried to use flash attention in encoder, but it is not available.")
            self.transformer.requires_grad_(False)
            self.transformer.eval()
        else:
            self.transformer.requires_grad_(True)
            self.transformer = get_peft_model(
                self.transformer,
                LoraConfig(**cfg.lora_cfg),
            )

        self.cfg = cfg

    def train(self, mode: bool = True):
        super().train(mode)
        # Keep the transformer in eval model if not finetuning
        if self.cfg.lora_cfg == None:
            self.transformer.eval()
        return self

    @property
    def latent_dim(self):
        return self._latent_dim
    
    @property
    def latent_len(self):
        return 1 if self.cfg.compressor_cfg is None else self.cfg.compressor_cfg.k

    def forward(self, input_ids, attention_mask=None, return_sem=False):
        # Maybe randomly substitute tokens (for DAE training)
        if (self.cfg.dropout_p > 0.0) and self.training:
            input_ids = random_substitution(
                input_ids, self.cfg.dropout_p, self.tokenizer.vocab_size
            )

        # Make the batch dict expected by sentence_transformers models
        batch = {"input_ids": input_ids, "attention_mask": attention_mask}

        # Run through transformer
        batch = self.transformer(batch)

        if self.cfg.compressor_cfg == None:
            # Pooling bottleneck + normalization/SEM
            batch = self.pooling(batch)

            if self.cfg.normalize:
                sentence_embedding = self.normalization(batch)["sentence_embedding"]
            else:
                sentence_embedding = batch["sentence_embedding"]
            
            if self.cfg.sem_cfg != None:
                if return_sem:
                    return self.sem(sentence_embedding, return_sem=True)
                sentence_embedding = self.sem(sentence_embedding)

            sentence_embedding = sentence_embedding[:, None]  

            # Maybe apply VAE heads
            if self.cfg.variational:
                if self.training:
                    mean = self.out_mean(sentence_embedding)
                    logvar = self.out_logvar(sentence_embedding)
                    return mean, logvar
                else:
                    return self.out_mean(sentence_embedding)
        else: 
            # Cross-attention bottleneck
            ph = einx.rearrange(
                "k d -> b k d", self.placeholder_tokens, b=batch["token_embeddings"].shape[0]
            )
            sentence_embedding = self.compressor(
                ph,
                context=batch["token_embeddings"],
                context_mask=batch["attention_mask"],
            )

        return sentence_embedding


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
    
    @property
    def latent_len(self):
        return self.cfg.compressor_cfg.k if self.cfg.compressor_cfg is not None else 1

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

        return latent
