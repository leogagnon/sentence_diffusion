from abc import ABC, abstractmethod
from functools import partial
import os
from peft.mapping_func import get_peft_model
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
from torch.utils.checkpoint import checkpoint
import hydra
from torch.distributions import Gamma
from entmax import entmax15


@dataclass
class SEMHeadConfig:
    L: int
    V: int
    temp: float
    input_dim: Optional[int] = None


class SEMHead(nn.Module):
    def __init__(self, cfg: Optional[SEMHeadConfig] = None, **kwargs):
        super().__init__()

        if cfg == None:
            cfg = SEMHeadConfig(**kwargs)

        assert cfg.input_dim is not None, "input_dim has to be set"
        self.proj_in = nn.Linear(cfg.input_dim, cfg.L * cfg.V, bias=False)
        self.norm = nn.LayerNorm(cfg.L * cfg.V, eps=1e-6)
        self.proj_out = nn.Linear(cfg.L * cfg.V, cfg.input_dim, bias=False)
        self.cfg = cfg

    @property
    def dlc_len(self):
        return self.cfg.L

    def forward(self, x, return_dlc=False):
        # Proj in DLC space
        x = self.proj_in(x)
        x = self.norm(x)
        x = einx.rearrange("b (l v) -> b l v", x, l=self.cfg.L, v=self.cfg.V)

        # Compute Softmax
        probs = torch.softmax(x / self.cfg.temp, dim=-1)

        # Compute output
        out = einx.rearrange("b l v -> b (l v)", probs)
        out = self.proj_out(out)

        if return_dlc:
            return out, self.encode(probs)
        else:
            return out, probs

    def encode(self, probs):
        dlc = probs.argmax(-1)
        return dlc


@dataclass
class HSEMHeadConfig:
    L: int
    V: int
    D: int
    temp: float
    input_dim: Optional[int] = None


class HSEMHead(nn.Module):
    def __init__(self, cfg: Optional[HSEMHeadConfig] = None, **kwargs):
        super().__init__()

        if cfg == None:
            cfg = HSEMHeadConfig(**kwargs)

        assert cfg.input_dim is not None, "input_dim has to be set"
        self.N = (cfg.V**cfg.D) // (cfg.V - 1)
        self.proj_in = nn.Linear(cfg.input_dim, cfg.L * cfg.V * self.N, bias=False)
        self.norm = nn.LayerNorm(cfg.L * cfg.V * self.N, eps=1e-6)
        self.proj_out = nn.Linear(cfg.L * cfg.V * self.N, cfg.input_dim, bias=False)
        self.cfg = cfg

    @property
    def dlc_len(self):
        return self.cfg.D * self.cfg.L

    def forward(self, x: torch.Tensor, return_dlc=False):
        bs = x.shape[0]

        # Compute conditional probabilities
        x = self.proj_in(x)
        x = self.norm(x)
        x = einx.rearrange(
            "b (L N V) -> b L N V",
            x,
            L=self.cfg.L,
            N=self.N,
            V=self.cfg.V,
        )
        x = F.softmax(x / self.cfg.temp, -1)

        # Compute DLC probs (i.e. the joint) by going down tree
        # E.g. p(x_0,x_1,x_2) = p(x_0) * p(x_1 | x_0) * p(x_2 | x_0, x_1)
        parent_probs = torch.ones(
            size=(bs, self.cfg.L, 1), device=x.device, dtype=x.dtype
        )
        start = 0
        probs = []
        for d in range(self.cfg.D):
            # Compute probs at level d by multiplying with parent probs
            end = start + self.cfg.V**d
            level = x[:, :, start:end] * parent_probs[..., None]
            probs.append(level)
            # Update parent and go down a level
            parent_probs = einx.rearrange("b L n V -> b L (n V)", level)
            start = end

        # Compute output
        out = torch.cat(probs, dim=2)
        out = einx.rearrange("b L N V -> b (L N V)", out)
        out = self.proj_out(out)

        if return_dlc:
            return out, self.encode(probs)
        else:
            return out, probs

    def encode(self, probs):

        # Argmax on first level
        dlc = [probs[0].squeeze(2).argmax(-1)]
        for i in range(self.cfg.D - 1):
            # Get the node at level i that was chosen at level i-1
            node = torch.gather(
                input=probs[i + 1],
                dim=2,
                index=einx.rearrange("b l -> b l n v", dlc[i], n=1, v=self.cfg.V),
            ).squeeze(2)

            # Argmax at level i (on the chosen node)
            dlc.append(node.argmax(-1))

        # Concatenate all levels
        # First L indices are level 1, second L are level 2, ...
        dlc = torch.cat(dlc, dim=1)

        return dlc


class SONARTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.auto_model = M2M100Encoder.from_pretrained(
            "cointegrated/SONAR_200_text_encoder",
        )

    def get_sentence_embedding_dimension(self):
        return self.auto_model.config.hidden_size

    def forward(self, features, **kwargs):
        token_embeddings = self.auto_model(**features).last_hidden_state
        features.update({"token_embeddings": token_embeddings})
        return features


@dataclass
class EncoderConfig:
    model_name: str
    sem: Optional[dict] = None
    prompt: Optional[str] = None


class EncoderModel(nn.Module):
    def __init__(self, cfg: Optional[EncoderConfig] = None, **kwargs):
        super().__init__()

        if cfg == None:
            cfg = EncoderConfig(**kwargs)

        # Initialize backbone
        if "SONAR" in cfg.model_name:
            self.transformer = SONARTransformer()
            self.pooling = sentence_transformers.models.Pooling(
                self.transformer.get_sentence_embedding_dimension(), pooling_mode="mean"
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                "cointegrated/SONAR_200_text_encoder"
            )
            self.tokenizer.src_lang = "eng_Latn"
            self._latent_dim = self.transformer.get_sentence_embedding_dimension()
        else:
            model_kwargs = {}

            # Some model-specific kwargs
            if "roberta" in cfg.model_name.lower():
                model_kwargs.update({"add_pooling_layer": False})

            if "qwen" in cfg.model_name.lower():
                model_kwargs.update({"attn_implementation": "flash_attention_2"})

            if "nemotron" in cfg.model_name.lower():
                model_kwargs.update(
                    {
                        "attn_implementation": "flash_attention_2",
                        "dtype": "bfloat16",
                    }
                )

            backbone = sentence_transformers.SentenceTransformer(
                cfg.model_name, model_kwargs=model_kwargs,trust_remote_code=True,
            )

            self.transformer = backbone[0]
            assert isinstance(
                self.transformer, sentence_transformers.models.Transformer
            ), "Expected Transformer as first module"

            self.pooling = backbone[1]
            assert isinstance(
                self.pooling, sentence_transformers.models.Pooling
            ), "Expected Pooling as second module"

            self.tokenizer = backbone.tokenizer
            self._latent_dim = (
                self.transformer.auto_model.get_input_embeddings().weight.shape[1]
            )

        if cfg.sem is not None:
            cfg.sem["input_dim"] = self.latent_dim
            self.sem = hydra.utils.instantiate(cfg.sem)
            self.sem: SEMHead | HSEMHead

        self.cfg = cfg

    def compile(self):
        # Only compile the SEM
        if self.cfg.sem is not None:
            self.sem.compile()

    @property
    def latent_dim(self):
        return self._latent_dim

    @property
    def latent_len(self):
        return 1

    def forward(self, input_ids, attention_mask, return_dlc=False):

        # Make the batch dict expected by sentence_transformers models
        batch = {"input_ids": input_ids, "attention_mask": attention_mask}

        # Run through transformer and pooling
        batch = self.transformer(batch)
        batch = self.pooling(batch)
        sentence_embedding = batch["sentence_embedding"]

        # Run through SEM
        if self.cfg.sem is not None:
            x_out, x_intern = self.sem(sentence_embedding, return_dlc=return_dlc)
            return x_out, x_intern
        else:
            return sentence_embedding
