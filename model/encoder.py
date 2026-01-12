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
from torch.nn.init import trunc_normal_
from torch.nn.utils import weight_norm
import torch.nn as nn


@dataclass
class SEMHeadConfig:
    L: int
    V: int
    temp: float
    output_dim: Optional[int] = None
    input_dim: Optional[int] = None
    ln: bool = True
    per_simpex_ln: bool = False
    hard_renorm: bool = False
    proj_out_test: bool = False


class SEMHead(nn.Module):
    def __init__(self, cfg: Optional[SEMHeadConfig] = None, **kwargs):
        super().__init__()

        if cfg == None:
            cfg = SEMHeadConfig(**kwargs)

        assert cfg.input_dim is not None, "input_dim has to be set"
        self.proj_in = nn.Linear(cfg.input_dim, cfg.L * cfg.V, bias=False)
        if cfg.ln:
            if cfg.per_simpex_ln:
                self.norm = nn.LayerNorm((cfg.V,))
            else:
                self.norm = nn.LayerNorm((cfg.L, cfg.V))
        else:
            self.norm = nn.Identity()

        if cfg.proj_out_test:
            self.proj_out = nn.Linear(cfg.L * cfg.V, cfg.output_dim * cfg.L, bias=False)
        else:
            self.proj_out = nn.Parameter(torch.empty(cfg.L, cfg.V, cfg.output_dim))
            torch.nn.init.kaiming_uniform_(self.proj_out, a=math.sqrt(5))
        self.latent_len = cfg.L

        self.cfg = cfg

    @property
    def dlc_len(self):
        return self.cfg.L

    def forward(
        self,
        x,
        return_dlc=False,
        return_count=False,
        noise: float = 0.0,
        temp: Optional[float] = 0.0,
    ):
        temp = self.cfg.temp if temp is None else temp
        # Proj in DLC space
        x = self.proj_in(x)
        x = einx.rearrange("b (l v) -> b l v", x, l=self.cfg.L, v=self.cfg.V)
        x = self.norm(x)

        # Compute Softmax
        probs = torch.softmax(x / temp, dim=-1)
        
        # Compute output (add noise, renorm, project out)
        out = probs
        if noise > 0.0:
            out = out + noise * torch.randn_like(out)
        if self.cfg.hard_renorm:
            out = torch.softmax(out * 10.0, dim=-1)
        
        if self.cfg.proj_out_test:
            out = einx.rearrange("b l v -> b (l v)", out)
            out = self.proj_out(out)
            out = einx.rearrange("b (l o) -> b l o", out, o=self.cfg.output_dim, l=self.cfg.L)
        else:
            out = torch.einsum("blv,lvo->blo", out, self.proj_out).contiguous()

        # Figure out what to return
        out_dict = {"c_out": out, "probs": probs}

        if return_dlc:
            out_dict.update({"dlc": self._encode(probs)})

        if return_count:
            out_dict.update({"usage_count": self._usage_count(probs)})

        return out_dict

    def _encode(self, probs):
        dlc = probs.argmax(-1)
        return dlc

    def _usage_count(self, probs):
        counts = torch.stack(
            [torch.sum(probs.argmax(-1) == i, dim=0) for i in range(self.cfg.V)], dim=-1
        )
        counts = einx.rearrange("L V -> (L V)", counts)
        return counts


@dataclass
class HSEMHeadConfig:
    L: int
    V: int
    D: int
    temp: float
    input_dim: Optional[int] = None
    per_simpex_ln: bool = False
    ln: bool = True
    out_normalization: str = "none"


class HSEMHead(nn.Module):
    def __init__(self, cfg: Optional[HSEMHeadConfig] = None, **kwargs):
        super().__init__()

        if cfg == None:
            cfg = HSEMHeadConfig(**kwargs)

        assert cfg.input_dim is not None, "input_dim has to be set"
        self.N = (cfg.V**cfg.D) // (cfg.V - 1)
        self.proj_in = nn.Linear(cfg.input_dim, cfg.L * self.N * cfg.V, bias=False)
        if cfg.ln:
            if cfg.per_simpex_ln:
                self.norm = nn.LayerNorm(
                    (cfg.V,),
                )
            else:
                self.norm = nn.LayerNorm((cfg.L, self.N, cfg.V))
        else:
            self.norm = nn.Identity()

        self.latent_len = cfg.D
        self.proj_out = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        nn.Linear(cfg.L * (cfg.V**l) * cfg.V, cfg.input_dim, bias=False)
                        for l in range(cfg.D)
                    ]
                )
            ]
        )
        self.cfg = cfg

    @property
    def dlc_len(self):
        return self.cfg.D * self.cfg.L

    def forward(
        self,
        x: torch.Tensor,
        return_dlc=False,
        return_count=True,
        noise: float = 0.0,
        temp: Optional[float] = None,
    ):
        bs = x.shape[0]
        temp = self.cfg.temp if temp is None else temp

        # Compute conditional probabilities
        x = self.proj_in(x)
        x = einx.rearrange(
            "b (L N V) -> b L N V",
            x,
            L=self.cfg.L,
            N=self.N,
            V=self.cfg.V,
        )
        x = self.norm(x)
        x = F.softmax(x / temp, -1)

        # Compute DLC probs (i.e. the joint) by going down tree
        # E.g. p(x_0,x_1,x_2) = p(x_0) * p(x_1 | x_0) * p(x_2 | x_0, x_1)
        with torch.autocast(device_type="cuda", dtype=torch.float32):
            parent_probs = torch.ones(
                size=(bs, self.cfg.L, 1), device=x.device, dtype=x.dtype
            )
            start = 0
            probs = []
            for d in range(self.cfg.D):
                # Compute probs at level d by multiplying with parent probs
                end = start + self.cfg.V**d
                level = x[:, :, start:end] * parent_probs[..., None]
                probs.append(einx.rearrange("b L n V -> b (L n V)", level))
                # Update parent and go down a level
                parent_probs = einx.rearrange("b L n V -> b L (n V)", level)
                start = end

        # Compute output
        out = torch.cat(
            [
                self.proj_out[i](probs[i] + noise * torch.randn_like(probs[i]))
                for i in range(len(probs))
            ],
            dim=2,
        )

        out = einx.rearrange("b L N V -> b (L N V)", out)
        out = self.proj_out(out)

        out_dict = {"c_out": out, "probs": probs}

        if return_dlc:
            out_dict.update({"dlc": self._encode(probs)})

        if return_count:
            out_dict.update({"usage_count": self._usage_count(probs)})

        return out_dict

    def _encode(self, probs):

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

    def _usage_count(self, probs):
        counts = torch.concatenate(
            [
                torch.stack(
                    [torch.sum(p.argmax(-1) == i, dim=0) for i in range(self.cfg.V)],
                    dim=-1,
                )
                for p in probs
            ],
            dim=1,
        )
        counts = einx.rearrange("L N V -> (L N V)", counts)
        return counts


@dataclass
class DINOHeadConfig:
    nlayers: int
    hidden_dim: int
    mlp_bias: bool
    dim: Optional[int] = None
    bottleneck_dim: Optional[int] = None
    normalize: bool = False


class DINOHead(nn.Module):
    def __init__(self, cfg: Optional[DINOHeadConfig] = None, **kwargs):
        super().__init__()
        if cfg == None:
            cfg = DINOHeadConfig(**kwargs)
        assert cfg.dim is not None
        self.mlp = _build_mlp(
            cfg.nlayers,
            cfg.dim,
            cfg.dim if cfg.bottleneck_dim is None else cfg.bottleneck_dim,
            hidden_dim=cfg.hidden_dim,
            use_bn=False,
            bias=cfg.mlp_bias,
        )
        self.apply(self._init_weights)
        if cfg.bottleneck_dim is not None:
            self.last_layer = weight_norm(
                nn.Linear(cfg.bottleneck_dim, cfg.dim, bias=False)
            )
            self.last_layer.weight_g.data.fill_(1)
        else:
            self.last_layer = nn.Identity()

        self.cfg = cfg

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.mlp(x)
        if self.cfg.normalize:
            x = nn.functional.normalize(x, dim=-1, p=2, eps=1e-5)
        x = self.last_layer(x)
        return x


def _build_mlp(
    nlayers, in_dim, bottleneck_dim, hidden_dim=None, use_bn=False, bias=True
):
    if nlayers == 1:
        return nn.Linear(in_dim, bottleneck_dim, bias=bias)
    else:
        layers = [nn.Linear(in_dim, hidden_dim, bias=bias)]
        if use_bn:
            layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(nn.GELU())
        for _ in range(nlayers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim, bias=bias))
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
        layers.append(nn.Linear(hidden_dim, bottleneck_dim, bias=bias))
        return nn.Sequential(*layers)


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
    latent_dim: Optional[int] = None
    sem: Optional[dict] = None
    dino_head: Optional[DINOHeadConfig] = None
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
            backbone_dim = self.transformer.get_sentence_embedding_dimension()
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
                cfg.model_name,
                model_kwargs=model_kwargs,
                trust_remote_code=True,
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
            backbone_dim = (
                self.transformer.auto_model.get_input_embeddings().weight.shape[1]
            )

        if cfg.sem is not None:
            cfg.sem["input_dim"] = backbone_dim
            cfg.sem["output_dim"] = cfg.latent_dim
            self.sem = hydra.utils.instantiate(cfg.sem)
            self.sem: SEMHead | HSEMHead

        if cfg.dino_head is not None:
            cfg.dino_head.dim = backbone_dim
            self.dino_head = DINOHead(cfg.dino_head).requires_grad_(True)

        self.cfg = cfg

    def compile(self):
        # Only compile the SEM
        if self.cfg.sem is not None:
            self.sem.compile()

    @property
    def latent_dim(self):
        return self.cfg.latent_dim

    @property
    def latent_len(self):
        return self.sem.latent_len

    def forward(
        self,
        input_ids,
        attention_mask,
        return_dlc=False,
        return_count=False,
        noise: float = 0.0,
        temp: Optional[float] = None,
    ):

        # Make the batch dict expected by sentence_transformers models
        batch = {"input_ids": input_ids, "attention_mask": attention_mask}

        # Run through transformer and pooling
        batch = self.transformer(batch)
        batch = self.pooling(batch)
        z = batch["sentence_embedding"]

        if self.cfg.sem is None:
            return z
        else:
            sem_out = self.sem(
                z,
                return_dlc=return_dlc,
                return_count=return_count,
                noise=noise,
                temp=temp,
            )
            z = sem_out.pop("c_out")

        if self.cfg.dino_head is not None:
            z = self.dino_head(z)

        return z, sem_out
