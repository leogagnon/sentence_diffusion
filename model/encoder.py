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


def get_sentence_encoder(name: str) -> SentenceTransformer:
    if name.startswith("thesephist"):
        # Custom wrapper around thesephist/contra-bottleneck-t5-{name}-wikipedia models
        return SentenceTransformer(modules=[PretrainedDAE(name=name)])
    else:
        # Regular sentence transformer model from HuggingFace
        return SentenceTransformer(
            name,
            model_kwargs={"torch_dtype": "float16"},
        )


from model.gaussian_diffusion import (
    get_sampling_schedule,
    right_pad_dims_to,
    time_to_alpha,
)


@dataclass
class PromptGeneratorConfig:
    n_layers: int
    n_heads: int


@dataclass
class EncoderConfig:
    name: str
    k: int
    prompt_generator_cfg: PromptGeneratorConfig
    out_proj_dim: Optional[int] = None
    lora_cfg: Optional[dict] = None


class EncoderModel(nn.Module):
    def __init__(self, cfg: EncoderConfig):
        super().__init__()
        self.cfg = cfg

        # Init backbone
        self.backbone = get_sentence_encoder(cfg.name)

        if self.cfg.lora_cfg != None:
            self.backbone = get_peft_model(
                self.backbone,
                LoraConfig(**self.cfg.lora_cfg),
            )
        else:
            self.backbone.requires_grad_(False)
            self.backbone.eval()

        assert cfg.out_proj_dim != None

        self.prompt_noise_embed = nn.Sequential(
            ScaledSinusoidalEmbedding(cfg.out_proj_dim),
            nn.Linear(cfg.out_proj_dim, cfg.out_proj_dim * 4),
            nn.GELU(),
            nn.Linear(cfg.out_proj_dim * 4, cfg.out_proj_dim),
        )

        self.prompt_gen_projector = nn.Linear(
            self.backbone.get_sentence_embedding_dimension(),
            cfg.out_proj_dim * cfg.k,
            bias=False,
        )

        self.prompt_gen = AttentionLayers(
            dim=cfg.out_proj_dim,
            depth=cfg.prompt_generator_cfg.n_layers,
            heads=cfg.prompt_generator_cfg.n_heads,
            causal=False,
            use_adaptive_rmsnorm=True,
            ff_swish=True,
            ff_glu=True,
        )

        self.aug_noise_schedule = partial(
            time_to_alpha, alpha_schedule=get_sampling_schedule("cosine"), scale=3.0
        )

    @property
    def latent_dim(self):
        return self.backbone.get_sentence_embedding_dimension()

    def forward(self, input_str):

        # If no LoRA, don't compute gradients through backbone
        if self.cfg.lora_cfg is None:
            with torch.no_grad():
                z = self.backbone.encode(
                    input_str,
                    convert_to_tensor=True,
                    show_progress_bar=False,
                ).detach()
        else:
            z = self.backbone.encode(
                input_str, convert_to_tensor=True, show_progress_bar=False
            )

        # Project to prompt space
        z = self.prompt_gen_projector(z)
        z = einx.rearrange("b (k d) -> b k d", z, k=self.cfg.k)

        # Sample augmentation noise using a scaled cosine schedule
        if self.training:
            times = torch.zeros((z.size(0),), device=z.device).float().uniform_(0, 1.0)
            alpha = self.aug_noise_schedule(times)
        else:
            alpha = torch.full(
                size=(z.size(0),), fill_value=0.97467943448, device=z.device
            )
        alpha = right_pad_dims_to(z, alpha)
        noise = torch.randn_like(z)
        z = alpha.sqrt() * z + (1 - alpha).sqrt() * noise

        # Process z with prompt generator (conditioned on noise level)
        noise_emb = self.prompt_noise_embed(alpha[None] * 1000)
        noise_emb = einx.rearrange("b d -> b 1 d", noise_emb)
        z = self.prompt_gen(z, condition=noise_emb)

        return z


class PretrainedDAE(InputModule):
    """
    SentenceEmbedding wrapper around thesephist/contra-bottleneck-t5-{name}-wikipedia models.
    """

    def __init__(self, name):
        super().__init__()
        self.model = AutoModelForCausalLM.from_pretrained(name, trust_remote_code=True)
        self.model = self.model.half()
        self.tokenizer = T5Tokenizer.from_pretrained(name)

        # Remove decoder and LM head, we only need the encoder + bottleneck
        del self.model.decoder, self.model.dec_emb, self.model.lm_head

    def tokenize(self, texts):
        return self.tokenizer.batch_encode_plus(
            texts, return_tensors="pt", padding=True
        )

    def save(self, path):
        pass

    def get_sentence_embedding_dimension(self):
        return self.model.bottleneck.out_proj.out_features

    def forward(self, features):
        hidden_states = self.model.encoder(**features).last_hidden_state
        attention_mask = features["attention_mask"]

        hidden_states = hidden_states.repeat(
            attention_mask.shape[0] // hidden_states.shape[0], 1, 1
        )  # during contrastive search, attn mask can have higher batch size than hidden_state
        mask_expanded = attention_mask.to(dtype=hidden_states.dtype).unsqueeze(-1).expand(hidden_states.shape)
        mean_pooled_embedding = torch.sum(
            hidden_states * mask_expanded, 1
        ) / torch.clamp(mask_expanded.sum(1), min=1e-9)
        unscaled_latent, attn_weights = self.model.bottleneck(
            mean_pooled_embedding.unsqueeze(1),
            hidden_states,
            hidden_states,
            need_weights=False,
            # torch MHA attn_mask has opposite signs to HF T5 masks... sigh
            attn_mask=attention_mask.to(dtype=hidden_states.dtype)
            .unsqueeze(1)
            .repeat_interleave(self.model.num_heads, dim=0),
        )
        latent = self.model.bottleneck_scale * F.normalize(unscaled_latent, p=2, dim=2)

        return {"sentence_embedding": latent.squeeze(1)}
