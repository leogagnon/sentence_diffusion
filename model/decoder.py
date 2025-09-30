from abc import ABC, abstractmethod
from functools import partial
import math
import os
import einx
from peft.mapping_func import get_peft_model
from peft.tuners.lora.config import LoraConfig
import torch.nn as nn
from dataclasses import dataclass
from typing import Optional
from sentence_transformers import SentenceTransformer
from transformers.models.auto.modeling_auto import AutoModelForCausalLM
from transformers.models.auto.tokenization_auto import AutoTokenizer
from x_transformers import Encoder
from x_transformers.x_transformers import AttentionLayers, ScaledSinusoidalEmbedding
import torch
from typing import Optional, Union
from einops import rearrange
from torch.nn import ModuleDict

from model.gaussian_diffusion import time_to_alpha


@dataclass
class PromptGeneratorConfig:
    n_layers: int
    n_heads: int
    k: int
    noise_conditioning: bool = False
    default_alpha: float = 0.95  # delta^2=0.05 like in DGLM


@dataclass
class DecoderConfig:
    name: str
    input_dim: Optional[int] = None  # has to be set
    prompt_generator_cfg: Optional[PromptGeneratorConfig] = None
    lora_cfg: Optional[dict] = None


class DecoderModel(nn.Module):
    """Wrapper around a pretrained decoder model (e.g. GPT) with optional LoRA adaptation and soft prompting."""

    def __init__(self, cfg: DecoderConfig):
        super().__init__()
        self.cfg = cfg

        # Init causal LM backbone
        self.backbone = AutoModelForCausalLM.from_pretrained(cfg.name)
        if self.cfg.lora_cfg != None:
            self.backbone = get_peft_model(
                self.backbone,
                LoraConfig(**self.cfg.lora_cfg),
            )
        else:
            self.backbone.requires_grad_(False)
            self.backbone.eval()

        # Init tokenizer and add pad token if missing
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.name)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})
            self.backbone.resize_token_embeddings(
                len(self.tokenizer), mean_resizing=True
            )

        # Init prompt generator (project up, chunk, project up again, process with transformer)
        if cfg.prompt_generator_cfg != None:
            pre_proj_dim = 128
            self.prompt_generator = nn.ModuleDict(
                {
                    "z_to_chunk": nn.Linear(
                        cfg.input_dim,
                        cfg.prompt_generator_cfg.k * pre_proj_dim,
                        bias=False,
                    ),
                    "chunk_to_embd": nn.Linear(
                        pre_proj_dim, self.backbone.config.hidden_size, bias=False
                    ),
                }
            )
            if cfg.prompt_generator_cfg.noise_conditioning:
                self.prompt_generator["encoder"] = AttentionLayers(
                    dim=self.backbone.config.hidden_size,
                    depth=cfg.prompt_generator_cfg.n_layers,
                    heads=cfg.prompt_generator_cfg.n_heads,
                    causal=False,
                    use_adaptive_rmsnorm=True,
                    use_adaptive_layerscale=True,
                    ff_swish=True,
                    ff_glu=True,
                    dim_condition=self.backbone.config.hidden_size * 4,
                    adaptive_condition_mlp=True,
                    attn_qk_norm=True,
                    attn_qk_norm_dim_scale=True,
                )
                self.prompt_generator["noise_embd"] = ScaledSinusoidalEmbedding(self.backbone.config.hidden_size)
            else:
                self.prompt_generator["encoder"] = Encoder(
                    dim=self.backbone.config.hidden_size,
                    depth=cfg.prompt_generator_cfg.n_layers,
                    heads=cfg.prompt_generator_cfg.n_heads,
                )
        else:
            self.in_proj = nn.Linear(
                cfg.input_dim, self.backbone.config.hidden_size, bias=False
            )

    def z_to_prompt(self, z: torch.Tensor, alpha: Optional[torch.Tensor] = None):
        if self.cfg.prompt_generator_cfg == None:
            # In this case z is already the prompt
            return self.in_proj(z)

        # Project z to prompt space (B, D) -> (B, k * d) -> (B, k, d) -> (B, k, embd_dim)
        prompt = self.prompt_generator["z_to_chunk"](z)
        prompt = rearrange(
            prompt,
            "b (k d) -> b k d",
            k=self.cfg.prompt_generator_cfg.k,
        )
        prompt = self.prompt_generator["chunk_to_embd"](prompt)

        # Process prompt with transformer (potentially conditioned on noise level)
        if self.cfg.prompt_generator_cfg.noise_conditioning:
            if alpha is None:
                # If no alpha is given, use default value (e.g. at inference)
                alpha = torch.full(
                    (z.shape[0], 1),
                    self.cfg.prompt_generator_cfg.default_alpha,
                    device=z.device,
                )
            noise_embd = self.prompt_generator["noise_embd"](alpha[None] * 1000)
            noise_embd = rearrange(noise_embd, "b d -> b 1 d")
            prompt = self.prompt_generator["encoder"](prompt, condition=noise_embd)
        else:
            prompt = self.prompt_generator["encoder"](prompt)

        return prompt

    def forward(
        self,
        input_ids,
        z: Optional[torch.Tensor] = None,
        alpha: Optional[torch.Tensor] = None,
    ):
        if z == None:
            return self.backbone(input_ids=input_ids).logits
        else:
            # Compute input embeddings
            prompt = self.z_to_prompt(z, alpha=alpha)
            input_embeds = self.backbone.get_input_embeddings()(input_ids)
            input_embeds = torch.cat([prompt, input_embeds], dim=1)

            # Forward pass
            output = self.backbone(inputs_embeds=input_embeds)
            logits = output.logits[:, prompt.shape[1] :]

        return logits

    @torch.no_grad()
    def generate(
        self,
        max_length: int,
        z: Optional[torch.Tensor] = None,
        alpha: Optional[torch.Tensor] = None,
    ):
        """Generate text using autoregressive decoding, potentially conditioned on soft prefix z"""

        if z != None:
            # Compute cache for z
            prompt = self.z_to_prompt(z, alpha=alpha)
            cache = self.backbone(
                inputs_embeds=prompt,
                use_cache=True,
            ).past_key_values
            # Position of the BOS should be after the prefix (like in training)
            cache_position = torch.tensor([prompt.shape[1]], device=z.device)
            attention_mask = torch.ones(
                (prompt.shape[0], prompt.shape[1] + 1), device=z.device
            )
        else:
            cache = None
            cache_position = None
            attention_mask = None

        # Autoregressive generation from BOS token with cached z (nucleus sampling)
        bos = torch.full(
            (z.shape[0], 1),
            self.tokenizer.bos_token_id,
            device=z.device,
            dtype=torch.long,
        )
        output = self.backbone.generate(
            input_ids=bos,
            past_key_values=cache,
            cache_position=cache_position,
            attention_mask=attention_mask,
            max_length=max_length,
            do_sample=True,
            top_p=0.92,
            top_k=50,
            num_beams=1,
            temperature=0.9,
            return_dict_in_generate=True,
            use_cache=True,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        output = output.sequences[:, 1:]  # Remove BOS token

        return output
