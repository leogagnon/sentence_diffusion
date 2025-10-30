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
from tokenizers.processors import TemplateProcessing
from lightning.pytorch.utilities.rank_zero import rank_zero_info
from transformers import GenerationConfig

@dataclass
class PromptGeneratorConfig:
    n_layers: int
    n_heads: int
    k: int


@dataclass
class DecoderConfig:
    name: str
    input_dim: Optional[int] = None
    prompt_generator_cfg: Optional[PromptGeneratorConfig] = None
    lora_cfg: Optional[dict] = None
    disable_dropout: bool = True


class DecoderModel(nn.Module):
    """Wrapper around a pretrained decoder model (e.g. GPT) with optional LoRA adaptation and soft prompting."""

    def __init__(self, cfg: DecoderConfig):
        super().__init__()
        self.cfg = cfg

        # Init causal LM backbone
        self.backbone = AutoModelForCausalLM.from_pretrained(cfg.name)
        try:
            self.backbone.set_attn_implementation("flash_attention_2")
        except:
            rank_zero_info(
                "Tried to use flash attention in encoder, but it is not available."
            )

        # Disable dropout in the backbone
        if self.cfg.disable_dropout:
            for module in self.backbone.modules():
                if isinstance(module, torch.nn.Dropout):
                    module.p = 0.0

        if self.cfg.lora_cfg != None:
            self.backbone = get_peft_model(
                self.backbone,
                LoraConfig(**self.cfg.lora_cfg),
            )

        # Init tokenizer, make it put BOS and EOS tokens around inputs, and add PAD and THINK tokens.
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.name)
        self.tokenizer._tokenizer.post_processor = TemplateProcessing(
            single=self.tokenizer.bos_token + " $A " + self.tokenizer.eos_token,
            special_tokens=[
                (self.tokenizer.eos_token, self.tokenizer.eos_token_id),
                (self.tokenizer.bos_token, self.tokenizer.bos_token_id),
            ],
        )
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
            self.prompt_generator["encoder"] = Encoder(
                dim=self.backbone.config.hidden_size,
                depth=cfg.prompt_generator_cfg.n_layers,
                heads=cfg.prompt_generator_cfg.n_heads,
            )
        else:
            # This means there is no z; the model is just a standard unconditional decoder
            pass

        # Whether the model is in DLC mode or not
        self.is_dlc = False

    def z_to_soft_prompt(self, z):
        # Project z to prompt space (B, D) -> (B, k * d) -> (B, k, d) -> (B, k, embd_dim)
        prompt = self.prompt_generator["z_to_chunk"](z)
        prompt = rearrange(
            prompt,
            "b 1 (k d) -> b k d",
            k=self.cfg.prompt_generator_cfg.k,
        )
        prompt = self.prompt_generator["chunk_to_embd"](prompt)
        prompt = self.prompt_generator["encoder"](prompt)
        return prompt

    def compile(self):
        # Only compile GPT2 backbone
        self.backbone.compile()

    def forward(
        self,
        input_ids,
        z: Optional[torch.Tensor] = None,
    ):
        if z == None:
            assert hasattr(self, "prompt_generator") == False
            return self.backbone(input_ids=input_ids).logits
        else:
            # Compute input embeddings
            prompt = self.z_to_soft_prompt(z)
            input_embeds = self.backbone.get_input_embeddings()(input_ids)
            input_embeds = torch.cat([prompt, input_embeds], dim=1)

            # Forward pass
            return self.backbone(inputs_embeds=input_embeds).logits[
                :, prompt.shape[1] :
            ]

    @torch.inference_mode()
    def generate(
        self,
        max_length: int,
        z: Optional[torch.Tensor] = None,
        dlc: Optional[torch.Tensor] = None,
        dlc_len: Optional[int] = None,
        batch_size: Optional[int] = None,
        gen_kwargs: Optional[dict] = None,
        gen_kwargs_dlc: Optional[dict] = None,
    ):
        """Generate text using autoregressive decoding, potentially conditioned on soft prefix z"""
        device = next(self.parameters()).device
        # If in DLC mode
        if self.is_dlc:
            assert z is None
            batch_size = batch_size if dlc is None else dlc.shape[0]
            think_token = torch.full(
                size=(batch_size, 1),
                fill_value=self.tokenizer.think_token_id,
                dtype=torch.long,
                device=device,
            )
            if dlc is None:
                # If no DLC is provided, generate it
                assert dlc_len is not None
                gen_cfg_dlc = {
                    "max_new_tokens": dlc_len + 1,  # 32-token DLC + closing <|bos|>
                    "do_sample": True,
                    "top_p": 1.0,
                    "top_k": 50,
                    "num_beams": 1,
                    "temperature": 1.0,
                    "return_dict_in_generate": True,
                    "pad_token_id": self.tokenizer.pad_token_id,
                    "eos_token_id": self.tokenizer.bos_token_id,
                    "use_cache": True

                }
                if gen_kwargs_dlc is not None:
                    gen_cfg_dlc.update(gen_kwargs_dlc)
                
                dlc = self.backbone.generate(
                    input_ids=think_token,
                    generation_config=GenerationConfig(**gen_cfg_dlc)
                ).sequences

                # Remove closing <|bos|>, will be re-added for continuation generation
                dlc = dlc[:, :-1]
            else:
                # Prepend <|think|> token
                dlc = torch.cat([think_token, dlc], dim=1)
            cache = self.backbone(
                input_ids=dlc,
                use_cache=True,
            ).past_key_values
            cache_position = torch.tensor([dlc.shape[1]], device=device)
            attention_mask = torch.ones((dlc.shape[0], dlc.shape[1] + 1), device=device)
        # If generating from latent (for auto-encoding phase)
        elif z != None:
            soft_prompt = self.z_to_soft_prompt(z)
            cache = self.backbone(
                inputs_embeds=soft_prompt,
                use_cache=True,
            ).past_key_values
            batch_size = z.shape[0]
            cache_position = torch.tensor([soft_prompt.shape[1]], device=device)
            attention_mask = torch.ones(
                (soft_prompt.shape[0], soft_prompt.shape[1] + 1), device=device
            )
        # If generating unconditionally (for baseline)
        else:
            assert (
                batch_size is not None
            ), "Must provide batch_size if no conditionning is given"
            cache = None
            cache_position = None
            attention_mask = None

        # Autoregressive generation from <|BOS|> token (maybe with cached prefix)
        bos = torch.full(
            (batch_size, 1),
            self.tokenizer.bos_token_id,
            device=device,
            dtype=torch.long,
        )
        gen_cfg = {
            "max_new_tokens": max_length,  # 32-token DLC + closing <|bos|>
            "do_sample": True,
            "top_p": 1.0,
            "top_k": 50,
            "num_beams": 1,
            "temperature": 1.0,
            "return_dict_in_generate": True,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "use_cache": True

        }
        if gen_kwargs is not None:
            gen_cfg.update(gen_kwargs)
        output = self.backbone.generate(
            input_ids=bos,
            generation_config=GenerationConfig(**gen_cfg),
            past_key_values=cache,
            cache_position=cache_position,
            attention_mask=attention_mask
        ).sequences

        return output
