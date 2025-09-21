from abc import ABC, abstractmethod
import os
from peft.mapping_func import get_peft_model
from peft.tuners.lora.config import LoraConfig
import torch.nn as nn
from dataclasses import dataclass
from typing import Optional
from sentence_transformers import SentenceTransformer
from transformers.models.auto.modeling_auto import AutoModelForCausalLM
from transformers.models.auto.tokenization_auto import AutoTokenizer
from x_transformers import Encoder
import torch
from typing import Optional
from einops import rearrange, repeat


@dataclass
class DecoderConfig:
    name: str
    lora_cfg: Optional[dict] = None


class DecoderModel(nn.Module):
    """Wrapper around a pretrained decoder model (e.g. GPT) with optional LoRA adaptation and soft prompting."""

    def __init__(self, cfg: DecoderConfig):
        super().__init__()
        self.cfg = cfg

        self.backbone = AutoModelForCausalLM.from_pretrained(
            cfg.name,
            attn_implementation="flash_attention_2",
            torch_dtype=torch.float16,
        )
        if self.cfg.lora_cfg != None:
            self.backbone = get_peft_model(
                self.backbone,
                LoraConfig(**self.cfg.lora_cfg),
            )

        # Init tokenizer and add pad token if missing
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.name)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})
            self.backbone.resize_token_embeddings(len(self.tokenizer))

    def forward(self, input_ids, z: Optional[torch.Tensor] = None):
        if z == None:
            return self.backbone(input_ids=input_ids).logits
        else:
            # compute input embeddings
            input_embeds = self.backbone.get_input_embeddings()(input_ids)
            input_embeds = torch.cat([z, input_embeds], dim=1)

            # Forward pass
            output = self.backbone(
                inputs_embeds=input_embeds
            )
            logits = output.logits[:, z.shape[1] :]

        return logits

    @torch.no_grad()
    def generate(self, max_length: int, z: Optional[torch.Tensor] = None):
        """Generate text using autoregressive decoding, potentially conditioned on soft prefix z"""

        if z != None:
            # Compute cache for z
            cache = self.backbone(
                inputs_embeds=z,
                use_cache=True,
            ).past_key_values
            # Position of the BOS should be after the prefix
            cache_position = torch.tensor([z.shape[1]])
        else:
            cache = None
            cache_position = None

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
