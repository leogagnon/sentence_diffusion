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


@dataclass
class DecoderConfig:
    name: str
    lora_cfg: Optional[dict] = None


class DecoderModel(nn.Module):
    def __init__(self, cfg: DecoderConfig):
        super().__init__()
        self.cfg = cfg

        self.backbone = AutoModelForCausalLM.from_pretrained(
            cfg.name,
            device_map="auto",
            attn_implementation="flash_attention_2",
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

    def forward(self, input_ids, z, attention_mask=None):

        # Compute tokens
        tokens = self.backbone.get_input_embeddings()(input_ids)
        tokens = torch.cat([z, tokens], dim=1)

        # Update attention_mask (bos and z)
        attention_mask = torch.cat(
            [
                torch.ones((z.shape[0], z.shape[1] + 1), device=tokens.device),
                attention_mask,
            ],
            dim=1,
        )
        self.backbone.get_input_embeddings()
        # Forward pass
        output = self.backbone(inputs_embeds=tokens)
        logits = output.logits[:, z.shape[1] :]

        return logits

    @torch.no_grad()
    def generate_from(self, z, max_length):
        """Generate text from latent code z using autoregressive decoding."""

        # Get cache from z
        prefill = self.backbone(
            inputs_embeds=z,
            use_cache=True,
        )
        cache = prefill.past_key_values

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
            cache_position=torch.tensor([z.shape[1]]),
            attention_mask=torch.ones(
                z.shape[0], z.shape[1] + 1, device=z.device, dtype=torch.long
            ),
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
