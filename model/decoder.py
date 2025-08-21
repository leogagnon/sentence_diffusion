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
            cfg.name, device_map="auto"
        )
        if self.cfg.lora_cfg != None:
            self.backbone = get_peft_model(
                self.backbone,
                LoraConfig(**self.cfg.lora_cfg),
            )
        self.bos_token = AutoTokenizer.from_pretrained(cfg.name).bos_token_id

    def forward(self, input_ids, z, attention_mask=None):

        # Prepend BOS
        input_ids = torch.cat(
            [
                torch.full_like(
                    input_ids[:, [0]],
                    self.bos_token,
                ),
                input_ids,
            ],
            dim=1,
        )

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
        logits = output.logits[:, z.shape[1] : -1, :]

        return logits.contiguous()
