from abc import ABC, abstractmethod
import os
from peft.mapping_func import get_peft_model
from peft.tuners.lora.config import LoraConfig
import torch.nn as nn
from dataclasses import dataclass
from typing import Optional
from sentence_transformers import SentenceTransformer
from transformers.models.auto.modeling_auto import AutoModelForCausalLM, AutoModel
from transformers.models.auto.tokenization_auto import AutoTokenizer
import torch


@dataclass
class EncoderConfig:
    name: str
    k: int
    lora_cfg: Optional[dict] = None


class EncoderModel(nn.Module):
    def __init__(self, cfg: EncoderConfig):
        super().__init__()
        self.cfg = cfg

        # Init backbone
        self.backbone = AutoModel.from_pretrained(
            cfg.name, device_map="auto"
        )
        if self.cfg.lora_cfg != None:
            self.backbone = get_peft_model(
                self.backbone,
                LoraConfig(**self.cfg.lora_cfg),
            )

        self.tokenizer = AutoTokenizer.from_pretrained(cfg.name)

    def forward(self, input_ids, attention_mask=None):
        
        # Run through backbone and get first k tokens
        z = self.backbone(input_ids, attention_mask=attention_mask)
        z = z["last_hidden_state"][:, : self.cfg.k, :]

        return z

        

