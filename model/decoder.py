from abc import ABC, abstractmethod
from functools import partial
import math
import os
import einx
from peft.mapping_func import get_peft_model
import torch.nn as nn
from dataclasses import dataclass
from typing import Optional
from sentence_transformers import SentenceTransformer
from transformers.models.auto.modeling_auto import AutoModelForCausalLM
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers import GPT2TokenizerFast
from x_transformers import Encoder
from x_transformers.x_transformers import AttentionLayers, ScaledSinusoidalEmbedding
import torch
from typing import Optional, Union, List
from einops import rearrange
from torch.nn import ModuleDict
from tokenizers.processors import TemplateProcessing
from lightning.pytorch.utilities.rank_zero import rank_zero_info
from transformers import GenerationConfig, GPT2LMHeadModel, GPT2Config
from copy import deepcopy


@dataclass
class DecoderConfig:
    name: str
    cross_attention: bool = False
    dlc_vocab_size: int = 0
    dlc_len: Optional[int] = None


class DecoderModel(nn.Module):

    def __init__(self, cfg: DecoderConfig):
        super().__init__()

        # Init causal LM backbone
        if cfg.cross_attention:
            self.backbone = GPT2LMHeadModel.from_pretrained(
                cfg.name,
                config=GPT2Config.from_pretrained(
                    cfg.name, add_cross_attention=True, is_decoder=True
                ),
            )
        else:
            self.backbone = AutoModelForCausalLM.from_pretrained(cfg.name)

        # Use better attention implementation
        try:
            self.backbone.set_attn_implementation("sdpa")
        except:
            rank_zero_info(
                "Tried to use SDPA attention in decoder, but it is not available."
            )

        # Disable dropout in the backbone
        for module in self.backbone.modules():
            if isinstance(module, torch.nn.Dropout):
                module.p = 0.0

        # Init tokenizer, make it put BOS and EOS tokens around inputs, and add PAD and THINK tokens.
        self.tokenizer = GPT2TokenizerFast.from_pretrained(cfg.name)
        
        # Add PAD and THINK tokens, resize embeddings
        self.tokenizer.add_special_tokens(
            {"pad_token": "<|pad|>", "additional_special_tokens": ["<|think|>"]}
        )
        self.tokenizer.think_token_id = self.tokenizer.convert_tokens_to_ids(
            "<|think|>"
        )

        # Resize token embeddings to account for new tokens
        # If using DLC, we add dlc_vocab_size tokens on top of that
        self.backbone.resize_token_embeddings(
            len(self.tokenizer) + cfg.dlc_vocab_size, mean_resizing=True
        )
        if cfg.dlc_vocab_size > 0:
            assert cfg.dlc_len is not None, "If using DLC, dlc_len must be specified"

        self.cfg = cfg

    @property
    def latent_dim(self):
        return self.backbone.get_input_embeddings().weight.shape[1]

    def compile(self):
        # Only compile GPT2 backbone
        self.backbone.compile()

    def forward(
        self,
        input_ids,
        attention_mask: Optional[torch.Tensor] = None,
        z: Optional[torch.Tensor] = None,
    ):
        return self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            encoder_hidden_states=z,
        ).logits

    @torch.inference_mode()
    def generate(
        self,
        max_length: int,
        z: Optional[torch.Tensor] = None,
        prefix: Optional[List[List[int]]] = None,
        dlc: Optional[List[List[int]]] = None,
        batch_size: Optional[int] = None,
        gen_kwargs: Optional[dict] = None,
        gen_kwargs_dlc: Optional[dict] = None,
    ):
        """
        Generate text, maybe along with DLC, maybe conditionned on a prompt
        Format of the generation is : prefix <|think|> DLC <|bos|> suffix
        """
        device = next(self.parameters()).device
        
        # Process prefix if provided
        if prefix is not None:
            assert isinstance(prefix, list), "Prefix should be a list of input_ids"
            # NOTE: We use left padding for generation
            prefix = self.tokenizer.pad(
                {"input_ids": prefix},
                padding=True,
                padding_side="left",
                return_tensors="pt",
            ).to(device=device)
            prefix_ids, prefix_attention_mask = (
                prefix["input_ids"],
                prefix["attention_mask"],
            )
            if batch_size is None:
                batch_size = prefix_ids.shape[0]

        if self.cfg.dlc_vocab_size > 0:
            # This means we are in DLC generation mode, no z conditioning
            assert z is None

            # Append <|think|> token to the prefix
            think_token = torch.full(
                size=(batch_size, 1),
                fill_value=self.tokenizer.think_token_id,
                dtype=torch.long,
                device=device,
            )
            if prefix is not None:
                input_ids = torch.cat([prefix_ids, think_token], dim=1)
                attention_mask = torch.cat(
                    [
                        prefix_attention_mask,
                        torch.ones_like(think_token, dtype=torch.bool),
                    ],
                    dim=1,
                )
            else:
                input_ids = think_token
                attention_mask = torch.ones_like(think_token, dtype=torch.bool)

            # Generate DLC if not provided
            if dlc is None:
                gen_cfg_dlc = {
                    "max_new_tokens": self.cfg.dlc_len,
                    "do_sample": True,
                    "top_p": 1.0,
                    "top_k": 50,
                    "num_beams": 1,
                    "temperature": 1.0,
                    "pad_token_id": self.tokenizer.pad_token_id,
                    "eos_token_id": self.tokenizer.bos_token_id,
                    "use_cache": False,
                }
                if gen_kwargs_dlc is not None:
                    gen_cfg_dlc.update(gen_kwargs_dlc)

                dlc = self.backbone.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    generation_config=GenerationConfig(**gen_cfg_dlc),
                )[:, -self.cfg.dlc_len :]
            else:
                dlc = torch.LongTensor(dlc).to(device=device)
                assert dlc.shape[1] == self.cfg.dlc_len

            # Append "DLC <|think|>"" to input_ids and update mask
            input_ids = torch.cat([input_ids, dlc, think_token], dim=1)
            attention_mask = torch.cat(
                [
                    attention_mask,
                    torch.ones(
                        (batch_size, dlc.shape[1] + 1), dtype=torch.bool, device=device
                    ),
                ],
                dim=1,
            )
            # NOTE: Final input_ids are : prefix <|think|> DLC <|think|>
        else:
            # This means we are in normal generation mode, maybe with z conditioning
            assert (
                prefix is not None
            ), "Right now, generation without DLC must be conditionned on a prefix"
            input_ids = prefix_ids
            attention_mask = prefix_attention_mask
            # NOTE: Final input_ids are : prefix

        gen_cfg = {
            "max_new_tokens": max_length,
            "do_sample": True,
            "top_p": 1.0,
            "top_k": 50,
            "num_beams": 1,
            "temperature": 1.0,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "use_cache": True,
        }
        if gen_kwargs is not None:
            gen_cfg.update(gen_kwargs)

        output = self.backbone.generate(
            input_ids=input_ids,
            generation_config=GenerationConfig(**gen_cfg),
            encoder_hidden_states=z,
            attention_mask=attention_mask,
        )

        # Remove the input_ids prefix
        output = output[:, input_ids.shape[1] :]

        return output
