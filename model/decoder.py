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

class DecoderModel(nn.Module):

    def __init__(self, cfg: DecoderConfig):
        super().__init__()
        self.cfg = cfg

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
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.name)
        self.tokenizer: GPT2TokenizerFast
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

        # Whether the model is in DLC mode or not
        self.is_dlc = False

    @property
    def dim(self):
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
        if z == None:
            assert hasattr(self, "prompt_generator") == False
            return self.backbone(
                input_ids=input_ids, attention_mask=attention_mask
            ).logits
        else:
            # Compute input embeddings
            return self.backbone(
                input_ids=input_ids,
                encoder_hidden_states=z,
            ).logits

    @torch.inference_mode()
    def generate(
        self,
        max_length: int,
        z: Optional[torch.Tensor] = None,
        prefix: Optional[List[int]] = None,
        dlc: Optional[torch.Tensor] = None,
        gen_dlc_len: Optional[int] = None,
        batch_size: Optional[int] = None,
        gen_kwargs: Optional[dict] = None,
        gen_kwargs_dlc: Optional[dict] = None,
    ):
        """
        Generate text, maybe along with DLC, maybe conditionned on a prompt
        Format of the generation is : prefix <|think|> DLC <|bos|> suffix
        """
        device = next(self.parameters()).device

        # If in DLC mode
        if self.is_dlc:
            assert z is None
            batch_size = batch_size if prefix is None else len(prefix)

            # if no prompt, set it to <|think|>
            if prefix is None:
                prefix = (
                    torch.full(
                        size=(batch_size, 1),
                        fill_value=self.tokenizer.think_token_id,
                        dtype=torch.long,
                        device=device,
                    )
                    if prefix is None
                    else prefix
                )
                prompt_mask = None
            else:
                prefix = self.tokenizer.pad(
                    {"input_ids": prefix},
                    padding=True,
                    padding_side="left",
                    return_tensors="pt",
                ).to(device=device)
                prefix, prompt_mask = prefix["input_ids"], prefix["attention_mask"]

            # if no DLC, generate it
            if dlc is None:
                assert gen_dlc_len is not None
                gen_cfg_dlc = {
                    "max_new_tokens": gen_dlc_len,
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
                    input_ids=prefix,
                    attention_mask=prompt_mask,
                    generation_config=GenerationConfig(**gen_cfg_dlc),
                )[:, prefix.shape[1] :]

            # Append DLC after prompt => ... <|think|> DLC
            prefix = torch.cat([prefix, dlc], dim=1)
            prompt_mask = (
                torch.ones_like(prefix, dtype=torch.bool)
                if prompt_mask is None
                else torch.cat(
                    [prompt_mask, torch.ones_like(dlc, dtype=torch.bool)], dim=1
                )
            )

            bos = torch.full(
                (batch_size, 1),
                self.tokenizer.bos_token_id,
                device=device,
                dtype=torch.long,
            )
            input_ids = torch.cat([prefix, bos])
            attention_mask = torch.cat(
                [prompt_mask, torch.ones_like(prompt_mask[:, [0]])], dim=1
            )

        # If generating from continuous latent (for auto-encoder)
        elif z != None:
            batch_size = z.shape[0]
            attention_mask = None
            input_ids = torch.full(
                (batch_size, 1),
                self.tokenizer.bos_token_id,
                device=device,
                dtype=torch.long,
            )
        # If generating unconditionally (for baseline)
        else:
            if prefix is not None:
                prefix = self.tokenizer.pad(
                    {"input_ids": prefix},
                    padding=True,
                    padding_side="left",
                    return_tensors="pt",
                ).to(device=device)
                input_ids, attention_mask = (
                    prefix["input_ids"],
                    prefix["attention_mask"],
                )
            else:
                input_ids = torch.full(
                    (batch_size, 1),
                    self.tokenizer.bos_token_id,
                    device=device,
                    dtype=torch.long,
                )
                attention_mask = None

        gen_cfg = {
            "max_new_tokens": max_length,
            "do_sample": True,
            "top_p": 1.0,
            "top_k": 50,
            "num_beams": 1,
            "temperature": 1.0,
            "return_dict_in_generate": True,
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
        ).sequences

        # Remove the input_ids prefix
        output = output[:, input_ids.shape[1] :]

        return output
