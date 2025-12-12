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
from transformers import GenerationConfig
from copy import deepcopy

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
    disable_dropout: bool = True


class DecoderModel(nn.Module):

    def __init__(self, cfg: DecoderConfig):
        super().__init__()
        self.cfg = cfg

        # Init causal LM backbone
        self.backbone = AutoModelForCausalLM.from_pretrained(cfg.name)
        try:
            self.backbone.set_attn_implementation("sdpa")
        except:
            rank_zero_info(
                "Tried to use flash attention in decoder, but it is not available."
            )

        # Disable dropout in the backbone
        if self.cfg.disable_dropout:
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
            "b (k d) -> b k d",
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
        attention_mask : Optional[torch.Tensor] = None
    ):
        if z == None:
            assert hasattr(self, "prompt_generator") == False
            return self.backbone(input_ids=input_ids, attention_mask=attention_mask).logits
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
        prompt: Optional[List[int]] = None,
        dlc: Optional[torch.Tensor] = None,
        gen_dlc_len: Optional[int] = None,
        batch_size: Optional[int] = None,
        gen_kwargs: Optional[dict] = None,
        gen_kwargs_dlc: Optional[dict] = None,
    ):
        """
        Generate text, maybe along with DLC, maybe conditionned on a prompt
        Format of the generation is : prompt <|think|> DLC <|bos|> text
        """
        device = next(self.parameters()).device

        # If in DLC mode
        if self.is_dlc:
            assert z is None
            batch_size = batch_size if prompt is None else len(prompt)

            # if not prompt, set it to <|think|>
            if prompt is None:
                prompt = (
                    torch.full(
                        size=(batch_size, 1),
                        fill_value=self.tokenizer.think_token_id,
                        dtype=torch.long,
                        device=device,
                    )
                    if prompt is None
                    else prompt
                )
                prompt_mask = None
            else:
                prompt = self.tokenizer.pad(
                    {"input_ids": prompt},
                    padding=True,
                    padding_side="left",
                    return_tensors="pt",
                ).to(device=device)
                prompt, prompt_mask = prompt["input_ids"], prompt["attention_mask"]

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
                    input_ids=prompt,
                    attention_mask=prompt_mask,
                    generation_config=GenerationConfig(**gen_cfg_dlc),
                )[:, prompt.shape[1]:]

            # Append DLC after prompt => ... <|think|> DLC
            prompt = torch.cat([prompt, dlc], dim=1)
            prompt_mask = (
                torch.ones_like(prompt, dtype=torch.bool)
                if prompt_mask is None
                else torch.cat(
                    [prompt_mask, torch.ones_like(dlc, dtype=torch.bool)], dim=1
                )
            )

            # Compute the KV cache of the prompt, along with import kwargs for generate
            cache = self.backbone(
                input_ids=prompt,
                attention_mask=prompt_mask,
                position_ids=(prompt_mask.cumsum(dim=1) - 1).masked_fill(prompt_mask == 0, 1),
                use_cache=True,
            ).past_key_values
            cache_position = torch.tensor([prompt_mask.shape[1]], device=device)
            attention_mask = torch.cat(
                [prompt_mask, torch.ones_like(prompt_mask[:, [0]])], dim=1
            )
            input_ids = torch.full(
                (batch_size, 1),
                self.tokenizer.bos_token_id,
                device=device,
                dtype=torch.long,
            )

        # If generating from continuous latent (for auto-encoder)
        elif z != None:
            batch_size = z.shape[0]
            soft_prompt = self.z_to_soft_prompt(z)
            cache = self.backbone(
                inputs_embeds=soft_prompt,
                use_cache=True,
            ).past_key_values
            cache_position = torch.tensor([soft_prompt.shape[1]], device=device)
            attention_mask = torch.ones(
                (soft_prompt.shape[0], soft_prompt.shape[1] + 1),
                device=device,
                dtype=torch.long,
            )
            input_ids = torch.full(
                (batch_size, 1),
                self.tokenizer.bos_token_id,
                device=device,
                dtype=torch.long,
            )
        # If generating unconditionally (for baseline)
        else:
            if prompt is not None:
                prompt = self.tokenizer.pad(
                    {"input_ids": prompt},
                    padding=True,
                    padding_side="left",
                    return_tensors="pt",
                ).to(device=device)
                input_ids, attention_mask = (
                    prompt["input_ids"],
                    prompt["attention_mask"],
                )
            else:
                input_ids = torch.full(
                    (batch_size, 1),
                    self.tokenizer.bos_token_id,
                    device=device,
                    dtype=torch.long,
                )
                attention_mask = None

            cache = None
            cache_position = None

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
            past_key_values=deepcopy(cache),
            cache_position=cache_position,
            attention_mask=attention_mask,
        ).sequences

        # Remove the input_ids prefix
        output = output[:, input_ids.shape[1] :]

        return output
