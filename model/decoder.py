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
        self.tokenizer.add_special_tokens({"additional_special_tokens": ["<|think|>"]})
        self.tokenizer.think_token_id = self.tokenizer.convert_tokens_to_ids(
            "<|think|>"
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
        prompt: Optional[torch.Tensor] = None,
        batch_size: Optional[int] = None,
        is_dlc: bool = False,
        dlc_len: Optional[int] = None,
    ):
        """Generate text using autoregressive decoding, potentially conditioned on soft prefix z"""
        # Maybe compute prompt and cache it
        if is_dlc:
            assert z is None
            if prompt is None:
                bot = torch.full(
                    size=(batch_size, 1),
                    fill_value=self.tokenizer.think_token_id,
                    dtype=torch.long,
                ).cuda()
                batch_size = None
                prompt = self.backbone.generate(
                    input_ids=bot,
                    max_new_tokens=dlc_len + 1,  # 32-token DLC + closing <|think|>
                    do_sample=True,
                    top_p=0.92,
                    top_k=50,
                    num_beams=1,
                    temperature=0.9,
                    return_dict_in_generate=False,
                    use_cache=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.think_token_id,
                )
                prompt = prompt[
                    :, :-1
                ]  # Remove closing <|think|>, will be re-added as BOS for next phase
                cache = self.backbone(
                    input_ids=prompt,
                    use_cache=True,
                ).past_key_values
                device = prompt.device
                cache_position = torch.tensor([prompt.shape[1]], device=prompt.device)
                attention_mask = torch.ones(
                    (prompt.shape[0], prompt.shape[1] + 1), device=device
                )
        elif z != None:
            assert prompt is None, "Cannot provide both z and prompt"
            prompt = self.z_to_soft_prompt(z)
            cache = self.backbone(
                inputs_embeds=prompt,
                use_cache=True,
            ).past_key_values
            device = prompt.device
            batch_size = z.shape[0]
            cache_position = torch.tensor([prompt.shape[1]], device=prompt.device)
            attention_mask = torch.ones(
                (prompt.shape[0], prompt.shape[1] + 1), device=device
            )
        else:
            cache = None
            cache_position = None
            attention_mask = None
            device = next(self.backbone.parameters()).device
            assert (
                batch_size is not None
            ), "Must provide batch_size if no conditionning is given"

        # Autoregressive generation from token (maybe with cached prompt)
        # Start with the new <|think|> token if conditionning on a DLC
        bos_token_id = (
            self.tokenizer.think_token_id if is_dlc else self.tokenizer.bos_token_id
        )
        bos = torch.full(
            (batch_size, 1),
            bos_token_id,
            device=device,
            dtype=torch.long,
        )
        output = self.backbone.generate(
            input_ids=bos,
            past_key_values=cache,
            cache_position=cache_position,
            attention_mask=attention_mask,
            max_new_tokens=max_length,
            do_sample=True,
            top_p=0.92,
            top_k=50,
            num_beams=1,
            temperature=0.9,
            return_dict_in_generate=False,
            use_cache=True,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        return output
