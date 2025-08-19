from functools import partial
import lightning as L
from omegaconf import OmegaConf
from dataclasses import dataclass
from typing import Any, List, Optional
from peft import LoraConfig
import torch
import random
import numpy as np
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from transformers.models.auto.modeling_auto import AutoModelForCausalLM
from peft import get_peft_model
from model.diffusion_transformer import DiTConfig, DiT
from model.prompt_generator import PromptGenerator, PromptGeneratorConfig
from data import StoriesDatasetConfig, StoriesDataset
from sentence_transformers import SentenceTransformer
import einx


@dataclass
class DecoderTaskConfig:
    n_samples: int
    lr: float
    lm_name: str
    semb_name: str
    lora_config: dict
    dataset: StoriesDatasetConfig

    prompt_generator: PromptGeneratorConfig


class DecoderTask(L.LightningModule):
    def __init__(self, cfg: Optional[DecoderTaskConfig] = None, **kwargs):
        super().__init__()

        if cfg == None:
            cfg = OmegaConf.to_object(
                OmegaConf.merge(
                    OmegaConf.create(DecoderTaskConfig),
                    OmegaConf.create(kwargs),
                )
            )

        self.cfg = cfg

        # Load language model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.lm_name, add_bos_token=False)
        model = AutoModelForCausalLM.from_pretrained(cfg.lm_name, device_map="auto")
        self.model = get_peft_model(model, LoraConfig(**cfg.lora_config))
        self.bos_token = self.tokenizer.bos_token_id

        # Load sentence embedding model
        self.semb = SentenceTransformer(cfg.semb_name)

        # Load prompt generator
        self.prompt_generator = PromptGenerator(cfg.prompt_generator)

        # Remove dropout
        for mod in self.model.modules():
            if isinstance(mod, torch.nn.Dropout):
                mod.p = 0.0

        self.dataset = StoriesDataset(cfg.dataset, self.tokenizer, self.semb)

    def train_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.cfg.batch_size, shuffle=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.cfg.lr)
        return optimizer

    def training_step(self, batch, batch_idx):

        # Get soft prompt
        z = self.prompt_generator(batch["input_emb"])

        # Get embeddings of input_ids
        input_ids = torch.cat(
            [
                torch.full_like(
                    batch["input_ids"][:, [0]],
                    self.bos_token,
                ),
                batch["input_ids"],
            ],
            dim=1,
        )
        tokens = self.model.get_input_embeddings()(input_ids)
        tokens = torch.cat([z, tokens], dim=1)

        # Forward pass
        logits = self.model(input_embeds=tokens)
        logits = logits.logits[:, z.shape[1] :, :]

        # Apply cross-entropy loss
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)), batch["input_ids"].view(-1)
        )
        return loss
