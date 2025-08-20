from abc import ABC, abstractmethod
import os
import torch.nn as nn
from dataclasses import dataclass
from typing import Optional
from sentence_transformers import SentenceTransformer
from transformers.models.auto.modeling_auto import AutoModelForCausalLM
from transformers.models.auto.tokenization_auto import AutoTokenizer
from x_transformers import Encoder
import torch


@dataclass
class SentenceEmbeddingEncoderConfig:
    name: str
    depth: int
    heads: int
    out_dim: Optional[int] = None
    input_dim: Optional[int] = None
    precomputed_path: Optional[str] = None


class SentenceEmbeddingEncoder(nn.Module):
    def __init__(self, cfg: SentenceEmbeddingEncoderConfig):
        super().__init__()

        self.backbone = SentenceTransformer(cfg.name)
        self.encoder = Encoder(
            dim=cfg.input_dim // 2,
            depth=cfg.depth,
            heads=cfg.heads,
        )
        self.in_proj = nn.Linear(cfg.input_dim, 4 * cfg.input_dim)
        self.out_proj = nn.Linear(cfg.input_dim // 2, cfg.out_dim)

        if cfg.precomputed_path is not None and os.path.exists(cfg.precomputed_path):
            self.sent_embs = torch.load(cfg.precomputed_path)

    def forward(self, x_dict):
        if hasattr(self, "sent_embs"):
            z = self.sent_embs[x_dict["indices"]]
        else:
            z = self.backbone.encode(
                x_dict["sentences"], convert_to_tensor=True, show_progress_bar=False
            )
        z = self.in_proj(z)
        z = self.encoder(z)
        z = self.out_proj(z)
        return z, None


@dataclass
class VariationalEncoderConfig:
    name: str
    k: int
    p: float
    beta: float


class VariationalEncoder(nn.Module):
    def __init__(self, cfg: VariationalEncoderConfig):
        super().__init__()
        self.cfg = cfg
        self.backbone = AutoModelForCausalLM.from_pretrained(
            cfg.name, device_map="auto"
        )
        tokenizer = AutoTokenizer.from_pretrained(cfg.name, add_bos_token=False)
        self.vocab_size = len(tokenizer)
        self.latent_size = self.backbone.config.hidden_size
        self.fc_mean = torch.nn.Linear(self.latent_size, self.latent_size)
        self.fc_log_var = torch.nn.Linear(self.latent_size, self.latent_size)

    def _noise(self, inputs):

        probability = torch.full(
            inputs.shape, self.cfg.p, dtype=torch.float32, device=inputs.device
        )

        masked_indices = torch.bernoulli(probability).bool()
        random_words = torch.randint(
            self.vocab_size, inputs.shape, dtype=torch.long
        )
        inputs[masked_indices] = random_words[masked_indices]

        return inputs

    def reparameterize(self, mean, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mean)

    def forward(self, x):
        x = self._noise(x)
        x = self.backbone(x)
        x = x["last_hidden_state"][:, : self.cfg.k, :]
        x = x.permute(0, 2, 1)
        mean = self.fc_mean(x)
        log_var = self.fc_log_var(x)

        KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
        z = self.reparameterize(mean, log_var)

        return z, self.cfg.beta * KLD
