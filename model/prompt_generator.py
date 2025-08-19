from dataclasses import dataclass
from x_transformers import Encoder
import torch.nn as nn
import torch


@dataclass
class PromptGeneratorConfig:
    dim: int
    depth: int
    heads: int


class PromptGenerator(nn.Module):
    def __init__(self, cfg: PromptGeneratorConfig):
        super().__init__()
        self.cfg = cfg
        self.encoder = Encoder(
            dim=cfg.dim,
            depth=cfg.depth,
            heads=cfg.heads,
        )
        self.proj = nn.Linear(cfg.dim, 4 * cfg.dim)

    def forward(self, z):
        z = self.proj(z)
        z = torch.chunk(z, 8, dim=-1)
        z = self.encoder(z)

        return z
