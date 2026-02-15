from dataclasses import dataclass
from typing import Optional

import einx
import hydra
import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import AutoModel, AutoModelForMaskedLM, AutoTokenizer
from transformers.models.m2m_100.modeling_m2m_100 import M2M100Encoder


@dataclass
class SEMHeadConfig:
    L: int
    V: int
    temp: float
    input_dim: Optional[int] = None


class SEMHead(nn.Module):
    def __init__(self, cfg: Optional[SEMHeadConfig] = None, **kwargs):
        super().__init__()

        if cfg == None:
            cfg = SEMHeadConfig(**kwargs)

        assert cfg.input_dim is not None, "input_dim has to be set"
        self.proj_in = nn.Linear(cfg.input_dim, cfg.L * cfg.V, bias=False)
        self.norm = nn.LayerNorm((cfg.L, cfg.V))

        self.cfg = cfg

    @property
    def out_dim(self):
        return self.cfg.L * self.cfg.V

    @property
    def dlc_len(self):
        return self.cfg.L

    def forward(
        self,
        x,
        return_dlc=False,
        return_count=False,
        noise: float = 0.0,
        temp: Optional[float] = None,
    ) -> dict:

        # Proj in DLC space and normalize
        x = self.proj_in(x)
        x = einx.rearrange("b (l v) -> b l v", x, l=self.cfg.L, v=self.cfg.V)
        x = self.norm(x)

        # Compute Softmax (with temperature)
        temp = self.cfg.temp if temp is None else temp
        probs = torch.softmax(x / temp, dim=-1)

        # Maybe add noise
        z = einx.rearrange("b l v -> b (l v)", probs)
        if noise > 0.0:
            z = z + noise * torch.randn_like(z)

        # Return stuff
        out_dict = {"z": z, "probs": probs}
        if return_dlc:
            out_dict.update({"dlc": self._encode(probs)})
        if return_count:
            out_dict.update({"usage_count": self._usage_count(probs)})

        return out_dict

    def _encode(self, probs):
        # DLC = argmax of each simplex
        dlc = probs.argmax(-1)
        return dlc

    def _usage_count(self, probs):
        # Count how many times each DLC word was used
        counts = torch.stack(
            [torch.sum(probs.argmax(-1) == i, dim=0) for i in range(self.cfg.V)], dim=-1
        )
        return counts


@dataclass
class HSEMHeadConfig:
    L: int
    V: int
    D: int
    temp: float
    input_dim: Optional[int] = None
    per_simpex_ln: bool = False
    ln: bool = True
    out_normalization: str = "none"


class HSEMHead(nn.Module):
    def __init__(self, cfg: Optional[HSEMHeadConfig] = None, **kwargs):
        super().__init__()

        if cfg == None:
            cfg = HSEMHeadConfig(**kwargs)

        assert cfg.input_dim is not None, "input_dim has to be set"
        self.N = (cfg.V**cfg.D) // (cfg.V - 1)
        self.proj_in = nn.Linear(cfg.input_dim, cfg.L * self.N * cfg.V, bias=False)
        if cfg.ln:
            if cfg.per_simpex_ln:
                self.norm = nn.LayerNorm(
                    (cfg.V,),
                )
            else:
                self.norm = nn.LayerNorm((cfg.L, self.N, cfg.V))
        else:
            self.norm = nn.Identity()

        self.latent_len = cfg.D
        self.proj_out = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        nn.Linear(cfg.L * (cfg.V**l) * cfg.V, cfg.input_dim, bias=False)
                        for l in range(cfg.D)
                    ]
                )
            ]
        )
        self.cfg = cfg

    @property
    def dlc_len(self):
        return self.cfg.D * self.cfg.L

    def forward(
        self,
        x: torch.Tensor,
        return_dlc=False,
        return_count=True,
        noise: float = 0.0,
        temp: Optional[float] = None,
    ):
        bs = x.shape[0]
        temp = self.cfg.temp if temp is None else temp

        # Compute conditional probabilities
        x = self.proj_in(x)
        x = einx.rearrange(
            "b (L N V) -> b L N V",
            x,
            L=self.cfg.L,
            N=self.N,
            V=self.cfg.V,
        )
        x = self.norm(x)
        x = F.softmax(x / temp, -1)

        # Compute DLC probs (i.e. the joint) by going down tree
        # E.g. p(x_0,x_1,x_2) = p(x_0) * p(x_1 | x_0) * p(x_2 | x_0, x_1)
        with torch.autocast(device_type="cuda", dtype=torch.float32):
            parent_probs = torch.ones(
                size=(bs, self.cfg.L, 1), device=x.device, dtype=x.dtype
            )
            start = 0
            probs = []
            for d in range(self.cfg.D):
                # Compute probs at level d by multiplying with parent probs
                end = start + self.cfg.V**d
                level = x[:, :, start:end] * parent_probs[..., None]
                probs.append(einx.rearrange("b L n V -> b (L n V)", level))
                # Update parent and go down a level
                parent_probs = einx.rearrange("b L n V -> b L (n V)", level)
                start = end

        # Compute output
        out = torch.cat(
            [
                self.proj_out[i](probs[i] + noise * torch.randn_like(probs[i]))
                for i in range(len(probs))
            ],
            dim=2,
        )

        out = einx.rearrange("b L N V -> b (L N V)", out)
        out = self.proj_out(out)

        out_dict = {"c_out": out, "probs": probs}

        if return_dlc:
            out_dict.update({"dlc": self._encode(probs)})

        if return_count:
            out_dict.update({"usage_count": self._usage_count(probs)})

        return out_dict

    def _encode(self, probs):

        # Argmax on first level
        dlc = [probs[0].squeeze(2).argmax(-1)]
        for i in range(self.cfg.D - 1):
            # Get the node at level i that was chosen at level i-1
            node = torch.gather(
                input=probs[i + 1],
                dim=2,
                index=einx.rearrange("b l -> b l n v", dlc[i], n=1, v=self.cfg.V),
            ).squeeze(2)

            # Argmax at level i (on the chosen node)
            dlc.append(node.argmax(-1))

        # Concatenate all levels
        # First L indices are level 1, second L are level 2, ...
        dlc = torch.cat(dlc, dim=1)

        return dlc

    def _usage_count(self, probs):
        counts = torch.concatenate(
            [
                torch.stack(
                    [torch.sum(p.argmax(-1) == i, dim=0) for i in range(self.cfg.V)],
                    dim=-1,
                )
                for p in probs
            ],
            dim=1,
        )
        counts = einx.rearrange("L N V -> (L N V)", counts)
        return counts


@dataclass
class EncoderConfig:
    model_name: str
    latent_length: int
    latent_dim: Optional[int] = None
    sem: Optional[dict] = None
    train_backbone: bool = True
    mlp_proj: bool = False


class EncoderModel(nn.Module):
    def __init__(self, cfg: Optional[EncoderConfig] = None, **kwargs):
        super().__init__()

        assert cfg.latent_dim is not None, "latent_dim has to be set"

        if cfg == None:
            cfg = EncoderConfig(**kwargs)

        # Initialize backbone
        if "SONAR" in cfg.model_name:
            self.transformer = M2M100Encoder.from_pretrained(
                "cointegrated/SONAR_200_text_encoder"
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                "cointegrated/SONAR_200_text_encoder"
            )
            self.tokenizer.src_lang = "eng_Latn"
            backbone_dim = self.transformer.config.hidden_size
        else:
            model_kwargs = {}

            if "roberta" in cfg.model_name.lower():
                self.transformer = AutoModelForMaskedLM.from_pretrained(
                    cfg.model_name,
                    trust_remote_code=True,
                    **model_kwargs,
                )
            else:
                self.transformer = AutoModel.from_pretrained(
                    cfg.model_name,
                    trust_remote_code=True,
                    **model_kwargs,
                )

            self.tokenizer = AutoTokenizer.from_pretrained(
                cfg.model_name,
                trust_remote_code=True,
            )
            backbone_dim = self.transformer.config.hidden_size

        # Maybe freeze backbone
        self.transformer = self.transformer.train(cfg.train_backbone).requires_grad_(
            cfg.train_backbone
        )

        # Initialize SEM/output projection
        if cfg.sem is None:
            dim_0 = backbone_dim
        else:
            cfg.sem["input_dim"] = backbone_dim
            self.sem = hydra.utils.instantiate(cfg.sem)
            self.sem: SEMHead | HSEMHead
            dim_0 = self.sem.out_dim

        # Build projection head
        dim_1 = cfg.latent_length * cfg.latent_dim
        if cfg.mlp_proj:
            self.out_proj = nn.Sequential(
                nn.Linear(dim_0, 1024, bias=True),
                nn.ReLU(),
                nn.Linear(
                    1024,
                    dim_1,
                    bias=False,
                ),
            )
        else:
            self.out_proj = nn.Linear(
                dim_0,
                dim_1,
                bias=False,
            )

        self.backbone_dim = backbone_dim

        self.cfg = cfg

    def train(self, mode=True):
        super().train(mode)
        if not self.cfg.train_backbone:
            self.transformer.train(False)
        return self

    def compile(self):
        # Only compile the SEM
        if self.cfg.sem is not None:
            self.sem.compile()

    def forward(
        self,
        input_ids,
        attention_mask,
        return_dlc=False,
        return_count=False,
        noise: float = 0.0,
        temp: Optional[float] = None,
        return_logits: bool = False,
        only_backbone: bool = False,
    ):

        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        token_embeddings = outputs.hidden_states[-1]

        z = self._mean_pool(token_embeddings, attention_mask)

        if only_backbone:
            return z

        out = {}

        if return_logits:
            out["logits"] = outputs.logits

        # Run through SEM / output projection
        if self.cfg.sem is None:
            out["latent"] = z.clone()
            z = self.out_proj(z)
        else:
            out.update(
                self.sem(
                    z,
                    return_dlc=return_dlc,
                    return_count=return_count,
                    noise=noise,
                    temp=temp,
                )
            )
            z = self.out_proj(out.pop("z"))

        # Reshape latent
        z = einx.rearrange(
            "b (l d) -> b l d", z, l=self.cfg.latent_length, d=self.cfg.latent_dim
        )

        # Maybe squeeze if latent_length == 1
        z = torch.squeeze(z, dim=1) if self.cfg.latent_length == 1 else z

        return z, out

    @staticmethod
    def _mean_pool(token_embeddings: torch.Tensor, attention_mask: torch.Tensor):
        mask = attention_mask.unsqueeze(-1).type_as(token_embeddings)
        summed = (token_embeddings * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp(min=1)
        return summed / denom
