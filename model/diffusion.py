import abc
import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import (AutoModelForMaskedLM, GPT2TokenizerFast)

# Flags required to enable jit fusion kernels
torch._C._jit_set_profiling_mode(False)
torch._C._jit_set_profiling_executor(False)
torch._C._jit_override_can_fuse_on_cpu(True)
torch._C._jit_override_can_fuse_on_gpu(True)


class Noise(abc.ABC, nn.Module):
    """
    Baseline forward method to get the total + rate of noise at a timestep
    """

    def forward(self, t):
        # Assume time goes from 0 to 1
        return self.total_noise(t), self.rate_noise(t)

    @abc.abstractmethod
    def rate_noise(self, t):
        """
        Rate of change of noise ie g(t)
        """

    @abc.abstractmethod
    def total_noise(self, t):
        """
        Total noise ie \int_0^t g(t) dt + g(0)
        """


class LogLinearNoise(Noise):
    """Log Linear noise schedule.

    Built such that 1 - 1/e^(n(t)) interpolates between 0 and
    ~1 when t varies from 0 to 1. Total noise is
    -log(1 - (1 - eps) * t), so the sigma will be
    (1 - eps) * t.
    """

    def __init__(self, eps=1e-3):
        super().__init__()
        self.eps = eps
        self.sigma_max = self.total_noise(torch.tensor(1.0))
        self.sigma_min = self.eps + self.total_noise(torch.tensor(0.0))

    def rate_noise(self, t):
        return (1 - self.eps) / (1 - (1 - self.eps) * t)

    def total_noise(self, t):
        return -torch.log1p(-(1 - self.eps) * t)

    def importance_sampling_transformation(self, t):
        f_T = torch.log1p(-torch.exp(-self.sigma_max))
        f_0 = torch.log1p(-torch.exp(-self.sigma_min))
        sigma_t = -torch.log1p(-torch.exp(t * f_T + (1 - t) * f_0))
        t = -torch.expm1(-sigma_t) / (1 - self.eps)
        return t


def sample_categorical(categorical_probs):
    gumbel_norm = 1e-10 - (torch.rand_like(categorical_probs) + 1e-10).log()
    return (categorical_probs / gumbel_norm).argmax(dim=-1)


@dataclass
class DiTConfig:
    name: str
    dlc_vocab_size: int = 0
    dlc_len: Optional[int] = None
    sigma_min: float = 1e-4
    sigma_max: float = 20.0
    sampling_steps: int = 1024
    noise_removal: bool = True
    max_seq_len: int = 1024
    time_conditioning: bool = False
    change_of_variables: bool = False
    importance_sampling: bool = False
    antithetic_sampling: bool = True
    sampling_mode: str = "mdlm"  # "mdlm", "remdm-cap", "remdm-loop"
    eta: Optional[float] = None  # Will be set depending on sampling_mode if None
    alpha_on: float = 0.9  
    t_on: float = 0.55 
    t_off: float = 0.05  
    T: int = 0  # Number of discrete timesteps. If 0, continuous time is used.
    nucleus_p: float = 0.9  # Nucleus sampling probability
    sampling_eps: float = 0.001  # Minimum time for


class DiTModel(nn.Module):

    def __init__(self, cfg: DiTConfig):
        super().__init__()

        self.backbone = AutoModelForMaskedLM.from_pretrained(
            cfg.name, trust_remote_code=True
        )

        # Init tokenizer, make it put BOS and EOS tokens around inputs, and add PAD and THINK tokens.
        self.tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

        # Add PAD and THINK tokens, resize embeddings
        self.tokenizer.add_special_tokens(
            {
                "additional_special_tokens": ["<|mask|>", "<|think|>"],
                "pad_token": "<|pad|>",
            }
        )
        self.tokenizer.think_token_id = self.tokenizer.convert_tokens_to_ids(
            "<|think|>"
        )
        self.tokenizer.mask_token_id = self.tokenizer.convert_tokens_to_ids("<|mask|>")
        assert self.tokenizer.mask_token_id == 50257

        # Resize token embeddings to account for new tokens
        # If using DLC, we add dlc_vocab_size tokens on top of that
        new_embeddings = torch.empty(
            (cfg.dlc_vocab_size + 3, self.backbone.config.hidden_dim),
            dtype=torch.float32,
        )
        new_embeddings = torch.nn.init.kaiming_uniform_(new_embeddings, a=math.sqrt(5))

        self.backbone.backbone.vocab_embed.embedding.data = torch.cat(
            [
                self.backbone.backbone.vocab_embed.embedding.data,
                new_embeddings,
            ],
            dim=0,
        )
        old_out_layer: nn.Linear = self.backbone.backbone.output_layer.linear
        new_out_layer = nn.Linear(
            in_features=old_out_layer.in_features,
            out_features=old_out_layer.out_features + 2 + cfg.dlc_vocab_size,
            bias=True,
        )
        new_out_layer.weight.data.zero_()
        new_out_layer.bias.data.zero_()
        with torch.no_grad():
            new_out_layer.weight[:old_out_layer.out_features].copy_(old_out_layer.weight)
            new_out_layer.bias[:old_out_layer.out_features].copy_(old_out_layer.bias)
        self.backbone.backbone.output_layer.linear = new_out_layer

        if cfg.dlc_vocab_size > 0:
            assert cfg.dlc_len is not None, "If using DLC, dlc_len must be specified"

        self.noise = LogLinearNoise()

        if cfg.eta is None and cfg.sampling_mode.startswith("remdm"):
            cfg.eta = 0.008 if cfg.sampling_mode == "remdm-cap" else 0.05

        self.cfg = cfg

    def q_xt(self, x, move_chance, cond_mask=None):
        """Computes the noisy sample xt.

        Args:
        x: int torch.Tensor with shape (batch_size,
            diffusion_model_input_length), input.
        move_chance: float torch.Tensor with shape (batch_size, 1).
        """
        move_indices = torch.rand(*x.shape, device=x.device) < move_chance

        if cond_mask is not None:
            # Do not move the conditioned tokens
            move_indices = move_indices & (~cond_mask)

        xt = torch.where(move_indices, self.tokenizer.mask_token_id, x)
        return xt

    def ddpm_caching_update(self, x, t, dt, p_x0=None, frozen_mask=None):
        sigma_t, _ = self.noise(t)
        if t.ndim > 1:
            t = t.squeeze(-1)
        assert t.ndim == 1

        move_chance_t = t[:, None, None]
        move_chance_s = (t - dt)[:, None, None]
        assert move_chance_t.ndim == 3, move_chance_t.shape
        if p_x0 is None:
            p_x0 = self.forward(x, sigma_t).exp()
            if self.cfg.nucleus_p < 1:
                sorted_probs, sorted_indices = torch.sort(p_x0, descending=True, dim=-1)
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                top_p_mask = cumulative_probs <= self.cfg.nucleus_p
                top_p_mask[..., 0] = True
                nucleus_probs = sorted_probs * top_p_mask
                nucleus_probs /= nucleus_probs.sum(dim=-1, keepdim=True)
                p_x0 = torch.zeros_like(p_x0).scatter_(-1, sorted_indices, nucleus_probs)

        assert move_chance_t.ndim == p_x0.ndim

        if self.cfg.sampling_mode == "mdlm":
            q_xs = p_x0 * (move_chance_t - move_chance_s)
            q_xs[:, :, self.tokenizer.mask_token_id] = move_chance_s[:, :, 0]
            _x = sample_categorical(q_xs)
            copy_flag = (x != self.tokenizer.mask_token_id).to(x.dtype)
            xs = copy_flag * x + (1 - copy_flag) * _x
        elif self.cfg.sampling_mode == "remdm-cap":
            alpha_t = (1 - move_chance_t)[0].item()
            alpha_s = (1 - move_chance_s)[0].item()
            if alpha_t > 0:
                sigma = min(self.cfg.eta, (1 - alpha_s) / alpha_t)
            else:
                sigma = self.cfg.eta
            q_xs = p_x0 * (1 - sigma)
            q_xs[..., self.tokenizer.mask_token_id] = sigma
            q_xs_2 = p_x0 * ((alpha_s - (1 - sigma) * alpha_t) / (1 - alpha_t))
            q_xs_2[..., self.tokenizer.mask_token_id] = (1 - alpha_s - sigma * alpha_t) / (1 - alpha_t)
            copy_flag = (x != self.tokenizer.mask_token_id).to(torch.bool)
            q_xs = torch.where(copy_flag.unsqueeze(-1), q_xs, q_xs_2)
            xs = sample_categorical(q_xs)
        elif self.cfg.sampling_mode == "remdm-loop":
            time = t[0].item()
            # compute alpha_t and alpha_s
            if time > self.cfg.t_on:
                move_chance_t = (1 - (1 - t) * self.cfg.alpha_on / (1 - self.cfg.t_on))[:, None, None]
                move_chance_s = (1 - (1 - t + dt) * self.cfg.alpha_on / (1 - self.cfg.t_on))[:, None, None]
            elif time <= self.cfg.t_off:
                move_chance_t = (t * (1 - self.cfg.alpha_on) / self.cfg.t_off)[:, None, None]
                move_chance_s = ((t - dt) * (1 - self.cfg.alpha_on) / self.cfg.t_off)[:, None, None]
            else:
                move_chance_t, move_chance_s = None, None
            # use MDLM
            if time > self.cfg.t_on or time <= self.cfg.t_off:
                q_xs = p_x0 * (move_chance_t - move_chance_s)
                q_xs[:, :, self.tokenizer.mask_token_id] = move_chance_s[:, :, 0]
                _x = sample_categorical(q_xs)
                copy_flag = (x != self.tokenizer.mask_token_id).to(x.dtype)
                xs = copy_flag * x + (1 - copy_flag) * _x
            else: # use ReMDM
                sigma = self.cfg.eta
                q_xs = p_x0 * (1 - sigma)
                q_xs[..., self.tokenizer.mask_token_id] = sigma
                q_xs_2 = p_x0 * ((self.cfg.alpha_on - (1 - sigma) * self.cfg.alpha_on) / (1 - self.cfg.alpha_on))
                q_xs_2[..., self.tokenizer.mask_token_id] = (1 - self.cfg.alpha_on - self.cfg.alpha_on * sigma) / (1 - self.cfg.alpha_on)
                copy_flag = (x != self.tokenizer.mask_token_id).to(torch.bool)
                q_xs = torch.where(copy_flag.unsqueeze(-1), q_xs, q_xs_2)
                xs = sample_categorical(q_xs)

        # Makes sure that frozen tokens remain unchanged (e.g. by remasking)
        if frozen_mask is not None:
            xs = torch.where(frozen_mask, x, xs)

        if torch.allclose(xs, x):
            p_x0_cache = p_x0
        else:
            p_x0_cache = None

        return p_x0_cache, xs


    def sample_t(self, n, device):
        _eps_t = torch.rand(n, device=device)
        if self.cfg.antithetic_sampling:
            offset = torch.arange(n, device=device) / n
            _eps_t = (_eps_t / n + offset) % 1
        t = (1 - self.cfg.sampling_eps) * _eps_t + self.cfg.sampling_eps

        if self.cfg.importance_sampling:
            return self.noise.importance_sampling_transformation(t)
        return t

    def forward_pass_diffusion(self, x0, cond_mask=None):
        t = self.sample_t(x0.shape[0], x0.device)
        if self.cfg.T > 0:
            t = (t * self.cfg.T).to(torch.int)
            t = t / self.cfg.T
            # t \in {1/T, 2/T, ..., 1}
            t += 1 / self.cfg.T

        if self.cfg.change_of_variables:
            unet_conditioning = t[:, None]
            f_T = torch.log1p(-torch.exp(-self.noise.sigma_max))
            f_0 = torch.log1p(-torch.exp(-self.noise.sigma_min))
            move_chance = torch.exp(f_0 + t * (f_T - f_0))
            move_chance = move_chance[:, None]
        else:
            sigma, dsigma = self.noise(t)
            unet_conditioning = sigma[:, None]
            move_chance = 1 - torch.exp(-sigma[:, None])

        xt = self.q_xt(x0, move_chance, cond_mask=cond_mask)
        model_output = self.forward(xt, unet_conditioning)
        if torch.isnan(model_output).any():
            print("model_output contains NaNs", model_output)

        if self.cfg.T > 0:
            return self.d3pm_loss(model_output=model_output, xt=xt, x0=x0, t=t)

        # SUBS parameterization, continuous time.
        log_p_theta = torch.gather(
            input=model_output, dim=-1, index=x0[:, :, None]
        ).squeeze(-1)

        if self.cfg.change_of_variables or self.cfg.importance_sampling:
            return log_p_theta * torch.log1p(-torch.exp(-self.noise.sigma_min))

        return -log_p_theta * (dsigma / torch.expm1(sigma))[:, None]

    def d3pm_loss(self, model_output, xt, x0, t):
        dt = 1 / self.cfg.T

        if torch.is_tensor(t):
            t = t[:, None]
            assert t.ndim == 2
            t = t.clamp(0.0, 1.0 - 1e-4)
        alpha_t = 1 - t + torch.zeros_like(xt)
        alpha_s = 1 - (t - dt) + torch.zeros_like(xt)

        log_x_theta_at_x0 = torch.gather(model_output, -1, x0[:, :, None]).squeeze(-1)
        log_x_theta_at_m = model_output[:, :, self.tokenizer.mask_token_id]
        x_theta_at_m = log_x_theta_at_m.exp()

        term_1_coef = dt / t
        term_1_log_nr = torch.log(alpha_t * x_theta_at_m / t + 1)
        term_1_log_dr = log_x_theta_at_x0

        term_2_coef = 1 - dt / t
        term_2_log_nr = term_1_log_nr
        term_2_log_dr = torch.log(alpha_s * x_theta_at_m / (t - dt) + 1)

        L_vb_masked = term_1_coef * (term_1_log_nr - term_1_log_dr) + term_2_coef * (
            term_2_log_nr - term_2_log_dr
        )

        L_vb = L_vb_masked * (xt == self.tokenizer.mask_token_id)

        return self.cfg.T * L_vb

    @property
    def latent_dim(self):
        return self.backbone.get_input_embeddings().weight.shape[1]

    def subs_parameterization(self, logits, xt):
        # log prob at the mask index = - infinity
        logits[:, :, self.tokenizer.mask_token_id] += -1000000.0

        # Normalize the logits such that x.exp() is
        # a probability distribution over vocab_size.
        logits = logits - torch.logsumexp(logits, dim=-1, keepdim=True)

        # Apply updates directly in the logits matrix.
        # For the logits of the unmasked tokens, set all values
        # to -infinity except for the indices corresponding to
        # the unmasked tokens.
        unmasked_indices = xt != self.tokenizer.mask_token_id
        logits[unmasked_indices] = -1000000.0
        logits[unmasked_indices, xt[unmasked_indices]] = 0
        return logits

    def compile(self):
        self.backbone.compile()

    def forward(self, x, sigma):
        if sigma.ndim > 1:
            sigma = sigma.squeeze(-1)
        if not self.cfg.time_conditioning:
            sigma = torch.zeros_like(sigma)

        with torch.amp.autocast(device_type="cuda", dtype=torch.float32):
            logits = self.backbone(x, sigma)

        return self.subs_parameterization(logits=logits, xt=x)

    @torch.no_grad()
    def sample(self, batch_size=None, num_steps=None, eps=1e-5, prior=None, frozen_mask=None):
        """Generate samples from the model."""
        device = next(self.parameters()).device

        # Lightning auto-casting is not working in this method for some reason
        if num_steps is None:
            num_steps = self.cfg.sampling_steps

        if prior is None:
            assert batch_size is not None
            x = self.tokenizer.mask_token_id * torch.ones(
                size=(batch_size, self.cfg.max_seq_len),
                dtype=torch.int64,
                device=device,
            )
        else:
            assert batch_size is None
            x = prior

        if frozen_mask is None:
            frozen_mask = prior != self.tokenizer.mask_token_id

        timesteps = torch.linspace(1, eps, num_steps + 1, device=device)
        dt = (1 - eps) / num_steps
        p_x0_cache = None

        for i in tqdm(range(num_steps)):
            t = timesteps[i] * torch.ones(x.shape[0], 1, device=device)
            p_x0_cache, x_next = self.ddpm_caching_update(x, t, dt, p_x0=p_x0_cache, frozen_mask=frozen_mask)
            x = x_next

        if self.cfg.noise_removal:
            t = timesteps[-1] * torch.ones(x.shape[0], 1, device=device)
            unet_conditioning = self.noise(t)[0]
            x_ = self.forward(x, unet_conditioning).argmax(dim=-1)
            x = torch.where(frozen_mask, x, x_)
        return x

    def compute_loss(self, x0, attention_mask, cond_mask=None, normalize=True):

        loss = self.forward_pass_diffusion(x0, cond_mask=cond_mask)

        nlls = loss * attention_mask
        batch_nll = nlls.sum()
        
        if normalize:
            return batch_nll / attention_mask.sum()
        else:
            return batch_nll