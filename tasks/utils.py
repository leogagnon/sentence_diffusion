import math

import einx
import numpy as np
import torch


def sem_entropy(dlc_probs) -> tuple[torch.Tensor, torch.Tensor]:
    if isinstance(dlc_probs, list):
        # Flatten levels l1 = p(x_0), l2 = p(x_0, x_1), ...
        levels = [einx.rearrange("b L N V -> b L (N V)", l) for l in dlc_probs]
        ent = sum(
            [
                compute_entropy(
                    level,
                    normalized=True,
                ).mean()
                for level in levels
            ]
        ) / len(levels)
        marginal_ent = sum(
            [compute_entropy(level.mean(0), normalized=True).mean() for level in levels]
        ) / len(levels)
    else:
        ent = compute_entropy(dlc_probs, normalized=True).mean()
        marginal_ent = compute_entropy(dlc_probs.mean(0), normalized=True).mean()

    return ent, marginal_ent


def sem_margin(dlc_probs, delta):
    if isinstance(dlc_probs, list):
        # Flatten levels l1 = p(x_0), l2 = p(x_0, x_1), ...
        levels = [einx.rearrange("b L N V -> b L (N V)", l) for l in dlc_probs]
        reg = sum([compute_margin_reg(level, delta) for level in levels]) / len(levels)
    else:
        reg = compute_margin_reg(dlc_probs, delta)

    return reg


def compute_margin_reg(dlc_probs, delta):
    sorted_probs = torch.sort(dlc_probs, dim=-1, descending=True)[0]
    margin = sorted_probs[..., 0] - sorted_probs[..., 1]
    reg = torch.mean(torch.clamp(delta - margin, min=0.0) * (1 / delta))
    return reg


def cosine_warmup_get_value(step, max_value, warmup_steps, exp=3):
    value = max_value

    if step < warmup_steps:
        value *= 0.5 * (1 - math.cos(math.pi * math.pow(step / warmup_steps, exp)))

    return value


def compute_entropy(probs, normalized=False):
    ent = -(probs * torch.clamp(probs, min=1e-12).log()).sum(-1)
    if normalized:
        return ent / math.log(probs.shape[-1])
    return ent


def split_index_from_offsets(
    batch_offsets,
    batch_char_prefix_lengths,
):
    B = len(batch_offsets)
    out = []

    for offsets, L in zip(batch_offsets, batch_char_prefix_lengths):
        split_idx = 0
        for i, (start, end) in enumerate(offsets):
            if end < L:
                split_idx = i + 1
            else:
                break
        out.append(split_idx)

    return out


class SEMUsageTracker:
    def __init__(self, ema_decay=0.99):
        self.ema_decay = ema_decay
        self.usage = None

    @torch.autocast(device_type="cuda", enabled=False)
    @torch.no_grad()
    def update(self, batch_counts, batch_size):

        # Compute relative frequencies
        batch_freq = batch_counts / batch_size

        if self.usage is None:
            self.usage = batch_freq.cpu()
        else:
            # EMA update
            self.usage = (
                self.ema_decay * self.usage + (1 - self.ema_decay) * batch_freq.cpu()
            )


class NTXentLoss(torch.nn.Module):

    def __init__(self, temperature):
        super(NTXentLoss, self).__init__()
        self.temperature = temperature
        self.softmax = torch.nn.Softmax(dim=-1)
        self.mask_samples_from_same_repr = None

    def _get_correlated_mask(self, batch_size):
        diag = np.eye(2 * batch_size)
        l1 = np.eye((2 * batch_size), 2 * batch_size, k=-batch_size)
        l2 = np.eye((2 * batch_size), 2 * batch_size, k=batch_size)
        mask = torch.from_numpy((diag + l1 + l2))
        mask = (1 - mask).bool()
        return mask

    @torch.amp.autocast(device_type="cuda", enabled=False)
    def forward(self, zis: torch.Tensor, zjs: torch.Tensor):
        batch_size = zis.shape[0]

        # Setup and cache the mask to filter out positive samples from the negatives
        if self.mask_samples_from_same_repr is None:
            self.mask_samples_from_same_repr = self._get_correlated_mask(batch_size)

        representations = torch.cat([zjs.float(), zis.float()], dim=0)

        similarity_matrix = torch.nn.functional.cosine_similarity(
            representations.unsqueeze(1), representations.unsqueeze(0), dim=-1
        )

        # filter out the scores from the positive samples
        l_pos = torch.diag(similarity_matrix, batch_size)
        r_pos = torch.diag(similarity_matrix, -batch_size)
        positives = torch.cat([l_pos, r_pos]).view(2 * batch_size, 1)

        negatives = similarity_matrix[self.mask_samples_from_same_repr].view(
            2 * batch_size, -1
        )
        logits = torch.cat((positives, negatives), dim=1)
        logits /= self.temperature

        labels = torch.zeros(2 * batch_size, device=logits.device).long()
        loss = torch.nn.functional.cross_entropy(logits, labels, reduction="sum")

        return loss / (2 * batch_size)
