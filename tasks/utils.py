import math

import einx
import torch


def eval_ppl(ppl_model, ppl_tok, prefix_str, suffix_str, device) -> float:
    """Compute conditional perplexity of suffix given prefix using a frozen LM."""
    ppl_batch = ppl_tok(
        [p + c for p, c in zip(prefix_str, suffix_str)],
        padding=True,
        return_tensors="pt",
        return_offsets_mapping=True,
        add_special_tokens=False,
        return_attention_mask=False,
    ).to(device)

    split_idx = split_index_from_offsets(
        ppl_batch["offset_mapping"],
        [len(p) for p in prefix_str],
    )

    labels = ppl_batch["input_ids"].clone()
    labels[labels == ppl_tok.pad_token_id] = -100
    for i in range(len(labels)):
        labels[i, : split_idx[i]] = -100
    ppl = torch.exp(ppl_model(input_ids=ppl_batch["input_ids"], labels=labels).loss)

    return ppl.item()


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