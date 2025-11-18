import torch
import einx
import math
from torch.utils.data import Sampler
from typing import Iterator, Optional
from transformers import AutoTokenizer
import os


class InfiniteDistributedUniformSampler(Sampler[int]):
    """
    Infinite, per-rank independent uniform sampling *with replacement*.

    - No epoch notion (no set_epoch).
    - Works in single-process and DDP.
    - Yields indices forever → use max_steps / manual break in training loop.
    - Deterministic if `seed` is set; otherwise non-deterministic.

    Args:
        dataset: map-style dataset (needs __len__).
        seed: optional base seed; if None, a random 64-bit seed is used.
        chunk_size: draw this many indices per RNG call (perf tweak).
    """

    def __init__(
        self,
        n: int,
        batch_size: int,
        seed: Optional[int] = None,
        chunk_size: int = 4096,
    ):

        self.n = n
        self.batch_size = batch_size

        # Rank/world
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            self.rank = torch.distributed.get_rank()
        else:
            self.rank = 0

        # Base seed (None → random)
        if seed is None:
            seed = int.from_bytes(os.urandom(8), "little", signed=False)

        # Mix in rank to get independent streams per process
        mixed = (int(seed) ^ (0x9E3779B97F4A7C15 * (self.rank + 1))) & ((1 << 63) - 1)

        self._g = torch.Generator()
        self._g.manual_seed(mixed)
        self._chunk = int(chunk_size)

    def __len__(self) -> int:
        # Sentinel for frameworks that read len(); loader never actually exhausts.
        return 2**31 - 1

    def __iter__(self) -> Iterator[int]:
        while True:
            # Vectorized draw, then yield scalars
            idx = torch.randint(
                0, self.n, (self._chunk, self.batch_size), generator=self._g
            )
            # Yield as Python lists (fast path in DataLoader)
            for row in idx:
                yield row.tolist()

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


def cosine_warmup_get_value(step, max_value, warmup_steps, exp=2):
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