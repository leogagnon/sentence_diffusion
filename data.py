from dataclasses import dataclass
from typing import Optional
from datasets.load import load_from_disk, load_dataset
from hydra.utils import to_absolute_path
from tokenizers import Tokenizer
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data.dataset import Dataset
from lightning.pytorch.utilities.rank_zero import rank_zero_info, rank_zero_only
from transformers import DataCollatorWithPadding, PreTrainedTokenizerFast
from functools import partial
from model.encoder import EncoderModel
from enum import Enum
import os
from transformers import AutoTokenizer
from typing import List, Tuple
import os
import torch
from torch.utils.data import IterableDataset, get_worker_info, DataLoader
from typing import Any, Callable, Dict, List, Optional, Sequence, Union
from torch.nn.utils.rnn import pad_sequence
import ftfy
import math
import random

class InfoLabel(Enum):
    SUFFIX = 0
    DLC = 1
    PAD = 2
    PREFIX = 3
    SPECIAL = 4


@dataclass
class WikipediaDatasetConfig:
    max_length: int


class WikipediaDataset(Dataset):
    VAL_SIZE = 8192
    SEED = 42
    def __init__(self, cfg: Optional[WikipediaDatasetConfig] = None, **kwargs):
        if cfg == None:
            cfg = WikipediaDatasetConfig(**kwargs)
        self.dataset = load_dataset(
            "leogagnon/wikipedia-short-paragraphs", split="train"
        )
        self.cfg = cfg
        self.max_length = self.cfg.max_length

    def get_train_val_indices(self, val_size):
        indices = torch.randperm(
            len(self.dataset),
            generator=torch.Generator().manual_seed(self.SEED),
        )
        train_indices = indices[:-self.VAL_SIZE]
        val_indices = indices[-self.VAL_SIZE:]

        return train_indices, val_indices

    def __len__(self):
        return len(self.dataset)

    def __getitems__(self, indices):
        return {"input_str": self.dataset[indices]["input_ids"]}

    def __getitem__(self, idx):
        return {"input_str": self.dataset[int(idx)]["input_ids"]}


class FineWebDataset(Dataset):
    VAL_SIZE = 16384
    SEED = 42

    def __init__(self):

        self.dataset = load_dataset(
            "leogagnon/fineweb_100BT_tokenized_gpt2", split="train"
        )
        self.pre_tokenizer = AutoTokenizer.from_pretrained("gpt2-large")
        indices = torch.randperm(
            len(self.dataset),
            generator=torch.Generator().manual_seed(self.SEED),
        )
        self.train_indices = indices[:-self.VAL_SIZE]
        self.val_indices = indices[-self.VAL_SIZE:]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return {"input_ids": self.dataset[int(idx)]["input_ids"]}


class SpanPoissonMasker:
    def __init__(
        self,
        mask_id: int,
        mask_ratio: float = 0.3,
        poisson_lambda: float = 3.5,
        max_span_len: int = 128,
        keep_bos_eos: bool = True,
    ):
        self.mask_id = mask_id
        self.mask_ratio = mask_ratio
        self.poisson_lambda = poisson_lambda
        self.max_span_len = max_span_len
        self.keep_bos_eos = keep_bos_eos
        self._poisson_probs = self._build_poisson_probs()

    def _build_poisson_probs(self) -> List[float]:
        lam = float(self.poisson_lambda)
        ps = []
        lam_to_k = 1.0
        e_minus_lam = math.exp(-lam)
        k_fact = 1.0
        for k in range(self.max_span_len):
            ps.append(e_minus_lam * lam_to_k / k_fact)
            lam_to_k *= lam
            k_fact *= k + 1
            if ps[-1] < 1e-7:
                break
        s = sum(ps)
        return [p / s for p in ps]

    def _sample_span_len(self, rng: random.Random) -> int:
        # Reject 0-length (we're not doing insertion noise)
        while True:
            k = rng.choices(
                range(len(self._poisson_probs)),
                weights=self._poisson_probs,
                k=1,
            )[0]
            if k > 0:
                return k

    def __call__(
        self,
        tokens: List[int],
        rng: Optional[random.Random] = None,
    ) -> List[int]:
        rng = random if rng is None else rng

        L = len(tokens)
        if L == 0:
            return tokens
        if self.keep_bos_eos and L <= 2:
            return tokens

        start = 1 if self.keep_bos_eos else 0
        end = (L - 1) if self.keep_bos_eos else L
        eligible = end - start
        if eligible <= 0:
            return tokens

        budget = int(math.ceil(eligible * self.mask_ratio))
        if budget <= 0:
            return tokens

        # Boolean mask over token positions (True => masked/deleted)
        masked = [False] * L

        covered = 0
        # We sample candidate start positions in random order and grow spans from them,
        # only counting newly covered positions toward the budget.
        candidates = list(range(start, end))
        rng.shuffle(candidates)

        # If we run out of candidates before reaching budget (rare when budget is high),
        # reshuffle and try again.
        cand_idx = 0
        while covered < budget:
            if cand_idx >= len(candidates):
                rng.shuffle(candidates)
                cand_idx = 0

            s = candidates[cand_idx]
            cand_idx += 1

            # If already masked, starting here would waste coverage—skip.
            if masked[s]:
                continue

            span_len = self._sample_span_len(rng)

            # Extend rightwards, marking unmasked positions until:
            # - we hit the span_len,
            # - we reach end,
            # - or we reach the remaining budget.
            remaining = budget - covered
            j = s
            newly = 0
            while j < end and newly < span_len and newly < remaining:
                if not masked[j]:
                    masked[j] = True
                    newly += 1
                j += 1

            covered += newly
            # If newly == 0, we basically hit a weird case; loop continues.

        # Now build the corrupted output with span-collapsing:
        # emit exactly ONE mask_id per contiguous masked region.
        out: List[int] = []
        i = 0
        while i < L:
            if masked[i]:
                out.append(self.mask_id)
                # skip the whole contiguous region
                i += 1
                while i < L and masked[i]:
                    i += 1

            else:
                out.append(tokens[i])
                i += 1

        return out


class PrefixSuffixIterable(IterableDataset):
    """
    Iterable which randomly sample WITH REPLACEMENT and formats training inputs from web dataset (e.g FineWeb)
    - Select a document with enough tokens
    - Select a random window from that document
    - Potentially split into context/prefix/suffix
    - Tokenizes and formats everything
    """

    def __init__(
        self,
        dataset,
        dec_tok,
        suffix_length: int,
        enc_tok: Optional[Any] = None,
        encoder_mode: str = "suffix",
        encoder_noise: bool = False,
        prefix_length: int = 0,
        context_length: int = 0,
        num_dlc_ph: int = 0,
        seed: Optional[int] = None,
    ):
        assert hasattr(dataset, "__len__") and hasattr(dataset, "__getitem__")
        self.ds = dataset
        self.N = len(dataset)
        assert self.N > 0
        self.context_length = context_length
        self.prefix_length = prefix_length
        if self.context_length > 0:
            assert (
                self.prefix_length > 0
            ), "Prefix length must be >0 if context length is >0"
        self.suffix_length = suffix_length
        self.total_length = context_length + prefix_length + suffix_length
        self.seed = seed
        self.enc_tok = enc_tok
        self.dec_tok = dec_tok
        self.encoder_mode = encoder_mode
        self.num_dlc_ph = num_dlc_ph
        assert self.encoder_mode in ["suffix", "context", "none"]
        if self.encoder_mode == "context":
            assert (
                self.context_length > 0
            ), "If encoding the context, must have context length >0"
        self.encoder_noise = encoder_noise
        if self.encoder_noise:
            self.span_masker = SpanPoissonMasker(mask_id=self.enc_tok.mask_token_id)

    def _make_generator(self) -> random.Random:
        # DDP rank
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            rank = torch.distributed.get_rank()
        else:
            rank = 0

        # DataLoader worker id
        wi = get_worker_info()
        wid = wi.id if wi is not None else 0

        base = self.seed
        if base is None:
            base = int.from_bytes(os.urandom(8), "little", signed=False)

        # mix base seed with rank + worker to get independent streams
        mixed = (
            int(base)
            ^ (0x9E3779B97F4A7C15 * (rank + 1))
            ^ (0xBF58476D1CE4E5B9 * (wid + 1))
        ) & ((1 << 63) - 1)

        generator = random.Random(mixed)
        return generator

    def __iter__(self):
        generator = self._make_generator()

        while True:
            # Sample a random document
            idx = generator.randint(0, self.N - 1)
            input_ids_dec = self.ds[idx]["input_ids"]
            seq_length = len(input_ids_dec)

            # Only consider documents with enough length
            if seq_length >= self.total_length:

                # Sample a random window
                window_start = generator.randint(0, seq_length - self.total_length)
                input_ids_dec = input_ids_dec[
                    window_start : window_start + self.total_length
                ]

                # Maybe extract the context
                if self.encoder_mode == "context":
                    context_ids_dec = input_ids_dec[
                        : self.context_length + self.prefix_length
                    ]
                    input_ids_dec = input_ids_dec[self.context_length :]
                prefix_ids_dec = input_ids_dec[: self.prefix_length]
                suffix_ids_dec = input_ids_dec[-self.suffix_length :]

                # Build input_ids and info_mask for decoder
                if self.num_dlc_ph > 0:
                    input_ids_dec = (
                        prefix_ids_dec
                        + [self.dec_tok.think_token_id]
                        + ([-1] * self.num_dlc_ph)
                        + [self.dec_tok.think_token_id]
                        + suffix_ids_dec
                    )
                    info_mask_dec = (
                        [InfoLabel.PREFIX.value] * self.prefix_length
                        + [InfoLabel.SPECIAL.value]
                        + [InfoLabel.DLC.value] * self.num_dlc_ph
                        + [InfoLabel.SPECIAL.value]
                        + [InfoLabel.SUFFIX.value] * self.suffix_length
                    )
                else:
                    input_ids_dec = prefix_ids_dec + suffix_ids_dec
                    info_mask_dec = ([InfoLabel.PREFIX.value] * self.prefix_length) + (
                        [InfoLabel.SUFFIX.value] * self.suffix_length
                    )

                # Decode prefix and suffix strings
                prefix_str = self.dec_tok.decode(
                    prefix_ids_dec, skip_special_tokens=True
                )
                suffix_str = self.dec_tok.decode(
                    suffix_ids_dec, skip_special_tokens=True
                )

                out_dict = {
                    "input_ids_dec": torch.LongTensor(input_ids_dec),
                    "info_mask_dec": torch.LongTensor(info_mask_dec),
                    "suffix_str": suffix_str,
                    "prefix_str": prefix_str,
                }

                # Build input_ids for encoder
                # We fix potential text encoding issues with ftfy (which could mess with SentencePiece tokenization)
                # We filter out samples where the encoder input gets too long compared to decoder input (to avoid OOM)
                if self.encoder_mode == "suffix":
                    suffix_str = ftfy.fix_text(suffix_str)
                    input_ids_enc = self.enc_tok.encode(suffix_str)
                    if len(input_ids_enc) > len(suffix_ids_dec) * 1.3:
                        continue
                elif self.encoder_mode == "context":
                    context_str = self.dec_tok.decode(
                        context_ids_dec,
                        skip_special_tokens=True,
                    )
                    context_str = ftfy.fix_text(context_str)
                    input_ids_enc = self.enc_tok.encode(context_str)
                    if len(input_ids_enc) > len(context_ids_dec) * 1.3:
                        continue
                else:
                    yield out_dict

                # Potentially apply span masking to encoder input
                if self.encoder_noise:
                    input_ids_enc = self.span_masker(input_ids_enc, generator)

                out_dict.update({"input_ids_enc": torch.LongTensor(input_ids_enc)})
                yield out_dict


def get_dataloader(
    dataset: Dataset,
    batch_size: int,
    prefix_length: int,
    suffix_length: int,
    context_length: int,
    enc_tokenizer: PreTrainedTokenizerFast,
    dec_tokenizer: PreTrainedTokenizerFast,
    encoder_mode: str,
    encoder_noise: bool,
    seed: int = 32,
    num_dlc_ph: int = 0,
):

    # Make the collation function
    def collate_fn(batch):
        # Merge dicts by key
        batch = {key: [item[key] for item in batch] for key in batch[0].keys()}

        # Pad decoder input ids
        input_ids_dec = dec_tokenizer.pad(
            {"input_ids": batch["input_ids_dec"]},
            padding=True,
            return_tensors="pt",
            return_attention_mask=False,
        )["input_ids"]
        info_mask_dec = pad_sequence(
            batch["info_mask_dec"], batch_first=True, padding_value=InfoLabel.PAD.value
        )

        batch_enc = enc_tokenizer.pad(
            {"input_ids": batch["input_ids_enc"]},
            padding=True,
            return_tensors="pt",
            return_attention_mask=True,
        )

        return {
            "input_ids_dec": input_ids_dec,
            "info_mask_dec": info_mask_dec,
            "input_ids_enc": batch_enc["input_ids"],
            "attention_mask_enc": batch_enc["attention_mask"],
            "prefix_str": batch["prefix_str"],
            "suffix_str": batch["suffix_str"],
        }

    iterable = PrefixSuffixIterable(
        dataset,
        dec_tok=dec_tokenizer,
        enc_tok=enc_tokenizer,
        prefix_length=prefix_length,
        suffix_length=suffix_length,
        context_length=context_length,
        encoder_mode=encoder_mode,
        encoder_noise=encoder_noise,
        seed=seed,
        num_dlc_ph=num_dlc_ph,
    )

    return DataLoader(
        iterable,
        batch_size=batch_size,
        collate_fn=collate_fn,
        num_workers=int(os.environ.get("TORCH_NUM_WORKERS", 1)),
        persistent_workers=False,
        prefetch_factor=2,
        pin_memory=False,
    )
