import math
import os
import random
from dataclasses import dataclass
from enum import Enum
from typing import Any, List, Optional, Tuple

import ftfy
import torch
from datasets.load import load_dataset
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, IterableDataset, get_worker_info
from torch.utils.data.dataset import Dataset
from transformers import PreTrainedTokenizerFast


class InfoLabel(Enum):
    SUFFIX = 0
    DLC = 1
    PAD = 2
    PREFIX = 3
    SPECIAL = 4


@dataclass
class LanguageDatasetConfig:
    name: str
    val_size: int
    seed: int


class LanguageDataset(Dataset):
    """
    Simple dataset wrapper around HuggingFace datasets for text data
    """

    def __init__(self, cfg: Optional[LanguageDatasetConfig] = None, **kwargs):

        if cfg == None:
            cfg = LanguageDatasetConfig(**kwargs)

        if cfg.name == "fineweb":
            self.dataset = load_dataset(
                "HuggingFaceFW/fineweb", name="sample-100BT", split="train"
            )
            self.dataset = self.dataset.select_columns(["text", "token_count"])
        elif cfg.name == "owt":
            self.dataset = load_dataset("Skylion007/openwebtext", split="train")
        elif cfg.name == "wiki":
            self.dataset = load_dataset(
                "leogagnon/wikipedia-short-paragraphs", split="train"
            )
        indices = torch.randperm(
            len(self.dataset),
            generator=torch.Generator().manual_seed(cfg.seed),
        )
        self.train_indices = indices[: -cfg.val_size]
        self.val_indices = indices[-cfg.val_size :]

        self.cfg = cfg

    def __len__(self):
        return len(self.dataset)

    def __getitems__(self, indices):
        return self.dataset[indices]

    def __getitem__(self, idx):
        return self.dataset[int(idx)]


class SpanPoissonMasker:
    """
    Classic span masker based on Poisson distribution. Each masked span is replaced by a single mask token.
    """

    def __init__(
        self,
        mask_id: int,
        mask_ratio_range: Tuple[float, float] = (0.2, 0.4),
        poisson_lambda: float = 3.5,
        max_span_len: int = 128,
        keep_bos_eos: bool = True,
    ):
        self.mask_id = mask_id
        self.mask_ratio_range = mask_ratio_range
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

        mask_ratio = rng.uniform(self.mask_ratio_range[0], self.mask_ratio_range[1])

        budget = int(math.ceil(eligible * mask_ratio))
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
        else:
            assert (
                self.context_length == 0
            ), "If not encoding the context, context length must be 0"
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

    @classmethod
    def get_dataloader(
        cls: "PrefixSuffixIterable",
        dataset: Dataset,
        batch_size: int,
        prefix_length: int,
        suffix_length: int,
        context_length: int,
        enc_tok: Optional[PreTrainedTokenizerFast],
        dec_tok: PreTrainedTokenizerFast,
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
            input_ids_dec = dec_tok.pad(
                {"input_ids": batch["input_ids_dec"]},
                padding=True,
                return_tensors="pt",
                return_attention_mask=False,
            )["input_ids"]
            info_mask_dec = pad_sequence(
                batch["info_mask_dec"],
                batch_first=True,
                padding_value=InfoLabel.PAD.value,
            )

            out = {
                "input_ids_dec": input_ids_dec,
                "info_mask_dec": info_mask_dec,
                "prefix_str": batch["prefix_str"],
                "suffix_str": batch["suffix_str"],
            }

            if "input_ids_enc" in batch.keys():
                batch_enc = enc_tok.pad(
                    {"input_ids": batch["input_ids_enc"]},
                    padding=True,
                    return_tensors="pt",
                    return_attention_mask=True,
                )

                out.update(
                    {
                        "input_ids_enc": batch_enc["input_ids"],
                        "attention_mask_enc": batch_enc["attention_mask"],
                    }
                )
            return out

        iterable = PrefixSuffixIterable(
            dataset,
            dec_tok=dec_tok,
            enc_tok=enc_tok,
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
            prefetch_factor=4,
            pin_memory=True,
        )

    def __iter__(self):
        generator = self._make_generator()

        while True:
            # Sample a random document
            indices = generator.choices(range(self.N), k=1024)
            item_batch = self.ds.dataset.dataset[indices]
            input_ids_batch = self.dec_tok.batch_encode_plus(
                item_batch["text"],
                add_special_tokens=False,
                return_attention_mask=False,
            )["input_ids"]

            for i in range(1024):
                input_ids_dec = input_ids_batch[i]

                # Only consider documents with enough length
                seq_length = len(input_ids_dec)
                if seq_length < self.total_length:
                    continue

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
                        + ([0] * self.num_dlc_ph)
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
                    continue

                # Potentially apply span masking to encoder input
                if self.encoder_noise:
                    input_ids_enc = self.span_masker(input_ids_enc, generator)

                out_dict.update({"input_ids_enc": torch.LongTensor(input_ids_enc)})
                yield out_dict


class DeCLUTRIterable(IterableDataset):
    """
    Samples WITH REPLACEMENT DeCLUTR-style training examples from a text dataset (https://github.com/JohnGiorgi/DeCLUTR/)
    - Select a document with enough tokens
    - Sample multiple anchor spans from the document (at distance > 2 * max_span_length)
    - For each anchor, sample multiple positive spans from the document (near the anchor, at distance < max_span_length)
    - Tokenizes and formats everything
    """

    def __init__(
        self,
        dataset,
        tokenizer,
        min_span_length: int,
        max_span_length: int,
        num_anchors: int,
        num_positives: int,
        adjacent_positives: bool,
        seed: Optional[int] = None,
    ):
        assert hasattr(dataset, "__len__") and hasattr(dataset, "__getitem__")
        self.ds = dataset
        self.N = len(dataset)
        assert self.N > 0
        assert min_span_length < max_span_length

        self.min_span_length = min_span_length
        self.max_span_length = max_span_length
        self.num_anchors = num_anchors
        self.num_positives = num_positives
        self.seed = seed
        self.tok = tokenizer
        self.adjacent_positives = adjacent_positives

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

    @classmethod
    def get_dataloader(
        cls: "DeCLUTRIterable",
        dataset: Dataset,
        tokenizer: PreTrainedTokenizerFast,
        batch_size: int,
        min_span_length: int,
        max_span_length: int,
        num_anchors: int,
        num_positives: int,
        adjacent_positives: bool,
        seed: int = 32,
    ) -> DataLoader:
        # Make the collation function
        def collate_fn(batch):
            # Merge dicts by key
            concat_batch = {}
            for key in batch[0].keys():
                concat_batch[key] = []
                for item in batch:
                    concat_batch[key].extend(item[key])

            concat_batch["anchor_ids"] = pad_sequence(
                concat_batch["anchor_ids"],
                batch_first=True,
                padding_value=tokenizer.pad_token_id,
            )
            concat_batch["positive_ids"] = pad_sequence(
                concat_batch["positive_ids"],
                batch_first=True,
                padding_value=tokenizer.pad_token_id,
            )

            out = {
                "anchor_ids": concat_batch["anchor_ids"],
                "positive_ids": concat_batch["positive_ids"],
            }
            return out

        iterable = cls(
            dataset,
            tokenizer=tokenizer,
            min_span_length=min_span_length,
            max_span_length=max_span_length,
            num_anchors=num_anchors,
            num_positives=num_positives,
            adjacent_positives=adjacent_positives,
            seed=seed,
        )

        return DataLoader(
            iterable,
            batch_size=batch_size,
            collate_fn=collate_fn,
            num_workers=int(os.environ.get("TORCH_NUM_WORKERS", 1)),
            persistent_workers=False,
            prefetch_factor=8,
            pin_memory=True,
        )

    def __iter__(self):
        generator = self._make_generator()

        while True:
            # Sample a batch of 1024 random documents for I/O efficiency
            indices = generator.choices(range(self.N), k=1024)
            item_batch = self.ds.dataset.dataset[indices]
            input_ids_batch = self.tok.batch_encode_plus(
                item_batch["text"],
                add_special_tokens=False,
                return_attention_mask=False,
            )["input_ids"]

            for i in range(1024):

                input_ids = input_ids_batch[i]
                seq_length = len(input_ids)

                if seq_length < self.num_anchors * self.max_span_length * 2:
                    continue

                anchors, positives = [], []
                valid_anchor_starts = list(
                    range(
                        0, seq_length - self.max_span_length + 1, self.max_span_length
                    )
                )

                # Sample anchors
                for i in range(self.num_anchors):
                    anchor_len = int(
                        generator.betavariate(4, 2)
                        * (self.max_span_length - self.min_span_length)
                        + self.min_span_length
                    )
                    # This check prevents an edge case were we run out of valid_anchor_starts.
                    if (
                        len(valid_anchor_starts) // (self.num_anchors - i)
                        < self.num_anchors - i
                    ):
                        anchor_start_idx = generator.choice(
                            [0, len(valid_anchor_starts) - 1]
                        )
                    else:
                        anchor_start_idx = generator.randrange(
                            0, len(valid_anchor_starts)
                        )
                    # When num_anchors = 1, this is equivalent to uniformly sampling that starting position.
                    anchor_start = generator.randint(
                        valid_anchor_starts[anchor_start_idx],
                        valid_anchor_starts[anchor_start_idx]
                        + self.max_span_length
                        - anchor_len,
                    )
                    # Once sampled, remove an anchor (and its immediate neighbours) from consideration.
                    del valid_anchor_starts[
                        max(0, anchor_start_idx - 1) : anchor_start_idx + 2
                    ]
                    anchor_end = anchor_start + anchor_len
                    anchors.append(input_ids[anchor_start:anchor_end])

                    # For each anchor, sample positives

                    for _ in range(self.num_positives):

                        if self.adjacent_positives:
                            max_positive_len = min(
                                self.max_span_length,
                                max(anchor_start, seq_length - anchor_end),
                            )

                            positive_len = int(
                                generator.betavariate(2, 4)
                                * (max_positive_len - self.min_span_length)
                                + self.min_span_length
                            )
                            # There are two types of adjacent positives, those that border the beginning of the
                            # anchor and those that border the end. The checks above guarantee at least one of
                            # these is valid. Here we just choose from the valid positive starts at random.
                            valid_starts = []
                            if anchor_start - positive_len > 0:
                                valid_starts.append(anchor_start - positive_len)
                            if anchor_end + positive_len <= seq_length:
                                valid_starts.append(anchor_end)
                            positive_start = generator.choice(valid_starts)

                        else:

                            # Sample positive length from a beta distribution skewed towards shorter spans. The
                            # idea is to promote diversity and minimize the amount of overlapping text.
                            positive_len = int(
                                generator.betavariate(2, 4)
                                * (self.max_span_length - self.min_span_length)
                                + self.min_span_length
                            )
                            # By default, spans may be adjacent or overlap with each other and the anchor.
                            # Careful not to run off the edges of the document (this error may pass silently).
                            positive_start = generator.randint(
                                max(0, anchor_start - positive_len),
                                min(anchor_end, seq_length - positive_len),
                            )

                        positive_end = positive_start + positive_len
                        positives.append(input_ids[positive_start:positive_end])

                out_dict = {
                    "anchor_ids": [torch.LongTensor(a) for a in anchors],
                    "positive_ids": [torch.LongTensor(p) for p in positives],
                }

                yield out_dict
