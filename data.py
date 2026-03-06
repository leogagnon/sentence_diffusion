import math
import os
import random
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import ftfy
import torch
from datasets.load import load_dataset
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, IterableDataset, get_worker_info
from torch.utils.data.dataset import Dataset
from transformers import PreTrainedTokenizerFast, GPT2Tokenizer


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

        num_workers = int(os.environ.get("TORCH_NUM_WORKERS", 0))
        return DataLoader(
            iterable,
            batch_size=batch_size,
            collate_fn=collate_fn,
            num_workers=num_workers,
            persistent_workers=True,
            prefetch_factor=4 if num_workers > 0 else None,
            pin_memory=True,
        )

    def __iter__(self):
        generator = self._make_generator()

        self.dec_tok : GPT2Tokenizer

        while True:
            # Sample a random document
            indices = generator.choices(range(self.N), k=1024)
            item_batch = self.ds.dataset.dataset[indices]
            input_ids_batch = self.dec_tok(
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
                    context_ids_dec = input_ids_dec
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


class SimCSEIterable(IterableDataset):
    """
    Samples contiguous text spans for SimCSE-style training.
    Each yielded item is a single tokenized span; the SimCSE loss encodes the same
    span twice using different dropout masks to form positive pairs.
    """

    def __init__(
        self,
        dataset,
        tokenizer,
        min_span_length: int,
        max_span_length: int,
        seed: Optional[int] = None,
    ):
        assert hasattr(dataset, "__len__") and hasattr(dataset, "__getitem__")
        self.ds = dataset
        self.N = len(dataset)
        assert self.N > 0
        self.min_span_length = min_span_length
        self.max_span_length = max_span_length
        self.seed = seed
        self.tok = tokenizer

    def _make_generator(self) -> random.Random:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            rank = torch.distributed.get_rank()
        else:
            rank = 0
        wi = get_worker_info()
        wid = wi.id if wi is not None else 0
        base = self.seed
        if base is None:
            base = int.from_bytes(os.urandom(8), "little", signed=False)
        mixed = (
            int(base)
            ^ (0x9E3779B97F4A7C15 * (rank + 1))
            ^ (0xBF58476D1CE4E5B9 * (wid + 1))
        ) & ((1 << 63) - 1)
        return random.Random(mixed)

    @classmethod
    def get_dataloader(
        cls,
        dataset,
        tokenizer: PreTrainedTokenizerFast,
        batch_size: int,
        min_span_length: int,
        max_span_length: int,
        seed: int = 32,
    ) -> DataLoader:
        def collate_fn(batch):
            return tokenizer.pad(
                {"input_ids": batch},
                padding=True,
                return_tensors="pt",
                return_attention_mask=True,
            )

        iterable = cls(dataset, tokenizer=tokenizer, min_span_length=min_span_length, max_span_length=max_span_length, seed=seed)
        num_workers = int(os.environ.get("TORCH_NUM_WORKERS", 0))
        return DataLoader(
            iterable,
            batch_size=batch_size,
            collate_fn=collate_fn,
            num_workers=num_workers,
            persistent_workers=False,
            prefetch_factor=8 if num_workers > 0 else None,
            pin_memory=True,
        )

    def __iter__(self):
        generator = self._make_generator()

        while True:
            indices = generator.choices(range(self.N), k=1024)
            item_batch = self.ds.dataset.dataset[indices]
            input_ids_batch = self.tok(
                item_batch["text"],
                add_special_tokens=False,
                return_attention_mask=False,
            )["input_ids"]

            for i in range(1024):
                input_ids = input_ids_batch[i]
                seq_length = len(input_ids)
                if seq_length < self.min_span_length:
                    continue
                span_length = generator.randint(self.min_span_length, min(self.max_span_length, seq_length))
                window_start = generator.randint(0, seq_length - span_length)
                span = input_ids[window_start : window_start + span_length]
                yield torch.LongTensor(span)


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
        masked_anchors: bool = False,
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
        self.masked_anchors = masked_anchors
        if self.masked_anchors:
            assert (
                self.tok.mask_token_id is not None
            ), "Tokenizer must define mask_token_id when masked_anchors is enabled"
            assert (
                self.tok.pad_token_id is not None
            ), "Tokenizer must define pad_token_id when masked_anchors is enabled"
            assert (
                self.tok.vocab_size is not None
            ), "Tokenizer must define vocab_size when masked_anchors is enabled"

    def _apply_anchor_mlm(
        self,
        anchor_ids: List[int],
        rng: random.Random,
    ) -> Tuple[List[int], List[int]]:
        if len(anchor_ids) == 0:
            return anchor_ids, []

        num_to_mask = int(round(len(anchor_ids) * 0.15))
        if num_to_mask <= 0:
            return anchor_ids, [-100] * len(anchor_ids)

        positions = rng.sample(range(len(anchor_ids)), k=num_to_mask)
        mlm_labels = [-100] * len(anchor_ids)
        masked_ids = list(anchor_ids)

        for pos in positions:
            mlm_labels[pos] = anchor_ids[pos]
            coin = rng.random()
            if coin < 0.8:
                masked_ids[pos] = self.tok.mask_token_id
            elif coin < 0.9:
                rand_id = rng.randrange(self.tok.vocab_size)
                while rand_id in {self.tok.pad_token_id, self.tok.mask_token_id}:
                    rand_id = rng.randrange(self.tok.vocab_size)
                masked_ids[pos] = rand_id
            else:
                masked_ids[pos] = anchor_ids[pos]

        return masked_ids, mlm_labels

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
        masked_anchors: bool = False,
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
            if "anchor_labels" in concat_batch:
                concat_batch["anchor_labels"] = pad_sequence(
                    concat_batch["anchor_labels"],
                    batch_first=True,
                    padding_value=-100,
                )

            out = {
                "anchor_ids": concat_batch["anchor_ids"],
                "positive_ids": concat_batch["positive_ids"],
            }
            if "anchor_labels" in concat_batch:
                out["anchor_labels"] = concat_batch["anchor_labels"]
            return out

        iterable = cls(
            dataset,
            tokenizer=tokenizer,
            min_span_length=min_span_length,
            max_span_length=max_span_length,
            num_anchors=num_anchors,
            num_positives=num_positives,
            adjacent_positives=adjacent_positives,
            masked_anchors=masked_anchors,
            seed=seed,
        )

        num_workers = int(os.environ.get("TORCH_NUM_WORKERS", 0))
        return DataLoader(
            iterable,
            batch_size=batch_size,
            collate_fn=collate_fn,
            num_workers=num_workers,
            persistent_workers=True,
            prefetch_factor=4 if num_workers > 0 else None,
            pin_memory=True,
        )

    def __iter__(self):
        generator = self._make_generator()

        while True:
            # Sample a batch of 1024 random documents for I/O efficiency
            indices = generator.choices(range(self.N), k=512)
            item_batch = self.ds.dataset.dataset[indices]
            input_ids_batch = self.tok(
                item_batch["text"],
                add_special_tokens=False,
                return_attention_mask=False,
            )["input_ids"]

            for i in range(512):

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

                bos = self.tok.bos_token_id
                eos = self.tok.eos_token_id
                out_dict = {
                    "anchor_ids": [torch.LongTensor([bos] + a + [eos]) for a in anchors],
                    "positive_ids": [torch.LongTensor([bos] + p + [eos]) for p in positives],
                }

                if self.masked_anchors:
                    masked_anchors = []
                    anchor_labels = []
                    for anchor in anchors:
                        masked_ids, mlm_labels = self._apply_anchor_mlm(
                            anchor, generator
                        )
                        masked_anchors.append(torch.LongTensor([bos] + masked_ids + [eos]))
                        anchor_labels.append(torch.LongTensor([-100] + mlm_labels + [-100]))
                    out_dict["anchor_ids"] = masked_anchors
                    out_dict["anchor_labels"] = anchor_labels

                yield out_dict


# ---------------------------------------------------------------------------
# Dataset loading helpers for E5-style mixture training
# ---------------------------------------------------------------------------


def load_vitaminc_pairs() -> Tuple[List[str], List[str]]:
    """Load VitaminC SUPPORTS pairs as (claim, evidence)."""
    ds = load_dataset("tals/vitaminc", split="train")
    anchors, positives = [], []
    for row in ds:
        if row["label"] == "SUPPORTS":
            anchors.append(row["claim"])
            positives.append(row["evidence"])
    return anchors, positives


def load_anli_pairs() -> Tuple[List[str], List[str]]:
    """Load ANLI entailment pairs (all rounds) as (premise, hypothesis)."""
    anchors, positives = [], []
    for round_name in ("r1", "r2", "r3"):
        ds = load_dataset("facebook/anli", split=f"train_{round_name}")
        for row in ds:
            if row["label"] == 0:  # entailment
                anchors.append(row["premise"])
                positives.append(row["hypothesis"])
    return anchors, positives


def load_paws_pairs() -> Tuple[List[str], List[str]]:
    """Load PAWS paraphrase pairs (label=1)."""
    ds = load_dataset("google-research-datasets/paws", "labeled_final", split="train")
    anchors, positives = [], []
    for row in ds:
        if row["label"] == 1:
            anchors.append(row["sentence1"])
            positives.append(row["sentence2"])
    return anchors, positives


def load_snli_pairs(split: str = "train") -> Tuple[List[str], List[str]]:
    """Load SNLI entailment pairs (premise, hypothesis). Skips label=-1 rows."""
    ds = load_dataset("stanfordnlp/snli", split=split)
    anchors, positives = [], []
    for row in ds:
        if row["label"] == 0:  # entailment; -1 means no majority label
            anchors.append(row["premise"])
            positives.append(row["hypothesis"])
    return anchors, positives


def load_mnli_pairs(split: str = "train") -> Tuple[List[str], List[str]]:
    """Load MNLI entailment pairs. Use 'validation_matched' for validation."""
    ds = load_dataset("nyu-mll/multi_nli", split=split)
    anchors, positives = [], []
    for row in ds:
        if row["label"] == 0:  # entailment
            anchors.append(row["premise"])
            positives.append(row["hypothesis"])
    return anchors, positives


def load_yelp_polarity_groups() -> Dict[int, List[str]]:
    """Load Yelp Polarity reviews grouped by sentiment label (0=neg, 1=pos)."""
    ds = load_dataset("fancyzhx/yelp_polarity", split="train")
    groups: Dict[int, List[str]] = {0: [], 1: []}
    for row in ds:
        groups[row["label"]].append(row["text"])
    return groups


def load_ibm_argq_groups() -> Dict[str, List[str]]:
    """Load IBM ArgQ arguments grouped by topic."""
    ds = load_dataset("ibm-research/argument_quality_ranking_30k", "argument_quality_ranking", split="train")
    groups: Dict[str, List[str]] = {}
    for row in ds:
        topic = row["topic"]
        if topic not in groups:
            groups[topic] = []
        groups[topic].append(row["argument"])
    # Drop topics with fewer than 2 arguments
    return {k: v for k, v in groups.items() if len(v) >= 2}


# ---------------------------------------------------------------------------
# PairDatasetIterable — for explicit (anchor, positive) text pairs
# ---------------------------------------------------------------------------


class PairDatasetIterable(IterableDataset):
    """
    Iterable for datasets that provide explicit (anchor, positive) text pairs
    (e.g. VitaminC, ANLI, PAWS).

    When replacement=True (default): samples with replacement indefinitely.
    When replacement=False: shuffles all indices and yields each exactly once
    per epoch. Lightning restarts the iterator each epoch, and _epoch_counter
    advances the shuffle seed so each epoch sees a different ordering.

    Yields DeCLUTR-format dicts compatible with the standard collate_fn:
        {"anchor_ids": [LongTensor], "positive_ids": [LongTensor]}
    """

    def __init__(
        self,
        anchors: List[str],
        positives: List[str],
        tokenizer,
        max_length: int = 128,
        seed: Optional[int] = None,
        replacement: bool = True,
    ):
        assert len(anchors) == len(positives) and len(anchors) > 0
        self.anchors = anchors
        self.positives = positives
        self.N = len(anchors)
        self.tok = tokenizer
        self.max_length = max_length
        self.seed = seed
        self.replacement = replacement
        self._epoch_counter = 0

    def _make_generator(self) -> random.Random:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            rank = torch.distributed.get_rank()
        else:
            rank = 0
        wi = get_worker_info()
        wid = wi.id if wi is not None else 0
        base = self.seed
        if base is None:
            base = int.from_bytes(os.urandom(8), "little", signed=False)
        mixed = (
            int(base)
            ^ (0x9E3779B97F4A7C15 * (rank + 1))
            ^ (0xBF58476D1CE4E5B9 * (wid + 1))
        ) & ((1 << 63) - 1)
        return random.Random(mixed)

    def __iter__(self):
        if self.replacement:
            rng = self._make_generator()
            while True:
                indices = rng.choices(range(self.N), k=256)
                for idx in indices:
                    anchor_ids = self.tok.encode(
                        self.anchors[idx],
                        add_special_tokens=True,
                        max_length=self.max_length,
                        truncation=True,
                    )
                    positive_ids = self.tok.encode(
                        self.positives[idx],
                        add_special_tokens=True,
                        max_length=self.max_length,
                        truncation=True,
                    )
                    yield {
                        "anchor_ids": [torch.LongTensor(anchor_ids)],
                        "positive_ids": [torch.LongTensor(positive_ids)],
                    }
        else:
            # No replacement: shuffle all indices and yield each exactly once.
            # Use epoch counter to vary shuffle order across epochs.
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                rank = torch.distributed.get_rank()
            else:
                rank = 0
            wi = get_worker_info()
            wid = wi.id if wi is not None else 0
            num_workers = wi.num_workers if wi is not None else 1

            base = self.seed if self.seed is not None else 0
            # Same shuffle order for all workers on this rank; epoch counter varies per epoch.
            shuffle_seed = (
                int(base)
                ^ (0x9E3779B97F4A7C15 * (rank + 1))
                ^ (0xBF58476D1CE4E5B9 * (self._epoch_counter + 1))
            ) & ((1 << 63) - 1)
            rng = random.Random(shuffle_seed)
            indices = list(range(self.N))
            rng.shuffle(indices)
            self._epoch_counter += 1

            # Interleave across workers so each sees a disjoint subset.
            for i, idx in enumerate(indices):
                if i % num_workers != wid:
                    continue
                anchor_ids = self.tok.encode(
                    self.anchors[idx],
                    add_special_tokens=True,
                    max_length=self.max_length,
                    truncation=True,
                )
                positive_ids = self.tok.encode(
                    self.positives[idx],
                    add_special_tokens=True,
                    max_length=self.max_length,
                    truncation=True,
                )
                yield {
                    "anchor_ids": [torch.LongTensor(anchor_ids)],
                    "positive_ids": [torch.LongTensor(positive_ids)],
                }

    @classmethod
    def get_dataloader(
        cls,
        anchors: List[str],
        positives: List[str],
        tokenizer,
        batch_size: int,
        max_length: int = 128,
        seed: int = 32,
        replacement: bool = True,
    ) -> DataLoader:
        def collate_fn(batch):
            concat_batch: Dict[str, List] = {key: [] for key in batch[0].keys()}
            for item in batch:
                for key in concat_batch:
                    concat_batch[key].extend(item[key])
            concat_batch["anchor_ids"] = pad_sequence(
                concat_batch["anchor_ids"], batch_first=True,
                padding_value=tokenizer.pad_token_id,
            )
            concat_batch["positive_ids"] = pad_sequence(
                concat_batch["positive_ids"], batch_first=True,
                padding_value=tokenizer.pad_token_id,
            )
            return concat_batch

        iterable = cls(anchors=anchors, positives=positives,
                       tokenizer=tokenizer, max_length=max_length, seed=seed,
                       replacement=replacement)
        num_workers = int(os.environ.get("TORCH_NUM_WORKERS", 0))
        return DataLoader(
            iterable,
            batch_size=batch_size,
            collate_fn=collate_fn,
            num_workers=num_workers,
            persistent_workers=num_workers > 0,
            prefetch_factor=4 if num_workers > 0 else None,
            pin_memory=True,
        )


# ---------------------------------------------------------------------------
# DistillPairDatasetIterable — dual-tokenizer variant for distillation tasks
# ---------------------------------------------------------------------------


class DistillPairDatasetIterable(IterableDataset):
    """
    Iterable for distillation tasks: yields both student and teacher tokenizations
    for the same (anchor, positive) text pair.

    Batch keys:
        anchor_ids            [B, S_s]  student tokenization of anchor
        positive_ids          [B, S_s]  student tokenization of positive
        teacher_anchor_ids    [B, S_t]  teacher tokenization of anchor
        teacher_positive_ids  [B, S_t]  teacher tokenization of positive
    """

    def __init__(
        self,
        anchors: List[str],
        positives: List[str],
        student_tokenizer,
        teacher_tokenizer,
        student_max_length: int = 128,
        teacher_max_length: int = 128,
        seed: Optional[int] = None,
    ):
        assert len(anchors) == len(positives) and len(anchors) > 0
        self.anchors = anchors
        self.positives = positives
        self.N = len(anchors)
        self.s_tok = student_tokenizer
        self.t_tok = teacher_tokenizer
        self.s_max = student_max_length
        self.t_max = teacher_max_length
        self.seed = seed

    def _make_generator(self) -> random.Random:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            rank = torch.distributed.get_rank()
        else:
            rank = 0
        wi = get_worker_info()
        wid = wi.id if wi is not None else 0
        base = self.seed
        if base is None:
            base = int.from_bytes(os.urandom(8), "little", signed=False)
        mixed = (
            int(base)
            ^ (0x9E3779B97F4A7C15 * (rank + 1))
            ^ (0xBF58476D1CE4E5B9 * (wid + 1))
        ) & ((1 << 63) - 1)
        return random.Random(mixed)

    def __iter__(self):
        rng = self._make_generator()
        while True:
            indices = rng.choices(range(self.N), k=256)
            for idx in indices:
                a, p = self.anchors[idx], self.positives[idx]
                yield {
                    "anchor_ids": [torch.LongTensor(self.s_tok.encode(
                        a, add_special_tokens=True, max_length=self.s_max, truncation=True,
                    ))],
                    "positive_ids": [torch.LongTensor(self.s_tok.encode(
                        p, add_special_tokens=True, max_length=self.s_max, truncation=True,
                    ))],
                    "teacher_anchor_ids": [torch.LongTensor(self.t_tok.encode(
                        a, add_special_tokens=True, max_length=self.t_max, truncation=True,
                    ))],
                    "teacher_positive_ids": [torch.LongTensor(self.t_tok.encode(
                        p, add_special_tokens=True, max_length=self.t_max, truncation=True,
                    ))],
                }

    @classmethod
    def get_dataloader(
        cls,
        anchors: List[str],
        positives: List[str],
        student_tokenizer,
        teacher_tokenizer,
        batch_size: int,
        student_max_length: int = 128,
        teacher_max_length: int = 128,
        seed: int = 32,
    ) -> DataLoader:
        s_pad = student_tokenizer.pad_token_id
        t_pad = teacher_tokenizer.pad_token_id

        def collate_fn(batch):
            concat_batch: Dict[str, List] = {key: [] for key in batch[0].keys()}
            for item in batch:
                for key in concat_batch:
                    concat_batch[key].extend(item[key])
            concat_batch["anchor_ids"] = pad_sequence(
                concat_batch["anchor_ids"], batch_first=True, padding_value=s_pad,
            )
            concat_batch["positive_ids"] = pad_sequence(
                concat_batch["positive_ids"], batch_first=True, padding_value=s_pad,
            )
            concat_batch["teacher_anchor_ids"] = pad_sequence(
                concat_batch["teacher_anchor_ids"], batch_first=True, padding_value=t_pad,
            )
            concat_batch["teacher_positive_ids"] = pad_sequence(
                concat_batch["teacher_positive_ids"], batch_first=True, padding_value=t_pad,
            )
            return concat_batch

        iterable = cls(
            anchors=anchors,
            positives=positives,
            student_tokenizer=student_tokenizer,
            teacher_tokenizer=teacher_tokenizer,
            student_max_length=student_max_length,
            teacher_max_length=teacher_max_length,
            seed=seed,
        )
        num_workers = int(os.environ.get("TORCH_NUM_WORKERS", 0))
        return DataLoader(
            iterable,
            batch_size=batch_size,
            collate_fn=collate_fn,
            num_workers=num_workers,
            persistent_workers=True,
            prefetch_factor=4 if num_workers > 0 else None,
            pin_memory=True,
        )


# ---------------------------------------------------------------------------
# SameLabelPairIterable — for datasets without explicit pairs (same-label)
# ---------------------------------------------------------------------------


class SameLabelPairIterable(IterableDataset):
    """
    Iterable for datasets without explicit positive pairs. Texts are pre-grouped
    by label/topic; the iterable samples two texts from the same group as the
    anchor/positive pair.

    groups: Dict[label, List[str]]  — e.g. {0: ["neg text", ...], 1: ["pos text", ...]}
    """

    def __init__(
        self,
        groups: Dict,
        tokenizer,
        max_length: int = 128,
        seed: Optional[int] = None,
    ):
        # Filter out singleton groups
        self.groups = {k: v for k, v in groups.items() if len(v) >= 2}
        assert len(self.groups) > 0
        self.group_keys = list(self.groups.keys())
        self.tok = tokenizer
        self.max_length = max_length
        self.seed = seed

    def _make_generator(self) -> random.Random:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            rank = torch.distributed.get_rank()
        else:
            rank = 0
        wi = get_worker_info()
        wid = wi.id if wi is not None else 0
        base = self.seed
        if base is None:
            base = int.from_bytes(os.urandom(8), "little", signed=False)
        mixed = (
            int(base)
            ^ (0x9E3779B97F4A7C15 * (rank + 1))
            ^ (0xBF58476D1CE4E5B9 * (wid + 1))
        ) & ((1 << 63) - 1)
        return random.Random(mixed)

    def __iter__(self):
        rng = self._make_generator()
        while True:
            # Pick a random group, then sample 2 distinct texts from it
            key = rng.choice(self.group_keys)
            group = self.groups[key]
            anchor_text, positive_text = rng.sample(group, k=2)
            anchor_ids = self.tok.encode(
                anchor_text,
                add_special_tokens=True,
                max_length=self.max_length,
                truncation=True,
            )
            positive_ids = self.tok.encode(
                positive_text,
                add_special_tokens=True,
                max_length=self.max_length,
                truncation=True,
            )
            yield {
                "anchor_ids": [torch.LongTensor(anchor_ids)],
                "positive_ids": [torch.LongTensor(positive_ids)],
            }


# ---------------------------------------------------------------------------
# ContrastiveMixtureIterable — uniform mixture over per-dataset iterables
# ---------------------------------------------------------------------------


class ContrastiveMixtureIterable(IterableDataset):
    """
    Uniformly samples from a list of contrastive iterables, one dataset per batch.

    DDP correctness: the selector RNG uses `seed` only — NOT mixed with rank or
    worker id. This means all ranks/workers advance the same selector sequence
    independently and always agree on which dataset to sample from. The per-dataset
    child iterables use rank+worker-mixed seeds so each rank draws different items.

    The `batch_size` argument must match the DataLoader batch_size so that this
    iterable yields exactly `batch_size` items from one dataset before switching,
    keeping every DataLoader batch homogeneous.
    """

    def __init__(
        self,
        iterables: List[IterableDataset],
        batch_size: int,
        seed: int = 0,
    ):
        assert len(iterables) > 0
        self.iterables = iterables
        self.batch_size = batch_size
        self.seed = seed  # NOT rank-mixed — shared across all ranks/workers

    def __iter__(self):
        # Selector: same seed on all ranks/workers for agreement on dataset choice
        selector_rng = random.Random(self.seed)
        item_iters = [iter(it) for it in self.iterables]

        while True:
            task_idx = selector_rng.randrange(len(self.iterables))
            it = item_iters[task_idx]
            for _ in range(self.batch_size):
                yield next(it)

    @classmethod
    def get_dataloader(
        cls,
        iterables: List[IterableDataset],
        batch_size: int,
        tokenizer: PreTrainedTokenizerFast,
        seed: int = 0,
    ) -> DataLoader:
        def collate_fn(batch):
            concat_batch: Dict[str, List] = {}
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
            return {
                "anchor_ids": concat_batch["anchor_ids"],
                "positive_ids": concat_batch["positive_ids"],
            }

        iterable = cls(iterables, batch_size=batch_size, seed=seed)
        num_workers = int(os.environ.get("TORCH_NUM_WORKERS", 0))
        return DataLoader(
            iterable,
            batch_size=batch_size,
            collate_fn=collate_fn,
            num_workers=num_workers,
            persistent_workers=True,
            prefetch_factor=4 if num_workers > 0 else None,
            pin_memory=True,
        )


# ---------------------------------------------------------------------------
# MixedItemIterable — item-level mixture, safe for pairwise losses (DINO)
# ---------------------------------------------------------------------------


class MixedItemIterable(IterableDataset):
    """
    Item-level uniform mixture over a list of contrastive iterables.

    Unlike ContrastiveMixtureIterable, the dataset selector is chosen per item
    and the selector RNG IS mixed with rank+worker id. This means DDP ranks
    independently sample different datasets — safe because DINO's loss is
    pairwise and does not require batch-level dataset homogeneity.
    """

    def __init__(
        self,
        iterables: List[IterableDataset],
        seed: int = 0,
    ):
        assert len(iterables) > 0
        self.iterables = iterables
        self.seed = seed  # rank+worker-mixed for DDP independence

    def _make_selector(self) -> random.Random:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            rank = torch.distributed.get_rank()
        else:
            rank = 0
        wi = get_worker_info()
        wid = wi.id if wi is not None else 0
        mixed = (
            int(self.seed)
            ^ (0x9E3779B97F4A7C15 * (rank + 1))
            ^ (0xBF58476D1CE4E5B9 * (wid + 1))
        ) & ((1 << 63) - 1)
        return random.Random(mixed)

    def __iter__(self):
        rng = self._make_selector()
        item_iters = [iter(it) for it in self.iterables]
        while True:
            yield next(item_iters[rng.randrange(len(self.iterables))])

    @classmethod
    def get_dataloader(
        cls,
        iterables: List[IterableDataset],
        batch_size: int,
        tokenizer: PreTrainedTokenizerFast,
        seed: int = 0,
    ) -> DataLoader:
        def collate_fn(batch):
            concat_batch: Dict[str, List] = {}
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
            return {
                "anchor_ids": concat_batch["anchor_ids"],
                "positive_ids": concat_batch["positive_ids"],
            }

        iterable = cls(iterables, seed=seed)
        num_workers = int(os.environ.get("TORCH_NUM_WORKERS", 0))
        return DataLoader(
            iterable,
            batch_size=batch_size,
            collate_fn=collate_fn,
            num_workers=num_workers,
            persistent_workers=True,
            prefetch_factor=4 if num_workers > 0 else None,
            pin_memory=True,
        )
