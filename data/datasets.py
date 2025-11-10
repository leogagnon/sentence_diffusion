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

DATA_SEED = 42


@dataclass
class WikipediaDatasetConfig:
    max_length: int


class WikipediaDataset(Dataset):
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
            generator=torch.Generator().manual_seed(DATA_SEED),
        )
        train_indices = indices[:-val_size]
        val_indices = indices[-val_size:]

        return train_indices, val_indices

    def __len__(self):
        return len(self.dataset)

    def __getitems__(self, indices):
        return {"input_str": self.dataset[indices]["input_ids"]}

    def __getitem__(self, idx):
        return {"input_str": self.dataset[int(idx)]["input_ids"]}


@dataclass
class FineWebDatasetConfig:
    length_interval: Tuple[int, int]


class FineWebDataset(Dataset):
    def __init__(self, cfg: Optional[FineWebDatasetConfig] = None, **kwargs):
        if cfg == None:
            cfg = FineWebDatasetConfig(**kwargs)
        self.dataset = load_dataset(
            "leogagnon/fineweb_100BT_tokenized_gpt2", split="train"
        )
        self.pre_tokenizer = AutoTokenizer.from_pretrained("gpt2-large")
        self.cfg = cfg
        self.max_length = self.cfg.length_interval[1]

    def get_train_val_indices(self, val_size):
        indices = torch.randperm(
            len(self.dataset),
            generator=torch.Generator().manual_seed(DATA_SEED),
        )
        train_indices = indices[:-val_size]
        val_indices = indices[-val_size:]

        return train_indices, val_indices

    def __len__(self):
        return len(self.dataset)

    def __getitems__(self, indices):
        input_ids = self.dataset[indices]["input_ids"]

        # Select a random window
        full_len = torch.Tensor([len(x) for x in input_ids])
        window_len = torch.randint(
            low=self.cfg.length_interval[0],
            high=self.cfg.length_interval[1]+1,
            size=(len(input_ids),)
        )
        window_start = (
            torch.rand(len(input_ids))
            * torch.clamp(full_len - window_len, min=0)
        ).int()
        input_ids = [
            x[s : s + l] for x, s, l in zip(input_ids, window_start, window_len)
        ]

        # Decode input_str
        input_str = self.pre_tokenizer.batch_decode(input_ids, skip_special_tokens=True)

        return {"input_ids": input_ids, "input_str": input_str}

    def __getitem__(self, idx):
        return {"input_ids": self.dataset[int(idx)]["input_ids"]}
