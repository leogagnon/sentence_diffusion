from hydra.utils import to_absolute_path
from torch.utils.data import Dataset
import torch
from dataclasses import dataclass
from typing import Callable, List, Optional
import pandas as pd
from tokenizers import Tokenizer
from sentence_transformers import SentenceTransformer
from torch.nn.utils.rnn import pad_sequence
import os


@dataclass
class StoriesDatasetConfig:
    path: str
    start_index: int = 100
    end_index: int = 1000


class StoriesDataset(Dataset):
    """
    PyTorch Dataset for the 'stories in prompt' style data.

    Each item is a dict with the following keys:
        - "context": the first 3 sentences (string)
        - "x": the 4th sentence (string)
        - "y": the 5th sentence (string)
        - "context_input_ids": optional tokenized context ids (torch.LongTensor) if tokenizer is provided
        - "x_input_ids": optional tokenized x ids if tokenizer is provided
        - "y_input_ids": optional tokenized y ids if tokenizer is provided

    This class does not move tensors to GPU; that should be handled by the training loop / collate_fn.
    """

    def __init__(self, cfg: StoriesDatasetConfig, tokenizers: List[Tokenizer]):
        """
        Args:
            csv_path (str): path to CSV file (will be passed through hydra.to_absolute_path)
            tokenizer (callable, optional): huggingface-style tokenizer. If provided, tokenization
                will be performed in __init__ and returned as torch tensors (not moved to cuda).
            start_index (int): first CSV row index to load (inclusive)
            end_index (int): last CSV row index to load (exclusive)
        """

        # store cfg and resolve paths
        self.cfg = cfg
        self.cfg.path = to_absolute_path(self.cfg.path)

        df = pd.read_csv(os.path.join(self.cfg.path, "stories.csv"))

        sentences = [df[f"sentence{i}"].tolist() for i in range(1, 6)]
        self.sentences = [
            " ".join([sentences[j][i] for j in range(5)])
            for i in range(len(sentences[0]))
        ]

        # Embedding cache path
        self.input_ids = [
            tokenizer.batch_encode_plus(self.sentences)["input_ids"]
            for tokenizer in tokenizers
        ]

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        item = {
            "input_str": self.sentences[idx],
            "input_ids": self.input_ids[idx],
        }
        return item

    def __getitems__(self, indices):
        batch = {
            "input_str": [self.sentences[idx] for idx in indices],
            "indices": torch.LongTensor(indices),
        }
        for i in range(len(self.input_ids)):
            batch[f"input_ids_{i}"] = [
                pad_sequence(
                    [torch.LongTensor(self.input_ids[i][idx]) for idx in indices],
                    padding_value=0,
                    batch_first=True,
                )
            ]

        return batch
