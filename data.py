from hydra.utils import to_absolute_path
from torch.utils.data import Dataset
import torch
from dataclasses import dataclass
from typing import Callable, Optional
import pandas as pd
from tokenizers import Tokenizer
from sentence_transformers import SentenceTransformer

@dataclass
class StoriesDatasetConfig:
    csv_path: str
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

    def __init__(self, cfg: StoriesDatasetConfig, tokenizer: Tokenizer, semb: SentenceTransformer):
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
        self.cfg.csv_path = to_absolute_path(self.cfg.csv_path)

        df = pd.read_csv(self.cfg.csv_path)

        # Build in-memory records
        self.data = []
        for idx in range(self.cfg.start_index, min(self.cfg.end_index, len(df))):
            row = df.iloc[idx]

            input_str = " ".join(row)
            item = {"input_str" : input_str,
                    "input_idx": tokenizer.encode(input_str)["input_ids"][0],
                    "input_emb": semb.encode([input_str])[0]}
            self.data.append(item)


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx].copy()
        # Expose data as-is. Token tensors are torch.LongTensor if tokenizer provided.
        return item