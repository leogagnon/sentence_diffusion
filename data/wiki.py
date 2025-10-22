from dataclasses import dataclass
from typing import Optional
from datasets.load import load_from_disk, load_dataset
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data.dataset import Dataset
from lightning.pytorch.utilities.rank_zero import rank_zero_info, rank_zero_only
from transformers import DataCollatorWithPadding

DATA_SEED = 42


def collate_fn(texts, dec_tokenizer, enc_tokenizer=None, max_length=None):
    out = {"input_str": texts}

    batch_dec = dec_tokenizer.batch_encode_plus(
        texts,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt",
    )

    out.update(
        {
            "input_ids_dec": batch_dec["input_ids"],
            "attention_mask_dec": batch_dec["attention_mask"].bool(),
        }
    )

    if enc_tokenizer is not None:
        batch_enc = enc_tokenizer.batch_encode_plus(
            texts,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )
        out.update(
            {
                "input_ids_enc": batch_enc["input_ids"],
                "attention_mask_enc": batch_enc["attention_mask"].bool(),
            }
        )

    return out


def collate_fn_conditional(texts, dec_tokenizer, enc_tokenizer, max_length=None):
    out = {"input_str": texts}

    # Tokenize with decoder determine split location
    batch_dec = dec_tokenizer.batch_encode_plus(texts)
    batch_len = torch.Tensor([len(t) for t in batch_dec])
    split_ratio = (0.75 - 0.25) * torch.rand(1024) + 0.25
    split_idx = (batch_len * split_ratio).int()

    # Collation + Padding part of the tokenizer
    dec_collator = DataCollatorWithPadding(
        dec_tokenizer, padding=True, return_tensors="pt"
    )

    # Tokenize prompt only with decoder
    batch_dec_prompt = dec_collator([t[:s] for t, s in zip(batch_dec, split_idx)])
    
    # Tokenize cont with encoder and decoder
    batch_dec_cont = [t[s:] for t, s in zip(batch_dec, split_idx)]
    batch_enc_cont = enc_tokenizer.batch_encode_plus(
        dec_tokenizer.batch_decode(batch_dec_prompt),
        truncation=True,
        padding=True,
        max_length=max_length,
        return_tensors="pt",
    )
    batch_dec_cont = dec_collator(batch_dec_cont)

    out.update(
        {   
            # Prompt and continuation for decoder
            "prompt_ids_dec": batch_dec_prompt["input_ids"],
            "prompt_mask_dec": batch_dec_prompt["attention_mask"].bool(),
            "cont_ids_dec": batch_dec_cont["input_ids"],
            "cont_mask_dec": batch_dec_cont["attention_mask"].bool(),
            # Only continuation for encoder
            "cont_ids_enc": batch_enc_cont["input_ids"],
            "cont_mask_enc": batch_enc_cont["attention_mask"].bool(),
        }
    )

    return out


@dataclass
class WikipediaDatasetConfig:
    max_length: int


class WikipediaDataset(Dataset):
    def __init__(self, cfg: Optional[WikipediaDatasetConfig] = None, **kwargs):
        if cfg == None:
            cfg = WikipediaDatasetConfig(**kwargs)
        self.dataset = load_from_disk("data/wikipedia-paragraphs-filtered")["train"]
        self.cfg = cfg

    def get_collate_and_tokenize_fn(self, dec_tokenizer=None, enc_tokenizer=None):
        def collate_fn(texts):

            out = {"input_str": texts}

            if enc_tokenizer is not None:
                batch_enc = enc_tokenizer.batch_encode_plus(
                    texts,
                    truncation=True,
                    padding="max_length",
                    max_length=self.cfg.max_length,
                    return_tensors="pt",
                )
                out.update(
                    {
                        "input_ids_enc": batch_enc["input_ids"],
                        "attention_mask_enc": batch_enc["attention_mask"].bool(),
                    }
                )

            if dec_tokenizer is not None:
                batch_dec = dec_tokenizer.batch_encode_plus(
                    texts,
                    truncation=True,
                    padding="max_length",
                    max_length=self.cfg.max_length,
                    return_tensors="pt",
                )

                out.update(
                    {
                        "input_ids_dec": batch_dec["input_ids"],
                        "attention_mask_dec": batch_dec["attention_mask"].bool(),
                    }
                )

            return out

        return collate_fn

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[int(idx)]["input_ids"]
