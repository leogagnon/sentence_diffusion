from dataclasses import dataclass
from typing import Optional
from datasets.load import load_from_disk, load_dataset
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data.dataset import Dataset
from lightning.pytorch.utilities.rank_zero import rank_zero_info, rank_zero_only
from transformers import DataCollatorWithPadding, PreTrainedTokenizerFast
from functools import partial

DATA_SEED = 42


def collate_fn(
    texts,
    max_length,
    dec_tokenizer: PreTrainedTokenizerFast,
    enc_tokenizer: Optional[PreTrainedTokenizerFast] = None,
):
    out = {"input_str": texts}

    batch_dec = dec_tokenizer.batch_encode_plus(
        texts,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt",
        add_special_tokens=True,
    )

    out.update(
        {
            "cont_ids_dec": batch_dec["input_ids"],
            "cont_mask_dec": batch_dec["attention_mask"].bool(),
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
                "cont_ids_enc": batch_enc["input_ids"],
                "cont_mask_enc": batch_enc["attention_mask"].bool(),
            }
        )

    return out


def collate_fn_conditional(
    texts,
    max_length,
    dec_tokenizer: PreTrainedTokenizerFast,
    enc_tokenizer: PreTrainedTokenizerFast,
):
    out = {"input_str": texts}

    # Tokenize with decoder determine split location
    batch_dec = dec_tokenizer.batch_encode_plus(
        texts,
        add_special_tokens=False,
        return_tensors="pt",
        padding="max_length",
        max_length=max_length,
    )
    batch_len = (batch_dec["input_ids"] != dec_tokenizer.pad_token_id).sum(-1)
    split_ratio = (0.75 - 0.25) * torch.rand(1024) + 0.25
    split_idx = (batch_len * split_ratio).int()

    # Collation + Padding part of the tokenizer
    dec_collator = DataCollatorWithPadding(
        dec_tokenizer, padding=True, return_tensors="pt"
    )

    # Tokenize prompt and continuation with decoder
    # prompt <|think|>
    # <|bos|> cont <|eos|>
    # DLC will be put in between
    batch_dec_prompt = dec_collator([t[:s] for t, s in zip(batch_dec, split_idx)])
    batch_dec_cont = dec_collator(
        [
            [dec_tokenizer.bos_token_id + t[s:] + dec_tokenizer.eos_token_id]
            for t, s in zip(batch_dec, split_idx)
        ]
    )

    # Tokenize continuation with encoder
    batch_enc_cont = enc_tokenizer.batch_encode_plus(
        dec_tokenizer.batch_decode(batch_dec_prompt),
        truncation=True,
        padding=True,
        max_length=max_length,
        return_tensors="pt",
    )

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

    def get_collate_and_tokenize_fn(
        self, conditional: bool, dec_tokenizer, enc_tokenizer=None
    ):
        f = collate_fn_conditional if conditional else collate_fn
        f = partial(
            f,
            max_length=self.cfg.max_length,
            dec_tokenizer=dec_tokenizer,
            enc_tokenizer=enc_tokenizer,
        )
        return f

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[int(idx)]["input_ids"]
