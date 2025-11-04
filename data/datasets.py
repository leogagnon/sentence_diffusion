from dataclasses import dataclass
from typing import Optional
from datasets.load import load_from_disk, load_dataset
from hydra.utils import to_absolute_path
import pandas as pd
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

DATA_SEED = 42


class InfoLabel(Enum):
    CONT = 0
    DLC = 1
    PAD = 2
    PROMPT = 3


@torch.no_grad()
@torch.autocast(device_type="cuda", dtype=torch.bfloat16)
def collate_fn(
    batch,
    max_length,
    dec_tokenizer: Optional[PreTrainedTokenizerFast] = None,
    enc_tokenizer: Optional[PreTrainedTokenizerFast] = None,
    teacher_tokenizer: Optional[PreTrainedTokenizerFast] = None,
    dlc_encoder: Optional[EncoderModel] = None,
):
    out = {}

    # Compute the input_ids for the decoder
    if "input_ids" in batch:
        assert dec_tokenizer is not None
        lens = torch.Tensor([len(x) for x in batch["input_ids"]])
        window_len = torch.randint(
            low=32, high=max_length - 32, size=(len(batch["input_ids"]),)
        )
        window_start = (
            torch.rand(len(batch["input_ids"])) * torch.clamp(lens - window_len, min=0)
        ).int()
        input_ids_dec = [
            [dec_tokenizer.bos_token_id] + x[s : s + l]
            for x, s, l in zip(batch["input_ids"], window_start, window_len)
        ]
        input_ids_dec = dec_tokenizer.pad(
            {"input_ids": input_ids_dec},
            padding="max_length",
            return_tensors="pt",
            max_length=max_length,
            return_attention_mask=False,
        )["input_ids"]
        input_str = dec_tokenizer.batch_decode(input_ids_dec, skip_special_tokens=True)
    elif "input_str" in batch:
        if dec_tokenizer is None:
            input_ids_dec = dec_tokenizer.batch_encode_plus(
                batch["input_str"],
                truncation=True,
                padding="max_length",
                max_length=max_length,
                return_tensors="pt",
                add_special_tokens=True,
                return_attention_mask=False,
            )["input_ids"]
        else:
            input_ids_dec = None
        input_str = batch["input_str"]
    
    out.update({"input_str": input_str})

    if dec_tokenizer is not None:
        if dlc_encoder is None:
            info_mask_dec = torch.full_like(
                input_ids_dec, fill_value=InfoLabel.CONT.value, dtype=torch.int32
            )
            info_mask_dec[input_ids_dec == dec_tokenizer.pad_token_id] = (
                InfoLabel.PAD.value
            )
        else:
            device = dlc_encoder.transformer.auto_model.device
            # Compute DLC
            batch_enc = dlc_encoder.tokenizer.batch_encode_plus(
                input_str,
                truncation=True,
                padding="max_length",
                max_length=max_length,
                return_tensors="pt",
            ).to(device=device)
            dlc_ids = dlc_encoder(
                batch_enc["input_ids"],
                batch_enc["attention_mask"].bool(),
                return_dlc=True,
            )[1] + len(dec_tokenizer)

            # Preprend "<|think|> DLC" to input_ids_dec
            think_token = torch.full(
                size=(dlc_ids.shape[0], 1),
                fill_value=dec_tokenizer.think_token_id,
                dtype=dlc_ids.dtype,
                device=device,
            )
            input_ids_dec = torch.cat(
                [think_token, dlc_ids, input_ids_dec.to(device=device)], dim=1
            )

            # Build the info mask
            info_mask_dec = torch.full_like(
                input_ids_dec,
                fill_value=InfoLabel.CONT.value,
                dtype=torch.int32,
                device=device,
            )
            info_mask_dec[input_ids_dec == dec_tokenizer.pad_token_id] = (
                InfoLabel.PAD.value
            )
            info_mask_dec[:, : dlc_ids.shape[1] + 1] = InfoLabel.DLC.value
            info_mask_dec[:, 0] = InfoLabel.PROMPT.value

        out.update(
            {
                "input_ids_dec": input_ids_dec,
                "info_mask_dec": info_mask_dec,
            }
        )

    # Compute the input ids for the encoder if any
    if enc_tokenizer is not None:
        batch_enc = enc_tokenizer.batch_encode_plus(
            input_str,
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

    # Compute the input ids for the teacher encoder if any
    if teacher_tokenizer is not None:
        batch_teacher = teacher_tokenizer.batch_encode_plus(
            input_str,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )
        out.update(
            {
                "input_ids_teacher": batch_teacher["input_ids"],
                "attention_mask_teacher": batch_teacher["attention_mask"].bool(),
            }
        )

    return out


@torch.no_grad
@torch.autocast(device_type="cuda", dtype=torch.bfloat16)
def collate_fn_conditional(
    batch,
    max_length,
    dec_tokenizer: PreTrainedTokenizerFast,
    dlc_encoder: EncoderModel,
    enc_tokenizer=None,
):
    assert (
        "input_ids" in batch
    ), "Conditional DLC only supports pre-tokenized datasets for now"

    # Pick a window and randomly split
    lens = torch.Tensor([len(x) for x in batch["input_ids"]])
    window_start = (
        torch.rand(len(batch["input_ids"])) * torch.clamp(lens - max_length, min=0)
    ).int()
    prompt_len = torch.randint(
        low=32, high=max_length - 32, size=(len(batch["input_ids"]),)
    )
    prompt_dec = [
        x[s : s + l] for x, s, l in zip(batch["input_ids"], window_start, prompt_len)
    ]
    continuation_dec = [
        x[s + l : s + max_length]
        for x, s, l in zip(batch["input_ids"], window_start, prompt_len)
    ]
    continuation_str = dec_tokenizer.batch_decode(
        continuation_dec, skip_special_tokens=True
    )

    # Compute DLC of continuation
    continuation_enc = dlc_encoder.tokenizer.batch_encode_plus(
        continuation_str,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt",
    )
    dlc_ids = dlc_encoder(
        continuation_enc["input_ids"],
        continuation_enc["attention_mask"].bool(),
        return_dlc=True,
    )[1] + len(dec_tokenizer)

    # Build the input_ids for the decoder
    # prompt <|think|> DLC <|bos|> continuation
    input_ids_dec = [
        prompt
        + dec_tokenizer.think_token_id
        + dlc
        + dec_tokenizer.bos_token_id
        + continuation
        for (prompt, dlc, continuation) in zip(prompt_dec, dlc_ids, continuation_dec)
    ]
    input_ids_dec = dec_tokenizer.pad(
        {"input_ids": input_ids_dec},
        padding="max_length",
        return_tensors="pt",
        max_length=max_length + 32,
        return_attention_mask=False,
    )

    # Build the info mask
    info_mask_dec = torch.full_like(
        input_ids_dec, fill_value=InfoLabel.CONT, type=torch.int32
    )
    info_mask_dec[input_ids_dec == dec_tokenizer.pad_token_id] = InfoLabel.PAD
    for i in range(len(info_mask_dec)):
        info_mask_dec[i, : len(prompt_dec[i]) + 1] = InfoLabel.PROMPT
        info_mask_dec[
            i, (len(prompt_dec[i]) + 1) : (len(prompt_dec[i]) + 1 + dlc_ids.shape[1]) :
        ] = InfoLabel.DLC

    out = {"input_ids_dec": input_ids_dec, "info_mask_dec": info_mask_dec}

    return out


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

    def get_collate_and_tokenize_fn(
        self, conditional: bool, dec_tokenizer, enc_tokenizer=None, dlc_encoder=None
    ):
        f = collate_fn_conditional if conditional else collate_fn
        f = partial(
            f,
            max_length=self.cfg.max_length,
            dec_tokenizer=dec_tokenizer,
            enc_tokenizer=enc_tokenizer,
            dlc_encoder=dlc_encoder,
        )
        return f

    def __len__(self):
        return len(self.dataset)

    def __getitems__(self, indices):
        return {"input_str": self.dataset[indices]["input_ids"]}

    def __getitem__(self, idx):
        return {"input_str": self.dataset[int(idx)]["input_ids"]}


@dataclass
class FineWebDatasetConfig:
    max_length: int


class FineWebDataset(Dataset):
    def __init__(self, cfg: Optional[FineWebDatasetConfig] = None, **kwargs):
        if cfg == None:
            cfg = FineWebDatasetConfig(**kwargs)
        self.dataset = load_from_disk("data/fineweb_tokenized")
        self.cfg = cfg

    def get_collate_and_tokenize_fn(
        self, conditional: bool, dec_tokenizer, enc_tokenizer=None, dlc_encoder=None
    ):
        f = collate_fn_conditional if conditional else collate_fn
        f = partial(
            f,
            max_length=self.cfg.max_length,
            dec_tokenizer=dec_tokenizer,
            enc_tokenizer=enc_tokenizer,
            dlc_encoder=dlc_encoder,
        )
        return f

    def __len__(self):
        return len(self.dataset)

    def __getitems__(self, indices):
        return {"input_ids": self.dataset[indices]["input_ids"]}

    def __getitem__(self, idx):
        return {"input_ids": self.dataset[int(idx)]["input_ids"]}


@dataclass
class StoriesDatasetConfig:
    path: str


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

    def __init__(
        self,
        cfg: StoriesDatasetConfig,
        dec_tokenizer: Tokenizer,
        enc_tokenizer: Optional[Tokenizer] = None,
    ):
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

        # Tokenize sentences
        self.input_ids = {
            "enc": (
                enc_tokenizer.batch_encode_plus(self.sentences)["input_ids"]
                if enc_tokenizer != None
                else None
            ),
            "dec": dec_tokenizer.batch_encode_plus(self.sentences)["input_ids"],
        }
        # For some reason add_special_tokens doesn't work, so add BOS/EOS manually to decoder
        self.input_ids["dec"] = [
            [dec_tokenizer.bos_token_id] + ids + [dec_tokenizer.eos_token_id]
            for ids in self.input_ids["dec"]
        ]

        self.padding_token_id = {
            "enc": enc_tokenizer.pad_token_id if enc_tokenizer != None else None,
            "dec": dec_tokenizer.pad_token_id,
        }

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
        for key in self.input_ids.keys():
            if self.input_ids[key] is not None:
                batch[f"input_ids_{key}"] = pad_sequence(
                    [torch.LongTensor(self.input_ids[key][idx]) for idx in indices],
                    padding_value=self.padding_token_id[key],
                    batch_first=True,
                )
                batch[f"attention_mask_{key}"] = (
                    batch[f"input_ids_{key}"] != self.padding_token_id[key]
                )

        return batch
