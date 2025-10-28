from dataclasses import dataclass
from typing import Optional
from datasets.load import load_from_disk, load_dataset
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data.dataset import Dataset
from lightning.pytorch.utilities.rank_zero import rank_zero_info, rank_zero_only
from transformers import DataCollatorWithPadding, PreTrainedTokenizerFast
from functools import partial
from model.encoder import EncoderModel
from enum import Enum

DATA_SEED = 42


class InfoLabel(Enum):
    CONT = 0
    DLC = 1
    PAD = 2
    PROMPT = 3


@torch.no_grad()
def collate_fn(
    texts,
    max_length,
    dec_tokenizer: PreTrainedTokenizerFast,
    enc_tokenizer: Optional[PreTrainedTokenizerFast] = None,
    dlc_encoder: Optional[EncoderModel] = None,
):
    out = {"input_str": texts}

    # No need for attention mask because the decoder is causal
    batch_dec = dec_tokenizer.batch_encode_plus(
        texts,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt",
        add_special_tokens=True,
        return_attention_mask=False,
    )

    out.update(
        {
            "input_ids_dec": batch_dec["input_ids"],
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

    # <|think|> DLC <|bos|> x <|eos|>
    if dlc_encoder is not None:
        device = dlc_encoder.transformer.auto_model.device
        # Compute DLC
        batch_enc = dlc_encoder.tokenizer.batch_encode_plus(
            texts,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt"
        ).to(device=device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            dlc_ids = dlc_encoder(
                batch_enc["input_ids"], batch_enc["attention_mask"].bool(), return_dlc=True
            ) + len(dec_tokenizer)

        # Preprent <|think|> DLC to <|bos|> x <|eos|>
        think_token = torch.full(
            size=(dlc_ids.shape[0], 1),
            fill_value=dec_tokenizer.think_token_id,
            dtype=dlc_ids.dtype,
            device=device,
        )
        input_ids_dec = torch.cat([think_token, dlc_ids, out["input_ids_dec"].cuda()], dim=1)

        # Build the info mask
        info_mask = torch.full_like(
            input_ids_dec, fill_value=InfoLabel.CONT.value, dtype=torch.int32
        )
        info_mask[input_ids_dec == dec_tokenizer.pad_token_id] = InfoLabel.PAD.value
        info_mask[:, : dlc_ids.shape[1] + 1] = InfoLabel.DLC.value
        info_mask[:, 0] = InfoLabel.PROMPT.value
        out.update(
            {
                "input_ids_dec": input_ids_dec,
                "info_mask_dec": info_mask,
            }
        )

    return out


def collate_fn_conditional(
    texts,
    max_length,
    dec_tokenizer: PreTrainedTokenizerFast,
    dlc_encoder: EncoderModel,
    enc_tokenizer = None
):
    out = {"input_str": texts}

    # Tokenize with decoder and split at random location
    # Keeps everything in lists and doesn't add BOS/EOS
    batch_dec = dec_tokenizer.batch_encode_plus(
        texts, add_special_tokens=False, return_attention_mask=False, return_length=True
    )
    split_ratio = (0.75 - 0.25) * torch.rand(1024) + 0.25
    split_idx = (torch.tensor(batch_dec["length"]) * split_ratio).int()
    prompt_enc = [t[:s] for t, s in zip(batch_dec, split_idx)]
    continuation_dec = [t[s:] for t, s in zip(batch_dec, split_idx)]

    # Compute DLC
    continuation_enc = dlc_encoder.batch_encode_plus(
        dec_tokenizer.batch_decode(continuation_dec, skip_special_tokens=True),
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt",
    )
    dlc_ids = dlc_encoder(
        continuation_enc["input_ids"],
        continuation_enc["attention_mask"].bool(),
        return_dlc=True,
    ) + len(dec_tokenizer)

    # Build the input_ids for the decoder
    # prompt <|think|> DLC <|bos|> continuation <|eos|>
    batch_dec = [
        prompt
        + dec_tokenizer.think_token_id
        + dlc
        + dec_tokenizer.bos_token_id
        + continuation
        + dec_tokenizer.eos_token_id
        for (prompt, dlc, continuation) in zip(prompt_enc, dlc_ids, continuation_dec)
    ]
    batch_dec = dec_tokenizer.pad(
        batch_dec,
        padding="max_length",
        return_tensors="pt",
        max_length=max_length + 32,
    )

    # Build the info mask
    info_mask = torch.full_like(
        batch_dec["input_ids"], fill_value=InfoLabel.CONT, type=torch.int32
    )
    info_mask[batch_dec["input_ids"] == dec_tokenizer.pad_token_id] = InfoLabel.PAD
    for i in range(len(info_mask)):
        info_mask[i, : len(prompt_enc[i]) + 1] = InfoLabel.PROMPT
        info_mask[
            i, (len(prompt_enc[i]) + 1) : (len(prompt_enc[i]) + 1 + dlc_ids.shape[1]) :
        ] = InfoLabel.DLC

    out.update(
        {
            # Prompt and continuation for decoder
            "input_ids_dec": batch_dec["input_ids"],
            "info_mask": info_mask,
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

    def __getitem__(self, idx):
        return self.dataset[int(idx)]["input_ids"]
