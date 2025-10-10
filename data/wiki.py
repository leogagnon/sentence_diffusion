from dataclasses import dataclass
from typing import Optional
from datasets.load import load_from_disk
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data.dataset import Dataset

@dataclass
class WikipediaDatasetConfig:
    max_length: int 

class WikipediaDataset(Dataset):
    def __init__(self, cfg: Optional[WikipediaDatasetConfig] = None, **kwargs):
        if cfg == None:
            cfg = WikipediaDatasetConfig(**kwargs)

        self.dataset = load_from_disk(
            "/network/scratch/l/leo.gagnon/sentence_diffusion/data/wikipedia-paragraphs-filtered"
        )['train']
        self.cfg = cfg 

    def get_collate_and_tokenize_fn(self, dec_tokenizer = None, enc_tokenizer = None, prompt=None):
        def collate_fn(texts):

            out = {"input_str": texts}

            if enc_tokenizer is not None:
                
                # Maybe add prompt to encoder input
                if prompt is not None:
                    texts_enc = [(prompt+text) for text in texts]
                else:
                    texts_enc = texts

                batch_enc = enc_tokenizer.batch_encode_plus(
                    texts_enc,
                    truncation=True,
                    padding=True,
                    max_length=self.cfg.max_length,
                    return_tensors="pt",
                )
                out.update({
                    "input_ids_enc": batch_enc["input_ids"],
                    "attention_mask_enc": batch_enc["attention_mask"].bool(),
                })
                
            if dec_tokenizer is not None:
                batch_dec = dec_tokenizer.batch_encode_plus(texts)
                
                # Add bos and eos tokens and pad
                input_ids_dec = pad_sequence(
                    [
                        torch.LongTensor(
                            [dec_tokenizer.bos_token_id]
                            + ids
                            + [dec_tokenizer.eos_token_id]
                        )
                        for ids in batch_dec["input_ids"]
                    ],
                    padding_value=dec_tokenizer.pad_token_id,
                    batch_first=True,
                )[:, :self.cfg.max_length]
                attention_mask_dec = (input_ids_dec != dec_tokenizer.pad_token_id).bool()
                out.update({
                    "input_ids_dec": input_ids_dec,
                    "attention_mask_dec": attention_mask_dec,
                })

            return out

        return collate_fn

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[int(idx)]["input_ids"]
        