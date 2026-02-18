"""
Precompute MAUVE reference features from training data for evaluation.
This script generates reference prefix+suffix samples from the training set.
"""

import os
import sys
import argparse
from pathlib import Path

import torch
import numpy as np
from omegaconf import OmegaConf
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data import LanguageDataset, LanguageDatasetConfig, PrefixSuffixIterable
from torch.utils.data.dataset import Subset
from mauve import get_features_from_input, get_tokenizer


def main(args):
    # Create dataset config
    dataset_cfg = LanguageDatasetConfig(
        name=args.dataset_name,
        val_size=args.dataset_val_size,
        seed=args.dataset_seed,
    )

    # Create dataset
    dataset = LanguageDataset(dataset_cfg)
    train_data = Subset(dataset, indices=dataset.train_indices)

    # Split total_size into prefix and suffix (we'll just use half/half)
    # For MAUVE we combine them anyway, so the split doesn't matter
    prefix_length = args.total_size // 2
    suffix_length = args.total_size - prefix_length

    # Load tokenizer for featurization
    print(f"Loading tokenizer for {args.model_name}...")
    tokenizer = get_tokenizer(args.model_name)

    # Add padding token if not present (GPT2 doesn't have one by default)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "<|pad|>"})

    # Create dataloader with GPT2-large tokenizer
    dataloader = PrefixSuffixIterable.get_dataloader(
        train_data,
        batch_size=args.batch_size,
        prefix_length=prefix_length,
        suffix_length=suffix_length,
        context_length=0,
        enc_tok=None,  # We don't need encoder tokenization
        dec_tok=tokenizer,  # Use GPT2-large tokenizer
        encoder_mode="none",
        encoder_noise=False,
        seed=args.seed,
        num_dlc_ph=0,  # No DLC placeholders needed
    )

    # Collect tokenized prefix+suffix texts
    print(f"Collecting {args.num_samples} reference samples from training data...")
    reference_tokenized_texts = []

    dataloader_iter = iter(dataloader)
    num_batches = (args.num_samples + args.batch_size - 1) // args.batch_size

    for _ in tqdm(range(num_batches), desc="Collecting samples"):
        batch = next(dataloader_iter)

        # Get tokenized sequences (prefix+suffix combined)
        # The batch contains input_ids_dec which is the full sequence
        for input_ids in batch["input_ids_dec"]:
            if len(reference_tokenized_texts) >= args.num_samples:
                break
            # Convert to tensor format expected by get_features_from_input
            reference_tokenized_texts.append(input_ids.unsqueeze(0))

        if len(reference_tokenized_texts) >= args.num_samples:
            break

    reference_tokenized_texts = reference_tokenized_texts[:args.num_samples]
    print(f"Collected {len(reference_tokenized_texts)} tokenized reference texts")

    # Featurize using GPT2-large
    print(f"Featurizing with {args.model_name} on device {args.device_id}...")
    reference_features = get_features_from_input(
        features=None,
        tokenized_texts=reference_tokenized_texts,
        texts=None,
        featurize_model_name=args.model_name,
        max_len=args.max_len,
        device_id=args.device_id,
        name="reference",
        batch_size=args.feature_batch_size,
        verbose=True,
    )

    # Save features
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Saving features to {output_path}...")
    np.save(output_path, reference_features)

    print(f"Done! Saved {reference_features.shape[0]} features of dimension {reference_features.shape[1]}")
    print(f"Feature shape: {reference_features.shape}")
    print(f"Feature dtype: {reference_features.dtype}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Precompute MAUVE reference features")

    # Dataset parameters
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="fineweb",
        help="Dataset name (e.g., 'fineweb')",
    )
    parser.add_argument(
        "--dataset_val_size",
        type=int,
        default=16384,
        help="Validation set size",
    )
    parser.add_argument(
        "--dataset_seed",
        type=int,
        default=42,
        help="Dataset random seed",
    )

    # Sequence parameters
    parser.add_argument(
        "--total_size",
        type=int,
        default=80,
        help="Total sequence size (prefix + suffix length, e.g., 80 for 16+64)",
    )

    # Output parameters
    parser.add_argument(
        "--output_path",
        type=str,
        default="mauve_reference_features.npy",
        help="Path to save reference features",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=5000,
        help="Number of reference samples to collect",
    )

    # Batch parameters
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size for data collection",
    )
    parser.add_argument(
        "--feature_batch_size",
        type=int,
        default=64,
        help="Batch size for feature extraction",
    )

    # Featurization parameters
    parser.add_argument(
        "--model_name",
        type=str,
        default="gpt2-large",
        help="Model name for featurization",
    )
    parser.add_argument(
        "--max_len",
        type=int,
        default=150,
        help="Maximum sequence length for tokenization",
    )
    parser.add_argument(
        "--device_id",
        type=int,
        default=0,
        help="GPU device ID (-1 for CPU)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for data sampling",
    )

    args = parser.parse_args()
    main(args)
