#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Extract BERT text features for text-only OAD experiments.

Input:
    text_manifest_for_oad.csv

Output:
    bert_text_features.npy
    bert_text_features_index.csv

Notes:
    - Features are extracted in the SAME row order as the input CSV.
    - Default text column is `text_strict`.
    - Mean pooling is used over valid tokens.
    - Model is frozen; this is offline feature extraction only.
"""

from __future__ import annotations

import argparse
import os
from typing import List

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


def mean_pooling(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """
    last_hidden_state: [B, L, H]
    attention_mask:    [B, L]
    return:            [B, H]
    """
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()  # [B, L, H]
    masked_embeddings = last_hidden_state * mask
    sum_embeddings = masked_embeddings.sum(dim=1)  # [B, H]
    sum_mask = mask.sum(dim=1).clamp(min=1e-9)     # [B, H]
    return sum_embeddings / sum_mask


def normalize_text(x) -> str:
    if x is None:
        return ""
    if pd.isna(x):
        return ""
    x = str(x).replace("\r\n", "\n").replace("\r", "\n").strip()
    return x


@torch.no_grad()
def encode_batch(
    texts: List[str],
    tokenizer,
    model,
    device: torch.device,
    max_length: int,
) -> np.ndarray:
    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )

    encoded = {k: v.to(device) for k, v in encoded.items()}

    outputs = model(**encoded)
    last_hidden_state = outputs.last_hidden_state  # [B, L, H]
    pooled = mean_pooling(last_hidden_state, encoded["attention_mask"])  # [B, H]
    pooled = pooled.detach().cpu().float().numpy()
    return pooled


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_csv",
        type=str,
        default="/home/hbxz_lzl/pro1_baseline/valText/data/output/text_manifest_for_oad.csv",
        help="Path to text_manifest_for_oad.csv",
    )
    parser.add_argument(
        "--text_col",
        type=str,
        default="text_strict",
        choices=["text_strict", "text_with_who", "text_full"],
        help="Which text column to encode",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        required=True,
        help="Local path or HF model name, e.g. /path/to/bert-base-uncased",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/home/hbxz_lzl/pro1_baseline/valText/data/output/bert_text_strict",
        help="Directory to save features and index csv",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size for offline encoding",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=128,
        help="Tokenizer max_length",
    )
    parser.add_argument(
        "--num_workers_note",
        type=int,
        default=0,
        help="Unused, only reserved for compatibility/logging",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device for encoding",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    feat_path = os.path.join(args.output_dir, "bert_text_features.npy")
    index_path = os.path.join(args.output_dir, "bert_text_features_index.csv")

    df = pd.read_csv(args.input_csv)

    required_cols = [
        "sample_id",
        "video_id",
        "frame_index_in_video",
        "seq_idx",
        "split",
        "gt_label",
        "gt_label_id",
        args.text_col,
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    texts = [normalize_text(x) for x in df[args.text_col].tolist()]

    # device
    if args.device == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA requested but not available. Falling back to CPU.")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    print("=" * 80)
    print("[INFO] loading tokenizer/model...")
    print(f"model_name_or_path = {args.model_name_or_path}")
    print(f"device             = {device}")
    print(f"text_col           = {args.text_col}")
    print(f"num_rows           = {len(df)}")
    print(f"batch_size         = {args.batch_size}")
    print(f"max_length         = {args.max_length}")
    print("=" * 80)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    model = AutoModel.from_pretrained(args.model_name_or_path)
    model.eval()
    model.to(device)

    hidden_size = model.config.hidden_size
    all_features = np.zeros((len(df), hidden_size), dtype=np.float32)

    num_batches = (len(df) + args.batch_size - 1) // args.batch_size
    for batch_idx in tqdm(range(num_batches), desc="Encoding text"):
        start = batch_idx * args.batch_size
        end = min((batch_idx + 1) * args.batch_size, len(df))
        batch_texts = texts[start:end]
        feats = encode_batch(
            texts=batch_texts,
            tokenizer=tokenizer,
            model=model,
            device=device,
            max_length=args.max_length,
        )
        all_features[start:end] = feats

    np.save(feat_path, all_features)

    index_df = df[
        [
            "sample_id",
            "video_id",
            "frame_name",
            "frame_path",
            "frame_index_in_video",
            "seq_idx",
            "split",
            "subset_name",
            "manifest_id",
            "gt_label",
            "gt_label_id",
            "is_background",
        ]
    ].copy()

    index_df["text_col"] = args.text_col
    index_df["feature_index"] = np.arange(len(index_df), dtype=np.int64)
    index_df["feature_dim"] = hidden_size
    index_df["model_name_or_path"] = args.model_name_or_path
    index_df["max_length"] = args.max_length

    index_df.to_csv(index_path, index=False, encoding="utf-8")

    print("=" * 80)
    print("[DONE] features saved:")
    print(feat_path)
    print("[DONE] index saved:")
    print(index_path)
    print("=" * 80)
    print(f"feature_shape = {all_features.shape}")
    print(f"feature_dtype = {all_features.dtype}")
    print("\n[Index preview]")
    print(index_df.head(3).to_string(index=False))


if __name__ == "__main__":
    main()