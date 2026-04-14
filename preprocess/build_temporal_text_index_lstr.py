#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build temporal text index for text-only OAD.

Given bert_text_features_index.csv, construct causal windows of size K
for each sample using seq_idx within each video.

Padding mode:
    - replicate: replicate the earliest available feature on the left

Output:
    temporal_text_index_k{K}.csv

Each row corresponds to one current frame/sample, and contains:
    - metadata of the current sample
    - K historical text feature indices: text_idx_0 ... text_idx_{K-1}
      where text_idx_{K-1} is the current frame index
"""

from __future__ import annotations

import argparse
import os
from typing import List

import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_csv",
        type=str,
        default="/home/hbxz_lzl/pro1_baseline/valText/data/output/bert_text_strict/bert_text_features_index.csv",
        help="Path to bert_text_features_index.csv",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="/home/hbxz_lzl/pro1_baseline/valText/data/output/bert_text_strict/temporal_text_index_k8.csv",
        help="Output CSV path",
    )
    parser.add_argument(
        "--window_size",
        type=int,
        default=8,
        help="Causal temporal window size",
    )
    parser.add_argument(
        "--pad_mode",
        type=str,
        default="replicate",
        choices=["replicate"],
        help="Left padding mode for early timesteps",
    )
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)

    df = pd.read_csv(args.input_csv)

    required_cols = [
        "sample_id",
        "video_id",
        "frame_index_in_video",
        "seq_idx",
        "split",
        "gt_label",
        "gt_label_id",
        "is_background",
        "feature_index",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Sort safely
    df = df.sort_values(["video_id", "seq_idx", "frame_index_in_video"]).reset_index(drop=True)

    K = args.window_size
    rows: List[dict] = []

    for video_id, g in df.groupby("video_id", sort=False):
        g = g.sort_values(["seq_idx", "frame_index_in_video"]).reset_index(drop=True)

        feat_indices = g["feature_index"].tolist()

        for i in range(len(g)):
            # causal history: [i-K+1, ..., i]
            start = max(0, i - K + 1)
            window = feat_indices[start:i + 1]

            if len(window) < K:
                if args.pad_mode == "replicate":
                    pad_val = window[0]
                    pad = [pad_val] * (K - len(window))
                    window = pad + window
                else:
                    raise ValueError(f"Unsupported pad_mode: {args.pad_mode}")

            assert len(window) == K

            cur = g.iloc[i]

            row = {
                "sample_id": int(cur["sample_id"]),
                "video_id": cur["video_id"],
                "frame_index_in_video": int(cur["frame_index_in_video"]),
                "seq_idx": int(cur["seq_idx"]),
                "split": cur["split"],
                "gt_label": cur["gt_label"],
                "gt_label_id": int(cur["gt_label_id"]),
                "is_background": int(cur["is_background"]),
                "current_feature_index": int(cur["feature_index"]),
                "window_size": K,
                "pad_mode": args.pad_mode,
            }

            for j in range(K):
                row[f"text_idx_{j}"] = int(window[j])

            rows.append(row)

    out_df = pd.DataFrame(rows)

    # sanity check
    expected_cols = [
        "sample_id",
        "video_id",
        "frame_index_in_video",
        "seq_idx",
        "split",
        "gt_label",
        "gt_label_id",
        "is_background",
        "current_feature_index",
        "window_size",
        "pad_mode",
    ] + [f"text_idx_{j}" for j in range(K)]

    out_df = out_df[expected_cols]
    out_df.to_csv(args.output_csv, index=False, encoding="utf-8")

    print("=" * 80)
    print("[DONE] temporal text index saved to:")
    print(args.output_csv)
    print("=" * 80)
    print(f"num_rows = {len(out_df)}")
    print(f"window_size = {K}")
    print(f"pad_mode = {args.pad_mode}")

    print("\n[Preview]")
    preview_cols = [
        "video_id",
        "seq_idx",
        "gt_label",
        "current_feature_index",
    ] + [f"text_idx_{j}" for j in range(K)]
    print(out_df[preview_cols].head(5).to_string(index=False))


if __name__ == "__main__":
    main()