#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import os
from typing import Dict, Any, List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class TemporalTextDatasetLSTR(Dataset):
    """
    Text-only temporal dataset for OAD.

    Each sample returns:
        - text_seq: [K, D]
        - label: scalar int64
        - meta: useful metadata for debugging/eval
    """

    def __init__(
        self,
        temporal_index_csv: str,
        text_feature_npy: str,
        split: str,
        num_classes: int = 22,
        ignore_index: int = 21,
        return_meta: bool = False,
        feature_dtype: str = "float32",
    ) -> None:
        super().__init__()

        if not os.path.isfile(temporal_index_csv):
            raise FileNotFoundError(f"temporal_index_csv not found: {temporal_index_csv}")
        if not os.path.isfile(text_feature_npy):
            raise FileNotFoundError(f"text_feature_npy not found: {text_feature_npy}")

        self.temporal_index_csv = temporal_index_csv
        self.text_feature_npy = text_feature_npy
        self.split = split
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.return_meta = return_meta

        self.df = pd.read_csv(self.temporal_index_csv)
        if "split" not in self.df.columns:
            raise ValueError("Missing `split` column in temporal index csv.")

        self.df = self.df[self.df["split"] == self.split].reset_index(drop=True)
        if len(self.df) == 0:
            raise ValueError(f"No rows found for split={self.split}")

        self.text_features = np.load(self.text_feature_npy, mmap_mode="r")

        if self.text_features.ndim != 2:
            raise ValueError(
                f"text features must be 2D [N, D], but got shape={self.text_features.shape}"
            )

        self.feature_dim = int(self.text_features.shape[1])

        self.temporal_cols: List[str] = sorted(
            [c for c in self.df.columns if c.startswith("text_idx_")],
            key=lambda x: int(x.split("_")[-1]),
        )
        if len(self.temporal_cols) == 0:
            raise ValueError("No temporal columns found: expected text_idx_0 ... text_idx_{K-1}")

        self.window_size = len(self.temporal_cols)

        self._sanity_check(feature_dtype=feature_dtype)

        print("=" * 80)
        print("[TemporalTextDatasetLSTR] initialized")
        print(f"split         = {self.split}")
        print(f"num_rows      = {len(self.df)}")
        print(f"window_size   = {self.window_size}")
        print(f"feature_shape = {self.text_features.shape}")
        print(f"feature_dim   = {self.feature_dim}")
        print("=" * 80)

    def _sanity_check(self, feature_dtype: str) -> None:
        if feature_dtype == "float32" and self.text_features.dtype != np.float32:
            print(
                f"[WARN] text feature dtype is {self.text_features.dtype}, "
                f"expected float32. Continuing anyway."
            )

        min_idx = int(self.df[self.temporal_cols].min().min())
        max_idx = int(self.df[self.temporal_cols].max().max())

        if min_idx < 0:
            raise ValueError(f"Found negative feature index: {min_idx}")
        if max_idx >= len(self.text_features):
            raise ValueError(
                f"Feature index out of range: max_idx={max_idx}, num_features={len(self.text_features)}"
            )

        if "gt_label_id" not in self.df.columns:
            raise ValueError("Missing `gt_label_id` column in temporal index csv.")

        label_min = int(self.df["gt_label_id"].min())
        label_max = int(self.df["gt_label_id"].max())
        if label_min < 0 or label_max >= self.num_classes:
            print(
                f"[WARN] label range seems unusual: min={label_min}, max={label_max}, "
                f"num_classes={self.num_classes}"
            )

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int):
        row = self.df.iloc[index]

        feat_indices = row[self.temporal_cols].to_numpy(dtype=np.int64)
        text_seq = self.text_features[feat_indices]
        text_seq = np.asarray(text_seq, dtype=np.float32)

        label = int(row["gt_label_id"])

        text_seq_tensor = torch.from_numpy(text_seq)
        label_tensor = torch.tensor(label, dtype=torch.long)

        if not self.return_meta:
            return text_seq_tensor, label_tensor

        meta: Dict[str, Any] = {
            "sample_id": int(row["sample_id"]) if "sample_id" in row else -1,
            "video_id": row["video_id"] if "video_id" in row else "",
            "frame_index_in_video": int(row["frame_index_in_video"]) if "frame_index_in_video" in row else -1,
            "seq_idx": int(row["seq_idx"]) if "seq_idx" in row else -1,
            "split": row["split"] if "split" in row else "",
            "gt_label": row["gt_label"] if "gt_label" in row else "",
            "gt_label_id": label,
            "is_background": int(row["is_background"]) if "is_background" in row else -1,
            "current_feature_index": int(row["current_feature_index"]) if "current_feature_index" in row else -1,
        }
        return text_seq_tensor, label_tensor, meta
