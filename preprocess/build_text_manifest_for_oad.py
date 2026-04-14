#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build a clean text manifest for text-only OAD experiments.

Input:
    manifest_lstr_official_clean.csv

Output:
    text_manifest_for_oad.csv

Generated text fields:
    - text_strict   : evidence + optional
    - text_with_who : who + evidence + optional
    - text_full     : original text_input

Notes:
    - We intentionally DO NOT include current_action / predicted_class
      in text_strict or text_with_who, to reduce label leakage.
"""

from __future__ import annotations

import argparse
import os
import re
from typing import Optional

import pandas as pd


def normalize_text(x: Optional[str]) -> str:
    """Normalize cell text safely."""
    if x is None:
        return ""
    if pd.isna(x):
        return ""
    x = str(x).replace("\r\n", "\n").replace("\r", "\n")
    # Collapse excessive blank lines
    x = re.sub(r"\n\s*\n+", "\n\n", x)
    # Strip trailing spaces on each line
    x = "\n".join(line.strip() for line in x.split("\n"))
    return x.strip()


def join_sections(*sections: str) -> str:
    """Join non-empty text sections with blank lines."""
    parts = []
    for s in sections:
        s = normalize_text(s)
        if s:
            parts.append(s)
    return "\n\n".join(parts).strip()


def build_text_strict(evidence: str, optional: str) -> str:
    """Strict version: only evidence + optional."""
    sec_evidence = f"Evidence:\n{normalize_text(evidence)}" if normalize_text(evidence) else ""
    sec_optional = f"Optional:\n{normalize_text(optional)}" if normalize_text(optional) else ""
    return join_sections(sec_evidence, sec_optional)


def build_text_with_who(who: str, evidence: str, optional: str) -> str:
    """Version with who + evidence + optional."""
    sec_who = f"Who:\n{normalize_text(who)}" if normalize_text(who) else ""
    sec_evidence = f"Evidence:\n{normalize_text(evidence)}" if normalize_text(evidence) else ""
    sec_optional = f"Optional:\n{normalize_text(optional)}" if normalize_text(optional) else ""
    return join_sections(sec_who, sec_evidence, sec_optional)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_csv",
        type=str,
        default="/home/hbxz_lzl/pro1_baseline/valText/data/manifest/manifest_lstr_official_clean.csv",
        help="Path to manifest_lstr_official_clean.csv",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="/home/hbxz_lzl/pro1_baseline/valText/data/output/text_manifest_for_oad.csv",
        help="Output CSV path",
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
        "text_input",
        "who",
        "evidence",
        "optional",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Build clean text fields
    df["text_strict"] = df.apply(
        lambda row: build_text_strict(
            evidence=row.get("evidence", ""),
            optional=row.get("optional", ""),
        ),
        axis=1,
    )

    df["text_with_who"] = df.apply(
        lambda row: build_text_with_who(
            who=row.get("who", ""),
            evidence=row.get("evidence", ""),
            optional=row.get("optional", ""),
        ),
        axis=1,
    )

    df["text_full"] = df["text_input"].apply(normalize_text)

    # Helpful length stats
    df["len_text_strict"] = df["text_strict"].apply(lambda x: len(x))
    df["len_text_with_who"] = df["text_with_who"].apply(lambda x: len(x))
    df["len_text_full"] = df["text_full"].apply(lambda x: len(x))

    keep_cols = [
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
        "gt_multihot",
        "is_background",
        "flow_path",
        "text_input",
        "who",
        "evidence",
        "optional",
        "text_strict",
        "text_with_who",
        "text_full",
        "len_text_strict",
        "len_text_with_who",
        "len_text_full",
    ]

    # Only keep existing columns among keep_cols
    keep_cols = [c for c in keep_cols if c in df.columns]
    out_df = df[keep_cols].copy()

    # Sort for safety
    sort_cols = [c for c in ["video_id", "seq_idx", "frame_index_in_video"] if c in out_df.columns]
    out_df = out_df.sort_values(sort_cols).reset_index(drop=True)

    out_df.to_csv(args.output_csv, index=False, encoding="utf-8")

    print("=" * 80)
    print("[DONE] text manifest saved to:")
    print(args.output_csv)
    print("=" * 80)
    print(f"num_rows = {len(out_df)}")
    print(f"text_strict empty rows    = {(out_df['len_text_strict'] == 0).sum()}")
    print(f"text_with_who empty rows  = {(out_df['len_text_with_who'] == 0).sum()}")
    print(f"text_full empty rows      = {(out_df['len_text_full'] == 0).sum()}")

    # Print a tiny preview
    preview_cols = [c for c in ["video_id", "frame_index_in_video", "gt_label", "text_strict"] if c in out_df.columns]
    print("\n[Preview]")
    print(out_df[preview_cols].head(3).to_string(index=False))


if __name__ == "__main__":
    main()