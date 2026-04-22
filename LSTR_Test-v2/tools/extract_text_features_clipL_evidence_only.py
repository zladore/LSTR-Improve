import os
import json
import argparse
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import CLIPModel, CLIPTokenizer


def clean_text(x):
    if pd.isna(x):
        return ""
    x = str(x).strip()
    if x.lower() in {"nan", "none", "null"}:
        return ""
    return x


def l2_normalize(x, eps=1e-6):
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.clip(norm, eps, None)


def find_first_existing(columns, candidates, required=True, name="column"):
    for c in candidates:
        if c in columns:
            return c
    if required:
        raise KeyError(
            f"Cannot find {name}. Tried {candidates}. Existing columns: {list(columns)}"
        )
    return None


@torch.no_grad()
def encode_unique_texts(texts, tokenizer, model, device, batch_size=256, max_length=77):
    unique_texts = sorted(set(texts))
    feats_dict = {}
    all_features = []

    for i in tqdm(range(0, len(unique_texts), batch_size), desc="Encoding unique texts"):
        batch = unique_texts[i:i + batch_size]
        tokens = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        tokens = {k: v.to(device) for k, v in tokens.items()}
        text_features = model.get_text_features(**tokens)
        text_features = text_features.float().cpu().numpy()
        text_features = l2_normalize(text_features).astype(np.float32)
        all_features.append(text_features)

    all_features = np.concatenate(all_features, axis=0)
    feature_dim = all_features.shape[1]

    for text, feat in zip(unique_texts, all_features):
        feats_dict[text] = feat

    return feats_dict, feature_dim


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest_path",
        type=str,
        default="/home/hbxz_lzl/LSTR_Test/data/THUMOS/manifest/manifest_lstr_official_clean.csv",
    )
    parser.add_argument(
        "--clip_path",
        type=str,
        default="/home/hbxz_lzl/clip-vit-large-patch14-pytorch-slim",
    )
    parser.add_argument(
        "--target_root",
        type=str,
        default="/home/hbxz_lzl/LSTR_Test/data/THUMOS/target_perframe",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/home/hbxz_lzl/LSTR_Test/data/Output_THUMOS/text_features/CLIP-L_evidence_only",
    )
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_length", type=int, default=77)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"[INFO] Loading manifest: {args.manifest_path}")
    df = pd.read_csv(args.manifest_path)
    print(f"[INFO] Manifest shape: {df.shape}")

    video_col = find_first_existing(
        df.columns,
        ["video_id", "video", "video_name", "session", "session_id"],
        required=True,
        name="video column",
    )
    frame_col = find_first_existing(
        df.columns,
        ["frame_idx", "frame_index", "frame", "frame_number", "frame_num", "frame_index_in_video"],
        required=False,
        name="frame column",
    )
    evidence_col = find_first_existing(
        df.columns,
        ["evidence", "evidence_text"],
        required=True,
        name="evidence column",
    )

    print(f"[INFO] video_col    = {video_col}")
    print(f"[INFO] frame_col    = {frame_col}")
    print(f"[INFO] evidence_col = {evidence_col}")

    df["_evidence"] = df[evidence_col].map(clean_text)
    df["_orig_idx"] = np.arange(len(df))

    if frame_col is not None:
        df = df.sort_values([video_col, frame_col, "_orig_idx"]).reset_index(drop=True)
    else:
        df = df.sort_values([video_col, "_orig_idx"]).reset_index(drop=True)

    all_texts = df["_evidence"].tolist()
    if "" not in all_texts:
        all_texts.append("")

    device = args.device
    print(f"[INFO] device = {device}")
    print(f"[INFO] Loading CLIP-L from: {args.clip_path}")

    tokenizer = CLIPTokenizer.from_pretrained(args.clip_path, local_files_only=True)
    model = CLIPModel.from_pretrained(args.clip_path, local_files_only=True).to(device)
    model.eval()

    print(f"[INFO] text hidden_size = {model.text_model.config.hidden_size}")
    print(f"[INFO] projection_dim  = {model.config.projection_dim}")

    text2feat, feature_dim = encode_unique_texts(
        all_texts,
        tokenizer,
        model,
        device,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )

    print(f"[INFO] feature_dim = {feature_dim}")

    features = np.zeros((len(df), feature_dim), dtype=np.float32)
    empty_count = 0

    for i, row in tqdm(df.iterrows(), total=len(df), desc="Building row features"):
        ev = row["_evidence"]
        if ev:
            features[i] = text2feat[ev]
        else:
            empty_count += 1
            features[i] = np.zeros((feature_dim,), dtype=np.float32)

    mismatch = []
    num_videos = 0

    for video_name, part in tqdm(df.groupby(video_col, sort=False), desc="Saving per-video npy"):
        idx = part.index.to_numpy()
        video_feat = features[idx]

        save_path = os.path.join(args.output_dir, f"{video_name}.npy")
        np.save(save_path, video_feat.astype(np.float32))
        num_videos += 1

        target_path = os.path.join(args.target_root, f"{video_name}.npy")
        if os.path.exists(target_path):
            target = np.load(target_path, mmap_mode="r")
            if video_feat.shape[0] != target.shape[0]:
                mismatch.append({
                    "video": video_name,
                    "text_len": int(video_feat.shape[0]),
                    "target_len": int(target.shape[0]),
                })

    summary = {
        "manifest_path": args.manifest_path,
        "clip_path": args.clip_path,
        "target_root": args.target_root,
        "output_dir": args.output_dir,
        "video_col": video_col,
        "frame_col": frame_col,
        "evidence_col": evidence_col,
        "feature_dim": int(feature_dim),
        "num_rows": int(len(df)),
        "num_videos": int(num_videos),
        "num_empty_evidence": int(empty_count),
    }

    summary_path = os.path.join(args.output_dir, "summary.json")
    mismatch_path = os.path.join(args.output_dir, "length_mismatch.json")

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    with open(mismatch_path, "w", encoding="utf-8") as f:
        json.dump(mismatch, f, ensure_ascii=False, indent=2)

    print(f"[DONE] summary saved to: {summary_path}")
    print(f"[DONE] mismatch saved to: {mismatch_path}")
    print(f"[DONE] total videos saved: {num_videos}")

    if len(mismatch) == 0:
        print("[CHECK] All video feature lengths match target_perframe.")
    else:
        print(f"[WARN] Found {len(mismatch)} videos with length mismatch.")


if __name__ == "__main__":
    main()
