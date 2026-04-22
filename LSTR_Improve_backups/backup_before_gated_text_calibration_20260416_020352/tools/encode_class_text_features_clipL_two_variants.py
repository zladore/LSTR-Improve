import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import CLIPModel, AutoTokenizer
from tqdm import tqdm

# =========================
# Paths
# =========================
MODEL_PATH = "/home/hbxz_lzl/clip-vit-large-patch14-pytorch-slim"
INPUT_CSV = "/home/hbxz_lzl/LSTR_Improve/data/Output_THUMOS/class_text_des/clip-l-v2/thumos_class_description_v2.csv"

OUT_DIR_V1 = "/home/hbxz_lzl/LSTR_Improve/data/Output_THUMOS/clss_text_features/v1-prototype_text"
OUT_DIR_V2 = "/home/hbxz_lzl/LSTR_Improve/data/Output_THUMOS/clss_text_features/v2-core_par_hard"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
MAX_LENGTH = 77  # CLIP standard


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def clean_text(x) -> str:
    if pd.isna(x):
        return ""
    return str(x).strip()


def join_nonempty(parts):
    parts = [p.strip() for p in parts if isinstance(p, str) and p.strip()]
    return "\n".join(parts)


@torch.no_grad()
def encode_texts(texts, model, tokenizer, batch_size=16, max_length=77, device="cuda"):
    feats = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Encoding text", ncols=100):
        batch = texts[i:i + batch_size]
        enc = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        text_features = model.get_text_features(**enc)
        text_features = F.normalize(text_features, dim=-1)
        feats.append(text_features.cpu().numpy())
    return np.concatenate(feats, axis=0).astype(np.float32)


def save_variant(output_dir, class_names, texts, features, variant_name):
    ensure_dir(output_dir)

    npy_path = os.path.join(output_dir, "class_text_features.npy")
    order_json_path = os.path.join(output_dir, "class_feature_order.json")
    text_json_path = os.path.join(output_dir, "class_texts.json")
    summary_csv_path = os.path.join(output_dir, "class_feature_summary.csv")

    np.save(npy_path, features)

    with open(order_json_path, "w", encoding="utf-8") as f:
        json.dump(class_names, f, ensure_ascii=False, indent=2)

    text_records = []
    for i, (cls, txt) in enumerate(zip(class_names, texts)):
        text_records.append({
            "class_name": cls,
            "feature_index": i,
            "text": txt
        })
    with open(text_json_path, "w", encoding="utf-8") as f:
        json.dump(text_records, f, ensure_ascii=False, indent=2)

    rows = []
    for i, (cls, txt) in enumerate(zip(class_names, texts)):
        rows.append({
            "class_name": cls,
            "feature_index": i,
            "text_length": len(txt),
            "variant": variant_name
        })
    pd.DataFrame(rows).to_csv(summary_csv_path, index=False, encoding="utf-8-sig")

    print(f"[DONE] {variant_name} npy   -> {npy_path} | shape={features.shape}")
    print(f"[DONE] {variant_name} order -> {order_json_path}")
    print(f"[DONE] {variant_name} text  -> {text_json_path}")
    print(f"[DONE] {variant_name} csv   -> {summary_csv_path}")


def main():
    print(f"[INFO] Loading CSV: {INPUT_CSV}")
    df = pd.read_csv(INPUT_CSV)
    print(f"[INFO] Rows: {len(df)}")

    class_names = [clean_text(x) for x in df["class_name"].tolist()]

    # v1: prototype_text only
    texts_v1 = [clean_text(x) for x in df["prototype_text"].tolist()]

    # v2: core + partial + hard only
    texts_v2 = []
    for _, row in df.iterrows():
        core = clean_text(row.get("core_cues", ""))
        partial = clean_text(row.get("partial_view_cues", ""))
        hard = clean_text(row.get("hard_cases", ""))
        txt = join_nonempty([core, partial, hard])
        texts_v2.append(txt)

    print(f"[INFO] Loading CLIP-L from: {MODEL_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = CLIPModel.from_pretrained(MODEL_PATH).to(DEVICE)
    model.eval()

    print(f"[INFO] Device: {DEVICE}")

    print("[INFO] Encoding v1: prototype_text")
    feats_v1 = encode_texts(
        texts_v1,
        model=model,
        tokenizer=tokenizer,
        batch_size=BATCH_SIZE,
        max_length=MAX_LENGTH,
        device=DEVICE
    )
    save_variant(OUT_DIR_V1, class_names, texts_v1, feats_v1, "v1-prototype_text")

    print("[INFO] Encoding v2: core + partial + hard")
    feats_v2 = encode_texts(
        texts_v2,
        model=model,
        tokenizer=tokenizer,
        batch_size=BATCH_SIZE,
        max_length=MAX_LENGTH,
        device=DEVICE
    )
    save_variant(OUT_DIR_V2, class_names, texts_v2, feats_v2, "v2-core_par_hard")


if __name__ == "__main__":
    main()