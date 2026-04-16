import os
import json
import csv
import numpy as np
import torch
import torch.nn.functional as F
from transformers import CLIPModel, AutoTokenizer
from tqdm import tqdm

MODEL_PATH = "/home/hbxz_lzl/clip-vit-large-patch14-pytorch-slim"
INPUT_CSV = "/home/hbxz_lzl/LSTR_Improve/data/Output_THUMOS/class_text_des/clip-l-core-only/thumos_class_description_core_only.csv"
OUTPUT_DIR = "/home/hbxz_lzl/LSTR_Improve/data/Output_THUMOS/clss_text_features/v3-core_only"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
MAX_LENGTH = 77


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def clean_text(x) -> str:
    if x is None:
        return ""
    return str(x).strip()


def load_csv_rows(path: str):
    rows = []
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


@torch.no_grad()
def encode_texts(texts, model, tokenizer, batch_size=16, max_length=77, device="cuda"):
    feats = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Encoding core_only", ncols=100):
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


def main():
    ensure_dir(OUTPUT_DIR)

    rows = load_csv_rows(INPUT_CSV)
    class_names = [clean_text(r.get("class_name", "")) for r in rows]
    texts = [clean_text(r.get("core_text", "")) for r in rows]

    print(f"[INFO] rows = {len(rows)}")
    print(f"[INFO] model = {MODEL_PATH}")
    print(f"[INFO] device = {DEVICE}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = CLIPModel.from_pretrained(MODEL_PATH).to(DEVICE)
    model.eval()

    feats = encode_texts(
        texts,
        model=model,
        tokenizer=tokenizer,
        batch_size=BATCH_SIZE,
        max_length=MAX_LENGTH,
        device=DEVICE
    )

    np.save(os.path.join(OUTPUT_DIR, "class_text_features.npy"), feats)

    with open(os.path.join(OUTPUT_DIR, "class_feature_order.json"), "w", encoding="utf-8") as f:
        json.dump(class_names, f, ensure_ascii=False, indent=2)

    with open(os.path.join(OUTPUT_DIR, "class_texts.json"), "w", encoding="utf-8") as f:
        json.dump(
            [{"class_name": c, "feature_index": i, "text": t} for i, (c, t) in enumerate(zip(class_names, texts))],
            f, ensure_ascii=False, indent=2
        )

    with open(os.path.join(OUTPUT_DIR, "class_feature_summary.csv"), "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["class_name", "feature_index", "text_length", "variant"]
        )
        writer.writeheader()
        for i, (c, t) in enumerate(zip(class_names, texts)):
            writer.writerow({
                "class_name": c,
                "feature_index": i,
                "text_length": len(t),
                "variant": "v3-core_only"
            })

    print(f"[DONE] features -> {os.path.join(OUTPUT_DIR, 'class_text_features.npy')} | shape={feats.shape}")


if __name__ == "__main__":
    main()
