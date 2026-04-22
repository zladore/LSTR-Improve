#!/usr/bin/env bash
set -euo pipefail

IN_DIR="/home/hbxz_lzl/LSTR_Improve/data/Output_THUMOS/class_text_des/clip-l-v2"
OUT_DIR="/home/hbxz_lzl/LSTR_Improve/data/Output_THUMOS/class_text_des/clip-l-core-only"

mkdir -p "$OUT_DIR"

python3 - <<'PY'
import os
import json
import csv
import shutil

IN_DIR = "/home/hbxz_lzl/LSTR_Improve/data/Output_THUMOS/class_text_des/clip-l-v2"
OUT_DIR = "/home/hbxz_lzl/LSTR_Improve/data/Output_THUMOS/class_text_des/clip-l-core-only"

IN_JSON = os.path.join(IN_DIR, "thumos_class_description_v2.json")
OUT_JSON = os.path.join(OUT_DIR, "thumos_class_description_core_only.json")
OUT_CSV = os.path.join(OUT_DIR, "thumos_class_description_core_only.csv")

with open(IN_JSON, "r", encoding="utf-8") as f:
    data = json.load(f)

final_json = []
rows = []

for item in data:
    class_name = item.get("class_name", "")
    core = item.get("core_cues", [])
    if not isinstance(core, list):
        core = []

    core_text = "\n".join([x.strip() for x in core if isinstance(x, str) and x.strip()])

    new_item = {
        "class_name": class_name,
        "core_cues": core,
        "core_text": core_text
    }
    final_json.append(new_item)

    rows.append({
        "class_name": class_name,
        "core_cues": "\n".join(core),
        "core_text": core_text
    })

with open(OUT_JSON, "w", encoding="utf-8") as f:
    json.dump(final_json, f, ensure_ascii=False, indent=2)

with open(OUT_CSV, "w", encoding="utf-8-sig", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=["class_name", "core_cues", "core_text"]
    )
    writer.writeheader()
    for row in rows:
        writer.writerow(row)

src_note = os.path.join(IN_DIR, "v2_notes.txt")
if os.path.exists(src_note):
    shutil.copy2(src_note, os.path.join(OUT_DIR, "source_v2_notes.txt"))

with open(os.path.join(OUT_DIR, "core_only_notes.txt"), "w", encoding="utf-8") as f:
    f.write("Built from clip-l-v2 by keeping only core_cues.\n")
    f.write("core_text = newline-joined core_cues.\n")

print(f"[DONE] JSON -> {OUT_JSON}")
print(f"[DONE] CSV  -> {OUT_CSV}")
PY
