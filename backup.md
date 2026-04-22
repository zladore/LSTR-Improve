#!/bin/bash

SRC="/home/hbxz_lzl/LSTR_Test-v2"
DST="/home/hbxz_lzl/LSTR_Test-v2-backup"

# 如果目标已存在，先删掉重建
rm -rf "$DST"
mkdir -p "$DST"

# 先备份其余正常内容（排除这三个目录的实际内容）
rsync -a \
  --exclude='checkpoints/***' \
  --exclude='data/Output_THUMOS/***' \
  --exclude='data/THUMOS/***' \
  "$SRC"/ "$DST"/

# 再把这三个目录下的“目录结构 + 文件名”补进去，但文件内容为空
cd "$SRC" || exit 1

for d in checkpoints data/Output_THUMOS data/THUMOS; do
  # 复制目录结构
  find "$d" -type d -exec mkdir -p "$DST/{}" \;

  # 复制文件名，创建空文件
  find "$d" -type f -exec bash -c '
    for f do
      mkdir -p "'"$DST"'"/"$(dirname "$f")"
      : > "'"$DST"'"/"$f"
    done
  ' bash {} +
done

echo "备份完成：$DST"
