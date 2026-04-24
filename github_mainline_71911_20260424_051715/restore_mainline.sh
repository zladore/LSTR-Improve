#!/usr/bin/env bash
set -e

BK_DIR="$(cd "$(dirname "$0")" && pwd)"
TARGET="/home/hbxz_lzl/LSTR_Test"

cp "$BK_DIR/_critical_files/rekognition_online_action_detection/config/defaults.py" \
   "$TARGET/rekognition_online_action_detection/config/"

cp "$BK_DIR/_critical_files/rekognition_online_action_detection/utils/text_calibration.py" \
   "$TARGET/rekognition_online_action_detection/utils/"

cp "$BK_DIR/_critical_files/rekognition_online_action_detection/utils/checkpointer.py" \
   "$TARGET/rekognition_online_action_detection/utils/"

cp "$BK_DIR/_critical_files/rekognition_online_action_detection/engines/base_inferences/perframe_det_batch_inference.py" \
   "$TARGET/rekognition_online_action_detection/engines/base_inferences/"

cp "$BK_DIR/_critical_files/rekognition_online_action_detection/models/lstr.py" \
   "$TARGET/rekognition_online_action_detection/models/"

echo "[DONE] Restored stable mainline files to $TARGET"
