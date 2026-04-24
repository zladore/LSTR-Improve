# LSTR_Test stable mainline backup
## 恢复该版本指令
/home/hbxz_lzl/LSTR_Test_backups/github_mainline_71911_20260424_051715/restore_mainline.sh

## Purpose
This backup preserves the currently verified stable classifier-only TTA mainline.

## Verified result
- Action detection perframe mAP: 71.911

## Mainline setting
- repo: /home/hbxz_lzl/LSTR_Test
- dataset: THUMOS
- model: trimodal clipL evidence_only
- checkpoint: epoch-14.pth
- inference mode: batch
- text calibration enabled: True
- mode: tta_kl_topk
- TTA_STEPS: 2
- TTA_LR: 5e-4
- TTA_TOPK: 3

## Important notes
This mainline is a matched working combination of:
- defaults.py
- text_calibration.py
- perframe_det_batch_inference.py

Do not casually mix these files with older backups, otherwise config/API mismatch may happen.

## Negative branches already rejected
- head_classifier: 59.646
- head_dec_classifier: 54.618
- work_adapter_classifier (lr=5e-4): 54.530
- work_adapter_classifier (lr=1e-3): 52.814

## Recommended future direction
Continue only on classifier-only mainline:
1. auto-stop / automatic convergence
2. selective update trigger
3. speed-performance tradeoff

## Verification command
python -u tools/test_net.py \
  --config_file /home/hbxz_lzl/LSTR_Test/configs/THUMOS/LSTR/lstr_long_512_work_8_kinetics_1x_trimodal_clipL_evidence_only.yaml \
  --gpu 1 \
  MODEL.CHECKPOINT /home/hbxz_lzl/LSTR_Test/checkpoints/THUMOS/LSTR/lstr_long_512_work_8_kinetics_1x_trimodal_clipL_evidence_only/clipL_evidence_only_v1/epoch-14.pth \
  MODEL.LSTR.INFERENCE_MODE batch \
  MODEL.TEXT_CALIBRATION.ENABLED True \
  MODEL.TEXT_CALIBRATION.MODE tta_kl_topk \
  MODEL.TEXT_CALIBRATION.TEXT_TEMP 0.07 \
  MODEL.TEXT_CALIBRATION.TTA_STEPS 2 \
  MODEL.TEXT_CALIBRATION.TTA_LR 5e-4 \
  MODEL.TEXT_CALIBRATION.TTA_TOPK 3 \
  MODEL.TEXT_CALIBRATION.CLASS_FEATURE_PATH /home/hbxz_lzl/LSTR_Improve/data/Output_THUMOS/clss_text_features/v1-prototype_text/class_text_features.npy \
  MODEL.TEXT_CALIBRATION.CLASS_ORDER_PATH /home/hbxz_lzl/LSTR_Improve/data/Output_THUMOS/clss_text_features/v1-prototype_text/class_feature_order.json \
  MODEL.TEXT_CALIBRATION.CLASS_TEXT_JSON_PATH /home/hbxz_lzl/LSTR_Improve/data/Output_THUMOS/clss_text_features/v1-prototype_text/class_texts.json \
  MODEL.TEXT_CALIBRATION.FRAME_FEATURE_ROOT /home/hbxz_lzl/LSTR_Test/data/Output_THUMOS/text_features/CLIP-L_evidence_only
