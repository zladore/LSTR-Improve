| 模态  | 最优设置                                       |        mAP |
| --- | ------------------------------------------ | ---------: |
| 三模态 | v1-prototype_text + step2 + lr4e-4 + topk3 | **71.950** |
| 双模态 | v2-core_par_hard + step1 + lr5e-4 + topk3  | **71.197** |

# LSTR/OAD Text Calibration 当前最终锚点记录

## 1. 实验目标

当前实验主要验证：在 LSTR 在线动作检测任务中，引入文本特征作为测试时校准信号后，是否能够提升 THUMOS per-frame mAP。

整体方法为 **classifier-only test-time adaptation**：

- 不更新 backbone；
- 不更新 feature head；
- 不更新 LSTR temporal encoder/decoder；
- 只在测试阶段对最终分类器 `classifier` 做少量梯度更新；
- 更新目标来自文本伪目标分布 `Q`；
- 使用 `tta_kl_topk`，即只在当前视觉预测 top-k 候选类别内构造文本校准目标；
- 官方评估仍然使用 per-frame mAP。

---

## 2. 当前最终锚点

| 模态 | 最优设置 | checkpoint | config | TTA mode | step | lr | topk | text bank | mAP |
|---|---|---|---|---|---:|---:|---:|---|---:|
| 三模态 | v1-prototype_text + step2 + lr4e-4 + topk3 | epoch14 | trimodal_clipL_evidence_only | tta_kl_topk | 2 | 4e-4 | 3 | v1-prototype_text | **71.950** |
| 双模态 | v2-core_par_hard + step1 + lr5e-4 + topk3 | epoch12 | lstr_long_512_work_8_kinetics_1x | tta_kl_topk | 1 | 5e-4 | 3 | v2-core_par_hard | **71.197** |

结论：当前主线以 **三模态 71.950** 作为最高锚点；双模态最优为 **71.197**，用于验证该文本校准方法在没有 text stream 的双模态 baseline 上同样有效。

---

## 3. 三模态最终设置细节

### 3.1 基本设置

| 项目 | 内容 |
|---|---|
| 项目路径 | `/home/hbxz_lzl/LSTR_Test` |
| 模态 | `threestream` |
| checkpoint | `/home/hbxz_lzl/LSTR_Test/checkpoints/THUMOS/LSTR/lstr_long_512_work_8_kinetics_1x_trimodal_clipL_evidence_only/clipL_evidence_only_v1/epoch-14.pth` |
| config | `/home/hbxz_lzl/LSTR_Test/configs/THUMOS/LSTR/lstr_long_512_work_8_kinetics_1x_trimodal_clipL_evidence_only.yaml` |
| frame-level text feature root | `/home/hbxz_lzl/LSTR_Test/data/Output_THUMOS/text_features/CLIP-L_evidence_only` |
| class text bank | `/home/hbxz_lzl/LSTR_Improve/data/Output_THUMOS/clss_text_features/v1-prototype_text` |
| TTA 方法 | `tta_kl_topk` |
| 更新参数 | 只更新 `classifier` |
| TTA step | 2 |
| TTA lr | 4e-4 |
| TTA topk | 3 |
| 最终 mAP | **71.950** |

### 3.2 三模态关键命令设置

```bash
source /home/hbxz_lzl/venvs/lstr_bw/bin/activate

cd /home/hbxz_lzl/LSTR_Test

python tools/test_net.py \
  --config_file configs/THUMOS/LSTR/lstr_long_512_work_8_kinetics_1x_trimodal_clipL_evidence_only.yaml \
  --gpu 0 \
  MODEL.CHECKPOINT /home/hbxz_lzl/LSTR_Test/checkpoints/THUMOS/LSTR/lstr_long_512_work_8_kinetics_1x_trimodal_clipL_evidence_only/clipL_evidence_only_v1/epoch-14.pth \
  INPUT.MODALITY threestream \
  MODEL.TEXT_CALIBRATION.ENABLED True \
  MODEL.TEXT_CALIBRATION.CLASS_FEATURE_PATH /home/hbxz_lzl/LSTR_Improve/data/Output_THUMOS/clss_text_features/v1-prototype_text/class_text_features.npy \
  MODEL.TEXT_CALIBRATION.CLASS_ORDER_PATH /home/hbxz_lzl/LSTR_Improve/data/Output_THUMOS/clss_text_features/v1-prototype_text/class_feature_order.json \
  MODEL.TEXT_CALIBRATION.CLASS_TEXT_JSON_PATH /home/hbxz_lzl/LSTR_Improve/data/Output_THUMOS/clss_text_features/v1-prototype_text/class_texts.json \
  MODEL.TEXT_CALIBRATION.FRAME_FEATURE_ROOT /home/hbxz_lzl/LSTR_Test/data/Output_THUMOS/text_features/CLIP-L_evidence_only \
  MODEL.TEXT_CALIBRATION.MODE tta_kl_topk \
  MODEL.TEXT_CALIBRATION.TTA_STEPS 2 \
  MODEL.TEXT_CALIBRATION.TTA_LR 4e-4 \
  MODEL.TEXT_CALIBRATION.TTA_TOPK 3
