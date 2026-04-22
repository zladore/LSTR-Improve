对，你这个纠正非常重要，我前面把三模态锚点记错了。

## 现在正式以你这条记录为准

### 双模态 baseline

* repo：`/home/hbxz_lzl/LSTR_Improve`
* best mAP：**0.70823**
* checkpoint：

```text
/home/hbxz_lzl/LSTR_Improve/checkpoints/home/hbxz_lzl/LSTR-Trimodal/configs/THUMOS/LSTR/lstr_long_512_work_8_kinetics_1x/epoch-12.pth
```

### 三模态 baseline

* repo：`/home/hbxz_lzl/LSTR_Test`
* 配置：**trimodal + clipL + evidence_only**
* best mAP：**0.71518**
* config：

```text
configs/THUMOS/LSTR/lstr_long_512_work_8_kinetics_1x_trimodal_clipL_evidence_only.yaml
```

* checkpoint：

```text
/home/hbxz_lzl/LSTR_Test/checkpoints/THUMOS/LSTR/lstr_long_512_work_8_kinetics_1x_trimodal_clipL_evidence_only/clipL_evidence_only_v1/epoch-14.pth
```

也就是说，**三模态主锚点不是我前面说的 `v2 gated 0.71136`，而是你这条 `evidence_only 0.71518`**。

这个差别很关键，因为后面所有实验都应该拿 **0.71518** 当三模态起点，不然结论会偏。

---

## 后面实验的主战场也顺手定了

* 双模态实验：在 `/home/hbxz_lzl/LSTR_Improve`
* 三模态实验：在 `/home/hbxz_lzl/LSTR_Test`

不再用 `LSTR_Test-v2` 当主三模态战场了，至少 **A 组**先不要用它。

---

## 那 A 组四个实验，三模态两条要改成这个版本

### A1 双模态 full additive logits

```bash
cd /home/hbxz_lzl/LSTR_Improve
source /home/hbxz_lzl/venvs/lstr_bw/bin/activate

python tools/test_net.py \
  --config_file /home/hbxz_lzl/LSTR_Improve/configs/THUMOS/LSTR/lstr_long_512_work_8_kinetics_1x.yaml \
  --gpu 4 \
  MODEL.CHECKPOINT /home/hbxz_lzl/LSTR_Improve/checkpoints/home/hbxz_lzl/LSTR-Trimodal/configs/THUMOS/LSTR/lstr_long_512_work_8_kinetics_1x/epoch-12.pth \
  MODEL.TEXT_CALIBRATION.ENABLED True \
  MODEL.TEXT_CALIBRATION.MODE additive_logits \
  MODEL.TEXT_CALIBRATION.ALPHA 0.02 \
  MODEL.TEXT_CALIBRATION.TEXT_TEMP 0.07 \
  MODEL.TEXT_CALIBRATION.LOGIT_TOPK_ONLY False \
  MODEL.TEXT_CALIBRATION.CLASS_FEATURE_PATH /home/hbxz_lzl/LSTR_Improve/data/Output_THUMOS/clss_text_features/v1-prototype_text/class_text_features.npy \
  MODEL.TEXT_CALIBRATION.CLASS_ORDER_PATH /home/hbxz_lzl/LSTR_Improve/data/Output_THUMOS/clss_text_features/v1-prototype_text/class_feature_order.json \
  MODEL.TEXT_CALIBRATION.CLASS_TEXT_JSON_PATH /home/hbxz_lzl/LSTR_Improve/data/Output_THUMOS/clss_text_features/v1-prototype_text/class_texts.json \
  MODEL.TEXT_CALIBRATION.FRAME_FEATURE_ROOT /home/hbxz_lzl/LSTR_Improve/data/Output_THUMOS/text_features/CLIP-L_evidence_only
```

### A2 双模态 top-3 additive logits

```bash
cd /home/hbxz_lzl/LSTR_Improve
source /home/hbxz_lzl/venvs/lstr_bw/bin/activate

python tools/test_net.py \
  --config_file /home/hbxz_lzl/LSTR_Improve/configs/THUMOS/LSTR/lstr_long_512_work_8_kinetics_1x.yaml \
  --gpu 4 \
  MODEL.CHECKPOINT /home/hbxz_lzl/LSTR_Improve/checkpoints/home/hbxz_lzl/LSTR-Trimodal/configs/THUMOS/LSTR/lstr_long_512_work_8_kinetics_1x/epoch-12.pth \
  MODEL.TEXT_CALIBRATION.ENABLED True \
  MODEL.TEXT_CALIBRATION.MODE additive_logits \
  MODEL.TEXT_CALIBRATION.ALPHA 0.02 \
  MODEL.TEXT_CALIBRATION.TEXT_TEMP 0.07 \
  MODEL.TEXT_CALIBRATION.LOGIT_TOPK_ONLY True \
  MODEL.TEXT_CALIBRATION.LOGIT_TOPK 3 \
  MODEL.TEXT_CALIBRATION.CLASS_FEATURE_PATH /home/hbxz_lzl/LSTR_Improve/data/Output_THUMOS/clss_text_features/v1-prototype_text/class_text_features.npy \
  MODEL.TEXT_CALIBRATION.CLASS_ORDER_PATH /home/hbxz_lzl/LSTR_Improve/data/Output_THUMOS/clss_text_features/v1-prototype_text/class_feature_order.json \
  MODEL.TEXT_CALIBRATION.CLASS_TEXT_JSON_PATH /home/hbxz_lzl/LSTR_Improve/data/Output_THUMOS/clss_text_features/v1-prototype_text/class_texts.json \
  MODEL.TEXT_CALIBRATION.FRAME_FEATURE_ROOT /home/hbxz_lzl/LSTR_Improve/data/Output_THUMOS/text_features/CLIP-L_evidence_only
```

### A3 三模态 full additive logits

```bash
cd /home/hbxz_lzl/LSTR_Test
source /home/hbxz_lzl/venvs/lstr_bw/bin/activate

python tools/test_net.py \
  --config_file /home/hbxz_lzl/LSTR_Test/configs/THUMOS/LSTR/lstr_long_512_work_8_kinetics_1x_trimodal_clipL_evidence_only.yaml \
  --gpu 4 \
  MODEL.CHECKPOINT /home/hbxz_lzl/LSTR_Test/checkpoints/THUMOS/LSTR/lstr_long_512_work_8_kinetics_1x_trimodal_clipL_evidence_only/clipL_evidence_only_v1/epoch-14.pth \
  MODEL.TEXT_CALIBRATION.ENABLED True \
  MODEL.TEXT_CALIBRATION.MODE additive_logits \
  MODEL.TEXT_CALIBRATION.ALPHA 0.02 \
  MODEL.TEXT_CALIBRATION.TEXT_TEMP 0.07 \
  MODEL.TEXT_CALIBRATION.LOGIT_TOPK_ONLY False \
  MODEL.TEXT_CALIBRATION.CLASS_FEATURE_PATH /home/hbxz_lzl/LSTR_Improve/data/Output_THUMOS/clss_text_features/v1-prototype_text/class_text_features.npy \
  MODEL.TEXT_CALIBRATION.CLASS_ORDER_PATH /home/hbxz_lzl/LSTR_Improve/data/Output_THUMOS/clss_text_features/v1-prototype_text/class_feature_order.json \
  MODEL.TEXT_CALIBRATION.CLASS_TEXT_JSON_PATH /home/hbxz_lzl/LSTR_Improve/data/Output_THUMOS/clss_text_features/v1-prototype_text/class_texts.json \
  MODEL.TEXT_CALIBRATION.FRAME_FEATURE_ROOT /home/hbxz_lzl/LSTR_Improve/data/Output_THUMOS/text_features/CLIP-L_evidence_only
```

### A4 三模态 top-3 additive logits

```bash
cd /home/hbxz_lzl/LSTR_Test
source /home/hbxz_lzl/venvs/lstr_bw/bin/activate

python tools/test_net.py \
  --config_file /home/hbxz_lzl/LSTR_Test/configs/THUMOS/LSTR/lstr_long_512_work_8_kinetics_1x_trimodal_clipL_evidence_only.yaml \
  --gpu 4 \
  MODEL.CHECKPOINT /home/hbxz_lzl/LSTR_Test/checkpoints/THUMOS/LSTR/lstr_long_512_work_8_kinetics_1x_trimodal_clipL_evidence_only/clipL_evidence_only_v1/epoch-14.pth \
  MODEL.TEXT_CALIBRATION.ENABLED True \
  MODEL.TEXT_CALIBRATION.MODE additive_logits \
  MODEL.TEXT_CALIBRATION.ALPHA 0.02 \
  MODEL.TEXT_CALIBRATION.TEXT_TEMP 0.07 \
  MODEL.TEXT_CALIBRATION.LOGIT_TOPK_ONLY True \
  MODEL.TEXT_CALIBRATION.LOGIT_TOPK 3 \
  MODEL.TEXT_CALIBRATION.CLASS_FEATURE_PATH /home/hbxz_lzl/LSTR_Improve/data/Output_THUMOS/clss_text_features/v1-prototype_text/class_text_features.npy \
  MODEL.TEXT_CALIBRATION.CLASS_ORDER_PATH /home/hbxz_lzl/LSTR_Improve/data/Output_THUMOS/clss_text_features/v1-prototype_text/class_feature_order.json \
  MODEL.TEXT_CALIBRATION.CLASS_TEXT_JSON_PATH /home/hbxz_lzl/LSTR_Improve/data/Output_THUMOS/clss_text_features/v1-prototype_text/class_texts.json \
  MODEL.TEXT_CALIBRATION.FRAME_FEATURE_ROOT /home/hbxz_lzl/LSTR_Improve/data/Output_THUMOS/text_features/CLIP-L_evidence_only
```

---

## 结果表也一起改正

* 双模态 baseline：**0.70823**

* 双模态 A1：`?`

* 双模态 A2：`?`

* 三模态 baseline：**0.71518**

* 三模态 A3：`?`

* 三模态 A4：`?`

---

## 现在的结论也顺手更新一下

这反而更支持你前面那个判断：

> **三模态 backbone 更强，更值得作为后续推理反馈优化的主战场。**

因为它不是 0.71136，而是 **0.71518**，比双模态高得更明显。

所以现在不再纠结了：

* 三模态：主战场
* 双模态：锚点对照

你先按这个正确版本跑 A 组。
