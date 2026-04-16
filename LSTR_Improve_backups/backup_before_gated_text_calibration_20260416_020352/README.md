# LSTR_Improve backup before gated text calibration

## 1. 备份目的
本备份用于保存“文本校准已跑通，但尚未继续做门控校准”这一阶段的代码与实验状态，便于后续继续修改前回滚。

## 2. 当前代码改动点
本备份包含以下关键文件：

- `rekognition_online_action_detection/config/defaults.py`
- `rekognition_online_action_detection/utils/text_calibration.py`
- `rekognition_online_action_detection/engines/base_inferences/perframe_det_batch_inference.py`
- `configs/THUMOS/LSTR/lstr_long_512_work_8_kinetics_1x.yaml`

以及本阶段用到的辅助脚本：

- `tools/encode_class_text_features_clipL_two_variants.py`
- `tools/make_class_text_core_only.sh`
- `tools/encode_class_text_features_clipL_core_only.py`

## 3. 本阶段主要过程

### 3.1 基础 baseline
- checkpoint: `epoch-12.pth`
- baseline perframe mAP: `0.70823`

### 3.2 文本校准代码接入
已完成：
1. 在 `defaults.py` 中加入 `MODEL.TEXT_CALIBRATION` 配置项
2. 新建 `text_calibration.py`
3. 修改 `perframe_det_batch_inference.py`
4. 使用 `query_indices` 对齐窗口级预测和整视频帧级文本特征
5. 采用概率级融合：
   - `p_mix = (1 - alpha) * p_vis + alpha * p_text`

### 3.3 类别文本版本
测试过：
- `v1-prototype_text`
- `v2-core_par_hard`
- `v3-core_only`

### 3.4 帧级文本版本
测试过：
- `CLIP-L_evi_opt`
- `CLIP-L_evidence_only`

当前观察：`evidence_only` 明显优于 `evi_opt`

## 4. 关键实验结果（THUMOS perframe mAP）

- baseline（无文本校准）: `0.70823`
- `v2-core_par_hard + CLIP-L_evi_opt + alpha=0.02`: `0.70951`
- `v2-core_par_hard + CLIP-L_evi_opt + alpha=0.03`: `0.70923`
- `v2-core_par_hard + CLIP-L_evi_opt + alpha=0.05`: `0.70861`
- `v2-core_par_hard + CLIP-L_evi_opt + alpha=0.10`: `0.70633`
- `v1-prototype_text + CLIP-L_evi_opt + alpha=0.02`: `0.70856`
- `v2-core_par_hard + CLIP-L_evidence_only + alpha=0.02`: `0.71063`
- `v3-core_only + CLIP-L_evidence_only + alpha=0.02`: `0.70942`
- `v1-prototype_text + CLIP-L_evidence_only + alpha=0.02`: `0.71129` （当前最好）

## 5. 当前结论
1. 文本校准方向有效
2. `alpha` 只能取很小，`0.02` 明显优于 `0.10`
3. frame 端去掉 `optional` 更好，`evidence_only > evi_opt`
4. class 端不是越短越好：
   - `prototype_text` 最好
   - `core_par_hard` 次之
   - `core_only` 最弱

当前 best setting：

- class = `v1-prototype_text`
- frame = `CLIP-L_evidence_only`
- alpha = `0.02`
- text_temp = `0.07`
- mAP = `0.71129`

## 6. 下一步原计划
在此备份之后，下一步准备尝试 gated text calibration（门控文本校准）。

