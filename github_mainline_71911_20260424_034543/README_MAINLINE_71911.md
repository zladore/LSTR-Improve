# LSTR_Test 当前主线说明

## 当前主线
恢复到三模态 classifier-only TTA 主线（最佳已知结果 71.911）

## 主线配置
- repo: /home/hbxz_lzl/LSTR_Test
- task: THUMOS online action detection
- backbone: trimodal clipL evidence_only
- checkpoint: epoch-14.pth
- TTA mode: classifier-only
- MODE: tta_kl_topk
- TTA_STEPS: 2
- TTA_LR: 5e-4
- TTA_TOPK: 3

## 当前最佳已知结果
- classifier-only + step=2 + lr=5e-4 : 71.911

## 已证伪方向
- head_classifier
- head_dec_classifier
- work_adapter_classifier

## 后续计划
基于 classifier-only 主线，继续研究：
1. 自动收敛（auto-stop）
2. 只在需要时更新（triggered update）
3. 速度-效果权衡
