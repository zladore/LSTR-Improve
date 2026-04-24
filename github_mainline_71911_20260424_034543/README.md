# LSTR_Test GitHub mainline snapshot

## Snapshot purpose
This snapshot preserves the stable classifier-only TTA mainline before adding
auto-stop and selective-update mechanisms.

## Stable setting
- Dataset: THUMOS
- Backbone: trimodal clipL evidence_only
- Checkpoint: epoch-14.pth
- TTA mode: classifier-only
- TTA objective: tta_kl_topk
- TTA_STEPS: 2
- TTA_LR: 5e-4
- TTA_TOPK: 3

## Best known result on this mainline
- Action detection perframe mAP: 71.911

## Negative branches already tested
- head_classifier: 59.646
- head_dec_classifier: 54.618
- work_adapter_classifier (5e-4): 54.530
- work_adapter_classifier (1e-3): 52.814

## Conclusion
Classifier-only is the only stable positive TTA direction at this stage.
Future experiments should focus on:
1. automatic convergence
2. selective update
3. speed-performance tradeoff
