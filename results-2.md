# 提升 0.695

只用evidence  不用optional

'OUTPUT_DIR': 'checkpoints/THUMOS/LSTR/lstr_long_512_work_8_kinetics_1x_trimodal_clipL_evidence_only/clipL_evidence_only_v1', 'SESSION': 'clipL_evidence_only_v1', 'VERBOSE': False, 'GPU': '1'})
Epoch  1 | train det_loss: 0.94594 | test det_loss: 0.71562 det_mAP: 0.55805 | running time: 39.96 sec
Epoch  2 | train det_loss: 0.49987 | test det_loss: 0.53947 det_mAP: 0.61290 | running time: 39.11 sec
Epoch  3 | train det_loss: 0.40371 | test det_loss: 0.53503 det_mAP: 0.65278 | running time: 37.48 sec
Epoch  4 | train det_loss: 0.36979 | test det_loss: 0.57514 det_mAP: 0.64088 | running time: 42.31 sec
Epoch  5 | train det_loss: 0.33763 | test det_loss: 0.52851 det_mAP: 0.66773 | running time: 48.36 sec
Epoch  6 | train det_loss: 0.32279 | test det_loss: 0.50869 det_mAP: 0.67503 | running time: 38.59 sec
Epoch  7 | train det_loss: 0.29614 | test det_loss: 0.59483 det_mAP: 0.69019 | running time: 44.65 sec
Epoch  8 | train det_loss: 0.29149 | test det_loss: 0.54684 det_mAP: 0.68261 | running time: 41.34 sec
Epoch  9 | train det_loss: 0.27576 | test det_loss: 0.54467 det_mAP: 0.69124 | running time: 41.10 sec
Epoch 10 | train det_loss: 0.26494 | test det_loss: 0.56232 det_mAP: 0.69768 | running time: 46.19 sec
Epoch 11 | train det_loss: 0.24658 | test det_loss: 0.52192 det_mAP: 0.69739 | running time: 42.84 sec
Epoch 12 | train det_loss: 0.23062 | test det_loss: 0.56246 det_mAP: 0.69686 | running time: 40.13 sec
Epoch 13 | train det_loss: 0.22045 | test det_loss: 0.50867 det_mAP: 0.70863 | running time: 44.83 sec
Epoch 14 | train det_loss: 0.21017 | test det_loss: 0.53301 det_mAP: 0.71446 | running time: 42.94 sec
Epoch 15 | train det_loss: 0.19827 | test det_loss: 0.51685 det_mAP: 0.70757 | running time: 38.75 sec
Epoch 16 | train det_loss: 0.18746 | test det_loss: 0.53061 det_mAP: 0.71071 | running time: 42.77 sec
Epoch 17 | train det_loss: 0.18320 | test det_loss: 0.53201 det_mAP: 0.70420 | running time: 41.44 sec
Epoch 18 | train det_loss: 0.17416 | test det_loss: 0.50790 det_mAP: 0.71091 | running time: 38.74 sec
Epoch 19 | train det_loss: 0.16780 | test det_loss: 0.52970 det_mAP: 0.70979 | running time: 40.33 sec
Epoch 20 | train det_loss: 0.16134 | test det_loss: 0.55288 det_mAP: 0.70792 | running time: 42.32 sec
Epoch 21 | train det_loss: 0.15680 | test det_loss: 0.52646 det_mAP: 0.71045 | running time: 38.45 sec
Epoch 22 | train det_loss: 0.15284 | test det_loss: 0.52893 det_mAP: 0.70947 | running time: 38.47 sec
Epoch 23 | train det_loss: 0.15074 | test det_loss: 0.52688 det_mAP: 0.71075 | running time: 39.91 sec
Epoch 24 | train det_loss: 0.14875 | test det_loss: 0.53154 det_mAP: 0.71138 | running time: 38.77 sec
Epoch 25 | train det_loss: 0.14845 | test det_loss: 0.52973 det_mAP: 0.71132 | running time: 40.05 sec

/home/hbxz_lzl/python311/bin/python3 tools/test_net.py \
  --config_file configs/THUMOS/LSTR/lstr_long_512_work_8_kinetics_1x_trimodal_clipL_evidence_only.yaml \
  --gpu 2\
  MODEL.CHECKPOINT /home/hbxz_lzl/LSTR_Test/checkpoints/THUMOS/LSTR/lstr_long_512_work_8_kinetics_1x_trimodal_clipL_evidence_only/clipL_evidence_only_v1/epoch-14.pth \
  MODEL.LSTR.INFERENCE_MODE batch

{'SCHEDULER_NAME': 'warmup_cosine', 'MILESTONES': [], 'GAMMA': 0.1, 'WARMUP_FACTOR': 0.3, 'WARMUP_EPOCHS': 10.0, 'WARMUP_METHOD': 'linear'}), 'PHASES': ['train', 'test']}), 'OUTPUT_DIR': 'checkpoints/THUMOS/LSTR/lstr_long_512_work_8_kinetics_1x_trimodal_clipL_evidence_only', 'SESSION': '', 'VERBOSE': False, 'GPU': '2'})
BatchInference: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████| 696/696 [09:15<00:00,  1.25it/s]
Action detection perframe mAP: 0.71518
(lstr_bw) hbxz_lzl@hqu:~/LSTR_Test$ 
