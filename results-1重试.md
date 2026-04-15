Epoch  1 | train det_loss: 0.97338 | test det_loss: 0.70363 det_mAP: 0.54422 | running time: 40.61 sec
Epoch  2 | train det_loss: 0.50023 | test det_loss: 0.65457 det_mAP: 0.62983 | running time: 29.43 sec
Epoch  3 | train det_loss: 0.40792 | test det_loss: 0.54044 det_mAP: 0.65597 | running time: 27.73 sec
Epoch  4 | train det_loss: 0.36565 | test det_loss: 0.55337 det_mAP: 0.66236 | running time: 28.25 sec
Epoch  5 | train det_loss: 0.33669 | test det_loss: 0.52796 det_mAP: 0.67666 | running time: 29.45 sec
Epoch  6 | train det_loss: 0.32672 | test det_loss: 0.58993 det_mAP: 0.68186 | running time: 28.01 sec
Epoch  7 | train det_loss: 0.30665 | test det_loss: 0.53395 det_mAP: 0.67880 | running time: 29.39 sec
Epoch  8 | train det_loss: 0.29399 | test det_loss: 0.71711 det_mAP: 0.67636 | running time: 29.25 sec
Epoch  9 | train det_loss: 0.29956 | test det_loss: 0.51461 det_mAP: 0.68695 | running time: 28.79 sec
Epoch 10 | train det_loss: 0.27111 | test det_loss: 0.50006 det_mAP: 0.69106 | running time: 28.51 sec
Epoch 11 | train det_loss: 0.25073 | test det_loss: 0.50270 det_mAP: 0.69917 | running time: 29.11 sec
Epoch 12 | train det_loss: 0.23544 | test det_loss: 0.50460 det_mAP: 0.70776 | running time: 27.42 sec
Epoch 13 | train det_loss: 0.22574 | test det_loss: 0.53979 det_mAP: 0.69295 | running time: 27.88 sec
Epoch 14 | train det_loss: 0.21126 | test det_loss: 0.54561 det_mAP: 0.70209 | running time: 28.46 sec
Epoch 15 | train det_loss: 0.19983 | test det_loss: 0.53567 det_mAP: 0.70597 | running time: 28.44 sec
Epoch 16 | train det_loss: 0.19209 | test det_loss: 0.55000 det_mAP: 0.70363 | running time: 29.28 sec
Epoch 17 | train det_loss: 0.18597 | test det_loss: 0.51722 det_mAP: 0.69777 | running time: 28.82 sec
Epoch 18 | train det_loss: 0.17633 | test det_loss: 0.54275 det_mAP: 0.70583 | running time: 28.06 sec
Epoch 19 | train det_loss: 0.17052 | test det_loss: 0.53923 det_mAP: 0.70384 | running time: 28.25 sec
Epoch 20 | train det_loss: 0.16533 | test det_loss: 0.55675 det_mAP: 0.70143 | running time: 29.06 sec
Epoch 21 | train det_loss: 0.16149 | test det_loss: 0.52359 det_mAP: 0.70280 | running time: 28.23 sec
Epoch 22 | train det_loss: 0.15736 | test det_loss: 0.53637 det_mAP: 0.70463 | running time: 28.28 sec
Epoch 23 | train det_loss: 0.15603 | test det_loss: 0.53711 det_mAP: 0.70364 | running time: 26.81 sec
Epoch 24 | train det_loss: 0.15291 | test det_loss: 0.53522 det_mAP: 0.70391 | running time: 27.49 sec
Epoch 25 | train det_loss: 0.15270 | test det_loss: 0.53551 det_mAP: 0.70384 | running time: 27.81 sec

BatchInference: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████| 696/696 [07:18<00:00,  1.59it/s]
Action detection perframe mAP: 0.70823
(lstr_bw) hbxz_lzl@hqu:~/LSTR_Test$ 


## 三模态训练指令
###训练
/home/hbxz_lzl/python311/bin/python3 tools/train_net.py \
  --config_file configs/THUMOS/LSTR/lstr_long_512_work_8_kinetics_1x_trimodal_clipL_evi_opt.yaml \
  --gpu 1 \
  SOLVER.PHASES "['train','test']" \
  SESSION clipL_evi_opt_v1

### 推理
/home/hbxz_lzl/python311/bin/python3 tools/test_net.py \
  --config_file configs/THUMOS/LSTR/lstr_long_512_work_8_kinetics_1x_trimodal_clipL_evi_opt.yaml \
  --gpu 1 \
  MODEL.CHECKPOINT /home/hbxz_lzl/LSTR_Test/checkpoints/THUMOS/LSTR/lstr_long_512_work_8_kinetics_1x_trimodal_clipL_evi_opt/clipL_evi_opt_v1/epoch-14.pth \
  MODEL.LSTR.INFERENCE_MODE batch


## 三模态结果
evidence + optional

'OUTPUT_DIR': 'checkpoints/THUMOS/LSTR/lstr_long_512_work_8_kinetics_1x_trimodal_clipL_evi_opt/clipL_evi_opt_v1', 'SESSION': 'clipL_evi_opt_v1', 'VERBOSE': False, 'GPU': '1'})
Epoch  1 | train det_loss: 0.94768 | test det_loss: 0.71699 det_mAP: 0.55630 | running time: 41.95 sec
Epoch  2 | train det_loss: 0.50133 | test det_loss: 0.54227 det_mAP: 0.61078 | running time: 39.00 sec
Epoch  3 | train det_loss: 0.40498 | test det_loss: 0.53573 det_mAP: 0.65086 | running time: 36.91 sec
Epoch  4 | train det_loss: 0.37080 | test det_loss: 0.57623 det_mAP: 0.63934 | running time: 35.95 sec
Epoch  5 | train det_loss: 0.33879 | test det_loss: 0.52834 det_mAP: 0.66635 | running time: 36.81 sec
Epoch  6 | train det_loss: 0.32380 | test det_loss: 0.50904 det_mAP: 0.67321 | running time: 36.42 sec
Epoch  7 | train det_loss: 0.29678 | test det_loss: 0.59283 det_mAP: 0.68977 | running time: 37.07 sec
Epoch  8 | train det_loss: 0.29279 | test det_loss: 0.54334 det_mAP: 0.68287 | running time: 35.79 sec
Epoch  9 | train det_loss: 0.27630 | test det_loss: 0.54443 det_mAP: 0.69075 | running time: 36.91 sec
Epoch 10 | train det_loss: 0.26577 | test det_loss: 0.56357 det_mAP: 0.69706 | running time: 37.17 sec
Epoch 11 | train det_loss: 0.24692 | test det_loss: 0.52560 det_mAP: 0.69730 | running time: 39.25 sec
Epoch 12 | train det_loss: 0.23148 | test det_loss: 0.56222 det_mAP: 0.69644 | running time: 37.96 sec
Epoch 13 | train det_loss: 0.22098 | test det_loss: 0.50493 det_mAP: 0.70856 | running time: 35.59 sec
Epoch 14 | train det_loss: 0.21031 | test det_loss: 0.53269 det_mAP: 0.71395 | running time: 36.44 sec
Epoch 15 | train det_loss: 0.19830 | test det_loss: 0.52011 det_mAP: 0.70633 | running time: 36.14 sec
Epoch 16 | train det_loss: 0.18818 | test det_loss: 0.52984 det_mAP: 0.71006 | running time: 36.62 sec
Epoch 17 | train det_loss: 0.18348 | test det_loss: 0.53116 det_mAP: 0.70349 | running time: 36.77 sec
Epoch 18 | train det_loss: 0.17461 | test det_loss: 0.50953 det_mAP: 0.71026 | running time: 36.21 sec
Epoch 19 | train det_loss: 0.16824 | test det_loss: 0.52999 det_mAP: 0.70921 | running time: 35.85 sec
Epoch 20 | train det_loss: 0.16184 | test det_loss: 0.55078 det_mAP: 0.70728 | running time: 35.83 sec
Epoch 21 | train det_loss: 0.15729 | test det_loss: 0.52552 det_mAP: 0.70996 | running time: 32.70 sec
Epoch 22 | train det_loss: 0.15330 | test det_loss: 0.52860 det_mAP: 0.70889 | running time: 34.40 sec
Epoch 23 | train det_loss: 0.15110 | test det_loss: 0.52596 det_mAP: 0.71017 | running time: 33.78 sec
Epoch 24 | train det_loss: 0.14932 | test det_loss: 0.53053 det_mAP: 0.71098 | running time: 33.42 sec
Epoch 25 | train det_loss: 0.14908 | test det_loss: 0.52863 det_mAP: 0.71088 | running time: 34.19 sec


BatchInference: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████| 696/696 [09:08<00:00,  1.27it/s]
Action detection perframe mAP: 0.71470
(lstr_bw) hbxz_lzl@hqu:~/LSTR_Test$ 

/home/hbxz_lzl/python311/bin/python3 tools/test_net.py \
  --config_file configs/THUMOS/LSTR/lstr_long_512_work_8_kinetics_1x_trimodal_clipL_evi_opt.yaml \
  --gpu 1 \
  MODEL.CHECKPOINT /home/hbxz_lzl/LSTR_Test/checkpoints/THUMOS/LSTR/lstr_long_512_work_8_kinetics_1x_trimodal_clipL_evi_opt/clipL_evi_opt_v1/epoch-14.pth \
  MODEL.LSTR.INFERENCE_MODE batch

