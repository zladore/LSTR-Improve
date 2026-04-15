Epoch  1 | train det_loss: 0.97338 | test det_loss: 0.70363 det_mAP: 0.54422 | running time: 133.24 sec
Epoch  2 | train det_loss: 0.50023 | test det_loss: 0.65457 det_mAP: 0.62983 | running time: 64.39 sec
Epoch  3 | train det_loss: 0.40792 | test det_loss: 0.54044 det_mAP: 0.65597 | running time: 40.72 sec
Epoch  4 | train det_loss: 0.36565 | test det_loss: 0.55337 det_mAP: 0.66236 | running time: 39.98 sec
Epoch  5 | train det_loss: 0.33669 | test det_loss: 0.52796 det_mAP: 0.67666 | running time: 39.96 sec
Epoch  6 | train det_loss: 0.32672 | test det_loss: 0.58993 det_mAP: 0.68186 | running time: 40.66 sec
Epoch  7 | train det_loss: 0.30665 | test det_loss: 0.53395 det_mAP: 0.67880 | running time: 40.72 sec
Epoch  8 | train det_loss: 0.29399 | test det_loss: 0.71711 det_mAP: 0.67636 | running time: 40.68 sec
Epoch  9 | train det_loss: 0.29956 | test det_loss: 0.51461 det_mAP: 0.68695 | running time: 49.26 sec
Epoch 10 | train det_loss: 0.27111 | test det_loss: 0.50006 det_mAP: 0.69106 | running time: 41.59 sec
Epoch 11 | train det_loss: 0.25073 | test det_loss: 0.50270 det_mAP: 0.69917 | running time: 40.95 sec
Epoch 12 | train det_loss: 0.23544 | test det_loss: 0.50460 det_mAP: 0.70776 | running time: 40.92 sec
Epoch 13 | train det_loss: 0.22574 | test det_loss: 0.53979 det_mAP: 0.69295 | running time: 44.73 sec
Epoch 14 | train det_loss: 0.21126 | test det_loss: 0.54561 det_mAP: 0.70209 | running time: 44.69 sec
Epoch 15 | train det_loss: 0.19983 | test det_loss: 0.53567 det_mAP: 0.70597 | running time: 44.32 sec
Epoch 16 | train det_loss: 0.19209 | test det_loss: 0.55000 det_mAP: 0.70363 | running time: 44.31 sec
Epoch 17 | train det_loss: 0.18597 | test det_loss: 0.51722 det_mAP: 0.69777 | running time: 44.50 sec
Epoch 18 | train det_loss: 0.17633 | test det_loss: 0.54275 det_mAP: 0.70583 | running time: 44.57 sec
Epoch 19 | train det_loss: 0.17052 | test det_loss: 0.53923 det_mAP: 0.70384 | running time: 45.67 sec
Epoch 20 | train det_loss: 0.16533 | test det_loss: 0.55675 det_mAP: 0.70143 | running time: 45.73 sec
Epoch 21 | train det_loss: 0.16149 | test det_loss: 0.52359 det_mAP: 0.70280 | running time: 43.44 sec
Epoch 22 | train det_loss: 0.15736 | test det_loss: 0.53637 det_mAP: 0.70463 | running time: 41.69 sec
Epoch 23 | train det_loss: 0.15603 | test det_loss: 0.53711 det_mAP: 0.70364 | running time: 41.37 sec
Epoch 24 | train det_loss: 0.15291 | test det_loss: 0.53522 det_mAP: 0.70391 | running time: 44.74 sec
Epoch 25 | train det_loss: 0.15270 | test det_loss: 0.53551 det_mAP: 0.70384 | running time: 42.75 sec


epoch 12
'', 'VERBOSE': False, 'GPU': '1'})
BatchInference: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████| 696/696 [08:46<00:00,  1.32it/s]
Action detection perframe mAP: 0.68170
(lstr_bw) hbxz_lzl@hqu:~/LSTR_Improve$ 


# trail 1
把 RGB + Flow + Text 直接在 feature_head 里拼起来，再丢进原始 LSTR
##效果
Epoch  1 | train det_loss: 0.94768 | test det_loss: 0.71699 det_mAP: 0.55630 | running time: 37.33 sec
Epoch  2 | train det_loss: 0.50133 | test det_loss: 0.54227 det_mAP: 0.61078 | running time: 36.14 sec
Epoch  3 | train det_loss: 0.40498 | test det_loss: 0.53573 det_mAP: 0.65086 | running time: 37.06 sec
Epoch  4 | train det_loss: 0.37080 | test det_loss: 0.57623 det_mAP: 0.63934 | running time: 37.12 sec
Epoch  5 | train det_loss: 0.33879 | test det_loss: 0.52834 det_mAP: 0.66635 | running time: 36.78 sec
Epoch  6 | train det_loss: 0.32380 | test det_loss: 0.50904 det_mAP: 0.67321 | running time: 36.52 sec
Epoch  7 | train det_loss: 0.29678 | test det_loss: 0.59283 det_mAP: 0.68977 | running time: 37.08 sec
Epoch  8 | train det_loss: 0.29279 | test det_loss: 0.54334 det_mAP: 0.68287 | running time: 36.48 sec
Epoch  9 | train det_loss: 0.27630 | test det_loss: 0.54443 det_mAP: 0.69075 | running time: 36.18 sec
Epoch 10 | train det_loss: 0.26577 | test det_loss: 0.56357 det_mAP: 0.69706 | running time: 36.20 sec
Epoch 11 | train det_loss: 0.24692 | test det_loss: 0.52560 det_mAP: 0.69730 | running time: 37.46 sec
Epoch 12 | train det_loss: 0.23148 | test det_loss: 0.56222 det_mAP: 0.69644 | running time: 38.36 sec
Epoch 13 | train det_loss: 0.22098 | test det_loss: 0.50493 det_mAP: 0.70856 | running time: 35.72 sec
Epoch 14 | train det_loss: 0.21031 | test det_loss: 0.53269 det_mAP: 0.71395 | running time: 35.68 sec
Epoch 15 | train det_loss: 0.19830 | test det_loss: 0.52011 det_mAP: 0.70633 | running time: 35.82 sec
Epoch 16 | train det_loss: 0.18818 | test det_loss: 0.52984 det_mAP: 0.71006 | running time: 35.18 sec
Epoch 17 | train det_loss: 0.18348 | test det_loss: 0.53116 det_mAP: 0.70349 | running time: 36.10 sec
Epoch 18 | train det_loss: 0.17461 | test det_loss: 0.50953 det_mAP: 0.71026 | running time: 36.00 sec
Epoch 19 | train det_loss: 0.16824 | test det_loss: 0.52999 det_mAP: 0.70921 | running time: 36.52 sec
Epoch 20 | train det_loss: 0.16184 | test det_loss: 0.55078 det_mAP: 0.70728 | running time: 36.81 sec
Epoch 21 | train det_loss: 0.15729 | test det_loss: 0.52552 det_mAP: 0.70996 | running time: 36.31 sec
Epoch 22 | train det_loss: 0.15330 | test det_loss: 0.52860 det_mAP: 0.70889 | running time: 36.45 sec
Epoch 23 | train det_loss: 0.15110 | test det_loss: 0.52596 det_mAP: 0.71017 | running time: 37.06 sec
Epoch 24 | train det_loss: 0.14932 | test det_loss: 0.53053 det_mAP: 0.71098 | running time: 36.54 sec
Epoch 25 | train det_loss: 0.14908 | test det_loss: 0.52863 det_mAP: 0.71088 | running time: 34.92 sec

