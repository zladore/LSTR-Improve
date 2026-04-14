# LSTR-改进

## 跑通指令

### 训练

CUDA_VISIBLE_DEVICES=0 python -u tools/train_net.py \
  --config_file ./configs/THUMOS/LSTR/lstr_long_512_work_8_kinetics_1x.yaml \
  --gpu 0 \
  SOLVER.PHASES "['train']" \
  DATA_LOADER.NUM_WORKERS 0 \
  DATA_LOADER.PIN_MEMORY False

### 测试

CUDA_VISIBLE_DEVICES=0 python -u tools/test_net.py \
  --config_file ./configs/THUMOS/LSTR/lstr_long_512_work_8_kinetics_1x.yaml \
  --gpu 0 \
  MODEL.CHECKPOINT ./checkpoints/configs/THUMOS/LSTR/lstr_long_512_work_8_kinetics_1x/epoch-25.pth \
  MODEL.LSTR.INFERENCE_MODE batch \
  DATA_LOADER.NUM_WORKERS 0 \
  DATA_LOADER.PIN_MEMORY False
