


low_kinetics_bninception', 'TARGET_PERFRAME': 'target_perframe'}), 'DATA_LOADER': CfgNode({'BATCH_SIZE': 16, 'NUM_WORKERS': 8, 'PIN_MEMORY': True}), 'SOLVER': CfgNode({'START_EPOCH': 1, 'NUM_EPOCHS': 25, 'OPTIMIZER': 'adam', 'BASE_LR': 7e-05, 'WEIGHT_DECAY': 5e-05, 'MOMENTUM': 0.9, 'SCHEDULER': CfgNode({'SCHEDULER_NAME': 'warmup_cosine', 'MILESTONES': [], 'GAMMA': 0.1, 'WARMUP_FACTOR': 0.3, 'WARMUP_EPOCHS': 10.0, 'WARMUP_METHOD': 'linear'}), 'PHASES': ['train', 'test']}), 'OUTPUT_DIR': 'checkpoints/home/hbxz_lzl/LSTR_Improve/configs/THUMOS/LSTR/lstr_long_512_work_8_kinetics_1x', 'SESSION': '', 'VERBOSE': False, 'GPU': '1'})
BatchInference: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████| 696/696 [27:21<00:00,  2.36s/it]
Action detection perframe mAP: 0.70951
(lstr_bw) hbxz_lzl@hqu:~/LSTR_Improve$ 

ER': CfgNode({'START_EPOCH': 1, 'NUM_EPOCHS': 25, 'OPTIMIZER': 'adam', 'BASE_LR': 7e-05, 'WEIGHT_DECAY': 5e-05, 'MOMENTUM': 0.9, 'SCHEDULER': CfgNode({'SCHEDULER_NAME': 'warmup_cosine', 'MILESTONES': [], 'GAMMA': 0.1, 'WARMUP_FACTOR': 0.3, 'WARMUP_EPOCHS': 10.0, 'WARMUP_METHOD': 'linear'}), 'PHASES': ['train', 'test']}), 'OUTPUT_DIR': 'checkpoints/home/hbxz_lzl/LSTR_Improve/configs/THUMOS/LSTR/lstr_long_512_work_8_kinetics_1x', 'SESSION': '', 'VERBOSE': False, 'GPU': '1'})
BatchInference: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████| 696/696 [27:19<00:00,  2.36s/it]
Action detection perframe mAP: 0.70923
(lstr_bw) hbxz_lzl@hqu:~/LSTR_Improve$ 


ER': CfgNode({'START_EPOCH': 1, 'NUM_EPOCHS': 25, 'OPTIMIZER': 'adam', 'BASE_LR': 7e-05, 'WEIGHT_DECAY': 5e-05, 'MOMENTUM': 0.9, 'SCHEDULER': CfgNode({'SCHEDULER_NAME': 'warmup_cosine', 'MILESTONES': [], 'GAMMA': 0.1, 'WARMUP_FACTOR': 0.3, 'WARMUP_EPOCHS': 10.0, 'WARMUP_METHOD': 'linear'}), 'PHASES': ['train', 'test']}), 'OUTPUT_DIR': 'checkpoints/home/hbxz_lzl/LSTR_Improve/configs/THUMOS/LSTR/lstr_long_512_work_8_kinetics_1x', 'SESSION': '', 'VERBOSE': False, 'GPU': '1'})
BatchInference: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████| 696/696 [27:21<00:00,  2.36s/it]
Action detection perframe mAP: 0.70861


基线是 0.70823。
alpha=0.01 → 0.70633
**alpha=0.02 → 0.70951**
alpha=0.03 → 0.70923
alpha=0.05 → 0.70861
