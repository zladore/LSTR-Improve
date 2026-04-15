# LSTR-改进

## 跑通指令

## 跑通指令
Step1.先进入虚拟环境
source /home/hbxz_lzl/venvs/lstr_bw/bin/activate

cd long-short-term-transformer
# Training from scratch
python tools/train_net.py --config_file $PATH_TO_CONFIG_FILE --gpu $CUDA_VISIBLE_DEVICES
# Finetuning from a pretrained model
python tools/train_net.py --config_file /home/hbxz_lzl/LSTR_Improve/configs/THUMOS/LSTR/lstr_long_512_work_8_kinetics_1x.yaml --gpu 1 \
    MODEL.CHECKPOINT /home/hbxz_lzl/LSTR_Improve/checkpoints/home/hbxz_lzl/LSTR-Trimodal/configs/THUMOS/LSTR/lstr_long_512_work_8_kinetics_1x/epoch-7.pth


# 推理
cd long-short-term-transformer
# Online inference in stream mode
python tools/test_net.py --config_file /home/hbxz_lzl/LSTR_Improve/configs/THUMOS/LSTR/lstr_long_512_work_8_kinetics_1x.yaml --gpu 1 \
    MODEL.CHECKPOINT /home/hbxz_lzl/LSTR_Improve/checkpoints/home/hbxz_lzl/LSTR-Trimodal/configs/THUMOS/LSTR/lstr_long_512_work_8_kinetics_1x/epoch-7.pth MODEL.LSTR.INFERENCE_MODE batch

## 类名
CLASS_NAMES = [
    'Background', 'BaseballPitch', 'BasketballDunk', 'Billiards', 'CleanAndJerk',
    'CliffDiving', 'CricketBowling', 'CricketShot', 'Diving', 'FrisbeeCatch',
    'GolfSwing', 'HammerThrow', 'HighJump', 'JavelinThrow', 'LongJump',
    'PoleVault', 'Shotput', 'SoccerPenalty', 'TennisSwing', 'ThrowDiscus',
    'VolleyballSpiking', 'Ambiguous'
]
# CLASS_NAMES = [
#     '背景', '棒球投球', '篮球扣篮', '台球', '举重挺举',
#     '悬崖跳水', '板球投球', '板球击球', '跳水', '飞盘接捕',
#     '高尔夫挥杆', '链球投掷', '跳高', '标枪投掷', '跳远',
#     '撑杆跳', '铅球投掷', '足球点球', '网球挥拍', '铁饼投掷',
#     '排球扣球', '模糊类'
# ]
