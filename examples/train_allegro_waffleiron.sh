#!/bin/bash
# Allegro Hand + Waffleiron 대규모 학습
# waffleiron 데모: 605 프레임, retarget 600 프레임 (frame 30-230 = 200 steps)
# contact_retarget 데이터 없음 → --use_retarget_contact 제외

HAND=allegro_hand
CLIP=waffleiron-30-230

# waffleiron은 geometry가 복잡해서 Jacobian int32 제한으로 ~3300 envs까지만 가능
python dexmachina/rl/train_rl_games.py -B 2048 -obf -obt --max_epochs 5000 \
    --actuate_object --retarget_name para --horizon 16 -imw 0.5 --gain_mode all --curr_schedule uniform --wait_epochs 100 --learning_rate 0.0003 \
    --contact_beta 10 --upper_ratios 0.9 0.9 1 --lower_ratios 0.8 0.8 1 --save_freq 5000 --fixed_mode uniform --uniform_mode slow \
    --action_penalty 0.01 --dialback_ep_len 80 --skip_grad --deque_len 30 --task_rew_betas 10 1 5 \
    --aux_reset_thres 0 0 0 --curr_rew_thres 0.6 0.01 0.01 0.01 -am hybrid --hybrid_scales 0.1 1.0 --kp_init 80 --kv_init 5 \
    --clip $CLIP -imi 0.3 -bc 0.3 -con 3 -ert 0.6 -exp allegro_waffleiron --hand $HAND

## 평가 (체크포인트 경로를 실제 경로로 교체):
# CK=logs/rl_games/allegro_hand/allegro-allegro_waffleiron_waffleiron30-230-s01-u01_B4096_hybrid_thres0.6_ho16_imi0.3_con3.0_bc0.3/nn/allegro_hand.pth
# python dexmachina/rl/eval_rl_games.py -B 1 --checkpoint $CK -v
