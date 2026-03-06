#!/bin/bash
# Allegro Hand — 6개 ARCTIC 물체 학습 스크립트
# 사용법:
#   bash examples/train_allegro_all_objects.sh              # 6개 모두 순차 학습
#   bash examples/train_allegro_all_objects.sh box ketchup   # 지정 물체만 학습
#
# contact_retarget 데이터 있는 물체: box, mixer
# contact_retarget 데이터 없는 물체: ketchup, laptop, notebook, waffleiron

set -e

HAND=allegro_hand

# 공통 하이퍼파라미터
COMMON_ARGS="-obf -obt --max_epochs 5000 \
    --actuate_object --retarget_name para --horizon 16 -imw 0.5 --gain_mode all --curr_schedule uniform --wait_epochs 100 --learning_rate 0.0003 \
    --contact_beta 10 --upper_ratios 0.9 0.9 1 --lower_ratios 0.8 0.8 1 --save_freq 5000 --fixed_mode uniform --uniform_mode slow \
    --action_penalty 0.01 --dialback_ep_len 80 --skip_grad --deque_len 30 --task_rew_betas 10 1 5 \
    --aux_reset_thres 0 0 0 --curr_rew_thres 0.6 0.01 0.01 0.01 -am hybrid --hybrid_scales 0.1 1.0 --kp_init 80 --kv_init 5 \
    -imi 0.3 -bc 0.3 -con 3 -ert 0.6 --hand $HAND"

# 물체별 설정: BATCH_SIZE
# Allegro hand의 충돌 geometry로 인해 constraint 수 ≈ 12638, DOF = 51
# Jacobian int32 제한: 12638 × 51 × B < 2^31 → B_max ≈ 3332
# 안전 마진을 두고 B=3072 사용 (waffleiron은 mesh 복잡도로 인해 2048)
declare -A BATCH_SIZES=(
    [box]=3072
    [ketchup]=3072
    [laptop]=3072
    [mixer]=3072
    [notebook]=3072
    [waffleiron]=2048
)

# contact_retarget 데이터가 있는 물체
declare -A HAS_CONTACT_RETARGET=(
    [box]=1
    [mixer]=1
)

ALL_OBJECTS="box ketchup laptop mixer notebook waffleiron"

# 인자가 있으면 해당 물체만, 없으면 전체 실행
if [ $# -gt 0 ]; then
    OBJECTS="$@"
else
    OBJECTS=$ALL_OBJECTS
fi

for OBJ in $OBJECTS; do
    B=${BATCH_SIZES[$OBJ]}
    if [ -z "$B" ]; then
        echo "ERROR: Unknown object '$OBJ'. Available: $ALL_OBJECTS"
        exit 1
    fi

    CLIP="${OBJ}-30-230"
    EXP="allegro_${OBJ}"

    EXTRA=""
    if [ -n "${HAS_CONTACT_RETARGET[$OBJ]}" ]; then
        EXTRA="--use_retarget_contact"
    fi

    echo "=========================================="
    echo " Training: $OBJ (B=$B) $EXTRA"
    echo "=========================================="

    python dexmachina/rl/train_rl_games.py -B $B $COMMON_ARGS \
        --clip $CLIP -exp $EXP $EXTRA

    echo ""
    echo "=== Completed: $OBJ ==="
    echo ""
done

echo "All training complete!"
