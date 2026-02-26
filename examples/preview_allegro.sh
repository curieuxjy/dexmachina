#!/bin/bash
# Allegro Hand 대규모 환경 데모 시각화
# kinematic 모드: 데모 궤적 그대로 재생 (학습 아님)
# GUI 뷰어로 여러 환경을 동시에 확인

# waffleiron 데모 재생 (환경 16개)
python examples/preview_training.py -B 16 \
    --hand allegro_hand \
    --clip waffleiron-30-230 \
    --retarget_name para \
    -am kinematic \
    --actuate_object --kp_init 300 --kv_init 30

## 다른 예시:
# box 데모 재생
# python examples/preview_training.py -B 16 --hand allegro_hand --clip box-30-230 --retarget_name para -am kinematic --actuate_object --kp_init 300 --kv_init 30

# hybrid 모드 + 랜덤 액션 (학습 시 에이전트의 행동 시뮬레이션)
# python examples/preview_training.py -B 16 --hand allegro_hand --clip waffleiron-30-230 --retarget_name para -am hybrid --hybrid_scales 0.1 1.0 --random_actions --actuate_object --kp_init 300 --kv_init 30
