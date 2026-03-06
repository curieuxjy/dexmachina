#!/bin/bash
# Allegro Hand 리타게팅 일괄 실행 (ketchup, laptop, notebook)
# 이미 완료된 물체: box, mixer, waffleiron
# 결과: assets/retargeted/allegro_hand/s01/{obj}_use_01_vector_para.pt
# 실행: bash examples/retarget_allegro_all.sh (프로젝트 루트에서)
#
# 주의: clip 범위는 반드시 frame 0부터 시작해야 함.
# RL 학습 시 {obj}-30-230 clip으로 [30:230] 슬라이싱하므로,
# retarget 데이터가 최소 230 프레임 이상이어야 함.
# (기존 30-230은 200프레임만 생성 → [30:230] = 170프레임 → 오류)

set -e

# parallel_retarget.py 내부에서 assets/ 를 상대경로로 참조하므로 dexmachina/ 에서 실행
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SCRIPT_DIR/dexmachina"

HAND=allegro_hand

for OBJ in ketchup laptop notebook; do
    echo "=== Retargeting: $OBJ ==="
    python retargeting/parallel_retarget.py \
        --hand $HAND --clip ${OBJ}-0-230 --save -sn para
    echo "=== Done: $OBJ ==="
    echo ""
done

echo "All retargeting complete. Verify with:"
echo "ls assets/retargeted/allegro_hand/s01/"
