# Allegro Hand — 6개 물체 리타게팅 및 학습

**날짜**: 2026-02-26

Allegro Hand에 대해 ARCTIC 데이터셋의 6개 물체 전체(box, ketchup, laptop, mixer, notebook, waffleiron)의 리타게팅을 완료하고, 일괄 학습 스크립트를 구성한 기록.

---

## 배경

DexMachina는 ARCTIC 데이터셋의 사람 손 조작 시연을 로봇 손에 리타게팅한 뒤, 강화학습(PPO)으로 정책을 학습한다. Allegro Hand에 대해 기존에 리타게팅이 완료된 물체는 3개(box, mixer, waffleiron)였으며, 나머지 3개(ketchup, laptop, notebook)를 추가 리타게팅하여 6개 물체 전체를 준비했다.

### ARCTIC 6개 물체

| 물체 | 설명 | mesh 복잡도 | 관절 |
|------|------|------------|------|
| box | 단순 상자 | 낮음 (38M) | 1 (뚜껑) |
| ketchup | 케찹 병 | 낮음 (21M) | 1 (캡) |
| laptop | 노트북 | 낮음 (9.9M) | 1 (덮개) |
| mixer | 핸드믹서 | 중간 (32M) | 1 (헤드) |
| notebook | 공책 | 중간 (37M) | 1 (표지) |
| waffleiron | 와플기계 | 높음 (53M) | 1 (뚜껑) |

---

## Phase 1: 리타게팅

### 개요

리타게팅은 ARCTIC의 MANO(사람 손) 궤적을 로봇 손 관절 궤적으로 변환하는 과정이다. `dex_retargeting` 라이브러리의 역운동학(IK) 최적화를 사용하며, 200개 병렬 환경에서 동시에 실행하여 프레임별 최적 관절 자세를 구한다.

### 파이프라인

```
ARCTIC 원본 (.npy)          리타게팅 결과 (.pt)
assets/arctic/processed/    →    assets/retargeted/allegro_hand/
  s01/box_use_01.npy                s01/box_use_01_vector_para.pt
  s01/ketchup_use_01.npy            s01/ketchup_use_01_vector_para.pt
  s01/laptop_use_01.npy             s01/laptop_use_01_vector_para.pt
  ...                               ...
```

### 실행 방법

```bash
# 개별 실행
cd dexmachina/   # 중요: assets/ 상대경로 때문에 dexmachina/ 서브디렉토리에서 실행
python retargeting/parallel_retarget.py \
  --hand allegro_hand --clip ketchup-30-230 --save -sn para

# 일괄 실행 스크립트 (프로젝트 루트에서)
bash examples/retarget_allegro_all.sh
```

### 클립 문자열 형식

`--clip ketchup-30-230` → 물체=ketchup, 프레임 범위=30~230 (200 스텝). subject와 clip은 미지정 시 s01, u01이 기본값.

### 리타게팅 결과 (.pt 파일 내용)

각 `.pt` 파일에는 다음이 포함된다:

- `obj_pos` — 물체 위치 궤적 `[T, 3]`
- `obj_quat` — 물체 회전 궤적 `[T, 4]`
- `obj_arti` — 물체 관절각 궤적 `[T, n_joints]`
- `left/right` 각각:
  - `kpt_pos` — 키포인트 위치 `[T, n_kpts, 3]`
  - `wrist_pose` — 손목 위치+쿼터니언 `[T, 7]`
  - `residual_qpos` — 관절 자세 `[T, n_dofs]`
  - `contact_links_left/right` — 접촉 링크 인덱스

### 리타게팅 결과 요약

| 물체 | 파일 크기 | control_err (left) | control_err (right) | 상태 |
|------|----------|-------------------|---------------------|------|
| box | 1.69 MB | 기존 완료 | 기존 완료 | 기존 |
| ketchup | 580 KB | mean=0.024, max=0.22 | mean=0.051, max=0.36 | 신규 |
| laptop | 582 KB | mean=0.041, max=0.21 | mean=0.050, max=0.26 | 신규 |
| mixer | 1.72 MB | 기존 완료 | 기존 완료 | 기존 |
| notebook | 587 KB | mean=0.048, max=0.38 | mean=0.071, max=1.13 | 신규 |
| waffleiron | 1.71 MB | 기존 완료 | 기존 완료 | 기존 |

notebook의 오른손 최대 에러가 1.13으로 다소 높으나, 이는 특정 프레임에서의 피크값이며 평균은 0.07로 양호하다.

---

## Phase 2: 학습 스크립트

### 물체별 환경 설정

| 물체 | -B (병렬 envs) | `--use_retarget_contact` | 비고 |
|------|---------------|--------------------------|------|
| box | 3072 | O | contact_retarget 데이터 있음 |
| ketchup | 3072 | X | |
| laptop | 3072 | X | |
| mixer | 3072 | O | contact_retarget 데이터 있음 |
| notebook | 3072 | X | |
| waffleiron | 2048 | X | mesh 복잡도로 추가 마진 |

**Jacobian int32 제한**: Allegro hand의 충돌 geometry로 인해 **모든 물체**에서 constraint 수가 약 12638이다 (물체가 아닌 손이 constraint를 지배). DOF=51과 조합하면:

```
12638 × 51 × B < 2^31 (2,147,483,648)
→ B_max ≈ 3332
→ B=4096은 오버플로우 (2.64B > 2.15B)
→ B=3072는 안전 (1.98B < 2.15B)
```

초기에 box를 `-B 4096`으로 시도했으나 Jacobian 오버플로우가 발생하여 전체를 `-B 3072`로 조정했다.

### contact_retarget 데이터

`--use_retarget_contact` 플래그는 리타게팅 시 기록된 접촉 패턴을 보상으로 활용한다. 현재 allegro hand에 대해 box와 mixer만 데이터가 존재한다:

```
assets/contact_retarget/allegro_hand/s01/box_use_01.npy
assets/contact_retarget/allegro_hand/s01/mixer_use_01.npy
```

### 공통 하이퍼파라미터

기존 `train_allegro_waffleiron.sh`와 동일한 설정을 사용:

| 파라미터 | 값 | 설명 |
|---------|-----|------|
| `-am hybrid` | hybrid | 손목=residual, 손가락=absolute |
| `--hybrid_scales` | 0.1 1.0 | 손목/손가락 스케일 |
| `--max_epochs` | 5000 | 최대 에폭 수 |
| `--horizon` | 16 | 롤아웃 길이 |
| `--learning_rate` | 0.0003 | 학습률 |
| `--curr_schedule` | uniform | 커리큘럼 스케줄 (균등) |
| `-imi` | 0.3 | imitation 보상 가중치 |
| `-bc` | 0.3 | behavior cloning 보상 가중치 |
| `-con` | 3.0 | contact 보상 가중치 |
| `-ert` | 0.6 | 리셋 임계값 |
| `--kp_init` / `--kv_init` | 80 / 5 | 물체 PD 게인 초기값 |

### 사용법

```bash
# 6개 물체 전체 순차 학습
bash examples/train_allegro_all_objects.sh

# 특정 물체만 학습
bash examples/train_allegro_all_objects.sh box ketchup

# 단일 물체 학습
bash examples/train_allegro_all_objects.sh laptop
```

학습 결과는 `logs/rl_games/allegro_hand/allegro-allegro_{obj}_...` 디렉토리에 저장된다.

---

## Phase 3: 검증

### 리타게팅 결과 시각적 확인

kinematic 모드로 리타게팅된 궤적을 재생하여 손-물체 상호작용을 확인:

```bash
python examples/preview_training.py -B 4 \
  --hand allegro_hand --clip ketchup-30-230 \
  --retarget_name para -am kinematic \
  --actuate_object --kp_init 300 --kv_init 30
```

- `-am kinematic`: 리타게팅 궤적을 그대로 재생 (RL 학습 없이)
- `--kp_init 300 --kv_init 30`: 높은 PD 게인으로 물체가 데모를 따라가게 설정
- `-B 4`: 4개 환경만 실행 (시각화용)

각 물체에 대해 위 명령의 `--clip`을 변경하여 확인:
- `ketchup-30-230`
- `laptop-30-230`
- `notebook-30-230`

### 학습 결과 평가

학습 완료 후 체크포인트로 평가:

```bash
python dexmachina/rl/eval_rl_games.py -B 1 \
  --checkpoint logs/rl_games/allegro_hand/{exp_dir}/nn/allegro_hand.pth -v
```

---

## 트러블슈팅: `batch_dofs_info` 호환성

### 문제

리타게팅 실행 시 Genesis 0.3.3에서 다음 에러 발생:
```
GenesisException: Expecting 1D output tensor.
```

### 원인

`parallel_retarget.py`에서 `batch_dofs_info=False`로 설정되어 있었으나, `robot.py`의 `set_joint_gains()`는 `num_envs > 0`일 때 2D 텐서 `[num_envs, num_dofs]`를 생성한다.

Genesis 0.3.3에서 `batch_dofs_info` 설정에 따른 `set_dofs_kp/kv/force_range`의 동작:

| `batch_dofs_info` | 기대 텐서 shape | 비고 |
|-------------------|----------------|------|
| `False` (기본값) | 항상 1D `[n_dofs]` | n_envs 무관 |
| `True` | n_envs>0이면 2D `[n_envs, n_dofs]`, 아니면 1D | 환경별 개별 설정 가능 |

학습 코드(`constructors.py`)에서는 `batch_dofs_info=True`를 설정하여 2D 텐서를 문제없이 사용했으나, 리타게팅 코드에서는 `False`로 되어 있어 불일치가 발생했다.

### 해결

`parallel_retarget.py`에서 `batch_dofs_info=True`로 변경:

```python
# dexmachina/retargeting/parallel_retarget.py (line 61)
# 변경 전
batch_dofs_info=False,
# 변경 후
batch_dofs_info=True,
```

### 의존성 추가 설치

리타게팅 실행에 필요한 패키지 2개가 설치되어 있지 않았다:

```bash
pip install lxml dex_retargeting
```

- `lxml`: URDF XML 파싱
- `dex_retargeting`: 역운동학 기반 리타게팅 최적화 라이브러리

---

## 생성/수정된 파일 목록

| 파일 | 유형 | 설명 |
|------|------|------|
| `examples/train_allegro_all_objects.sh` | 신규 | 6개 물체 일괄 학습 스크립트 |
| `examples/retarget_allegro_all.sh` | 신규 | 미완성 3개 물체 리타게팅 스크립트 |
| `dexmachina/retargeting/parallel_retarget.py` | 수정 | `batch_dofs_info=True` |
| `CHANGELOG.md` | 수정 | 항목 #11 추가 |
| `assets/retargeted/allegro_hand/s01/ketchup_use_01_vector_para.pt` | 신규 | 리타게팅 결과 |
| `assets/retargeted/allegro_hand/s01/laptop_use_01_vector_para.pt` | 신규 | 리타게팅 결과 |
| `assets/retargeted/allegro_hand/s01/notebook_use_01_vector_para.pt` | 신규 | 리타게팅 결과 |
