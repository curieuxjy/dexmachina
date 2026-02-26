# DexMachina Study Guide

## 프로젝트 개요

DexMachina는 **양손 로봇 손(bimanual dexterous hands)**으로 물체를 조작하는 정책을 학습하는 프로젝트.
ARCTIC 데이터셋의 사람 손 시연(demonstration)을 로봇 손에 리타겟팅(retargeting)하고, 이를 기반으로 강화학습(RL)으로 조작 정책을 학습한다.

```
사람 손 시연 (ARCTIC) → 리타겟팅 → RL 학습 → 평가/시각화
```

---

## 1. 프로젝트 구조

```
dexmachina/
├── assets/                     # 로봇 손 URDF, 물체, 시연 데이터
│   ├── allegro_hand/           # Allegro Hand (22 DOF)
│   ├── inspire_hand/           # Inspire Hand (18 DOF)
│   ├── dex3_hand/              # Dex3 Hand (13 DOF)
│   ├── xhand/                  # XHand
│   ├── schunk_hand/            # Schunk Hand
│   ├── ability_hand/           # Ability Hand
│   ├── dexrobot_hand/          # DexRobot Hand
│   ├── mano_hand/              # MANO (사람 손 레퍼런스)
│   ├── arctic/                 # ARCTIC 데이터셋 물체 + 처리된 시연
│   │   ├── box/                # 물체 URDF/메시
│   │   ├── ketchup/, laptop/, mixer/, notebook/, waffleiron/
│   │   └── processed/          # 처리된 MANO 시연 (.npy)
│   ├── retargeted/             # 리타겟팅된 관절 데이터 (.pt)
│   └── contact_retarget/       # 리타겟팅된 접촉 데이터
│
├── envs/                       # RL 환경
│   ├── base_env.py             # 핵심 환경 (양손 + 물체 시뮬레이션)
│   ├── robot.py                # 로봇 손 컨트롤러
│   ├── object.py               # 물체 컨트롤러
│   ├── rewards.py              # 보상 함수 (task + imitation + contact)
│   ├── contacts.py             # 접촉 감지/필터링
│   ├── curriculum.py           # 커리큘럼 학습 (게인 스케줄링)
│   ├── demo_data.py            # 시연 데이터 로딩
│   ├── randomizations.py       # 도메인 랜덤화
│   └── hand_cfgs/              # 각 로봇 손별 설정
│       ├── allegro.py, inspire.py, dex3.py, xhand.py, ...
│
├── rl/                         # RL 학습/평가
│   ├── train_rl_games.py       # 학습 메인 스크립트
│   ├── eval_rl_games.py        # 평가 스크립트
│   ├── rl_games_wrapper.py     # RL-Games 환경 래퍼
│   └── configs/
│       └── rl_games_ppo_cfg.yaml  # PPO 하이퍼파라미터
│
├── retargeting/                # 리타겟팅 파이프라인
│   ├── retarget_utils.py       # 리타겟팅 핵심 함수
│   ├── parallel_retarget.py    # 병렬 리타겟팅
│   ├── map_contacts.py         # 접촉 매핑
│   └── process_arctic.py       # ARCTIC 원본 데이터 처리
│
├── hand_proc/                  # 새 로봇 손 처리 도구
│   ├── inspect_raw_urdf.py     # URDF 분석
│   ├── minimal_retarget.py     # 리타겟팅 데모
│   ├── add_wrist_dof.py        # 손목 DOF 추가
│   └── tune_gains.py           # 컨트롤러 게인 튜닝
│
└── examples/                   # 예제 스크립트
    ├── inspect_hand.py         # 로봇 손 시각화
    ├── load_object.py          # 물체 로딩/시연 재생
    ├── train_rl.sh             # 학습 예제 (inspire_hand, 대규모)
    └── train_dex3.sh           # 학습 예제 (dex3_hand, 소규모)
```

---

## 2. 데이터 파이프라인 이해

### 사용 가능한 데이터

| 물체 | 처리된 시연 | inspire 리타겟팅 | dex3 리타겟팅 | allegro 리타겟팅 | xhand 리타겟팅 |
|------|-----------|-----------------|-------------|----------------|--------------|
| box | box_use_01 | O | O | O | O |
| mixer | mixer_use_01 | O | - | O | - |
| ketchup | ketchup_use_01/02 | - | - | - | - |
| laptop | laptop_use_01 | - | - | - | - |
| notebook | notebook_use_01 | - | - | - | - |
| waffleiron | waffleiron_use_01 | - | - | O | - |

> 리타겟팅 데이터가 없는 조합은 `retargeting/` 파이프라인으로 직접 생성해야 함.

### 데이터 흐름

```
[ARCTIC 원본]  사람이 물체를 조작하는 MANO 손 파라미터
     ↓  process_arctic.py
[processed/]   정리된 시연: 물체 위치/회전/관절 + MANO 관절 위치
     ↓  parallel_retarget.py / minimal_retarget.py
[retargeted/]  로봇 손 관절 각도로 변환된 궤적 (.pt)
[contact_retarget/]  접촉 정보 매핑
     ↓
[BaseEnv]      시뮬레이션에서 RL 학습에 사용
```

---

## 3. 핵심 모듈 설명

### `envs/base_env.py` - 환경 핵심
- 왼손 + 오른손 로봇과 물체를 Genesis 시뮬레이터에서 관리
- `step()`: 정책 액션 → 로봇/물체 제어 → 물리 시뮬 → 관측/보상 반환
- `reset_idx()`: 에피소드 리셋 (시연 데이터의 시작 프레임으로)

### `envs/robot.py` - 로봇 컨트롤러
- **Action Mode** 4가지:
  - `absolute`: 정책이 관절 위치를 직접 출력
  - `residual`: 시연 궤적 위에 delta를 더함
  - `hybrid`: residual + absolute 혼합 (학습에 주로 사용)
  - `kinematic`: 역운동학 기반
- PD 컨트롤러로 관절 위치/속도 제어

### `envs/rewards.py` - 보상 함수
- **Task Reward**: 물체가 시연 궤적을 따르는 정도 (위치 + 회전 + 관절)
- **Imitation Reward** (`-imi`): 손가락 끝 위치가 시연과 일치하는 정도
- **Contact Reward** (`-con`): 접촉 패턴이 시연과 일치하는 정도
- **BC Reward** (`-bc`): 관절 각도가 리타겟팅 결과와 일치하는 정도

### `envs/curriculum.py` - 커리큘럼 학습
- 초기에는 물체의 PD 게인(kp, kv)을 높게 → 물체가 시연 궤적에 고정됨
- 학습 진행에 따라 게인을 줄임 → 로봇이 실제로 물체를 조작해야 함
- `uniform` 스케줄: 에포크마다 게인을 랜덤 샘플링

---

## 4. 단계별 실행 가이드

### Step 1: 로봇 손 시각화 (기본 동작 확인)

```bash
# Allegro Hand 확인 (기본)
python examples/inspect_hand.py -v

# 다른 손 모델 확인
python examples/inspect_hand.py --hand inspire_hand -v
python examples/inspect_hand.py --hand dex3_hand -v

# 무중력에서 확인
python examples/inspect_hand.py --hand allegro_hand --zero_gravity -v
```

### Step 2: 물체 시각화 및 시연 재생

```bash
# 물체만 로딩 (랜덤 위치)
python examples/load_object.py --obj_name box -v

# 시연 데이터로 물체 궤적 재생
python examples/load_object.py --obj_name box --load_demo -v

# 관절 구동 활성화
python examples/load_object.py --obj_name box --load_demo --actuate_object -v

# 다른 물체
python examples/load_object.py --obj_name mixer --load_demo -v
```

### Step 3: RL 학습 (소규모 테스트)

```bash
# dex3_hand + box, 환경 40개 (빠른 테스트)
bash examples/train_dex3.sh
```

또는 직접 실행:

```bash
python dexmachina/rl/train_rl_games.py \
    -B 40 \
    --hand dex3_hand \
    --clip box-30-230 \
    --retarget_name para \
    --actuate_object \
    --max_epochs 500 \
    --horizon 16 \
    -am hybrid --hybrid_scales 0.1 1.0 \
    --kp_init 80 --kv_init 5 \
    --curr_schedule uniform \
    --wait_epochs 100 \
    -imw 0.5 -imi 0.3 -bc 0.3 -con 3 \
    --contact_beta 10 \
    --task_rew_betas 10 1 5 \
    -ert 0.6 \
    --use_retarget_contact \
    --group_collisions \
    -obf -obt \
    -exp my_first_test
```

### Step 4: RL 학습 (대규모, 본격 학습)

```bash
# inspire_hand + box, 환경 4096개 (GPU 메모리 충분시)
bash examples/train_rl.sh
```

### Step 5: 학습 결과 평가

```bash
# 체크포인트 경로 확인
ls logs/rl_games/

# 예: dex3_hand 체크포인트로 평가 (뷰어 활성화)
CK=logs/rl_games/dex3_hand/<실험_디렉토리>/nn/dex3_hand.pth
python dexmachina/rl/eval_rl_games.py -B 1 --checkpoint $CK -v

# 비디오 녹화
python dexmachina/rl/eval_rl_games.py -B 1 --checkpoint $CK --record_video
```

---

## 5. 주요 학습 파라미터 해설

| 파라미터 | 의미 | 기본/권장값 |
|---------|------|-----------|
| `-B` | 병렬 환경 수 (배치 크기) | 40 (테스트), 4096 (본격) |
| `--hand` | 로봇 손 모델 | inspire_hand, dex3_hand, allegro_hand |
| `--clip` | 시연 클립 `물체-시작-끝` | box-30-230 |
| `-am` | 액션 모드 | hybrid (권장) |
| `--hybrid_scales` | hybrid 모드 스케일 | 0.1 1.0 |
| `--max_epochs` | 최대 학습 에포크 | 5000 |
| `--horizon` | 에피소드당 스텝 수 | 16 |
| `-imw` | imitation 보상 가중치 | 0.5 |
| `-imi` | fingertip imitation 계수 | 0.3 |
| `-bc` | behavior cloning 계수 | 0.3 |
| `-con` | contact 보상 계수 | 3.0 |
| `-ert` | 조기 리셋 보상 임계값 | 0.6 |
| `--kp_init` | 초기 위치 게인 (커리큘럼) | 80 |
| `--kv_init` | 초기 속도 게인 (커리큘럼) | 5 |
| `--curr_schedule` | 커리큘럼 스케줄 | uniform |
| `--task_rew_betas` | task 보상 스케일 (pos rot arti) | 10 1 5 |
| `--contact_beta` | 접촉 보상 스케일 | 10 |
| `-exp` | 실험 이름 (로그 디렉토리) | - |
| `--actuate_object` | 물체 관절 제어 활성화 | 사용 권장 |
| `-obf -obt` | 물체 관측에 force/torque 포함 | 사용 권장 |

---

## 6. 소스코드 읽기 순서 (권장)

```
1. examples/inspect_hand.py          # 가장 단순. 손 로딩 → 랜덤 액션
2. examples/load_object.py           # 물체 로딩 + 시연 데이터 재생
3. dexmachina/envs/robot.py          # BaseRobot: 액션 모드, PD 제어
4. dexmachina/envs/object.py         # ArticulatedObject: 물체 제어
5. dexmachina/envs/demo_data.py      # 시연 데이터 로딩 구조
6. dexmachina/envs/rewards.py        # 보상 함수 구성 이해
7. dexmachina/envs/base_env.py       # 전체 환경 통합 (가장 큼)
8. dexmachina/rl/train_rl_games.py   # 학습 루프 + 인자 파싱
9. dexmachina/rl/eval_rl_games.py    # 평가 + 시각화
10. dexmachina/envs/curriculum.py    # 커리큘럼 학습 전략
```

---

## 7. 빠른 동작 확인 커맨드 요약

```bash
# 1) 손 시각화
python examples/inspect_hand.py --hand allegro_hand -v

# 2) 물체 + 시연 재생
python examples/load_object.py --obj_name box --load_demo -v

# 3) 소규모 학습 (dex3 + box)
bash examples/train_dex3.sh

# 4) 학습 로그 확인
ls logs/rl_games/

# 5) 평가 (체크포인트 경로 교체)
python dexmachina/rl/eval_rl_games.py -B 1 --checkpoint <체크포인트.pth> -v
```

---

## 8. 트러블슈팅

| 문제 | 해결 |
|------|------|
| CUDA sm_120 에러 (RTX 5090) | `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128` |
| GUI 안 뜸 | `-v` 플래그 추가 |
| `dofs_idx_local` 경고 | Genesis API 변경, 무시 가능 |
| `frictionloss, damping` 경고 | URDF free joint 설정, 무시 가능 |
| 메모리 부족 | `-B` 값 줄이기 (40 → 8) |
| 리타겟팅 데이터 없음 | `dexmachina/retargeting/parallel_retarget.py`로 생성 |
