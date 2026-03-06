# DexMachina Training Process

DexMachina의 전체 학습 과정을 단계별로 설명합니다.

---

## 전체 흐름 요약

```
ARCTIC 데모 → 리타게팅 → RL 학습 (PPO) → 평가
```

1. **데이터 준비**: ARCTIC 데이터셋에서 사람 손 조작 데모를 로봇 손(Allegro/Inspire)에 맞게 리타게팅
2. **환경 구성**: Genesis 시뮬레이터에서 병렬 환경 생성 (로봇 손 + 물체 + 테이블)
3. **RL 학습**: PPO(Proximal Policy Optimization)로 정책 학습
4. **평가**: 학습된 체크포인트로 물체 조작 평가

---

## Phase 1: 초기화

### 1.1 진입점

```
python dexmachina/rl/train_rl_games.py [args]
```

`train_rl_games.py`가 전체 학습의 진입점입니다.

### 1.2 데이터 로딩

`get_all_env_cfg(args, device)` 함수(`constructors.py`)가 모든 설정을 조립합니다.

**클립 파싱**: `--clip box-30-230-s01-u01` 형태의 문자열을 `parse_clip_string()`으로 분해:
- `obj_name`: box
- `start/end`: 30-230 (프레임 범위 → 200 스텝)
- `subject_name`: s01
- `use_clip`: u01

**데모 데이터** (`demo_data.py`의 `load_genesis_retarget_data()`):
- 경로: `assets/retargeted/{hand}/{subject}/{obj}_use_{clip}_vector_{save_name}.pt`
- 내용: 물체의 프레임별 위치(`obj_pos`), 회전(`obj_quat`), 관절각도(`obj_arti`), 접촉 정보(`contact_links_left/right`)

**리타게팅 데이터**:
- 각 손(left/right)의 키포인트 궤적(`kpt_pos`), 손목 자세(`wrist_pose`), 관절 자세(`residual_qpos`)

### 1.3 Genesis 씬 구성

`BaseEnv.__init__()`에서 Genesis 물리 시뮬레이터를 초기화합니다.

```python
gs.init(backend=gs.gpu, logging_level='warning')
```

**씬 구성 요소**:
- 테이블 (고정 rigid body)
- 물체 (URDF 로드, 관절이 있는 articulated body)
- 로봇 손 × 2 (left, right — 각각 URDF 로드)
- (선택) 접촉 시각화 마커

**병렬 환경**: `-B 2048` → `scene.build(n_envs=2048)` — 동일한 씬을 2048개 복제하여 GPU에서 병렬 시뮬레이션.

### 1.4 PPO 에이전트 설정

`rl_games_ppo_cfg.yaml`에서 PPO 하이퍼파라미터를 로드합니다.

| 파라미터 | 값 | 설명 |
|---------|-----|------|
| 네트워크 | MLP [512, 512, 256, 128] | Actor-Critic 공유 네트워크 |
| 활성화함수 | ELU | |
| gamma | 0.99 | 할인율 |
| tau (GAE λ) | 0.95 | Generalized Advantage Estimation |
| learning_rate | 3e-4 | 적응형 스케줄 (KL divergence 기반) |
| e_clip | 0.2 | PPO 클리핑 범위 |
| grad_norm | 1.0 | 그래디언트 클리핑 |
| reward_shaper scale_value | 0.1 | 보상값 스케일링 |
| normalize_value | True | 값 함수 정규화 |
| value_bootstrap | True | 에피소드 끝에서 값 부트스트래핑 |

**동적 설정** (`train_rl_games.py`에서 계산):
```python
minibatch_size = num_envs × 8        # 예: 2048 × 8 = 16384
mini_epochs = max(1, num_envs/4096 × 5)  # 예: 2048/4096 × 5 ≈ 3
horizon_length = args.horizon           # 기본 16 스텝
```

### 1.5 래퍼 등록

```python
env = RlGamesVecEnvWrapper(env, rl_device, clip_obs=5.0, clip_actions=1.0)
```

`RlGamesVecEnvWrapper`는 `BaseEnv`를 RL-Games 라이브러리가 요구하는 인터페이스(`IVecEnv`)로 변환합니다.
- 관찰값 클리핑: ±5.0
- 액션 클리핑: ±1.0
- GPU 버퍼 간 전송 관리

### 1.6 W&B 로깅 초기화

```python
wandb.init(project="dexmachina", config=wandb_cfg, name=exp_name)
```

실험 설정, 보상 곡선, 커리큘럼 진행 상황 등을 Weights & Biases로 추적합니다.

---

## Phase 2: 학습 루프

RL-Games의 `Runner`가 PPO 학습 루프를 실행합니다.

```python
runner = Runner(IsaacAlgoObserver())
runner.load(agent_cfg)
runner.reset()
runner.run({"train": True})
```

### 2.1 하나의 Epoch 구조

각 epoch는 다음 순서로 진행:

```
[Rollout 데이터 수집] → [PPO 정책 업데이트] → [커리큘럼 조정] → [로깅]
```

### 2.2 Rollout 데이터 수집

`horizon` (기본 16) 스텝 동안 현재 정책으로 환경과 상호작용하여 경험 데이터를 수집합니다.

**한 스텝의 흐름** (`BaseEnv.step()`):

```
1. 액션 클램핑:    actions = clamp(actions, -clip, clip) × action_scale
2. 로봇 액션 적용:  robot.step(actions)      # 관절 위치/토크 명령
3. 물체 스텝:      object.step()             # PD 컨트롤러로 물체 구동
4. 물리 시뮬레이션:  scene.step()              # Genesis GPU 물리 연산
5. 상태 업데이트:   _compute_intermediate_values()  # NaN 체크, 접촉 정보 갱신
6. 종료 판단:      _get_dones()              # 타임아웃, 물체 낙하, early reset
7. 보상 계산:      _get_rewards()            # 복합 보상 함수
8. 리셋:          reset_idx(done_envs)       # 종료된 환경 리셋
9. 관찰 반환:      get_observations()         # 다음 스텝을 위한 관찰값
```

### 2.3 액션 모드

`--action_mode` (또는 `-am`)에 따라 정책이 출력하는 액션의 의미가 달라집니다:

| 모드 | 설명 |
|------|------|
| `kinematic` | 데모 궤적을 그대로 재생 (학습 아님) |
| `residual` | 데모 관절각도에 대한 잔차(delta)를 출력 |
| `hybrid` | 손목=residual, 손가락=absolute |
| `absolute` | 관절 위치를 직접 출력 |

학습 시 주로 `hybrid` 모드를 사용하며, `--hybrid_scales 0.1 1.0`으로 손목/손가락의 액션 스케일을 조절합니다.

### 2.4 보상 함수

총 보상은 여러 구성 요소의 합입니다:

```
total_reward = task_rew + imi_rew + con_rew + bc_rew - penalties
```

#### Task Reward (물체 추적)

데모 궤적을 얼마나 잘 따라가는지 측정합니다.

```
task_rew = w_task × exp(-β_pos·‖pos_err‖) × exp(-β_rot·rot_err) × exp(-β_arti·arti_err)
```

| 항목 | β (기본값) | 의미 |
|------|-----------|------|
| pos_err | β=20.0 | 물체 위치 오차 (L2 거리) |
| rot_err | β=5.0 | 물체 회전 오차 (쿼터니언 거리) |
| arti_err | β=20.0 | 관절 각도 오차 |

세 항의 **곱**으로 계산됨 (`multiply_task_rew=True`): 하나라도 크게 틀리면 전체 보상이 급감합니다.

#### Imitation Reward (키포인트 모방)

손의 키포인트(손가락 끝 등)가 데모의 키포인트를 얼마나 잘 따라가는지 측정합니다.

```
imi_rew = w_imi × mean(exp(-β_finger · ‖kpt_pos - demo_kpt_pos‖))
```

- β_finger = 20.0 (기본값)
- 좌우 손의 키포인트 평균
- `--imi_wrist_weight` > 0이면 손목 자세도 포함

#### Contact Reward (접촉 패턴)

로봇 손과 물체의 접촉 위치가 데모의 접촉 패턴과 얼마나 유사한지 측정합니다.

두 가지 방식:
1. **Chamfer Distance 기반** (`use_retarget_contact=False`): 접촉 점군 간 양방향 최소 거리
2. **Matched Contact 기반** (`use_retarget_contact=True`): 리타게팅된 접촉점과 1:1 매칭 거리

```
con_rew = w_con × mean(exp(-β_contact · distance))
```

- β_contact = 10~30 (기본값)
- 물체 프레임과 손목 프레임 각각에서 계산 후 곱함 (`multiply_frame_contact=True`)

#### BC Reward (행동 복제)

관절 각도가 데모의 관절 각도에서 얼마나 벗어났는지의 역수:

```
bc_rew = w_bc × mean(exp(-β_bc · ‖q - q_demo‖²))
```

- β_bc = 500.0 (기본값)

#### 패널티

- **Action penalty**: `w_action × mean(action²)` — 불필요한 큰 액션 억제
- **Force penalty**: `w_force × mean(max(0, ‖F‖ - 500))` — 과도한 접촉력 억제

### 2.5 종료 조건 (_get_dones)

환경이 리셋되는 조건:

1. **타임아웃**: 에피소드 길이가 `max_episode_length`에 도달
2. **물체 낙하**: 물체의 z 좌표가 테이블 높이 아래로 떨어짐
3. **NaN 발생**: 시뮬레이션 수치 불안정
4. **Early Reset**: `--early_reset_threshold` > 0 일 때, 누적 보상이 임계값 미만이면 일찍 종료

### 2.6 PPO 정책 업데이트

`horizon` 스텝의 데이터를 모은 후 PPO 업데이트를 수행합니다.

**Generalized Advantage Estimation (GAE)**:
```
δ_t = r_t + γ·V(s_{t+1}) - V(s_t)
A_t = Σ_{l=0}^{T-t} (γλ)^l · δ_{t+l}
```
- γ = 0.99 (할인율)
- λ = 0.95 (GAE 파라미터)

**PPO 클리핑 목적함수**:
```
L_CLIP = min(r_t(θ)·A_t, clip(r_t(θ), 1-ε, 1+ε)·A_t)
```
- ε = 0.2 (클리핑 범위)
- r_t(θ) = π_θ(a|s) / π_θ_old(a|s) (확률비)

**미니배치 학습**:
- 전체 데이터: `num_envs × horizon` 개 (예: 2048 × 16 = 32,768 전이)
- 미니배치 크기: `num_envs × 8` (예: 16,384)
- 미니 에폭 수: `max(1, num_envs/4096 × 5)` (예: 3)

**학습률 적응형 스케줄**:
```
KL divergence > 0.008 → 학습률 감소
KL divergence < 0.008 → 학습률 증가
```

### 2.7 커리큘럼 학습

DexMachina의 핵심 아이디어: **물체 PD 제어기의 gain을 점진적으로 줄여가며 난이도를 높임**.

```
[높은 kp/kv: 물체가 데모를 따라감] → ... → [kp/kv = 0: 자유 조작]
```

#### 왜 gain=0이 최종 목표인가

DexMachina의 최종 목표는 **로봇 손이 물리적 접촉만으로 물체를 조작하는 정책**을 학습하는 것입니다. 현실 세계에서 물체에는 PD 컨트롤러가 없으므로, 학습된 정책이 실제로 배포 가능하려면 gain=0 조건에서 동작해야 합니다.

PD 컨트롤러는 학습을 위한 **보조 장치**이지 최종 목표가 아닙니다. 자전거의 보조 바퀴에 비유할 수 있습니다:

| 단계 | 자전거 | DexMachina |
|------|--------|-----------|
| 초기 | 보조 바퀴 장착 | 높은 PD gain — 물체가 스스로 데모 궤적을 따라감 |
| 중간 | 보조 바퀴 느슨하게 | PD gain 점진 감소 — 손의 접촉이 점점 더 중요해짐 |
| 최종 | 보조 바퀴 제거 | gain = 0 — 물체는 오직 손의 물리적 접촉으로만 움직임 |

**처음부터 gain=0으로 학습하면 안 되는 이유 (Sparse Reward 문제)**:

```
gain=0, 학습 초기:
  → 정책이 랜덤 액션 출력
  → 손이 물체를 못 잡음
  → 물체가 중력으로 떨어짐
  → 보상 ≈ 0 (데모 궤적과 완전히 다름)
  → 학습 가능한 기울기 신호 없음
  → 학습 불가
```

커리큘럼이 이 문제를 해결합니다. 높은 gain에서는 물체가 알아서 데모를 따라가므로, 정책은 "어디에 손을 두어야 하는지"부터 쉽게 학습할 수 있습니다. gain이 줄어들면서 점차 "실제로 잡고 조작하는 기술"을 학습하게 됩니다.

평가 시에도 항상 gain=0입니다 (`eval_rl_games.py`):
```python
if obj.actuated:
    print("Setting eval time obj gains to 0.0")
    obj.set_joint_gains(0.0, 0.0, force_range=0.0)
```

#### 커리큘럼 단계별 학습 내용

```
[Phase A] gain=80 (높음)
  물체가 PD 컨트롤러에 의해 데모 궤적을 스스로 따라감
  → 손은 대충 맞는 위치에 있기만 해도 보상을 받음
  → "어디에 손을 두어야 하는지" 학습

[Phase B] gain=40 (중간)
  물체가 어느 정도 따라가지만, 손의 접촉이 없으면 흔들림
  → "잡는 힘과 접촉 위치" 학습

[Phase C] gain=5 (낮음)
  물체가 거의 안 따라감, 손이 실제로 잡고 있어야 유지됨
  → "정밀한 조작 기술" 학습

[Phase D] gain=0 (최종)
  PD 컨트롤러 완전 제거 = 현실과 동일한 조건
  → 물체는 오직 중력 + 로봇 손의 접촉력으로만 움직임
  → 이것이 실제로 배포할 수 있는 정책
```

#### 동작 원리

1. **초기**: 물체에 강한 PD 컨트롤러 (`kp=80, kv=5`) 적용 → 물체가 데모 궤적을 거의 그대로 따라감
2. **학습 안정화**: 보상 deque가 임계값 이상이고, 에피소드 길이가 최대에 가까우면 gain 감소 트리거
3. **Gain 감소**: `uniform` 스케줄 — 상한/하한 범위를 점진적으로 축소하며 랜덤 샘플링
4. **최종**: 모든 gain이 0 → 물체가 오직 로봇 손의 물리적 접촉으로만 움직임

#### 스케줄 종류

| 스케줄 | 동작 |
|--------|------|
| `fixed` | 에폭 수에 따라 고정 비율로 감소 (`interval` 마다) |
| `exp` | 보상 기울기 확인 후 지수 감소 |
| `uniform` | 상한·하한 범위를 점진 축소, 범위 내 랜덤 샘플링 |

#### Uniform 스케줄 상세 (주로 사용)

```
감소 트리거 시:
  upper[k] *= upper_ratio[k]    # 예: kp × 0.9
  lower[k] = upper[k] × lower_ratio[k]  # 예: kp × 0.9 × 0.8 (slow 모드)

에피소드 리셋 시:
  gain = uniform(lower, upper)  # 환경마다 다른 gain
```

- `--upper_ratios 0.9 0.9 1`: kp × 0.9, kv × 0.9, force_range × 1.0
- `--lower_ratios 0.8 0.8 1`: 하한은 상한의 80%
- `--uniform_mode slow`: 하한 = 현재 상한 × lower_ratio (보수적)

#### 감소 조건 (`determine_decay`)

모든 조건이 동시에 만족되어야 gain을 줄입니다:

1. `epoch > wait_epochs` (기본 100~2000)
2. 각 보상 종류의 deque 평균 ≥ 임계값 (`task≥0.5`, `con≥0.05` 등)
3. (skip_grad=False일 때) 보상 기울기가 안정적
4. 마지막 감소 후 최소 40 에폭 경과
5. 에피소드 평균 길이 ≥ `max_episode_length - 2`

#### Dialback 메커니즘

gain을 줄인 후 성능이 급락하면 (에피소드 길이가 `dialback_ep_len` 미만으로 오래 유지되면), 이전 gain으로 되돌립니다:

```python
if achieved_len < dialback_ep_len and epochs_since_decay > dialback_min_epochs:
    curr_gains = prev_gains × dialback_ratio  # 98%로 약간 줄여서 복원
```

---

## Phase 3: 평가

### 3.1 평가 실행

```bash
python dexmachina/rl/eval_rl_games.py -B 1 --checkpoint $CK -v
```

### 3.2 평가 과정

1. 저장된 환경 설정(`env.pkl`)에서 동일한 환경 재구성
2. 물체 PD gain을 0으로 설정 (자유 조작)
3. `early_reset_threshold = 0.0` (중간 리셋 없음)
4. `is_eval = True` — 항상 마지막 프레임까지 실행
5. 랜덤화 비활성화
6. 결정론적 정책 사용 (`deterministic=True`)

### 3.3 평가 지표

매 스텝마다 기록:
- `pos_dist`: 물체 위치 오차 (m)
- `rot_dist`: 물체 회전 오차 (rad)
- `arti_dist`: 관절 각도 오차

### 3.4 비디오 녹화

`--record_video` 옵션으로 moviepy를 통해 MP4 비디오 저장.

---

## 부록: 학습 실행 예시

### Allegro Hand + Waffleiron

```bash
python dexmachina/rl/train_rl_games.py -B 2048 -obf -obt --max_epochs 5000 \
    --actuate_object --retarget_name para --horizon 16 -imw 0.5 \
    --gain_mode all --curr_schedule uniform --wait_epochs 100 \
    --learning_rate 0.0003 --contact_beta 10 \
    --upper_ratios 0.9 0.9 1 --lower_ratios 0.8 0.8 1 \
    --save_freq 5000 --fixed_mode uniform --uniform_mode slow \
    --action_penalty 0.01 --dialback_ep_len 80 --skip_grad \
    --deque_len 30 --task_rew_betas 10 1 5 \
    --aux_reset_thres 0 0 0 --curr_rew_thres 0.6 0.01 0.01 0.01 \
    -am hybrid --hybrid_scales 0.1 1.0 \
    --kp_init 80 --kv_init 5 \
    --clip waffleiron-30-230 -imi 0.3 -bc 0.3 -con 3 -ert 0.6 \
    -exp allegro_waffleiron --hand allegro_hand
```

**주요 인자 해석**:

| 인자 | 값 | 의미 |
|------|-----|------|
| `-B 2048` | 2048 | 병렬 환경 수 (waffleiron은 Jacobian 제한으로 4096 불가) |
| `-am hybrid` | hybrid | 손목=residual, 손가락=absolute |
| `--hybrid_scales 0.1 1.0` | | 손목 스케일 0.1, 손가락 스케일 1.0 |
| `--actuate_object` | | 물체에 PD 컨트롤러 부착 (커리큘럼용) |
| `--kp_init 80 --kv_init 5` | | 초기 물체 PD gain |
| `--curr_schedule uniform` | | 커리큘럼 스케줄: uniform 분포 |
| `--uniform_mode slow` | | 보수적 하한 설정 |
| `-imi 0.3 -bc 0.3 -con 3` | | 모방 0.3, BC 0.3, 접촉 3.0 가중치 |
| `-ert 0.6` | | 누적 보상 < 0.6 × step이면 early reset |
| `--skip_grad` | | 보상 기울기 체크 건너뜀 (gain 감소 조건 완화) |
| `--horizon 16` | | 16 스텝마다 PPO 업데이트 |
| `--max_epochs 5000` | | 최대 5000 에폭 학습 |

---

## 부록: 핵심 파일 구조

```
dexmachina/
├── rl/
│   ├── train_rl_games.py          # 학습 진입점
│   ├── eval_rl_games.py           # 평가 스크립트
│   ├── rl_games_wrapper.py        # BaseEnv ↔ RL-Games 브릿지
│   └── configs/
│       └── rl_games_ppo_cfg.yaml  # PPO 하이퍼파라미터
├── envs/
│   ├── base_env.py                # 핵심 환경 (step, reset, reward 통합)
│   ├── robot.py                   # 로봇 손 시뮬레이션
│   ├── object.py                  # 물체 시뮬레이션
│   ├── rewards.py                 # 보상 함수 모듈
│   ├── curriculum.py              # 커리큘럼 학습 (gain 감소)
│   ├── constructors.py            # 설정 조립 파이프라인
│   └── demo_data.py               # 데모/리타게팅 데이터 로딩
└── retargeting/                   # 리타게팅 파이프라인
```

---

## 부록: GPU 메모리 참고

| 설정 | 대략적 GPU 메모리 |
|------|-----------------|
| 2048 envs (waffleiron) | ~22 GB |
| 4096 envs (box) | ~20 GB |

RTX 5090 (32 GB) 기준으로 충분히 구동 가능합니다.
