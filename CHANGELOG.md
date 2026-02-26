# DexMachina Compatibility Updates

**Date**: 2026-02-26
**Environment**: Ubuntu, NVIDIA GeForce RTX 5090, CUDA 13.0, Genesis 0.3.3

기존 DexMachina 코드가 최신 하드웨어(RTX 5090) 및 Genesis 0.3.3 API 변경에 대응하지 못하는 문제를 수정함.

---

## 1. PyTorch CUDA 호환성 (RTX 5090 / Blackwell)

**문제**: RTX 5090(Blackwell, sm_120)은 기존 `torch==2.5.1+cu124`에서 지원하지 않음.
```
RuntimeError: CUDA error: no kernel image is available for execution on the device
```

**해결**: PyTorch를 CUDA 12.8+ 빌드로 업그레이드.
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

**수정 파일**: `README.md` — RTX 5090 사용자를 위한 설치 안내 추가.

---

## 2. wandb / NumPy 2.0 호환성

**문제**: `wandb==0.12.21`이 NumPy 2.0에서 제거된 `np.float_`를 참조.
```
AttributeError: `np.float_` was removed in the NumPy 2.0 release.
```

**해결**: wandb 업그레이드.
```bash
pip install --upgrade wandb
```

---

## 3. Genesis API: `dof_idx_local` → `dofs_idx_local`

**문제**: Genesis 0.3.3에서 `joint.dof_idx_local` 속성이 deprecated됨. 접근할 때마다 경고 출력.
```
[WARNING] This property is deprecated. Please use 'dofs_idx_local' instead.
```

**변경 사항**:
- `dof_idx_local` (deprecated): 1-DOF joint일 때 `int` 반환
- `dofs_idx_local` (신규): 항상 `list` 반환

**해결**: 모든 `joint.dof_idx_local` → `joint.dofs_idx_local[0]` 으로 교체.

**수정 파일** (6개):
| 파일 | 수정 개소 |
|------|----------|
| `dexmachina/envs/robot.py` | 8개소 |
| `dexmachina/envs/object.py` | 2개소 |
| `dexmachina/retargeting/map_contacts.py` | 1개소 |
| `dexmachina/hand_proc/inspect_raw_urdf.py` | 2개소 |
| `dexmachina/hand_proc/minimal_retarget.py` | 1개소 |
| `dexmachina/hand_proc/tune_gains.py` | 2개소 |

---

## 4. Genesis API: `show_FPS` → `profiling_options`

**문제**: `Scene(show_FPS=...)` 파라미터가 deprecated됨.
```
[WARNING] Scene.show_FPS is deprecated. Please use Scene.profiling_options.show_FPS
```

**해결**: `show_FPS=value` → `profiling_options=gs.options.ProfilingOptions(show_FPS=value)` 로 교체.

**수정 파일** (6개):
- `dexmachina/envs/base_env.py`
- `dexmachina/retargeting/parallel_retarget.py`
- `dexmachina/retargeting/map_contacts.py`
- `dexmachina/hand_proc/minimal_retarget.py`
- `dexmachina/hand_proc/tune_gains.py`
- `examples/inspect_hand.py`, `examples/load_object.py`

---

## 5. Genesis API: `set_dofs_kp/kv/force_range` 텐서 shape

**문제**: Genesis 0.3.3은 병렬 환경(`n_envs > 0`)에서 `set_dofs_kp()` 등에 2D 텐서 `[num_envs, num_dofs]`를 요구. 기존 코드는 1D `[num_dofs]`를 전달하여 에러 발생.
```
GenesisException: Invalid input shape: torch.Size([1, 7]).
First dimension does not match length (40) of `envs_idx`.
```

**해결**: `robot.py`의 `set_joint_gains()`에서 텐서를 `.unsqueeze(0).expand(num_envs, -1)` 로 확장.

**수정 파일**: `dexmachina/envs/robot.py`

참고: `dexmachina/envs/object.py`의 `fill_gain_tensor()`는 이미 올바르게 2D 텐서를 생성하고 있었음.

---

## 6. Genesis API: `self_collision_group_filter` 미지원

**문제**: `RigidOptions.self_collision_group_filter`와 `link_group_mapping`이 Genesis 0.3.3에 존재하지 않음. `--group_collisions` 플래그 사용 시 에러 발생.
```
ValueError: "RigidOptions" object has no field "self_collision_group_filter"
```

**배경**: Genesis 0.3.3은 충돌 필터링을 `contype`/`conaffinity` 비트마스크(MuJoCo 방식)로 처리. `self_collision_group_filter`는 DexMachina가 기대하는 미구현 기능.

**해결**: `hasattr()` 가드를 추가하여, 해당 필드가 존재할 때만 설정하고 없으면 경고 출력 후 건너뜀.

**수정 파일**:
- `dexmachina/envs/base_env.py`
- `dexmachina/retargeting/parallel_retarget.py`
- `examples/train_dex3.sh` — `--group_collisions` 플래그 제거

---

## 7. `max_collision_pairs` 부족

**문제**: 기본값 100이 이론적 최대 충돌 쌍(426)보다 작아 충돌 누락 가능.
```
[WARNING] max_collision_pairs 100 is smaller than the theoretical maximal possible pairs 426
```

**해결**: 100 → 500으로 증가.

**수정 파일**:
- `dexmachina/envs/base_env.py`
- `dexmachina/retargeting/parallel_retarget.py`

---

## 미수정 경고 (무시 가능)

| 경고 | 원인 | 비고 |
|------|------|------|
| `frictionloss, damping or armature` on free joint | URDF 파일 내 free joint 설정 | 비물리적이지만 동작에 영향 없음. URDF 수정 필요 |
| `Reference robot position exceeds joint limits` | 초기 자세가 관절 한계 밖 | 시뮬레이션 시작 후 자동 보정됨 |

---

## 수정 파일 전체 목록

```
README.md                                    (+8)
dexmachina/envs/base_env.py                  (+16, -5)
dexmachina/envs/object.py                    (+4, -2)
dexmachina/envs/robot.py                     (+26, -13)
dexmachina/hand_proc/inspect_raw_urdf.py     (+4, -2)
dexmachina/hand_proc/minimal_retarget.py     (+4, -2)
dexmachina/hand_proc/tune_gains.py           (+6, -3)
dexmachina/retargeting/map_contacts.py       (+4, -2)
dexmachina/retargeting/parallel_retarget.py  (+19, -8)
examples/inspect_hand.py                     (+2, -1)
examples/load_object.py                      (+2, -1)
examples/train_dex3.sh                       (+2, -1)
```
