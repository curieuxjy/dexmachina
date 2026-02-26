"""
학습 환경 미리보기 스크립트.
BaseEnv를 GUI로 띄워서 양손 로봇 + 물체 시뮬레이션을 시각적으로 확인.

사용 방법:
    # 기본 (kinematic 모드 = 데모 그대로 재생, inspire_hand + box)
    python examples/preview_training.py

    # allegro hand + box
    python examples/preview_training.py --hand allegro_hand

    # 다른 클립 지정
    python examples/preview_training.py --clip mixer-30-230

    # 환경 2개 동시
    python examples/preview_training.py -B 2

    # hybrid 모드로 랜덤 액션 적용
    python examples/preview_training.py -am hybrid --random_actions

    # actuated object (데모 궤적을 따르도록 물체에 제어기 부착)
    python examples/preview_training.py -act --kp_init 300 --kv_init 30

참고:
    get_all_env_cfg() 를 사용해 constructors.py의 표준 설정 파이프라인을 따르므로,
    실제 학습 시와 동일한 환경을 미리보기 할 수 있음.
"""
import sys
import torch
import genesis as gs
from dexmachina.envs.base_env import BaseEnv
from dexmachina.envs.constructors import get_all_env_cfg, get_common_argparser


def main():
    parser = get_common_argparser()
    parser.add_argument('--random_actions', action='store_true',
                        help="Apply small random actions instead of zero actions")
    args = parser.parse_args()

    # 미리보기를 위한 기본 설정 오버라이드
    args.vis = True
    args.show_fps = True
    args.use_rl_games = True

    # retarget_name: 기본값 'genesis'이지만, 현재 데이터는 'para' 형식
    if '--retarget_name' not in sys.argv and '-rn' not in sys.argv:
        args.retarget_name = 'para'

    # kinematic 모드가 기본 (zero action = 데모 그대로 재생)
    if args.action_mode == 'residual':
        # get_common_argparser 기본값이 residual이므로, 명시적으로 지정하지 않았으면 kinematic으로
        if '--action_mode' not in sys.argv and '-am' not in sys.argv:
            args.action_mode = 'kinematic'

    device = torch.device('cuda:0')
    gs.init(backend=gs.gpu)

    # 표준 설정 파이프라인으로 모든 config 생성
    env_kwargs = get_all_env_cfg(args, device, load_retarget_data=True)

    # 미리보기용 조정
    env_cfg = env_kwargs['env_cfg']
    env_cfg['early_reset_threshold'] = 0.0  # 조기 리셋 비활성화

    # 환경 생성
    env = BaseEnv(**env_kwargs)

    ep_len = env.max_episode_length
    print(f"\n=== Environment Preview ===")
    print(f"  Hand: {args.hand}")
    print(f"  Clip: {args.clip}")
    print(f"  Envs: {env.num_envs}")
    print(f"  Episode length: {ep_len}")
    print(f"  Action dim: {env.action_dim}")
    print(f"  Action mode: {args.action_mode}")
    print(f"  Obs dim: {env.obs_dim}")
    if args.actuate_object:
        print(f"  Object gains: kp={args.kp_init}, kv={args.kv_init}, fr={args.force_range_init}")
    print(f"  Random actions: {args.random_actions}")
    print(f"===========================\n")

    # 환경 리셋
    env.reset()
    step = 0
    episode = 0

    while True:
        if args.random_actions:
            action = torch.randn(env.num_envs, env.action_dim, device=device) * 0.1
        else:
            action = torch.zeros(env.num_envs, env.action_dim, device=device)

        obs, rew, terminated, timeouts, extras = env.step(action)
        step += 1

        if step <= 3 or step % 50 == 0:
            rew_val = rew.mean().item()
            nan_flag = env.nan_envs.any().item()
            info = f"  step {step:4d}/{ep_len} | rew: {rew_val:.4f} | nan: {nan_flag}"
            if env.n_objects > 0:
                obj = list(env.objects.values())[0]
                p = obj.root_pos[0]
                info += f" | obj: [{p[0]:.3f},{p[1]:.3f},{p[2]:.3f}]"
            print(info)

        if (terminated | timeouts).any():
            episode += 1
            step = 0
            print(f"\n  Episode {episode} starting...\n")


if __name__ == '__main__':
    main()
