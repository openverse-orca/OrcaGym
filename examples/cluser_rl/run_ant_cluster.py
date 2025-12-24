import argparse
from orca_gym.scripts.rllib_appo_rl import env_creator, test_model, run_training, setup_cuda_environment

from orca_gym.log.orca_log import get_orca_logger
_logger = get_orca_logger()





def main(
        orcagym_addr: str,
        env_name: str,
        agent_name: str,
        max_episode_steps: int,
        num_env_runners: int,
        num_envs_per_env_runner: int,
        run_mode: str,
        iter: int,
        checkpoint_path: str,
        render_mode: str
    ):

    # 在脚本开头调用
    if setup_cuda_environment():
        _logger.info("CUDA 环境验证通过")
    else:
        _logger.info("CUDA 环境设置失败，GPU 加速可能不可用")

    if run_mode == 'training':
        run_training(
            orcagym_addr=orcagym_addr,
            env_name=env_name,
            agent_name=agent_name,
            max_episode_steps=max_episode_steps,
            num_env_runners=num_env_runners,
            num_envs_per_env_runner=num_envs_per_env_runner,
            iter=iter,
            render_mode=render_mode
        )
    elif run_mode == 'testing':
        if not checkpoint_path:
            raise ValueError("Checkpoint path must be provided for testing.")
        test_model(
            checkpoint_path=checkpoint_path,
            orcagym_addr=orcagym_addr,
            env_name=env_name,
            agent_name=agent_name,
            max_episode_steps=max_episode_steps,
            use_onnx_for_inference=False,
            explore_during_inference=False   
        )
    else:
        raise ValueError("Invalid run mode. Use 'training' or 'testing'.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run Ant OrcaGym environment with APPO training.')
    parser.add_argument('--num_env_runners', type=int, default=24, help='The number of environment runners')
    parser.add_argument('--num_envs_per_env_runner', type=int, default=128, help='The number of environments per environment runner')
    parser.add_argument('--run_mode', type=str, default='training', choices=['training', 'testing'], help='The mode to run (training / testing)')
    parser.add_argument('--iter', type=int, default=50, help='The number of iterations to run')
    parser.add_argument('--checkpoint_path', type=str, help='The path to the checkpoint file for testing, if not provided, will run training')
    parser.add_argument('--render_mode', type=str, default='human', choices=['human', 'none'], help='The render mode (human / none)')
    args = parser.parse_args()

    main(
        orcagym_addr="localhost:50051",
        env_name="Ant_OrcaGymEnv",
        agent_name="ant_usda",
        max_episode_steps=1000,
        num_env_runners=args.num_env_runners,
        num_envs_per_env_runner=args.num_envs_per_env_runner,
        run_mode=args.run_mode,
        iter=args.iter,
        checkpoint_path=args.checkpoint_path if args.run_mode == 'testing' else None,
        render_mode=args.render_mode
    )