import argparse
from orca_gym.scripts.rllib_appo_rl import env_creator, test_model, run_training, setup_cuda_environment
import yaml
import os

def main(config: dict,
    run_mode: str,
    checkpoint_path: str
):

    # 在脚本开头调用
    if setup_cuda_environment():
        print("CUDA 环境验证通过")
    else:
        print("CUDA 环境设置失败，GPU 加速可能不可用")

    if run_mode == 'training':
        run_training(
            orcagym_addr=config['orcagym_addr'],
            env_name=config['env_name'],
            agent_name=config['agent_name'],
            max_episode_steps=config['max_episode_steps'],
            num_env_runners=config['num_env_runners'],
            num_envs_per_env_runner=config['num_envs_per_env_runner'],
            async_env_runner=config['async_env_runner'],
            iter=config['iter'],
            render_mode=config['render_mode']
        )
    elif run_mode == 'testing':
        if not checkpoint_path:
            raise ValueError("Checkpoint path must be provided for testing.")
        test_model(
            checkpoint_path=checkpoint_path,
            orcagym_addr=config['orcagym_addr'],
            env_name=config['env_name'],
            agent_name=config['agent_name'],
            max_episode_steps=config['max_episode_steps'],
            use_onnx_for_inference=False,
            explore_during_inference=False,
            render_mode=config['render_mode']
        )
    else:
        raise ValueError("Invalid run mode. Use 'training' or 'testing'.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run Ant OrcaGym environment with APPO training.')
    parser.add_argument('--config_file', type=str, default='ant_local_config.yaml', help='The path of the config file')
    parser.add_argument('--run_mode', type=str, help='The mode to run (training / testing)')
    parser.add_argument('--checkpoint', type=str, help='The path to the checkpoint file for testing. no need for training')
    args = parser.parse_args()

    config_path = os.path.join(os.path.dirname(__file__), args.config_file)
    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    if args.run_mode == 'training':
        config = config['train_ant_local']
    elif args.run_mode == 'testing':
        config = config['test_ant_local']
    else:
        raise ValueError("Invalid run mode. Use 'training' or 'testing'.")

    main(
        config=config,
        run_mode=args.run_mode,
        checkpoint_path=args.checkpoint
    )
