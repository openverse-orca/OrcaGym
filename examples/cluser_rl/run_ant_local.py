import ray
from ray import tune
from ray.tune import RunConfig, CheckpointConfig
from ray.rllib.algorithms.appo import APPOConfig
from ray.tune.registry import register_env
import argparse
from orca_gym.adapters.rllib.metrics_callback import OrcaMetricsCallback
from orca_gym.adapters.rllib.appo_catalog import DictAPPOTorchRLModule, DictAPPOCatalog
from orca_gym.scripts.rllib_appo_rl import env_creator, test_model
from ray.rllib.algorithms.appo.appo_learner import APPOLearner
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from ray.rllib.algorithms.ppo.torch.default_ppo_torch_rl_module import DefaultPPOTorchRLModule
from ray.rllib.algorithms.ppo.ppo_catalog import PPOCatalog
import gymnasium as gym
import torch

def config_appo_tuner(
        num_env_runners: int, 
        num_envs_per_env_runner: int,
        iter: int,
        env: gym.Env
    ) -> tune.Tuner:
    config = (
        APPOConfig()
        .environment(
            env="OrcaGymEnv",
            env_config={},
            disable_env_checking=False,  # 启用环境检查
        )
        .env_runners(
            num_env_runners=num_env_runners,          
            num_envs_per_env_runner=num_envs_per_env_runner,   
            num_cpus_per_env_runner=1,  
            rollout_fragment_length=64,
            # create_local_env_runner=True,  # For debugging
        )
        .rl_module(
            rl_module_spec=RLModuleSpec(
                observation_space=env.observation_space,
                action_space=env.action_space,
                # module_class=DefaultPPOTorchRLModule,
                model_config={
                    "fcnet_hiddens": [256, 256],
                    "fcnet_activation": "relu",
                    "post_fcnet_activation": None,
                    "vf_share_layers": True,
                },
                # catalog_class=PPOCatalog,
            )
        )
        .learners(
            num_learners=1,
            num_gpus_per_learner=1,
        )
        .training(
            train_batch_size_per_learner=4096,
            # use_kl_loss=True,
            # kl_coeff=0.3,
            # entropy_coeff=0.01,
        )
        .resources(
            num_cpus_for_main_process=1,
        )
        .api_stack(
            enable_rl_module_and_learner=True,
            enable_env_runner_and_connector_v2=True
        )
        .callbacks(
            OrcaMetricsCallback
        )
    )
    
    # print("APPO Configuration:", config.to_dict())
    print(f"总环境数: {config.num_env_runners * config.num_envs_per_env_runner}")
    print(f"总CPU需求: {config.num_cpus_for_main_process + config.num_env_runners * config.num_cpus_per_env_runner}")
    
    return tune.Tuner(
        "APPO",
        param_space=config.to_dict(),
        run_config=RunConfig(
            name="APPO_OrcaGym_Training",
            stop={"training_iteration": iter},
            checkpoint_config=CheckpointConfig(
                num_to_keep=1,
                checkpoint_score_attribute="episode_reward_mean",
                checkpoint_at_end=True
            ),
            verbose=3,
        ),
    )

def run_training(
        orcagym_addr: str,
        env_name: str,
        agent_name: str,
        max_episode_steps: int,
        num_env_runners: int,
        num_envs_per_env_runner: int,
        iter: int,
        render_mode: str
    ):
    register_env(
        "OrcaGymEnv", 
        lambda env_context: env_creator(
            env_context=env_context,
            orcagym_addr=orcagym_addr,
            env_name=env_name,
            agent_name=agent_name,
            max_episode_steps=max_episode_steps,
            render_mode=render_mode,
        )
    )

    # 优化Ray初始化参数
    ray.init(
        num_cpus=num_env_runners + 1, 
        num_gpus=1,
        object_store_memory=num_env_runners * 1024**3, 
        include_dashboard=False, 
        ignore_reinit_error=True
    )

    # 创建一个样本环境实例
    env = env_creator(
        env_context=None,
        orcagym_addr=orcagym_addr,
        env_name=env_name,
        agent_name=agent_name,
        max_episode_steps=max_episode_steps,
        render_mode=render_mode
    )

    print("\nStarting training...")
    tuner = config_appo_tuner(
        num_env_runners=num_env_runners,
        num_envs_per_env_runner=num_envs_per_env_runner,
        iter=iter,
        env=env
    )
    results = tuner.fit()
    

    exit(-1)

    # 训练运行后清理
    if torch.distributed.is_initialized():
        print("Cleaning up distributed process group...")
        torch.distributed.destroy_process_group()
        print("Process group destroyed")


    if results:
        print("\nTraining completed. Best results:")
        best_result = results.get_best_result()
        if best_result.checkpoint:
            print(f"Best checkpoint: {best_result.checkpoint}")

    ray.shutdown()
    print("Process completed.")

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

    import torch
    import os

    # 确保PyTorch看到GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    # 显式设置设备
    if torch.cuda.is_available():
        torch.cuda.init()
        device = torch.device("cuda")
        print(f"CUDA initialized. Device: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("Using CPU")


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
            max_episode_steps=max_episode_steps
        )
    else:
        raise ValueError("Invalid run mode. Use 'training' or 'testing'.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run Ant OrcaGym environment with APPO training.')
    parser.add_argument('--num_env_runners', type=int, default=16, help='The number of environment runners')
    parser.add_argument('--num_envs_per_env_runner', type=int, default=64, help='The number of environments per environment runner')
    parser.add_argument('--run_mode', type=str, default='training', choices=['training', 'testing'], help='The mode to run (training / testing)')
    parser.add_argument('--iter', type=int, default=100, help='The number of iterations to run')
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