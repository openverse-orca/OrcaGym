import ray
from ray import tune
from ray.tune import RunConfig, CheckpointConfig
from ray.train import ScalingConfig
from ray.train.torch import TorchTrainer
from ray.rllib.algorithms.appo import APPOConfig
from ray.tune.registry import register_env
import argparse

import torch.version
from orca_gym.adapters.rllib.metrics_callback import OrcaMetricsCallback
from orca_gym.adapters.rllib.appo_catalog import DictAPPOTorchRLModule, DictAPPOCatalog
from orca_gym.scripts.rllib_appo_rl import env_creator, test_model, create_demo_env_instance
from ray.rllib.algorithms.appo.appo_learner import APPOLearner
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from ray.rllib.algorithms.ppo.torch.default_ppo_torch_rl_module import DefaultPPOTorchRLModule
from ray.rllib.algorithms.ppo.ppo_catalog import PPOCatalog
import gymnasium as gym
import torch
from ray.rllib.algorithms.appo.torch.default_appo_torch_rl_module import DefaultAPPOTorchRLModule
import os
import sys
import subprocess
from ray.runtime_env import RuntimeEnv
import glob

def setup_cuda_environment():
    """设置并验证 CUDA 环境"""
    # 获取 Conda 环境路径
    conda_prefix = os.environ.get("CONDA_PREFIX", "")
    if not conda_prefix:
        print("警告: 未检测到 Conda 环境")
        return False
    
    # 设置环境变量
    os.environ["CUDA_HOME"] = conda_prefix
    os.environ["LD_LIBRARY_PATH"] = f"{conda_prefix}/lib:{os.environ.get('LD_LIBRARY_PATH', '')}"
    
    # 验证 CUDA 和 cuDNN
    cuda_available = torch.cuda.is_available()
    cudnn_version = torch.backends.cudnn.version() if cuda_available else 0
    
    print("="*50)
    print("CUDA 环境验证")
    print(f"CUDA_HOME: {conda_prefix}")
    print(f"CUDA 可用: {cuda_available}")
    if cuda_available:
        print(f"GPU 设备: {torch.cuda.get_device_name(0)}")
        print(f"CUDA 版本: {torch.version.cuda}")
        print(f"cuDNN 版本: {cudnn_version}")
    print("="*50)
    
    return cuda_available


def verify_pytorch_cuda():
    """验证 PyTorch 是否能正确使用 CUDA"""
    print("="*50)
    print("PyTorch CUDA 验证")
    print(f"PyTorch 版本: {torch.__version__}")
    print(f"CUDA 可用: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"GPU 设备: {torch.cuda.get_device_name(0)}")
        print(f"CUDA 版本: {torch.version.cuda}")
        print(f"cuDNN 版本: {torch.backends.cudnn.version()}")
        
        # 运行简单的 CUDA 计算
        try:
            a = torch.tensor([1.0], device="cuda")
            b = torch.tensor([2.0], device="cuda")
            c = a + b
            print(f"CUDA 计算测试成功: 1.0 + 2.0 = {c.item()}")
        except Exception as e:
            print(f"CUDA 计算测试失败: {str(e)}")
    else:
        print("PyTorch 无法访问 CUDA")
    
    print("="*50)


def worker_env_check():
    import os
    import torch
    import traceback
    
    # 确保使用正确的库路径
    conda_prefix = os.environ.get("CONDA_PREFIX", "")
    if conda_prefix:
        os.environ["LD_LIBRARY_PATH"] = f"{conda_prefix}/lib:{os.environ.get('LD_LIBRARY_PATH', '')}"
    
    # 尝试初始化 CUDA
    try:
        torch.cuda.init()
        available = torch.cuda.is_available()
        device_name = torch.cuda.get_device_name(0) if available else "N/A"
        cudnn_version = torch.backends.cudnn.version() if available else "N/A"
        
        # 运行简单的 CUDA 计算
        if available:
            a = torch.tensor([1.0], device="cuda")
            b = torch.tensor([2.0], device="cuda")
            c = a + b
            calc_test = f"计算成功: {c.item()}"
        else:
            calc_test = "未测试"
    except Exception as e:
        available = False
        device_name = f"初始化失败: {str(e)}"
        cudnn_version = "N/A"
        calc_test = "未测试"
    
    return {
        "pid": os.getpid(),
        "cuda": available,
        "device_info": device_name,
        "cudnn_version": cudnn_version,
        "calc_test": calc_test,
        "pytorch_version": torch.__version__,
        "cuda_version": torch.version.cuda if hasattr(torch.version, "cuda") else "N/A",
        "ld_path": os.environ.get("LD_LIBRARY_PATH", ""),
        "cuda_home": os.environ.get("CUDA_HOME", ""),
    }


def config_appo_tuner(
        num_env_runners: int, 
        num_envs_per_env_runner: int,
        iter: int,
        env: gym.Env
    ) -> tune.Tuner:
    # 重要：获取系统的实际GPU数量
    num_gpus_available = torch.cuda.device_count()

    os.environ['CUDA_VISIBLE_DEVICES'] = "0,1"
    print(f"CUDA_VISIBLE_DEVICES: {os.environ['CUDA_VISIBLE_DEVICES']}")

    config = (
        APPOConfig()
        .environment(
            env="OrcaGymEnv",
            env_config={"worker_index": 1},
            disable_env_checking=False,
        )
        .env_runners(
            num_env_runners=num_env_runners,          
            num_envs_per_env_runner=num_envs_per_env_runner,
            num_cpus_per_env_runner=1,
            num_gpus_per_env_runner=(num_gpus_available - 0.1) * 1 / num_env_runners,  # 每个环境runner分配的GPU数量
            rollout_fragment_length=64,
        )
        .rl_module(
            rl_module_spec=RLModuleSpec(
                observation_space=env.observation_space,
                action_space=env.action_space,
                module_class=DefaultAPPOTorchRLModule,
                model_config={
                    "fcnet_hiddens": [256, 256],
                    "fcnet_activation": "relu",
                    "post_fcnet_activation": None,
                    "vf_share_layers": True,
                    "use_gpu": num_gpus_available > 0,
                },
            )
        )
        .learners(
            num_learners=0,
            num_gpus_per_learner=num_gpus_available * 0.1,
        )
        .training(
            train_batch_size_per_learner=4096,
        )
        .resources(
            num_cpus_for_main_process=1,
            num_gpus=num_gpus_available,
        )
        .api_stack(
            enable_rl_module_and_learner=True,
            enable_env_runner_and_connector_v2=True
        )
        .callbacks(
            OrcaMetricsCallback
        )
    )
    
    print(f"总环境数: {config.num_env_runners * config.num_envs_per_env_runner}")
    print(f"总CPU需求: {config.num_cpus_for_main_process + config.num_env_runners * config.num_cpus_per_env_runner}")
    # 重要：添加GPU资源报告
    print(f"总GPU需求: {num_gpus_available} (实际检测到 {num_gpus_available} 个GPU)")
    print(f"Learner GPU配置: {config.num_gpus_per_learner}")
    print(f"模型是否使用GPU: {config.rl_module_spec.model_config['use_gpu']}")
    

    # scaling_config = ScalingConfig(num_workers=2, use_gpu=True)
    # trainer = TorchTrainer(train_func, scaling_config=scaling_config)
    # result = trainer.fit()

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
        # scaling_config=ScalingConfig(
        #     num_workers=num_env_runners,
        #     use_gpu=True
        # )
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


    # 在环境设置后调用
    verify_pytorch_cuda()

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
        ignore_reinit_error=True,
        runtime_env={
            "env_vars": {
                "CUDA_VISIBLE_DEVICES": "0,1",
                "NVIDIA_VISIBLE_DEVICES": "all"
            }
        }
    )

    @ray.remote(num_gpus=0.1)
    def worker_env_check_remote():
        return worker_env_check()


    # 调用并打印结果
    results = ray.get([worker_env_check_remote.remote() for _ in range(3)])
    print("\n" + "="*50)
    print("工作进程 CUDA 验证结果")
    print("="*50)

    for i, res in enumerate(results):
        print(f"\n工作进程 #{i+1} (PID={res['pid']}):")
        print(f"  CUDA 可用: {res['cuda']}")
        print(f"  设备信息: {res['device_info']}")
        print(f"  cuDNN 版本: {res['cudnn_version']}")
        print(f"  计算测试: {res['calc_test']}")
        print(f"  CUDA_HOME: {res['cuda_home']}")
        print(f"  LD_LIBRARY_PATH: {res['ld_path']}")
        print(f"  PyTorch 版本: {res['pytorch_version']}")
        print(f"  CUDA 版本: {res['cuda_version']}")

    # 创建一个样本环境实例
    demo_env = create_demo_env_instance(
        orcagym_addr="localhost:50051",
        env_name="Ant_OrcaGymEnv",
        agent_name="ant_usda",
        max_episode_steps=1000,
    )

    print("\nStarting training...")

    # scaling_config = ScalingConfig(num_workers=2, use_gpu=True)
    # trainer = TorchTrainer(train_func, scaling_config=scaling_config)
    # result = trainer.fit()

    tuner = config_appo_tuner(
        num_env_runners=num_env_runners,
        num_envs_per_env_runner=num_envs_per_env_runner,
        iter=iter,
        env=demo_env
    )
    results = tuner.fit()
    

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

    # 在脚本开头调用
    if setup_cuda_environment():
        print("CUDA 环境验证通过")
    else:
        print("CUDA 环境设置失败，GPU 加速可能不可用")

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