import ray
from ray import tune
from ray.tune import RunConfig, CheckpointConfig
from ray.rllib.algorithms.appo import APPOConfig
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env
import ray
from ray.util.placement_group import placement_group, remove_placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

import torch.version
from orca_gym.adapters.rllib.metrics_callback import OrcaMetricsCallback
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
import gymnasium as gym
import torch
from ray.rllib.algorithms.appo.torch.default_appo_torch_rl_module import DefaultAPPOTorchRLModule
import os
from datetime import datetime
from ray.rllib.core.rl_module.rl_module import RLModule
from ray.rllib.core import DEFAULT_MODULE_ID
from ray.rllib.core.columns import Columns
import numpy as np
from ray.rllib.utils.numpy import convert_to_numpy
import tree
import time
import gymnasium as gym
import json
from ray.rllib.env.single_agent_env_runner import (
    SingleAgentEnvRunner,
)
from orca_gym.environment.async_env.single_agent_env_runner import (
    OrcaGymAsyncSingleAgentEnvRunner
)
from envs.legged_gym.legged_config import LeggedRobotConfig, LeggedObsConfig, CurriculumConfig, LeggedEnvConfig
from ray.rllib.core.rl_module.default_model_config import DefaultModelConfig

ENV_ENTRY_POINT = {
    "Ant_OrcaGymEnv": "envs.mujoco.ant_orcagym:AntOrcaGymEnv",
    "LeggedGym": "envs.legged_gym.legged_gym_env:LeggedGymEnv",
}

ENV_RUNNER_CLS = {
    "Ant_OrcaGymEnv": SingleAgentEnvRunner,
    "LeggedGym": OrcaGymAsyncSingleAgentEnvRunner,
}

from ray.rllib.utils.metrics import (
    ENV_RUNNER_RESULTS,
    EPISODE_RETURN_MEAN,
    NUM_ENV_STEPS_SAMPLED_LIFETIME,
)


def get_orca_gym_register_info(
        orcagym_addr : str,
        env_name : str, 
        agent_name : str, 
        agent_num: int,
        render_mode: str,
        worker_idx: int,
        vector_idx: int,
        async_env_runner: bool,
        height_map_file: str,
        max_episode_steps: int,
        task: str,
        frame_skip: int,
        action_skip: int,
        time_step: float
    ) -> tuple[ str, dict ]:

    orcagym_addr_str = orcagym_addr.replace(":", "-")
    env_id = env_name + "-OrcaGym-" + orcagym_addr_str + f"-{worker_idx:03d}"
    agent_names = [f"{agent_name}_{i:03d}" for i in range(agent_num)]
    kwargs = {
        'Ant_OrcaGymEnv': {
            'frame_skip': frame_skip,
            'orcagym_addr': orcagym_addr,
            'agent_names': agent_names,
            'time_step': time_step,
            'render_mode': render_mode,
            'max_episode_steps': max_episode_steps,
        },
        'LeggedGym': {
            'frame_skip': frame_skip,
            'action_skip': action_skip,
            'orcagym_addr': orcagym_addr,
            'agent_names': agent_names,
            'time_step': time_step,
            'render_mode': render_mode,
            'height_map_file': height_map_file,
            'env_id': env_id,
            'max_episode_steps': max_episode_steps,
            'is_subenv': True,
            'run_mode': 'training',
            'task': task,
            'robot_config': LeggedRobotConfig[agent_name],
            'legged_obs_config': LeggedObsConfig,
            'curriculum_config': CurriculumConfig,
            'legged_env_config': LeggedEnvConfig,
        },
    }

    return env_id, kwargs

def create_demo_env_instance(
    orcagym_addr: str,
    env_name: str,
    agent_name: str,
    agent_num: int,
    max_episode_steps: int,
    async_env_runner: bool,
    height_map_file: str,
    render_mode: str,
    task: str,
    frame_skip: int,
    action_skip: int,
    time_step: float
):
    """
    创建一个演示环境实例，主要用于测试和验证。
    """
    env_id, kwargs = get_orca_gym_register_info(
        orcagym_addr=orcagym_addr,
        env_name=env_name,
        agent_name=agent_name,
        agent_num=agent_num,
        render_mode=render_mode,
        worker_idx=999,
        vector_idx=1,
        async_env_runner=async_env_runner,
        height_map_file=height_map_file,
        max_episode_steps=max_episode_steps,
        task=task,
        frame_skip=frame_skip,
        action_skip=action_skip,
        time_step=time_step
    )

    if env_id not in gym.envs.registry:
        print(f"Registering environment: {env_id}")
        gym.register(
            id=env_id,
            entry_point=ENV_ENTRY_POINT[env_name],
            kwargs=kwargs[env_name],
            max_episode_steps=max_episode_steps,
            reward_threshold=0.0,
            vector_entry_point=ENV_ENTRY_POINT[env_name],
        )
    
    print(f"Creating environment {env_id} with kwargs={kwargs}")
    env = gym.make(env_id, **kwargs[env_name])
    
    return env, kwargs[env_name]

def get_config(
    agent_config: dict,
    task: str,
    num_learners: int,
    num_env_runners: int, 
    num_envs_per_env_runner: int,
    num_cpus_per_learner: int,
    num_gpus_per_learner: int,
    num_cpus_per_env_runner: int,
    num_gpus_per_env_runner: int,
    env: gym.Env,
    num_gpus_available: float,
    async_env_runner: bool,
    env_name: str,
    env_kwargs: dict,
    total_steps: int,
):
    # env_name 是用 - 分隔的字符串，去掉最后一段，保留前面的
    env_name_prefix = "-".join(env.spec.id.split("-")[:-1])
    print("env_name_prefix: ", env_name_prefix)
    print("action_space: ", env.action_space, "observation_space: ", env.observation_space)

    # 将SB3的学习率缩放一个因子
    lr_scale_factor = 1
    rl_initial_value = agent_config["lr_schedule"]["initial_value"] / lr_scale_factor
    rl_final_value = agent_config["lr_schedule"]["final_value"] / lr_scale_factor
    rl_end_fraction = agent_config["lr_schedule"]["end_fraction"]
    rl_lr_schedule = [
        [0, rl_initial_value],
        [total_steps * rl_end_fraction, rl_final_value],
    ]

    ent_coef_initial_value = agent_config["ent_coef_schedule"]["initial_value"]
    ent_coef_final_value = agent_config["ent_coef_schedule"]["final_value"]
    ent_coef_end_fraction = agent_config["ent_coef_schedule"]["end_fraction"]
    ent_coef_schedule = [
        [0, ent_coef_initial_value],
        [total_steps * ent_coef_end_fraction, ent_coef_final_value],
    ]

    # evaluation_num_env_runners = num_env_runners // 8
    # num_env_runners = num_env_runners - evaluation_num_env_runners

    rollout_fragment_length = agent_config.get("rollout_fragment_length", 64)
    minibatch_size = agent_config.get("minibatch_size", 1024)
    train_batch_size_per_learner = agent_config.get("train_batch_size", 4096)
    print(f"num_learners: {num_learners}, rollout_fragment_length: {rollout_fragment_length}, \
            minibatch_size: {minibatch_size}, train_batch_size_per_learner: {train_batch_size_per_learner}")

    config = (
        APPOConfig()
        .environment(
            env=env_name,
            env_config={
                "worker_index": 1, 
                "vector_index": 1, 
                "num_env_runners": num_env_runners, 
                "num_envs_per_env_runner": num_envs_per_env_runner, 
                "env_kwargs": env_kwargs, 
                "entry_point": ENV_ENTRY_POINT[env_name]
            },
            disable_env_checking=False,
            # render_env=True,
            # action_space=env.action_space,
            # observation_space=env.observation_space,
            normalize_actions=False,
            clip_actions=True,
        )
        .env_runners(
            env_runner_cls=ENV_RUNNER_CLS[env_name],
            num_env_runners=num_env_runners,          
            num_envs_per_env_runner=num_envs_per_env_runner,
            num_cpus_per_env_runner=num_cpus_per_env_runner,  
            num_gpus_per_env_runner=num_gpus_per_env_runner,
            rollout_fragment_length=rollout_fragment_length,
            gym_env_vectorize_mode=gym.envs.registration.VectorizeMode.ASYNC if async_env_runner else gym.envs.registration.VectorizeMode.SYNC,  # default is `SYNC`
            # observation_filter="MeanStdFilter",
            compress_observations=True,
        )
        .rl_module(
            model_config=DefaultModelConfig(
                # ====================================================
                # MLP 编码器 
                # ====================================================
                fcnet_hiddens=agent_config["fcnet_hiddens"],       
                fcnet_activation="silu",
                fcnet_kernel_initializer="orthogonal_",
                fcnet_kernel_initializer_kwargs={"gain": 1.0},
                fcnet_bias_initializer= "zeros_",

                # 输出层使用线性激活函数，确保输出范围正确
                head_fcnet_activation="linear",

                # 尝试共享编码器提高训练效率
                # 不共享编码器，是 SB3 PPO的默认方式
                vf_share_layers=agent_config.get("vf_share_layers", True),

                # 使用更稳定的训练策略
                free_log_std=agent_config.get("free_log_std", True),

                # ====================================================
                # LSTM 层配置
                # ====================================================
                use_lstm=agent_config.get("use_lstm", False), 
                lstm_cell_size=agent_config.get("lstm_cell_size", 256),
                max_seq_len=agent_config.get("max_seq_len", 10),

                # 卷积网络无关，保持None
                conv_filters=agent_config.get("conv_filters", None),

                # ====================================================
                # 使用GPU加速
                # ====================================================
                # use_gpu=True if num_gpus_available > 0 else False,
            ),
            # rl_module_spec=RLModuleSpec(
            #     observation_space=env.observation_space["observation"],
            #     action_space=env.action_space,
            #     module_class=DefaultAPPOTorchRLModule,
            #     model_config=model_config,
            # )
        )
        .learners(
            num_learners=num_learners,
            num_cpus_per_learner=num_cpus_per_learner, 
            num_gpus_per_learner=num_gpus_per_learner, 
            # 禁用 DDP 以避免 NCCL 冲突
            # _enable_learner_api=True,
        )
        .training(
            train_batch_size_per_learner=train_batch_size_per_learner,
            minibatch_size=minibatch_size,
            lr=rl_lr_schedule,
            grad_clip=agent_config.get("grad_clip", 40.0),
            grad_clip_by=agent_config.get("grad_clip_by", "global_norm"),
            gamma=agent_config["gamma"],
            # lambda_=agent_config.get("gae_lambda", 0.95),
            clip_param=agent_config.get("clip_param", 0.4),
            vf_loss_coeff=agent_config.get("vf_loss_coeff", 0.5), 
            entropy_coeff=ent_coef_schedule, 
            use_kl_loss=agent_config.get("use_kl_loss", False),
            circular_buffer_num_batches=agent_config.get("circular_buffer_num_batches", 16),
            circular_buffer_iterations_per_batch=agent_config.get("circular_buffer_iterations_per_batch", 20)
        )
        # .evaluation(
        #     evaluation_interval=10,
        #     evaluation_parallel_to_training=True,
        #     evaluation_num_env_runners=evaluation_num_env_runners,
        # )
        .resources(
            num_cpus_for_main_process=1,
            # num_gpus=num_gpus_available,
            # placement_strategy="SPREAD",
        )
        .api_stack(
            enable_rl_module_and_learner=True,
            enable_env_runner_and_connector_v2=True
        )
        .callbacks(
            OrcaMetricsCallback
        )
        .reporting(
            # min_sample_timesteps_per_iteration=timesteps_per_iteration,
            # 0 metrics reporting delay, this makes sure timestep,
            # which entropy coeff depends on, is updated after each worker rollout.
            min_time_s_per_iteration=5,
        )
    )
    return config

def config_appo_tuner(
        agent_config: dict,
        task: str,
        num_learners: int,
        num_env_runners: int, 
        num_envs_per_env_runner: int,
        num_cpus_per_learner: int,
        num_gpus_per_learner: int,
        num_cpus_per_env_runner: int,
        num_gpus_per_env_runner: int,
        iter: int,
        total_steps: int,
        env: gym.Env,
        async_env_runner: bool,
        env_name: str,
        env_kwargs: dict,
        num_gpus_available: float,
        model_dir: str = None,
    ) -> tune.Tuner:
        
    # 设置 NCCL 环境变量来避免 GPU 冲突
    os.environ["NCCL_DEBUG"] = "WARN"
    os.environ["NCCL_IB_DISABLE"] = "1"
    os.environ["NCCL_P2P_DISABLE"] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 只使用第一个GPU
    
    config = get_config(
        agent_config=agent_config,
        task=task,
        num_learners=num_learners,
        num_env_runners=num_env_runners,
        num_envs_per_env_runner=num_envs_per_env_runner,
        num_cpus_per_learner=num_cpus_per_learner,
        num_gpus_per_learner=num_gpus_per_learner,
        num_cpus_per_env_runner=num_cpus_per_env_runner,
        num_gpus_per_env_runner=num_gpus_per_env_runner,
        env=env,
        num_gpus_available=num_gpus_available,
        async_env_runner=async_env_runner,
        env_name=env_name,
        env_kwargs=env_kwargs,
        total_steps=total_steps,
    )
    
    print(f"总环境数: {config.num_env_runners * config.num_envs_per_env_runner}")
    print(f"总CPU需求: {config.num_cpus_for_main_process + config.num_env_runners * config.num_cpus_per_env_runner}")
    # 重要：添加GPU资源报告
    print(f"总GPU需求: {num_gpus_available} (实际检测到 {num_gpus_available} 个GPU)")
    print(f"Learner GPU配置: {config.num_gpus_per_learner}")
    # print(f"模型是否使用GPU: {config.rl_module_spec.model_config['use_gpu']}")

    # 设置存储路径 - 使用NFS共享目录，确保head和worker节点都可以访问
    # 支持通过软链接方式共享 ./trained_models_tmp 目录
    nfs_base_path = os.environ.get('ORCA_NFS_BASE_PATH', '/mnt/nfs')
    
    if model_dir:
        # 如果提供了model_dir，检查是否为NFS路径
        if model_dir.startswith('/mnt/nfs') or model_dir.startswith('/shared'):
            # 已经是NFS路径，直接使用
            storage_path = os.path.abspath(model_dir)
        elif model_dir.startswith('./trained_models_tmp') or 'trained_models_tmp' in model_dir:
            # 如果是trained_models_tmp相关路径，转换为NFS路径
            # 获取相对于trained_models_tmp的路径部分
            if model_dir.startswith('./trained_models_tmp/'):
                relative_path = model_dir[len('./trained_models_tmp/'):]
            elif 'trained_models_tmp/' in model_dir:
                relative_path = model_dir.split('trained_models_tmp/')[-1]
            else:
                relative_path = os.path.basename(model_dir)
            
            # 构建NFS路径
            storage_path = os.path.join(nfs_base_path, 'trained_models_tmp', relative_path)
        else:
            # 其他情况，将model_dir名称放到NFS的trained_models_tmp下
            model_name = os.path.basename(model_dir)
            storage_path = os.path.join(nfs_base_path, 'trained_models_tmp', model_name)
    else:
        # 如果没有提供model_dir，使用NFS共享目录作为默认路径
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        default_model_name = f"appo_training_{timestamp}"
        storage_path = os.path.join(nfs_base_path, 'trained_models_tmp', default_model_name)
    
    # 确保NFS目录存在
    try:
        os.makedirs(storage_path, exist_ok=True)
        print(f"使用NFS共享存储路径: {storage_path}")
        # 验证软链接是否正常工作
        if os.path.islink(os.path.join(nfs_base_path, 'trained_models_tmp')):
            link_target = os.readlink(os.path.join(nfs_base_path, 'trained_models_tmp'))
            print(f"检测到软链接: {nfs_base_path}/trained_models_tmp -> {link_target}")
    except PermissionError:
        print(f"警告: 无法创建NFS目录 {storage_path}，请检查权限和NFS挂载状态")
        # 回退到本地目录
        storage_path = os.path.abspath(model_dir if model_dir else os.getcwd())
        os.makedirs(storage_path, exist_ok=True)
        print(f"回退到本地存储路径: {storage_path}")
    except Exception as e:
        print(f"警告: NFS路径设置失败: {e}")
        # 回退到本地目录
        storage_path = os.path.abspath(model_dir if model_dir else os.getcwd())
        os.makedirs(storage_path, exist_ok=True)
        print(f"回退到本地存储路径: {storage_path}")
    
    # 创建 Tuner
    tuner = tune.Tuner(
        "APPO",
        param_space=config.to_dict(),
        run_config=RunConfig(
            name="APPO_OrcaGym_Training",
            stop={"training_iteration": iter},
            storage_path=storage_path,
            checkpoint_config=CheckpointConfig(
                num_to_keep=3,
                checkpoint_frequency=100,
                checkpoint_score_attribute="env_runners/episode_return_mean",
                checkpoint_score_order="max",
                checkpoint_at_end=True
            ),
            verbose=2,
        ),
    )
    
    return tuner




def env_creator(
        env_context,
        orcagym_addr: str,
        env_name: str,
        agent_name: str,
        agent_num: int,
        max_episode_steps: int,
        render_mode: str,
        async_env_runner: bool,
        height_map_file: str,
        task: str,
        frame_skip: int,
        action_skip: int,
        time_step: float
    ):

    if env_context is None:
        worker_idx = 1
        vector_idx = 0
        print("Creating environment in main thread, no env_context provided.")
    else:
        worker_idx = env_context.worker_index
        vector_idx = env_context.vector_index
        print(f"Creating environment in worker {worker_idx}, vector {vector_idx}.")

    env_id, kwargs = get_orca_gym_register_info(
        orcagym_addr=orcagym_addr,
        env_name=env_name,
        agent_name=agent_name,
        agent_num=agent_num,
        render_mode=render_mode,
        worker_idx=worker_idx,
        vector_idx=vector_idx,
        async_env_runner=async_env_runner,
        height_map_file=height_map_file,
        max_episode_steps=max_episode_steps,
        task=task,
        frame_skip=frame_skip,
        action_skip=action_skip,
        time_step=time_step
    )

    # if vector_idx == 0:
    # 避免重复注册冲突
    if env_id not in gym.envs.registry:
        print(f"Registering environment: {env_id}")
        gym.register(
            id=env_id,
            entry_point=ENV_ENTRY_POINT[env_name],
            kwargs=kwargs[env_name],
            max_episode_steps=max_episode_steps,
            reward_threshold=0.0,
        )
    else:
        print(f"Environment {env_id} already registered, skipping registration.")

    # 创建环境并验证
    try:
        print(f"Worker {worker_idx}, vector {vector_idx}: Creating environment {env_id}, kwargs={kwargs}")
        env = gym.make(env_id, **kwargs[env_name])
        
        # 验证观测空间
        print(f"Observation space for {env_id}: {env.observation_space}")
        print(f"Action space for {env_id}: {env.action_space}")
        
        # 测试采样观测值
        sample_obs = env.observation_space.sample()
        if not env.observation_space.contains(sample_obs):
            print(f"WARNING: Sampled observation is not within observation space!")
        
        print(f"Environment {env_id} created successfully in worker {worker_idx}, vector {vector_idx}."
              f" Render mode: {render_mode}"
              f" ProcessID: {os.getpid()}")
        return env
        
    except Exception as e:
        print(f"ERROR: Failed to create environment {env_id} in worker {worker_idx}")
        print(f"Exception: {str(e)}")
        import traceback
        traceback.print_exc()
        raise


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
        print(f"GPU 设备数量: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"CUDA 版本: {torch.version.cuda}")
        print(f"cuDNN 版本: {cudnn_version}")
        
        # 检查nvidia-smi
        try:
            import subprocess
            result = subprocess.run(['nvidia-smi', '--list-gpus'], 
                                 capture_output=True, text=True)
            if result.returncode == 0:
                nvidia_gpu_count = len(result.stdout.strip().split('\n'))
                print(f"nvidia-smi检测到GPU数量: {nvidia_gpu_count}")
            else:
                print("nvidia-smi执行失败")
        except Exception as e:
            print(f"nvidia-smi检查失败: {e}")
    print("="*50)
    
    return cuda_available


def verify_pytorch_cuda():
    """验证 PyTorch 是否能正确使用 CUDA"""
    print("="*50)
    print("PyTorch CUDA 验证")
    print(f"PyTorch 版本: {torch.__version__}")
    print(f"CUDA 可用: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"GPU 设备数量: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
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


def detect_ray_gpu_resources() -> float:
    # 从 ray 集群获取总共可用的GPU数量
    # 方法1: 获取集群中所有可用的GPU资源（推荐）
    try:
        cluster_resources = ray.available_resources()
        num_gpus_available = int(cluster_resources.get('GPU', 0))
        print(f"Ray集群可用GPU资源: {num_gpus_available}")
    except Exception as e:
        print(f"获取Ray集群资源失败: {e}")
        num_gpus_available = 0
    
    # 方法2: 如果方法1失败，尝试获取集群总资源
    if num_gpus_available == 0:
        try:
            total_resources = ray.cluster_resources()
            num_gpus_available = int(total_resources.get('GPU', 0))
            print(f"Ray集群总GPU资源: {num_gpus_available}")
        except Exception as e:
            print(f"获取Ray集群总资源失败: {e}")
            num_gpus_available = 0
    
    # 方法3: 如果Ray方法都失败，使用PyTorch检测
    if num_gpus_available == 0:
        if torch.cuda.is_available():
            num_gpus_available = torch.cuda.device_count()
            print(f"PyTorch检测到GPU数量: {num_gpus_available}")
        else:
            print("PyTorch无法检测到GPU")
    
    # 方法4: 最后尝试使用nvidia-smi
    if num_gpus_available == 0:
        try:
            import subprocess
            result = subprocess.run(['nvidia-smi', '--list-gpus'], 
                                 capture_output=True, text=True)
            if result.returncode == 0:
                num_gpus_available = len(result.stdout.strip().split('\n'))
                print(f"nvidia-smi检测到GPU数量: {num_gpus_available}")
            else:
                print("nvidia-smi执行失败")
        except Exception as e:
            print(f"nvidia-smi检查失败: {e}")
    
    print(f"最终检测到的GPU数量: {num_gpus_available}")
    return num_gpus_available


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


def run_training(
        orcagym_addr: str,
        env_name: str,
        agent_name: str,
        agent_config: dict,
        task: str,
        max_episode_steps: int,
        num_learners: int,
        num_env_runners: int,
        num_envs_per_env_runner: int,
        num_gpus_available: float,
        num_node_cpus: dict,
        num_cpus_per_learner: int,
        num_gpus_per_learner: int,
        num_cpus_per_env_runner: int,
        num_gpus_per_env_runner: int,
        async_env_runner: bool,
        iter: int,
        total_steps: int,
        render_mode: str,
        height_map_file: str,
        frame_skip: int,
        action_skip: int,
        time_step: float,
        model_dir: str = None
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
            agent_num=32,   # 一个Mujoco Instance支持 32 个agent是最合理的，这是默认配置
            max_episode_steps=max_episode_steps,
            render_mode=render_mode,
            async_env_runner=async_env_runner,
            height_map_file=height_map_file,
            task=task,
            frame_skip=frame_skip,
            action_skip=action_skip,
            time_step=time_step
        )
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
    demo_env, demo_env_kwargs = create_demo_env_instance(
        orcagym_addr=orcagym_addr,
        env_name=env_name,
        agent_name=agent_name,
        agent_num=32,   # 一个Mujoco Instance支持 32 个agent是最合理的，这是默认配置
        max_episode_steps=max_episode_steps,
        async_env_runner=async_env_runner,
        height_map_file=height_map_file,
        render_mode=render_mode,
        task=task,
        frame_skip=frame_skip,
        action_skip=action_skip,
        time_step=time_step
    )

    print("\nStarting training...")

    # scaling_config = ScalingConfig(num_workers=2, use_gpu=True)
    # trainer = TorchTrainer(train_func, scaling_config=scaling_config)
    # result = trainer.fit()

    tuner = config_appo_tuner(
        agent_config=agent_config,
        task=task,
        num_learners=num_learners,
        num_env_runners=num_env_runners,
        num_envs_per_env_runner=num_envs_per_env_runner,
        num_cpus_per_learner=num_cpus_per_learner,
        num_gpus_per_learner=num_gpus_per_learner,
        num_cpus_per_env_runner=num_cpus_per_env_runner,
        num_gpus_per_env_runner=num_gpus_per_env_runner,
        iter=iter,
        total_steps=total_steps,
        env=demo_env,
        num_gpus_available=num_gpus_available,
        async_env_runner=async_env_runner,
        env_name=env_name,
        env_kwargs=demo_env_kwargs,
        model_dir=model_dir,
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
            checkpoint_path = best_result.checkpoint.path
            print(f"Best checkpoint directory: {checkpoint_path}")
            


    demo_env.close()

    ray.shutdown()
    print("Process completed.")

def test_model(
        checkpoint_path,
        orcagym_addr: str,
        env_name: str,
        agent_name: str,
        max_episode_steps: int,
        use_onnx_for_inference: bool = False,
        explore_during_inference: bool = False,
        render_mode: str = 'human',
        async_env_runner: bool = False,
        height_map_file: str = None,
        task: str = None,
        frame_skip: int = 5,
        action_skip: int = 4,
        time_step: float = 0.001
    ):


    env = env_creator(
        env_context=None,
        orcagym_addr=orcagym_addr,
        env_name=env_name,
        agent_name=agent_name,
        agent_num=1,   
        max_episode_steps=max_episode_steps,
        render_mode=render_mode,
        async_env_runner=async_env_runner,
        height_map_file=height_map_file,
        task=task,
        frame_skip=frame_skip,
        action_skip=action_skip,
        time_step=time_step
    )
    
    # Create new RLModule and restore its state from the last algo checkpoint.
    # Note that the checkpoint for the RLModule can be found deeper inside the algo
    # checkpoint's subdirectories ([algo dir] -> "learner/" -> "module_state/" ->
    # "[module ID]):
    print("Restore RLModule from checkpoint ...", end="")
    rl_module = RLModule.from_checkpoint(
        os.path.join(
            checkpoint_path,
            "learner_group",
            "learner",
            "rl_module",
            DEFAULT_MODULE_ID,
        )
    )
    ort_session = None
    print(" ok")

    # Create an env to do inference in.
    random_seed = np.random.randint(0, 1000000)
    _, info = env.reset(seed=random_seed)
    obs = info['env_obs']['observation'][0]

    num_episodes = 0
    episode_return = 0.0

    while num_episodes < max_episode_steps:
        start_time = time.time()

        # Compute an action using a B=1 observation "batch".
        input_dict = {Columns.OBS: np.expand_dims(obs, 0)}
        if not use_onnx_for_inference:
            input_dict = {Columns.OBS: torch.from_numpy(obs).unsqueeze(0)}

        # If ONNX and module has not been exported yet, do thexplore_during_inferenceis here using
        # the input_dict as example input.
        elif ort_session is None:
            tensor_input_dict = {Columns.OBS: torch.from_numpy(obs).unsqueeze(0)}
            torch.onnx.export(rl_module, {"batch": tensor_input_dict}, f="test.onnx")
            ort_session = onnxruntime.InferenceSession(
                "test.onnx", providers=["CPUExecutionProvider"]
            )

        # No exploration (using ONNX).
        if ort_session is not None:
            rl_module_out = ort_session.run(
                None,
                {
                    key.name: val
                    for key, val in dict(
                        zip(
                            tree.flatten(ort_session.get_inputs()),
                            tree.flatten(input_dict),
                        )
                    ).items()
                },
            )
            # [0]=encoder outs; [1]=action logits
            rl_module_out = {Columns.ACTION_DIST_INPUTS: rl_module_out[1]}
        # No exploration (using RLModule).
        elif not explore_during_inference:
            rl_module_out = rl_module.forward_inference(input_dict)
        # W/ exploration (using RLModule).
        else:
            rl_module_out = rl_module.forward_exploration(input_dict)

        # For discrete action spaces used here, normally, an RLModule "only"
        # produces action logits, from which we then have to sample.
        # However, you can also write custom RLModules that output actions
        # directly, performing the sampling step already inside their
        # `forward_...()` methods.
        # print("rl_module_out: ", rl_module_out)
        logits = convert_to_numpy(rl_module_out[Columns.ACTION_DIST_INPUTS])

        # 最佳方案 (推荐)：显式拆分参数
        mu = logits[:, :env.action_space.shape[0]]

        # 直接使用均值作为最终动作
        action = np.clip(mu[0], env.action_space.low, env.action_space.high)

        # p=softmax(logits[0])
        # print(f"logits.shape: {logits.shape}", "env.action_space.shape:", env.action_space.shape, "p.shape:", p.shape)
        # Perform the sampling step in numpy for simplicity.
        # action = np.random.choice(a=16, size=p.shape[0], p=p)
        # print(f"action: {action}")
        # Send the computed action `a` to the env.
        _, _, _, _, info = env.step(action)
        env.render()
        obs = info['env_obs']['observation'][0]
        reward = info['reward'][0]
        is_success = info['is_success'][0]
        terminated = info['terminated'][0]
        truncated = info['truncated'][0]

        end_time = time.time()
        if end_time - start_time < time_step * frame_skip * action_skip:
            # print(f"sleep: {REALTIME_STEP - (end_time - start_time)}")
            time.sleep(time_step * frame_skip * action_skip - (end_time - start_time))
            
        episode_return += reward
        # Is the episode `done`? -> Reset.
        if terminated or truncated:
            print(f"Episode done: Total reward = {episode_return}")
            random_seed = np.random.randint(0, 1000000)
            _, info = env.reset(seed=random_seed)
            obs = info['env_obs']['observation'][0]
            num_episodes += 1
            episode_return = 0.0

    print(f"Done performing action inference through {num_episodes} Episodes")
    
    env.close()