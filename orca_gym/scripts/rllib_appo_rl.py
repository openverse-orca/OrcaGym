import ray
from ray import tune
from ray.tune import RunConfig, CheckpointConfig
from ray.rllib.algorithms.appo import APPOConfig
from ray.tune.registry import register_env

import torch.version
from orca_gym.adapters.rllib.metrics_callback import OrcaMetricsCallback
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
import gymnasium as gym
import torch
from ray.rllib.algorithms.appo.torch.default_appo_torch_rl_module import DefaultAPPOTorchRLModule
import os
from ray.rllib.core.rl_module.rl_module import RLModule
from ray.rllib.core import DEFAULT_MODULE_ID
from ray.rllib.core.columns import Columns
import numpy as np
from ray.rllib.utils.numpy import convert_to_numpy
import tree
import time
import os
import gymnasium as gym
import json
from ray.rllib.env.single_agent_env_runner import (
    SingleAgentEnvRunner,
)
from orca_gym.environment.async_env.single_agent_env_runner import (
    OrcaGymAsyncSingleAgentEnvRunner
)

ENV_ENTRY_POINT = {
    "Ant_OrcaGymEnv": "envs.mujoco.ant_orcagym:AntOrcaGymEnv",
    "LeggedGym": "envs.legged_gym.legged_gym_env:LeggedGymEnv",
}

ENV_RUNNER_CLS = {
    "Ant_OrcaGymEnv": SingleAgentEnvRunner,
    "LeggedGym": OrcaGymAsyncSingleAgentEnvRunner,
}

TIME_STEP = 0.001
FRAME_SKIP = 5
ACTION_SKIP = 4
REALTIME_STEP = TIME_STEP * FRAME_SKIP * ACTION_SKIP
CONTROL_FREQ = 1 / REALTIME_STEP

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
        max_episode_steps: int
    ) -> tuple[ str, dict ]:

    orcagym_addr_str = orcagym_addr.replace(":", "-")
    env_id = env_name + "-OrcaGym-" + orcagym_addr_str + f"-{worker_idx:03d}"
    agent_names = [f"{agent_name}_{i:03d}" for i in range(agent_num)]
    kwargs = {
        'Ant_OrcaGymEnv': {
            'frame_skip': FRAME_SKIP,
            'orcagym_addr': orcagym_addr,
            'agent_names': agent_names,
            'time_step': TIME_STEP,
            'render_mode': render_mode,
            'max_episode_steps': max_episode_steps,
        },
        'LeggedGym': {
            'frame_skip': FRAME_SKIP,
            'action_skip': ACTION_SKIP,
            'orcagym_addr': orcagym_addr,
            'agent_names': agent_names,
            'time_step': TIME_STEP,
            'render_mode': render_mode,
            'height_map_file': height_map_file,
            'env_id': env_id,
            'max_episode_steps': max_episode_steps,
            'is_subenv': True,
            'run_mode': 'training',
            'task': 'follow_command',
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
    render_mode: str
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
        max_episode_steps=max_episode_steps
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
    num_env_runners: int, 
    num_envs_per_env_runner: int,
    env: gym.Env,
    num_gpus_available: int,
    async_env_runner: bool,
    env_name: str,
    env_kwargs: dict
):
    # env_name 是用 - 分隔的字符串，去掉最后一段，保留前面的
    env_name_prefix = "-".join(env.spec.id.split("-")[:-1])
    print("env_name_prefix: ", env_name_prefix)
    print("action_space: ", env.action_space, "observation_space: ", env.observation_space)
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
            normalize_actions=True,
            clip_actions=True,
        )
        .env_runners(
            env_runner_cls=ENV_RUNNER_CLS[env_name],
            num_env_runners=num_env_runners,          
            num_envs_per_env_runner=num_envs_per_env_runner,
            num_cpus_per_env_runner=1,
            num_gpus_per_env_runner=(num_gpus_available - 0.1) * 1 / num_env_runners,  # 每个环境runner分配的GPU数量
            rollout_fragment_length=64,
            gym_env_vectorize_mode=gym.envs.registration.VectorizeMode.ASYNC if async_env_runner else gym.envs.registration.VectorizeMode.SYNC,  # default is `SYNC`
        )
        .rl_module(
            rl_module_spec=RLModuleSpec(
                observation_space=env.observation_space["observation"],
                action_space=env.action_space,
                module_class=DefaultAPPOTorchRLModule,
                model_config={
                    "fcnet_hiddens": [512, 256, 128],
                    "fcnet_activation": "relu",
                    "post_fcnet_activation": "tanh",
                    "vf_share_layers": False,
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
    return config

def config_appo_tuner(
        num_env_runners: int, 
        num_envs_per_env_runner: int,
        iter: int,
        env: gym.Env,
        async_env_runner: bool,
        env_name: str,
        env_kwargs: dict
    ) -> tune.Tuner:
    # 重要：获取系统的实际GPU数量
    num_gpus_available = torch.cuda.device_count()

    os.environ['CUDA_VISIBLE_DEVICES'] = "0,1"
    print(f"CUDA_VISIBLE_DEVICES: {os.environ['CUDA_VISIBLE_DEVICES']}")
    
    config = get_config(
        num_env_runners=num_env_runners,
        num_envs_per_env_runner=num_envs_per_env_runner,
        env=env,
        num_gpus_available=num_gpus_available,
        async_env_runner=async_env_runner,
        env_name=env_name,
        env_kwargs=env_kwargs
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
                num_to_keep=3,
                checkpoint_frequency=100,
                checkpoint_score_attribute="env_runners/episode_return_mean",
                checkpoint_score_order="max",
                checkpoint_at_end=True
            ),
            verbose=3,
        ),
        # scaling_config=ScalingConfig(
        #     num_workers=num_env_runners,
        #     use_gpu=True
        # )
    )




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
        max_episode_steps=max_episode_steps
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


def run_training(
        orcagym_addr: str,
        env_name: str,
        agent_name: str,
        max_episode_steps: int,
        num_env_runners: int,
        num_envs_per_env_runner: int,
        async_env_runner: bool,
        iter: int,
        render_mode: str,
        height_map_file: str
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
            agent_num=num_envs_per_env_runner,
            max_episode_steps=max_episode_steps,
            render_mode=render_mode,
            async_env_runner=async_env_runner,
            height_map_file=height_map_file
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
        agent_num=num_envs_per_env_runner,
        max_episode_steps=max_episode_steps,
        async_env_runner=async_env_runner,
        height_map_file=height_map_file,
        render_mode=render_mode,
    )

    print("\nStarting training...")

    # scaling_config = ScalingConfig(num_workers=2, use_gpu=True)
    # trainer = TorchTrainer(train_func, scaling_config=scaling_config)
    # result = trainer.fit()

    tuner = config_appo_tuner(
        num_env_runners=num_env_runners,
        num_envs_per_env_runner=num_envs_per_env_runner,
        iter=iter,
        env=demo_env,
        async_env_runner=async_env_runner,
        env_name=env_name,
        env_kwargs=demo_env_kwargs
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
    ):


    env = env_creator(
        env_context=None,  # 空上下文
        orcagym_addr=orcagym_addr,
        env_name=env_name,
        agent_name=agent_name,
        max_episode_steps=max_episode_steps,
        render_mode=render_mode,
        async_env_runner=False,
        height_map_file=None
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
    obs, info = env.reset(seed=random_seed)

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
        mu = logits[:, :env.action_space.shape[0]]  # 前8个: 动作均值
        # log_sigma = logits[:, 8:]  # 后8个: 标准差参数 (推理时不需要)

        # 直接使用均值作为最终动作
        action = mu[0]

        # p=softmax(logits[0])
        # print(f"logits.shape: {logits.shape}", "env.action_space.shape:", env.action_space.shape, "p.shape:", p.shape)
        # Perform the sampling step in numpy for simplicity.
        # action = np.random.choice(a=16, size=p.shape[0], p=p)
        # print(f"action: {action}")
        # Send the computed action `a` to the env.
        obs, reward, terminated, truncated, _ = env.step(action)
        # print("obs: ", obs)

        end_time = time.time()
        if end_time - start_time < REALTIME_STEP:
            # print(f"sleep: {REALTIME_STEP - (end_time - start_time)}")
            time.sleep(REALTIME_STEP - (end_time - start_time))
            
        episode_return += reward
        # Is the episode `done`? -> Reset.
        if terminated or truncated:
            print(f"Episode done: Total reward = {episode_return}")
            random_seed = np.random.randint(0, 1000000)
            obs, info = env.reset(seed=random_seed)
            num_episodes += 1
            episode_return = 0.0

    print(f"Done performing action inference through {num_episodes} Episodes")
    
    env.close()