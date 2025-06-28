
import ray
from ray import air
from ray import tune
from ray.tune import RunConfig, CheckpointConfig
from ray.rllib.algorithms.appo import APPOConfig
from ray.tune.registry import register_env
import numpy as np
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.core.rl_module.default_model_config import DefaultModelConfig
import numpy as np
import gymnasium as gym
import threading
import argparse
from ray.rllib.algorithms.callbacks import DefaultCallbacks
import time
from ray.rllib.policy.policy import Policy
from orca_gym.adapters.rllib.metrics_callback import OrcaMetricsCallback


ENV_ENTRY_POINT = {
    "Ant_OrcaGymEnv": "envs.mujoco.ant_orcagym:AntOrcaGymEnv",
}

TIME_STEP = 0.001
FRAME_SKIP = 20
REALTIME_STEP = TIME_STEP * FRAME_SKIP
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
        render_mode: str,
    ) -> tuple[ str, dict ]:
    orcagym_addr_str = orcagym_addr.replace(":", "-")
    env_id = env_name + "-OrcaGym-" + orcagym_addr_str + "-" + render_mode
    agent_names = [f"{agent_name}"]
    kwargs = {
        'frame_skip': FRAME_SKIP,
        'orcagym_addr': orcagym_addr,
        'agent_names': agent_names,
        'time_step': TIME_STEP,
        'render_mode': render_mode,
    }

    return env_id, kwargs



def env_creator(
        env_context,
        orcagym_addr: str,
        env_name: str,
        agent_name: str,
        max_episode_steps: int,
        render_mode: str
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
        render_mode=render_mode
    )

    # 只有第一个worker的第一个环境渲染
    if render_mode == 'human':
        # render_mode = 'human' if worker_idx == 1 and vector_idx == 0 else 'none'
        if worker_idx == 1 and env_id not in gym.envs.registry:
            # 如果是第一个worker的第一个环境，使用人类渲染模式
            render_mode = 'human'
        else:
            render_mode = 'none'
            kwargs['render_mode'] = 'none'  # 确保其他worker不渲染
    else:
        render_mode = 'none'
    


    if vector_idx == 0:
        # 避免重复注册冲突
        if env_id not in gym.envs.registry:
            print(f"Registering environment: {env_id}")
            gym.register(
                id=env_id,
                entry_point=ENV_ENTRY_POINT[env_name],
                kwargs=kwargs,
                max_episode_steps=max_episode_steps,
                reward_threshold=0.0,
            )
        else:
            print(f"Environment {env_id} already registered, skipping registration.")

    # 创建环境并验证
    try:
        print(f"Worker {worker_idx}, vector {vector_idx}: Creating environment {env_id}, kwargs={kwargs}")
        env = gym.make(env_id, **kwargs)
        
        # 验证观测空间
        print(f"Observation space for {env_id}: {env.observation_space}")
        print(f"Action space for {env_id}: {env.action_space}")
        
        # 测试采样观测值
        sample_obs = env.observation_space.sample()
        if not env.observation_space.contains(sample_obs):
            print(f"WARNING: Sampled observation is not within observation space!")
        
        print(f"Environment {env_id} created successfully in worker {worker_idx}, vector {vector_idx}."
              f" Render mode: {render_mode}")
        return env
        
    except Exception as e:
        print(f"ERROR: Failed to create environment {env_id} in worker {worker_idx}")
        print(f"Exception: {str(e)}")
        import traceback
        traceback.print_exc()
        raise


def test_model(
        checkpoint_path,
        orcagym_addr: str,
        env_name: str,
        agent_name: str,
        max_episode_steps: int
    ):
    
    # 1. 首先注册训练时使用的环境
    register_env(
        "OrcaGymEnv", 
        lambda env_context: env_creator(
            env_context=env_context,
            orcagym_addr=orcagym_addr,
            env_name=env_name,
            agent_name=agent_name,
            max_episode_steps=max_episode_steps,
            render_mode='human',
        )
    )
    
    # 2. 使用注册的环境创建实例
    from ray.rllib.env import EnvContext
    env = env_creator(
        env_context=None,  # 空上下文
        orcagym_addr=orcagym_addr,
        env_name=env_name,
        agent_name=agent_name,
        max_episode_steps=max_episode_steps,
        render_mode='human',
    )
    
    # 加载策略

    policies = Policy.from_checkpoint(checkpoint_path)
    default_policy_id = "default_policy"
    if default_policy_id not in policies:
        default_policy_id = next(iter(policies.keys()))
    policy = policies[default_policy_id]


    # 确保策略支持字典观测
    if not hasattr(policy.model, "branches"):
        raise TypeError("Loaded policy does not support dictionary observations")
    
    for _ in range(5):  # 运行5个测试episode
        state_dict, _ = env.reset()
        
        # 处理字典观测
        batch_state = {}
        for key, value in state_dict.items():
            if isinstance(value, np.ndarray):
                value = np.expand_dims(value, axis=0)
            batch_state[key] = value
        
        episode_reward = 0
        done = False
        
        while not done:
            time_start = time.time()
            
            # 计算动作
            action = policy.compute_single_action(batch_state, explore=False)[0]
            
            # 环境步进
            next_state_dict, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            
            # 准备下一个状态
            batch_state = {}
            for key, value in next_state_dict.items():
                if isinstance(value, np.ndarray):
                    value = np.expand_dims(value, axis=0)
                batch_state[key] = value
                
            # 渲染和计时控制
            env.render()
            time_end = time.time()
            elapsed_time = time_end - time_start
            if elapsed_time < REALTIME_STEP:
                time.sleep(REALTIME_STEP - elapsed_time)
        
        print(f"Test episode reward: {episode_reward}")
    
    env.close()