import os
import sys
import argparse
import time

current_file_path = os.path.abspath('')
project_root = os.path.dirname(os.path.dirname(current_file_path))

# 将项目根目录添加到 PYTHONPATH
if project_root not in sys.path:
    sys.path.append(project_root)


import gymnasium as gym
from stable_baselines3 import PPO, SAC, DDPG
from gymnasium.envs.registration import register
from datetime import datetime
import torch
import torch.nn as nn
from stable_baselines3.common.vec_env import SubprocVecEnv
from sb3_contrib import TQC
from stable_baselines3.her import GoalSelectionStrategy, HerReplayBuffer
from stable_baselines3.common.noise import NormalActionNoise
import numpy as np


def register_env(orcagym_addr, env_name, env_index, task, max_episode_steps, frame_skip):
    env_ids = []
    for env_name in env_names:
        env_id = f"PandaMocap-v0-OrcaGym-{env_name}"
        print("register_env: ", env_id)
        gym.register(
            id=env_id,
            entry_point=f"envs.panda_mocap.{task}",
            kwargs={'frame_skip': frame_skip, 
                    'reward_type': "sparse",
                    'orcagym_addr': orcagym_addr, 
                    'agent_names': ['Panda'], 
                    'time_step': 0.01,
                    'render_mode': "human",},
            max_episode_steps=max_episode_steps,
            reward_threshold=0.0,
        )
        env_ids.append(env_id)
    return env_ids

def make_env(orcagym_addr, env_index, agents_per_env, task, max_episode_steps, frame_skip):
    print("make_env: ", orcagym_addr, env_index)
    def _init():
        # 注册环境，确保子进程中也能访问
        env_name = "PandaMocap-v0"
        register_env([orcagym_addr], agents_per_env, task, max_episode_steps, frame_skip)
        env = gym.make(env_id)
        seed = int(env_id[-3:])
        env.unwrapped.set_seed_value(seed)
        return env
    return _init

def training_model(model, total_timesteps, model_file):
    # 训练模型，每一百万步保存一次check point
    CKP_LEN = 1000000
    training_loop = []
    if total_timesteps <= CKP_LEN:
        training_loop.append(total_timesteps)
    else:
        if total_timesteps % CKP_LEN == 0:
            training_loop = [CKP_LEN] * (total_timesteps // CKP_LEN)
        else:
            training_loop = [CKP_LEN] * (total_timesteps // CKP_LEN)
            training_loop.append(total_timesteps % CKP_LEN)
    
    for i, loop in enumerate(training_loop):
        model.learn(loop)

        if i < len(training_loop) - 1:
            model.save(f"{model_file}_ckp{(i + 1) * loop}")
            print(f"-----------------Save Model Checkpoint: {(i + 1) * loop}-----------------")

    print(f"-----------------Save Model-----------------")
    model.save(model_file)
        
def continue_training_ppo(env, env_num, total_timesteps, model_file):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 加载已有模型或初始化新模型
    if os.path.exists(f"{model_file}.zip"):
        model = PPO.load(model_file, env=env, device=device)
    else:
        # 定义自定义策略网络
        policy_kwargs = dict(
            net_arch=dict(
                pi=[128, 128, 128],  # 策略网络结构
                vf=[128, 128, 128]   # 值函数网络结构
            ),
            ortho_init=True,
            activation_fn=nn.ReLU
        )
        model = PPO("MultiInputPolicy", env, verbose=1, learning_rate=0.0003, n_steps=2048, batch_size=128, gamma=0.95, clip_range=0.2, policy_kwargs=policy_kwargs, device=device)
        
    training_model(model, total_timesteps, model_file)


def continue_training_sac(env, env_num, total_timesteps, max_episode_steps, model_file):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if os.path.exists(f"{model_file}.zip"):
        model = SAC.load(model_file, env=env, device=device)
    else:        
        # https://arxiv.org/html/2312.13788v2
        # Open-Source Reinforcement Learning Environments Implemented in MuJoCo with Franka Manipulator。        
        # Training phase: Use HER
        goal_selection_strategy = "future"  # 选择 'future' 策略

        policy_kwargs = dict(
            net_arch=[256, 256, 256],  # 三层隐藏层，每层256个神经元
            n_critics=2,
            # n_quantiles=25,
            # activation_fn=nn.ReLU
        )

        replay_buffer_class = HerReplayBuffer
        replay_buffer_kwargs = dict(
            n_sampled_goal=4,
            goal_selection_strategy=goal_selection_strategy
        )

        # Initialize the model
        model = SAC(
            "MultiInputPolicy", 
            env, 
            replay_buffer_class=replay_buffer_class,
            replay_buffer_kwargs=replay_buffer_kwargs,
            verbose=1, 
            learning_rate=0.001,  # 学习率
            buffer_size=1000000,  # 重播缓冲区大小
            batch_size=512,  # 批量大小
            tau=0.005, 
            gamma=0.99,  # 折扣因子
            learning_starts= max_episode_steps * env_num * 8,
            policy_kwargs=policy_kwargs, 
            device=device
        )

    training_model(model, total_timesteps, model_file)

def continue_training_ddpg(env, env_num, total_timesteps, max_episode_steps, model_file):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if os.path.exists(f"{model_file}.zip"):
        model = DDPG.load(model_file, env=env, device=device)
    else:        
        # https://arxiv.org/html/2312.13788v2
        # Open-Source Reinforcement Learning Environments Implemented in MuJoCo with Franka Manipulator。        
        n_actions = env.action_space.shape[0]
        noise_std = 0.2
        action_noise = NormalActionNoise(
            mean=np.zeros(n_actions), sigma=noise_std * np.ones(n_actions)
        )

        model = DDPG(
            "MultiInputPolicy",
            env,
            replay_buffer_class=HerReplayBuffer,
            replay_buffer_kwargs=dict(
                n_sampled_goal=4,
                goal_selection_strategy="future",
            ),
            verbose=1,
            buffer_size=int(1e6),
            learning_rate=1e-3,
            action_noise=action_noise,
            gamma=0.95,
            batch_size=512,
            learning_starts= max_episode_steps * env_num * 8,
            policy_kwargs=dict(net_arch=[256, 256, 256]),
        )

    training_model(model, total_timesteps, model_file)


def continue_training_tqc(env, env_num, total_timesteps, max_episode_steps, model_file):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if os.path.exists(f"{model_file}.zip"):
        model = TQC.load(model_file, env=env, device=device)
    else:        
        # https://arxiv.org/html/2312.13788v2
        # Open-Source Reinforcement Learning Environments Implemented in MuJoCo with Franka Manipulator。        
        # Training phase: Use HER
        goal_selection_strategy = "future"  # 选择 'future' 策略

        policy_kwargs = dict(
            net_arch=[512, 512, 512],  # 三层隐藏层，每层256个神经元
            n_critics=2,
            n_quantiles=25,
            activation_fn=torch.nn.ReLU
        )

        replay_buffer_class = HerReplayBuffer
        replay_buffer_kwargs = dict(
            n_sampled_goal=4,
            goal_selection_strategy=goal_selection_strategy
        )

        # Initialize the model
        model = TQC(
            "MultiInputPolicy", 
            env, 
            replay_buffer_class=replay_buffer_class,
            replay_buffer_kwargs=replay_buffer_kwargs,
            verbose=1, 
            learning_rate=0.001,  # 学习率
            buffer_size=1000000,  # 重播缓冲区大小
            batch_size=2048,  # 批量大小
            tau=0.05, 
            gamma=0.95,  # 折扣因子
            learning_starts= max_episode_steps * env_num * 8,
            policy_kwargs=policy_kwargs, 
            device=device
        )

    training_model(model, total_timesteps, model_file)


def testing_model(env, model):
    # 测试模型
    observation, info = env.reset()
    for test in range(10):
        total_reward = 0
        for _ in range(1000):
            start_time = datetime.now()

            # predict_start = datetime.now()
            action, _states = model.predict(observation, deterministic=True)
            # predict_time = datetime.now() - predict_start
            # print("Predict Time: ", predict_time.total_seconds(), flush=True)

            # setp_start = datetime.now()
            observation, reward, terminated, truncated, info = env.step(action)
            env.render()
            # step_time = datetime.now() - setp_start
            # print("Step Time: ", step_time.total_seconds(), flush=True)

            total_reward += reward

            # 
            elapsed_time = datetime.now() - start_time
            if elapsed_time.total_seconds() < 0.01:
                time.sleep(0.01 - elapsed_time.total_seconds())

            if terminated or truncated:
                print(f"----------------Test: {test}----------------")
                print("Successed: ", terminated)
                print("Total Reward: ", total_reward)
                print("---------------------------------------")
                observation, info = env.reset()
                total_reward = 0
                break

    env.close()

def train_model(grpc_addresses, agents_per_env, task, max_episode_steps, frame_skip, model_type, total_timesteps, model_file):
    try:
        print("simulation running... , grpc_addresses: ", grpc_addresses)
        env_ids = register_env(env_name, task, max_episode_steps, frame_skip)
        env_num = len(env_ids)

        env_fns = [make_env(orcagym_addr, agents_per_env, task, max_episode_steps, frame_skip, env_id) for orcagym_addr, env_id in zip(grpc_addresses, env_ids)]
        env = SubprocVecEnv(env_fns)

        print("Start Simulation!")
        if model_type == "ppo":
            continue_training_ppo(env, env_num, total_timesteps, max_episode_steps, model_file)
        elif model_type == "tqc":
            continue_training_tqc(env, env_num, total_timesteps, max_episode_steps, model_file)
        elif model_type == "sac":
            continue_training_sac(env, env_num, total_timesteps, max_episode_steps, model_file)
        elif model_type == "ddpg":
            continue_training_ddpg(env, env_num, total_timesteps, max_episode_steps, model_file)
        else:
            raise ValueError("Invalid model type")
    except KeyboardInterrupt:
        print("退出仿真环境")
        env.close()

def test_model(orcagym_addr, task, max_episode_steps, frame_skip, model_type, model_file):
    try:
        print("simulation running... , orcagym_addr: ", orcagym_addr)
        env_id = register_env([orcagym_addr], task, max_episode_steps, frame_skip)[0]
        env = gym.make(env_id)
        seed = int(env_id[-3:])
        env.unwrapped.set_seed_value(seed)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if model_type == "ppo":
            model = PPO.load(model_file, env=env, device=device)
        elif model_type == "tqc":
            model = TQC.load(model_file, env=env, device=device)
        elif model_type == "sac":
            model = SAC.load(model_file, env=env, device=device)
        elif model_type == "ddpg":
            model = DDPG.load(model_file, env=env, device=device)
        else:
            raise ValueError("Invalid model type")

        testing_model(env, model)
    except KeyboardInterrupt:
        print("退出仿真环境")
        env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run multiple instances of the script with different gRPC addresses.')
    parser.add_argument('--grpc_addresses', type=str, nargs='+', default=['localhost:50051'], help='The gRPC addresses to connect to')
    parser.add_argument('--agents_per_env', type=int, default=1, help='The number of agents per env')
    parser.add_argument('--task', type=str, default='reach', help='The task to run (reach or pick_and_place)')
    parser.add_argument('--model_type', type=str, default='tqc', help='The model to use (ppo/tqc/sac/ddpg)')
    parser.add_argument('--run_mode', type=str, default='training', help='The mode to run (training or testing)')
    parser.add_argument('--total_timesteps', type=int, default=100000, help='The total timesteps to train the model')
    args = parser.parse_args()

    grpc_addresses = args.grpc_addresses
    agents_per_env = args.agents_per_env
    task = args.task
    model_type = args.model_type
    run_mode = args.run_mode
    total_timesteps = args.total_timesteps

    model_file = f"panda_mocap_{task}_{model_type}_{total_timesteps}_model"

    if task == 'reach':
        task = 'reach:FrankaReachEnv'
    elif task == 'pick_and_place':
        task = 'pick_and_place:FrankaPickAndPlaceEnv'
    elif task == 'push':
        task = 'push:FrankaPushEnv'
    elif task == 'slide':
        task = 'slide:FrankaSlideEnv'
    else:
        raise ValueError("Invalid task")

    # 训练需要skip跨度大一点，可以快一点，测试skip跨度小一点，流畅一些
    MAX_EPISODE_STEPS_TRAINING = 50
    FRAME_SKIP_TRAINING = 50

    MAX_EPISODE_STEPS_TESTING = 250
    FRAME_SKIP_TESTING = 1


    if run_mode == "training":
        train_model(grpc_addresses, agents_per_env, task, MAX_EPISODE_STEPS_TRAINING, FRAME_SKIP_TRAINING, model_type, total_timesteps, model_file)
        test_model(grpc_addresses[0], task, MAX_EPISODE_STEPS_TESTING, FRAME_SKIP_TESTING, model_type, model_file)
    elif run_mode == "testing":
        test_model(grpc_addresses[0], task, MAX_EPISODE_STEPS_TESTING, FRAME_SKIP_TESTING, model_type, model_file)    
    else:
        raise ValueError("Invalid run mode")

