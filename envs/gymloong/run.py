import os
import sys
import argparse
import time

current_file_path = os.path.realpath(__file__)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))

# 将项目根目录添加到 PYTHONPATH
if project_root not in sys.path:
    sys.path.append(project_root)

print(f'Project root: {project_root}')

import gymnasium as gym
from stable_baselines3 import PPO, SAC, DDPG
from gymnasium.envs.registration import register
from datetime import datetime
import torch
import torch.nn as nn
from envs.orca_gym_env import ActionSpaceType
from stable_baselines3.common.vec_env import SubprocVecEnv
from sb3_contrib import TQC
from stable_baselines3.her import GoalSelectionStrategy, HerReplayBuffer
from stable_baselines3.common.noise import NormalActionNoise
import numpy as np


def register_env(grpc_addresses, task, max_episode_steps, frame_skip):
    env_ids = []
    for grpc_address in grpc_addresses:
        env_id = f"AzureLoong-v0-OrcaGym-{grpc_address[-3:]}"
        print("register_env: ", env_id)
        gym.register(
            id=env_id,
            entry_point=f"envs.gymloong.environ.Azure_Loong.{task}",
            kwargs={'frame_skip': frame_skip, 
                    'reward_type': "sparse",
                    'action_space_type': ActionSpaceType.CONTINUOUS,
                    'action_step_count': 0,
                    'grpc_address': grpc_address, 
                    'agent_names': ['Panda'], 
                    'time_step': 0.01},
            max_episode_steps=max_episode_steps,
            reward_threshold=0.0,
        )
        env_ids.append(env_id)
    return env_ids

def make_env(grpc_address, task, max_episode_steps, frame_skip, env_id):
    print("make_env: ", grpc_address, env_id)
    def _init():
        # 注册环境，确保子进程中也能访问
        register_env([grpc_address], task, max_episode_steps, frame_skip)
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

def train_model(grpc_addresses, task, max_episode_steps, frame_skip, model_type, total_timesteps, model_file):
    try:
        print("simulation running... , grpc_addresses: ", grpc_addresses)
        env_ids = register_env(grpc_addresses, task, max_episode_steps, frame_skip)
        env_num = len(env_ids)

        env_fns = [make_env(grpc_address, task, max_episode_steps, frame_skip, env_id) for grpc_address, env_id in zip(grpc_addresses, env_ids)]
        env = SubprocVecEnv(env_fns)

        print("Start Simulation!")
        if model_type == "ppo":
            continue_training_ppo(env, env_num, total_timesteps, max_episode_steps, model_file)
        # elif model_type == "tqc":
        #     continue_training_tqc(env, env_num, total_timesteps, max_episode_steps, model_file)
        # elif model_type == "sac":
        #     continue_training_sac(env, env_num, total_timesteps, max_episode_steps, model_file)
        # elif model_type == "ddpg":
        #     continue_training_ddpg(env, env_num, total_timesteps, max_episode_steps, model_file)
        # else:
        #     raise ValueError("Invalid model type")
    except KeyboardInterrupt:
        print("退出仿真环境")
        env.close()

def test_model(grpc_address, task, max_episode_steps, frame_skip, model_type, model_file):
    try:
        print("simulation running... , grpc_address: ", grpc_address)
        env_id = register_env([grpc_address], task, max_episode_steps, frame_skip)[0]
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
    parser.add_argument('--task', type=str, default='Azure_Loong_env', help='The task to run (reach or pick_and_place)')
    parser.add_argument('--model_type', type=str, default='ppo', help='The model to use (ppo/tqc/sac/ddpg)')
    parser.add_argument('--run_mode', type=str, default='training', help='The mode to run (training or testing)')
    parser.add_argument('--total_timesteps', type=int, default=100000, help='The total timesteps to train the model')
    args = parser.parse_args()

    grpc_addresses = args.grpc_addresses
    task = args.task
    model_type = args.model_type
    run_mode = args.run_mode
    total_timesteps = args.total_timesteps

    model_file = f"azureloong_{task}_{model_type}_{total_timesteps}_model"

    # if task == 'reach':
    #     task = 'reach:FrankaReachEnv'
    # elif task == 'pick_and_place':
    #     task = 'pick_and_place:FrankaPickAndPlaceEnv'
    # elif task == 'push':
    #     task = 'push:FrankaPushEnv'
    # elif task == 'slide':
    #     task = 'slide:FrankaSlideEnv'
    # else:
    #     raise ValueError("Invalid task")
    
    task = 'Azure_Loong_env:AzureLoongEnv'

    # 训练需要skip跨度大一点，可以快一点，测试skip跨度小一点，流畅一些
    MAX_EPISODE_STEPS_TRAINING = 50
    FRAME_SKIP_TRAINING = 50

    MAX_EPISODE_STEPS_TESTING = 250
    FRAME_SKIP_TESTING = 1


    if run_mode == "training":
        train_model(grpc_addresses, task, MAX_EPISODE_STEPS_TRAINING, FRAME_SKIP_TRAINING, model_type, total_timesteps, model_file)
        test_model(grpc_addresses[0], task, MAX_EPISODE_STEPS_TESTING, FRAME_SKIP_TESTING, model_type, model_file)
    elif run_mode == "testing":
        test_model(grpc_addresses[0], task, MAX_EPISODE_STEPS_TESTING, FRAME_SKIP_TESTING, model_type, model_file)    
    else:
        raise ValueError("Invalid run mode")

