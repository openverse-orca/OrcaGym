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
# from stable_baselines3.common.vec_env import SubprocVecEnv
from orca_gym.multi_agent import SubprocVecEnvMA
from sb3_contrib import TQC
from stable_baselines3.her import GoalSelectionStrategy, HerReplayBuffer
from stable_baselines3.common.noise import NormalActionNoise
import numpy as np

from envs.legged_gym.legged_robot_config import LeggedEnvConfig


def register_env(orcagym_addr, env_name, env_index, agent_num, agent_name, task, entry_point, time_step, max_episode_steps, frame_skip, render_remote) -> str:
    orcagym_addr_str = orcagym_addr.replace(":", "-")
    env_id = env_name + "-OrcaGym-" + orcagym_addr_str + f"-{env_index:03d}"
    gym.register(
        id=env_id,
        entry_point=entry_point,
        kwargs={'frame_skip': frame_skip, 
                'task': task,
                'orcagym_addr': orcagym_addr, 
                'agent_names': [f"{agent_name}_{agent_id:03d}" for agent_id in range(agent_num)], 
                'time_step': time_step,
                'max_episode_steps': max_episode_steps, # 环境永不停止，agent有最大步数
                'render_mode': "human",
                'render_remote': render_remote,
                'env_id': env_id},
        max_episode_steps=sys.maxsize,      # 环境永不停止，agent有最大步数
        reward_threshold=0.0,
    )
    return env_id


def make_env(orcagym_addr, env_name, env_index, agent_num, agent_name, task, entry_point, time_step, max_episode_steps, frame_skip, render_remote):
    def _init():
        # 注册环境，确保子进程中也能访问
        env_id = register_env(orcagym_addr, env_name, env_index, agent_num, agent_name, task, entry_point, time_step, max_episode_steps, frame_skip, render_remote)
        print("Registering environment with id: ", env_id)

        env = gym.make(env_id)
        seed = int(env_id[-3:])
        env.unwrapped.set_seed_value(seed)
        return env
    return _init

def training_model(model, total_timesteps, model_file):
    # 训练模型，每10亿步保存一次check point
    try:
        CKP_LEN = 1000000000
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
    finally:
        print(f"-----------------Save Model-----------------")
        model.save(model_file)
        
# def setup_model_ppo(env, env_num, agent_num, total_timesteps, start_episode, max_episode_steps, model_file, load_existing_model):
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     # 加载已有模型或初始化新模型
#     if os.path.exists(f"{model_file}.zip") and load_existing_model:
#         model = PPO.load(model_file, env=env, device=device)
#     else:
#         # 定义自定义策略网络
#         policy_kwargs = dict(
#             net_arch=dict(
#                 pi=[512, 256, 128],  # 策略网络结构
#                 vf=[512, 256, 128]   # 值函数网络结构
#             ),
#             ortho_init=True,
#             # activation_fn=nn.ReLU,
#             activation_fn=nn.ELU,
#             # log_std_init=2,  # 设置log_std的初始值
#         )
#         model = PPO("MultiInputPolicy", 
#                     env, 
#                     verbose=1, 
#                     learning_rate=0.0003, 
#                     n_steps=2048, 
#                     batch_size=2048, 
#                     gamma=0.99, 
#                     clip_range=0.2, 
#                     ent_coef=0.01,
#                     # target_kl=0.01,
#                     policy_kwargs=policy_kwargs, 
#                     device=device)
        
#     return model


def setup_model_ppo(env, 
                    env_num, 
                    agent_num, 
                    total_timesteps, 
                    start_episode, 
                    max_episode_steps, 
                    model_file, 
                    load_existing_model):
    """
    设置或加载 PPO 模型。

    参数:
    - env: 训练环境
    - env_num: 环境数量
    - agent_num: 每个环境中的智能体数量
    - total_timesteps: 总时间步数
    - start_episode: 开始的回合数
    - max_episode_steps: 每回合最大步数
    - model_file: 模型文件路径
    - load_existing_model: 是否加载现有模型标志

    返回:
    - model: 初始化的或加载的 PPO 模型
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 如果存在模型文件且指定加载现有模型，则加载模型
    if os.path.exists(f"{model_file}.zip") and load_existing_model:
        print(f"加载现有模型：{model_file}")
        model = PPO.load(model_file, env=env, device=device)
    else:
        # 定义自定义策略网络
        policy_kwargs = dict(
            net_arch=dict(
                pi=[1024, 512, 256],  # 策略网络结构
                vf=[1024, 512, 256]   # 值函数网络结构
            ),
            ortho_init=True,       # 正交初始化
            activation_fn=nn.ELU,  # 激活函数
        )

        # 根据环境数量和智能体数量计算批次大小和采样步数
        total_envs = env_num * agent_num
        n_steps = 256  # 每个环境采样步数
        batch_size = 65536  # 批次大小

        # 确保 batch_size 是 total_envs * n_steps 的因数
        assert (total_envs * n_steps) % batch_size == 0, \
            f"batch_size ({batch_size}) 应该是 total_envs * n_steps ({total_envs * n_steps}) 的因数。"

        model = PPO(
            policy="MultiInputPolicy",  # 多输入策略
            env=env, 
            verbose=1, 
            learning_rate=3e-4, 
            n_steps=n_steps, 
            batch_size=batch_size, 
            gamma=0.99, 
            clip_range=0.2, 
            ent_coef=0.01, 
            policy_kwargs=policy_kwargs, 
            device=device
        )

    # 打印模型摘要
    print(f"模型已设置：\n- Device: {device}\n- Batch Size: {model.batch_size}\n- n_steps: {model.n_steps}")

    return model

def setup_model_sac(env, env_num, agent_num, total_timesteps, start_episode, max_episode_steps, model_file, load_existing_model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if os.path.exists(f"{model_file}.zip") and load_existing_model:
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
            learning_starts= max_episode_steps * env_num * agent_num * start_episode,
            policy_kwargs=policy_kwargs, 
            device=device
        )

    return model

def setup_model_ddpg(env, env_num, agent_num, total_timesteps, start_episode, max_episode_steps, model_file, load_existing_model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if os.path.exists(f"{model_file}.zip") and load_existing_model:
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
            learning_starts= max_episode_steps * env_num * agent_num * start_episode,
            policy_kwargs=dict(net_arch=[256, 256, 256]),
        )

    return model


def setup_model_tqc(env, env_num, agent_num, total_timesteps, start_episode, max_episode_steps, model_file, load_existing_model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if os.path.exists(f"{model_file}.zip") and load_existing_model:
        model = TQC.load(model_file, env=env, device=device)
    else:        
        # https://arxiv.org/html/2312.13788v2
        # Open-Source Reinforcement Learning Environments Implemented in MuJoCo with Franka Manipulator。        
        # Training phase: Use HER
        goal_selection_strategy = "future"  # 选择 'future' 策略

        policy_kwargs = dict(
            net_arch=[256, 256, 128],
            n_critics=2,
            n_quantiles=25,
            activation_fn=torch.nn.ReLU
        )

        replay_buffer_class = HerReplayBuffer
        replay_buffer_kwargs = dict(
            n_sampled_goal = 10,    
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
            learning_starts= max_episode_steps * env_num * agent_num * start_episode,
            policy_kwargs=policy_kwargs, 
            device=device
        )

    return model



def generate_env_list(orcagym_addresses, subenv_num):
    orcagym_addr_list = []
    env_index_list = []
    render_remote_list = []
    
    for orcagym_addr in orcagym_addresses:
        for i in range(subenv_num):
            orcagym_addr_list.append(orcagym_addr)
            env_index_list.append(i)
            render_remote_list.append(True if i == 0 else False)

    return orcagym_addr_list, env_index_list, render_remote_list


def train_model(orcagym_addresses, subenv_num, agent_num, agent_name, task, entry_point, time_step, max_episode_steps, frame_skip, model_type, total_timesteps, start_episode, model_file, load_existing_model):
    try:
        print("simulation running... , orcagym_addresses: ", orcagym_addresses)

        env_name = "LeggedGym-v0"
        orcagym_addr_list, env_index_list, render_mode_list = generate_env_list(orcagym_addresses, subenv_num)
        env_num = len(orcagym_addr_list)
        print("env num: ", env_num)
        env_fns = [make_env(orcagym_addr, env_name, env_index, agent_num, agent_name, task, entry_point, time_step, max_episode_steps, frame_skip, render_remote) for orcagym_addr, env_index, render_remote in zip(orcagym_addr_list, env_index_list, render_mode_list)]
        env = SubprocVecEnvMA(env_fns, agent_num)

        print("Start Simulation!")
        if model_type == "ppo":
            model = setup_model_ppo(env, env_num, agent_num, total_timesteps, start_episode, max_episode_steps, model_file, load_existing_model)
        elif model_type == "tqc":
            model = setup_model_tqc(env, env_num, agent_num, total_timesteps, start_episode, max_episode_steps, model_file, load_existing_model)
        elif model_type == "sac":
            model = setup_model_sac(env, env_num, agent_num, total_timesteps, start_episode, max_episode_steps, model_file, load_existing_model)
        elif model_type == "ddpg":
            model = setup_model_ddpg(env, env_num, agent_num, total_timesteps, start_episode, max_episode_steps, model_file, load_existing_model)
        else:
            raise ValueError("Invalid model type")
        
        training_model(model, total_timesteps, model_file)

    finally:
        print("退出仿真环境")
        print(f"-----------------Save Model-----------------")
        model.save(model_file)
        env.close()

def test_model(orcagym_addresses, agent_num, agent_name, task, entry_point, time_step, max_episode_steps, frame_skip, model_type, model_file):
    try:
        print("simulation running... , orcagym_addr: ", orcagym_addresses)

        env_name = "LeggedGym-v0"
        orcagym_addr_list, env_index_list, render_mode_list = generate_env_list(orcagym_addresses, 1)
        env_num = len(orcagym_addr_list)
        print("env num: ", env_num)
        env_fns = [make_env(orcagym_addr, env_name, env_index, agent_num, agent_name, task, entry_point, time_step, max_episode_steps, frame_skip, render_remote) for orcagym_addr, env_index, render_remote in zip(orcagym_addr_list, env_index_list, render_mode_list)]
        env = SubprocVecEnvMA(env_fns, agent_num)

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

        testing_model(env, agent_num, model, time_step, max_episode_steps, frame_skip)
    except KeyboardInterrupt:
        print("退出仿真环境")
        env.close()

def _segment_observation(observation, agent_num):
    # 将观测数据分割成多个agent的数据
    obs_list = []
    for i in range(agent_num):
        agent_obs = {}
        for key, value in observation.items():
            agent_value_len = len(value) // agent_num
            agent_obs[key] = value[i * agent_value_len : (i + 1) * agent_value_len]
        obs_list.append(agent_obs)
    return obs_list

def _output_test_info(test, total_rewards, rewards, dones, infos):
    print(f"----------------Test: {test}----------------")
    print("Total Reward: ", total_rewards)
    print("Reward: ", rewards)
    print("Done: ", dones)
    print("is_success: ", [agent_info["is_success"] for agent_info in infos])
    print("---------------------------------------")

def testing_model(env : SubprocVecEnvMA, agent_num, model, time_step, max_episode_steps, frame_skip):
    # 测试模型
    observations = env.reset()
    test = 0
    total_rewards = np.zeros(agent_num)
    step = 0
    dt = time_step * frame_skip
    print("Start Testing!")
    try:
        while True:
            step += 1
            start_time = datetime.now()

            obs_list = _segment_observation(observations, agent_num)
            action_list = []
            for agent_obs in obs_list:
                # predict_start = datetime.now()
                action, _states = model.predict(agent_obs, deterministic=True)
                action_list.append(action)
                # predict_time = datetime.now() - predict_start
                # print("Predict Time: ", predict_time.total_seconds(), flush=True)

            action = np.concatenate(action_list).flatten()
            # print("action: ", action)
            # setp_start = datetime.now()
            observations, rewards, dones, infos = env.step(action)

            # print("obs, reward, terminated, truncated, info: ", observation, reward, terminated, truncated, info)

            env.render()
            # step_time = datetime.now() - setp_start
            # print("Step Time: ", step_time.total_seconds(), flush=True)

            total_rewards += rewards

            # 
            elapsed_time = datetime.now() - start_time
            if elapsed_time.total_seconds() < dt:
                time.sleep(dt - elapsed_time.total_seconds())

            if step == max_episode_steps:
                _output_test_info(test, total_rewards, rewards, dones, infos)
                step = 0
                test += 1
                total_rewards = np.zeros(agent_num)
            
    finally:
        print("退出仿真环境")
        env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run multiple instances of the script with different gRPC addresses.')
    parser.add_argument('--orcagym_addresses', type=str, nargs='+', default=['localhost:50051'], help='The gRPC addresses to connect to')
    parser.add_argument('--subenv_num', type=int, default=1, help='The number of subenvs for each gRPC address')
    parser.add_argument('--agent_num', type=int, default=1, help='The number of agents for each subenv')
    parser.add_argument('--agent_name', type=str, default='go2', help='The name of the agent')
    parser.add_argument('--task', type=str, default='follow_command', help='The task to run')
    parser.add_argument('--model_type', type=str, default='ppo', help='The model to use (ppo/tqc/sac/ddpg)')
    parser.add_argument('--run_mode', type=str, default='training', help='The mode to run (training or testing)')
    parser.add_argument('--model_file', type=str, help='The model file to save/load. If not provided, a new model file will be created while training')
    parser.add_argument('--load_existing_model', type=bool, default=False, help='Load existing model')
    parser.add_argument('--training_episode', type=int, default=100, help='The number of training episodes for each agent')
    parser.add_argument('--start_her_episode', type=float, default=1.0, help='Before start HER training, run each agent for some episodes to get experience')
    args = parser.parse_args()

    TIME_STEP = LeggedEnvConfig["TIME_STEP"]

    FRAME_SKIP_REALTIME = LeggedEnvConfig["FRAME_SKIP_REALTIME"]
    FRAME_SKIP_SHORT = LeggedEnvConfig["FRAME_SKIP_SHORT"]
    FRAME_SKIP_LONG = LeggedEnvConfig["FRAME_SKIP_LONG"]

    EPISODE_TIME_VERY_SHORT = LeggedEnvConfig["EPISODE_TIME_VERY_SHORT"]
    EPISODE_TIME_SHORT = LeggedEnvConfig["EPISODE_TIME_SHORT"]
    EPISODE_TIME_LONG = LeggedEnvConfig["EPISODE_TIME_LONG"]

    orcagym_addresses = args.orcagym_addresses
    subenv_num = args.subenv_num
    agent_num = args.agent_num
    agent_name = args.agent_name
    task = args.task
    model_type = args.model_type
    run_mode = args.run_mode
    load_existing_model = args.load_existing_model
    training_episode = args.training_episode
    start_her_episode = args.start_her_episode

    entry_point = 'envs.legged_gym.legged_gym_env:LeggedGymEnv'

    if task == 'stand' or task == 'move_forward' or task == 'no_action' or task == 'follow_command':
        frame_skip = FRAME_SKIP_SHORT # FRAME_SKIP_REALTIME
        max_episode_steps = int(1 / (TIME_STEP * frame_skip) * EPISODE_TIME_LONG)
    else:
        raise ValueError("Invalid task")

    total_timesteps = training_episode * subenv_num * agent_num * max_episode_steps

    if args.model_file is not None:
        model_file = args.model_file
    elif run_mode == "training" and not load_existing_model:
        formatted_now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        model_file = f"legged_{task}_{model_type}_{subenv_num}_{agent_num}_{training_episode}_model_{formatted_now}"
    else:
        raise ValueError("Invalid model file! Please provide a model file for testing, or set `load_existing_model` to False for training")


    if run_mode == "training":
        print("Start Training! task: ", task, " subenv_num: ", subenv_num, " agent_num: ", agent_num, " agent_name: ", agent_name)
        print("Model Type: ", model_type, " Total Timesteps: ", total_timesteps, " HER Start Episode: ", start_her_episode)
        print("Max Episode Steps: ", max_episode_steps, " Frame Skip: ", frame_skip)
        train_model(orcagym_addresses, subenv_num, agent_num, agent_name, task, entry_point, TIME_STEP, max_episode_steps, frame_skip, model_type, total_timesteps, start_her_episode, model_file, load_existing_model)
    elif run_mode == "testing":
        test_model(orcagym_addresses, agent_num, agent_name, task, entry_point, TIME_STEP, max_episode_steps, frame_skip, model_type, model_file)    
    else:
        raise ValueError("Invalid run mode")
