
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
import os
import sys
import time
from envs.legged_gym.legged_config import LeggedRobotConfig

from stable_baselines3.common.callbacks import BaseCallback
class SnapshotCallback(BaseCallback):
    def __init__(self, 
                 save_interval : int, 
                 save_path : str,
                 model_nstep : int,
                 verbose=0):
        super().__init__(verbose)
        self.save_interval = save_interval  # 保存间隔迭代数
        self.steps_count = 0                # 步数计数器
        self.iteration_count = 0            # 迭代计数器
        self.save_path = save_path          # 保存路径前缀
        self.model_nstep = model_nstep      # 模型每迭代执行步数

    def _on_step(self) -> bool:
        # 每步递增计数器
        self.steps_count += 1
        if self.steps_count % self.model_nstep == 0:
            self.iteration_count += 1
        # print("callback step: ", self.steps_count, " iteration: ", self.iteration_count)
        # 满足间隔条件时保存
        if self.iteration_count % self.save_interval == 0 and self.steps_count % self.model_nstep == 0:
            model_path = f"{self.save_path}_iteration_{self.iteration_count}.zip"
            self.logger.info(f"保存模型到 {model_path}")
            self.model.save(model_path)
        return True  # 确保训练继续

def register_env(
    orcagym_addr: str,
    env_name: str,
    env_index: int,
    agent_num: int,
    agent_name: str,
    task: str,
    run_mode: str,
    nav_ip: str,
    entry_point: str,
    time_step: float,
    max_episode_steps: int,
    frame_skip: int,
    is_subenv: bool,
    height_map_file: str
) -> str:
    orcagym_addr_str = orcagym_addr.replace(":", "-")
    env_id = env_name + "-OrcaGym-" + orcagym_addr_str + f"-{env_index:03d}"
    gym.register(
        id=env_id,
        entry_point=entry_point,
        kwargs={
            'frame_skip': frame_skip,
            'task': task,
            'orcagym_addr': orcagym_addr,
            'agent_names': [f"{agent_name}_{agent_id:03d}" for agent_id in range(agent_num)],
            'time_step': time_step,
            'max_episode_steps': max_episode_steps,  # 环境永不停止，agent有最大步数
            'render_mode': "human",
            'is_subenv': is_subenv,
            'height_map_file': height_map_file,
            'run_mode': run_mode,
            'ip': nav_ip,
            'env_id': env_id
        },
        max_episode_steps=sys.maxsize,  # 环境永不停止，agent有最大步数
        reward_threshold=0.0,
    )
    return env_id


def make_env(
    orcagym_addr: str, 
    env_name: str, 
    env_index: int, 
    agent_num: int, 
    agent_name: str, 
    task: str, 
    run_mode: str, 
    nav_ip: str, 
    entry_point: str, 
    time_step: float, 
    max_episode_steps: int, 
    frame_skip: int, 
    is_subenv: bool, 
    height_map_file: str
) -> callable:
    def _init():
        # 注册环境，确保子进程中也能访问
        env_id = register_env(
            orcagym_addr=orcagym_addr, 
            env_name=env_name, 
            env_index=env_index, 
            agent_num=agent_num, 
            agent_name=agent_name, 
            task=task, 
            run_mode=run_mode, 
            nav_ip=nav_ip, 
            entry_point=entry_point, 
            time_step=time_step, 
            max_episode_steps=max_episode_steps, 
            frame_skip=frame_skip, 
            is_subenv=is_subenv, 
            height_map_file=height_map_file
        )
        print("Registering environment with id: ", env_id)

        env = gym.make(env_id)
        seed = int(env_id[-3:])
        env.unwrapped.set_seed_value(seed)
        return env
    return _init

def training_model(
    model : PPO, 
    total_timesteps: int, 
    model_file: str,
):
    # 训练模型，每10亿步保存一次check point
    try:
        snapshot_callback = SnapshotCallback(save_interval=100, 
                                             save_path=model_file,
                                             model_nstep=model.n_steps)
        
        
        model.learn(total_timesteps=total_timesteps, 
                    callback=snapshot_callback)
    finally:
        print(f"-----------------Save Model-----------------")
        model.save(model_file)

def setup_model_ppo(
    env: SubprocVecEnvMA, 
    env_num: int, 
    agent_num: int, 
    agent_name: str,
    model_file: str, 
    load_existing_model: bool
) -> PPO:
    """
    设置或加载 PPO 模型。

    参数:
    - env: 训练环境
    - env_num: 环境数量
    - agent_num: 每个环境中的智能体数量
    - model_file: 模型文件路径
    - load_existing_model: 是否加载现有模型标志

    返回:
    - model: 初始化的或加载的 PPO 模型
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 如果存在模型文件且指定加载现有模型，则加载模型
    if os.path.exists(f"{model_file}") and load_existing_model:
        print(f"加载现有模型：{model_file}")
        model = PPO.load(model_file, env=env, device=device)
    else:
        print("初始化新模型")
        # 定义自定义策略网络
        
        robot_config = LeggedRobotConfig[agent_name]

        policy_kwargs = dict(
            net_arch=dict(
                pi=robot_config["pi"],  # 策略网络结构
                vf=robot_config["vf"]   # 值函数网络结构
            ),
            ortho_init=True,       # 正交初始化
            activation_fn=nn.ELU,  # 激活函数
        )

        # 根据环境数量和智能体数量计算批次大小和采样步数
        total_envs = env_num * agent_num

        n_steps = robot_config["n_steps"]  # 每个环境采样步数

        batch_size = robot_config["batch_size"]  # 批次大小

        # 确保 batch_size 是 total_envs * n_steps 的因数
        if (total_envs * n_steps) % batch_size != 0:
            Warning(f"batch_size ({batch_size}) 应该是 total_envs * n_steps ({total_envs * n_steps}) 的因数。")
            batch_size = (total_envs * n_steps) // 4  # 设置为总环境数量的四分之一
            

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
            device=device,
            tensorboard_log="./ppo_tensorboard/",  # TensorBoard 日志目录
        )

    # 打印模型摘要
    print(f"模型已设置：\n- Device: {device}\n- Total Envs: {total_envs}\n- Batch Size: {model.batch_size}\n- n_steps: {model.n_steps}")

    return model



def generate_env_list(orcagym_addresses, subenv_num):
    orcagym_addr_list = []
    env_index_list = []
    is_subenv_list = []
    
    for orcagym_addr in orcagym_addresses:
        for i in range(subenv_num):
            orcagym_addr_list.append(orcagym_addr)
            env_index_list.append(i)
            is_subenv_list.append(False if i == 0 else True)

    return orcagym_addr_list, env_index_list, is_subenv_list


def train_model(
    orcagym_addresses: str,
    subenv_num: int,
    agent_num: int,
    agent_name: str,
    task: str,
    entry_point: str,
    time_step: float,
    max_episode_steps: int,
    frame_skip: int,
    model_type: str,
    total_timesteps: int,
    start_her_episode: int,
    model_file: str,
    height_map_file: str,
    load_existing_model: bool,
):
    try:
        print("simulation running... , orcagym_addresses: ", orcagym_addresses)

        env_name = "LeggedGym-v0"
        orcagym_addr_list, env_index_list, render_mode_list = generate_env_list(orcagym_addresses, subenv_num)
        env_num = len(orcagym_addr_list)
        print("env num: ", env_num)
        env_fns = [
            make_env(
                orcagym_addr=orcagym_addr,
                env_name=env_name,
                env_index=env_index,
                agent_num=agent_num,
                agent_name=agent_name,
                task=task,
                run_mode="training",
                nav_ip="localhost",
                entry_point=entry_point,
                time_step=time_step,
                max_episode_steps=max_episode_steps,
                frame_skip=frame_skip,
                is_subenv=is_subenv,
                height_map_file=height_map_file,
            )
            for orcagym_addr, env_index, is_subenv in zip(orcagym_addr_list, env_index_list, render_mode_list)
        ]
        env = SubprocVecEnvMA(env_fns, agent_num)

        print("Start Simulation!")
        if model_type == "ppo":
            model = setup_model_ppo(
                env=env,
                env_num=env_num,
                agent_num=agent_num,
                agent_name=agent_name,
                model_file=model_file,
                load_existing_model=load_existing_model,
            )
        else:
            raise ValueError("Invalid model type")

        training_model(model, total_timesteps, model_file)

    finally:
        print("退出仿真环境")
        print(f"-----------------Save Model-----------------")
        model.save(model_file)
        env.close()

def test_model(
    orcagym_addresses: str,
    agent_num: int,
    agent_name: str,
    task: str,
    run_mode: str,
    nav_ip: str,
    entry_point: str,
    time_step: float,
    max_episode_steps: int,
    frame_skip: int,
    model_type: str,
    model_file: str,
    height_map_file: str
):
    try:
        print("simulation running... , orcagym_addr: ", orcagym_addresses)

        env_name = "LeggedGym-v0"
        orcagym_addr_list, env_index_list, render_mode_list = generate_env_list(orcagym_addresses, 1)
        env_num = len(orcagym_addr_list)
        print("env num: ", env_num)
        env_fns = [
            make_env(
                orcagym_addr=orcagym_addr,
                env_name=env_name,
                env_index=env_index,
                agent_num=agent_num,
                agent_name=agent_name,
                task=task,
                run_mode=run_mode,
                nav_ip=nav_ip,
                entry_point=entry_point,
                time_step=time_step,
                max_episode_steps=max_episode_steps,
                frame_skip=frame_skip,
                is_subenv=is_subenv,
                height_map_file=height_map_file,
            )
            for orcagym_addr, env_index, is_subenv in zip(orcagym_addr_list, env_index_list, render_mode_list)
        ]
        env = SubprocVecEnvMA(env_fns, agent_num)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if model_type == "ppo":
            model: PPO = PPO.load(model_file, env=env, device=device)
        else:
            raise ValueError("Invalid model type")
        
        testing_model(
            env=env,
            agent_num=agent_num,
            model=model,
            time_step=time_step,
            max_episode_steps=max_episode_steps,
            frame_skip=frame_skip
        )

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

def _output_test_info(
    test: int,
    total_rewards: np.ndarray,
    rewards: np.ndarray,
    dones: np.ndarray,
    infos: list
):
    print(f"----------------Test: {test}----------------")
    print("Total Reward: ", total_rewards)
    print("Reward: ", rewards)
    print("Done: ", dones)
    print("is_success: ", [agent_info["is_success"] for agent_info in infos])
    print("---------------------------------------")

def testing_model(
    env: SubprocVecEnvMA, 
    agent_num: int, 
    model, 
    time_step: float, 
    max_episode_steps: int, 
    frame_skip: int
):
    """
    测试模型。

    参数:
    - env: SubprocVecEnvMA
        测试环境
    - agent_num: int
        智能体数量
    - model: Any
        训练好的模型
    - time_step: float
        时间步长
    - max_episode_steps: int
        每回合最大步数
    - frame_skip: int
        帧跳跃数
    """
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
                action, _states = model.predict(agent_obs, deterministic=True)
                action_list.append(action)

            action = np.concatenate(action_list).flatten()
            observations, rewards, dones, infos = env.step(action)

            env.render()

            total_rewards += rewards

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
        