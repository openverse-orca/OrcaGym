
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
    model, 
    total_timesteps: int, 
    model_file: str
):
    # 训练模型，每10亿步保存一次check point
    try:
        # CKP_LEN = 1000000000
        CKP_LEN = 50000000
        # CKP_LEN = 100000

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

def setup_model_ppo(
    env: SubprocVecEnvMA, 
    env_num: int, 
    agent_num: int, 
    total_timesteps: int, 
    start_her_episode: int, 
    max_episode_steps: int, 
    model_file: str, 
    load_existing_model: bool
) -> PPO:
    """
    设置或加载 PPO 模型。

    参数:
    - env: 训练环境
    - env_num: 环境数量
    - agent_num: 每个环境中的智能体数量
    - total_timesteps: 总时间步数
    - start_her_episode: 开始的回合数
    - max_episode_steps: 每回合最大步数
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

        n_steps = 128  # 每个环境采样步数

        batch_size = total_envs * 16  # 批次大小

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
            device=device,
            tensorboard_log="./ppo_tensorboard/",  # TensorBoard 日志目录
        )

    # 打印模型摘要
    print(f"模型已设置：\n- Device: {device}\n- Batch Size: {model.batch_size}\n- n_steps: {model.n_steps}")

    return model

def setup_model_sac(
    env: SubprocVecEnvMA, 
    env_num: int, 
    agent_num: int, 
    total_timesteps: int, 
    start_her_episode: int, 
    max_episode_steps: int, 
    model_file: str, 
    load_existing_model: bool
) -> SAC:
    """
    设置或加载 SAC 模型。

    参数:
    - env: SubprocVecEnvMA
        训练环境（应为 VecEnv 类型，支持并行环境）
    - env_num: int
        环境数量
    - agent_num: int
        每个环境中的智能体数量
    - total_timesteps: int
        总时间步数
    - start_her_episode: int
        开始的回合数
    - max_episode_steps: int
        每回合最大步数
    - model_file: str
        模型文件路径
    - load_existing_model: bool
        是否加载现有模型标志

    返回:
    - model: SAC
        初始化的或加载的 SAC 模型
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 如果存在模型文件且指定加载现有模型，则加载模型
    if os.path.exists(f"{model_file}.zip") and load_existing_model:
        print(f"加载现有模型：{model_file}")
        model = SAC.load(model_file, env=env, device=device)
    else:
        # 定义自定义策略网络
        policy_kwargs = dict(
            net_arch=dict(
                pi=[1024, 512, 256],  # 策略网络结构
                qf=[1024, 512, 256]   # Q 函数网络结构
            ),
            activation_fn=nn.ELU,  # 激活函数
        )

        # 计算总环境数量
        total_envs = env_num * agent_num

        # SAC 参数设置
        buffer_size = 1000000  # 经验回放缓冲区大小
        batch_size = 256  # 从缓冲区中采样的批次大小

        model = SAC(
            policy="MultiInputPolicy",  # 多输入策略，适用于多智能体环境
            env=env, 
            verbose=1, 
            learning_rate=3e-4, 
            buffer_size=buffer_size, 
            learning_starts=max_episode_steps * env_num * agent_num * start_her_episode,
            batch_size=batch_size, 
            gamma=0.99, 
            tau=0.005, 
            ent_coef="auto",  # 自动调整熵系数
            target_update_interval=1,  # 目标网络更新间隔
            train_freq=(1, "step"),  # 每步训练一次
            gradient_steps=1,  # 每步进行一次梯度更新
            policy_kwargs=policy_kwargs, 
            device=device
        )

    # 打印模型摘要
    print(f"模型已设置：\n- Device: {device}\n- Batch Size: {model.batch_size}\n- Buffer Size: {model.replay_buffer.buffer_size}")

    return model

def setup_model_ddpg(
    env: SubprocVecEnvMA,
    env_num: int,
    agent_num: int,
    total_timesteps: int,
    start_her_episode: int,
    max_episode_steps: int,
    model_file: str,
    load_existing_model: bool
) -> DDPG:
    """
    设置或加载 DDPG 模型。

    参数:
    - env: SubprocVecEnvMA
        训练环境（应为 VecEnv 类型，支持并行环境）
    - env_num: int
        环境数量
    - agent_num: int
        每个环境中的智能体数量
    - total_timesteps: int
        总时间步数
    - start_her_episode: int
        开始的回合数
    - max_episode_steps: int
        每回合最大步数
    - model_file: str
        模型文件路径
    - load_existing_model: bool
        是否加载现有模型标志

    返回:
    - model: DDPG
        初始化的或加载的 DDPG 模型
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if os.path.exists(f"{model_file}.zip") and load_existing_model:
        print(f"加载现有模型：{model_file}")
        model = DDPG.load(model_file, env=env, device=device)
    else:
        print("初始化新模型")
        # 动作噪声
        n_actions = env.action_space.shape[0]
        noise_std = 0.2
        action_noise = NormalActionNoise(
            mean=np.zeros(n_actions), sigma=noise_std * np.ones(n_actions)
        )

        # 初始化 DDPG 模型
        model = DDPG(
            policy="MultiInputPolicy",
            env=env,
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
            learning_starts=max_episode_steps * env_num * agent_num * start_her_episode,
            policy_kwargs=dict(net_arch=[256, 256, 256]),
            device=device
        )

    # 打印模型摘要
    print(f"模型已设置：\n- Device: {device}\n- Batch Size: {model.batch_size}\n- Buffer Size: {model.replay_buffer.buffer_size}")

    return model


def setup_model_tqc(
    env: SubprocVecEnvMA, 
    env_num: int, 
    agent_num: int, 
    total_timesteps: int, 
    start_her_episode: int, 
    max_episode_steps: int, 
    model_file: str, 
    load_existing_model: bool
) -> TQC:
    """
    设置或加载 TQC 模型。

    参数:
    - env: SubprocVecEnvMA
        训练环境（应为 VecEnv 类型，支持并行环境）
    - env_num: int
        环境数量
    - agent_num: int
        每个环境中的智能体数量
    - total_timesteps: int
        总时间步数
    - start_her_episode: int
        开始的回合数
    - max_episode_steps: int
        每回合最大步数
    - model_file: str
        模型文件路径
    - load_existing_model: bool
        是否加载现有模型标志

    返回:
    - model: TQC
        初始化的或加载的 TQC 模型
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if os.path.exists(f"{model_file}.zip") and load_existing_model:
        model = TQC.load(model_file, env=env, device=device)
    else:
        # HER 策略
        goal_selection_strategy = "future"  # 选择 'future' 策略

        policy_kwargs = dict(
            net_arch=[256, 256, 128],
            n_critics=2,
            n_quantiles=25,
            activation_fn=torch.nn.ReLU
        )

        replay_buffer_class = HerReplayBuffer
        replay_buffer_kwargs = dict(
            n_sampled_goal=10,
            goal_selection_strategy=goal_selection_strategy
        )

        # 初始化 TQC 模型
        model = TQC(
            policy="MultiInputPolicy",
            env=env,
            replay_buffer_class=replay_buffer_class,
            replay_buffer_kwargs=replay_buffer_kwargs,
            verbose=1,
            learning_rate=0.001,  # 学习率
            buffer_size=1000000,  # 重播缓冲区大小
            batch_size=2048,  # 批量大小
            tau=0.05,
            gamma=0.95,  # 折扣因子
            learning_starts=max_episode_steps * env_num * agent_num * start_her_episode,
            policy_kwargs=policy_kwargs,
            device=device
        )

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
                total_timesteps=total_timesteps,
                start_her_episode=start_her_episode,
                max_episode_steps=max_episode_steps,
                model_file=model_file,
                load_existing_model=load_existing_model,
            )
        elif model_type == "tqc":
            model = setup_model_tqc(
                env=env,
                env_num=env_num,
                agent_num=agent_num,
                total_timesteps=total_timesteps,
                start_her_episode=start_her_episode,
                max_episode_steps=max_episode_steps,
                model_file=model_file,
                load_existing_model=load_existing_model,
            )
        elif model_type == "sac":
            model = setup_model_sac(
                env=env,
                env_num=env_num,
                agent_num=agent_num,
                total_timesteps=total_timesteps,
                start_her_episode=start_her_episode,
                max_episode_steps=max_episode_steps,
                model_file=model_file,
                load_existing_model=load_existing_model,
            )
        elif model_type == "ddpg":
            model = setup_model_ddpg(
                env=env,
                env_num=env_num,
                agent_num=agent_num,
                total_timesteps=total_timesteps,
                start_her_episode=start_her_episode,
                max_episode_steps=max_episode_steps,
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
        elif model_type == "tqc":
            model: TQC = TQC.load(model_file, env=env, device=device)
        elif model_type == "sac":
            model: SAC = SAC.load(model_file, env=env, device=device)
        elif model_type == "ddpg":
            model: DDPG = DDPG.load(model_file, env=env, device=device)
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
        