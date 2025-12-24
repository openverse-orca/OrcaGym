
import gymnasium as gym
from stable_baselines3 import PPO
from gymnasium.envs.registration import register
from datetime import datetime
import torch
import torch.nn as nn
# from stable_baselines3.common.vec_env import SubprocVecEnv
from orca_gym.environment.async_env import OrcaGymAsyncSubprocVecEnv
from sb3_contrib import TQC
from stable_baselines3.her import GoalSelectionStrategy, HerReplayBuffer
from stable_baselines3.common.noise import NormalActionNoise
import numpy as np
import os
import sys
import time

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import get_schedule_fn, get_linear_fn

from envs.legged_gym.legged_config import LeggedRobotConfig, LeggedObsConfig, CurriculumConfig, LeggedEnvConfig

from orca_gym.log.orca_log import get_orca_logger
_logger = get_orca_logger()


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

class CurriculumCallback(BaseCallback):
    def __init__(self, 
                 save_path : str,
                 model_nstep : int,
                 curriculum_list : list[dict[str, int]],
                 verbose=0):
        super().__init__(verbose)
        self.steps_count = 0                # 步数计数器
        self.iteration_count = 0            # 迭代计数器
        self.save_path = save_path          # 保存路径前缀
        self.model_nstep = model_nstep      # 模型每迭代执行步数
        self.curriculum_list = curriculum_list        # 课程名称和持续迭代数
        self.curriculum_index = 0
        
    def _on_step(self) -> bool:
        # 每步递增计数器
        self.steps_count += 1
        if self.steps_count % self.model_nstep == 0:
            self.iteration_count += 1
        # print("callback step: ", self.steps_count, " iteration: ", self.iteration_count)
        # 满足间隔条件时保存
        if self.iteration_count > 0 and self.iteration_count % self.curriculum_list[self.curriculum_index]["milestone"] == 0 and self.steps_count % self.model_nstep == 0:
            # 保存模型
            model_path = f"{self.save_path}_{self.curriculum_list[self.curriculum_index]['name']}_iteration_{self.iteration_count}.zip"
            self.logger.info(f"保存模型到 {model_path}")
            self.model.save(model_path)

            # 更新课程
            self.curriculum_index += 1
            if self.curriculum_index >= len(self.curriculum_list):
                self.curriculum_index = 0
            self.logger.info(f"更新课程: {self.curriculum_list[self.curriculum_index]['name']}")
            self.model.env.setup_curriculum(self.curriculum_list[self.curriculum_index]["name"])
            self.model.env.reset()

            
        return True  # 确保训练继续


def register_env(
    orcagym_addr: str,
    env_name: str,
    env_index: int,
    agent_num: int,
    agent_name: str,
    task: str,
    run_mode: str,
    entry_point: str,
    time_step: float,
    max_episode_steps: int,
    frame_skip: int,
    action_skip: int,
    is_subenv: bool,
    height_map_file: str,
    render_mode: str
) -> str:
    orcagym_addr_str = orcagym_addr.replace(":", "-")
    env_id = env_name + "-OrcaGym-" + orcagym_addr_str + f"-{env_index:03d}"
    gym.register(
        id=env_id,
        entry_point=entry_point,
        kwargs={
            'frame_skip': frame_skip,
            'action_skip': action_skip,
            'task': task,
            'orcagym_addr': orcagym_addr,
            'agent_names': [f"{agent_name}_{agent_id:03d}" for agent_id in range(agent_num)],
            'time_step': time_step,
            'max_episode_steps': max_episode_steps,  # 环境永不停止，agent有最大步数
            'render_mode': render_mode,
            'is_subenv': is_subenv,
            'height_map_file': height_map_file,
            'run_mode': run_mode,
            'env_id': env_id,
            'robot_config': LeggedRobotConfig[agent_name],
            'legged_obs_config': LeggedObsConfig,
            'curriculum_config': CurriculumConfig,
            'legged_env_config': LeggedEnvConfig,
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
    entry_point: str, 
    time_step: float, 
    max_episode_steps: int, 
    frame_skip: int, 
    action_skip: int,
    is_subenv: bool, 
    height_map_file: str,
    render_mode: str
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
            entry_point=entry_point, 
            time_step=time_step, 
            max_episode_steps=max_episode_steps, 
            frame_skip=frame_skip, 
            action_skip=action_skip,
            is_subenv=is_subenv, 
            height_map_file=height_map_file,
            render_mode=render_mode
        )
        _logger.info(f"Registering environment with id:  {env_id}")

        env = gym.make(env_id)
        seed = int(env_id[-3:])
        env.unwrapped.set_seed_value(seed)
        return env
    return _init

def training_model(
    model : PPO, 
    total_timesteps: int, 
    model_file: str,
    curriculum_list: list[dict[str, int]],
):
    try:
        snapshot_callback = SnapshotCallback(save_interval=100, 
                                             save_path=model_file,
                                             model_nstep=model.n_steps)

        curriculum_callback = CurriculumCallback(save_path=model_file, 
                                                 model_nstep=model.n_steps, 
                                                 curriculum_list=curriculum_list)
        
        
        model.learn(total_timesteps=total_timesteps, 
                    callback=[snapshot_callback, curriculum_callback])
    finally:
        _logger.info(f"-----------------Save Model-----------------")
        model.save(model_file)

def setup_model_ppo(
    env: OrcaGymAsyncSubprocVecEnv, 
    env_num: int, 
    agent_num: int, 
    agent_config : dict,
    model_file: str, 
) -> PPO:
    """
    设置或加载 PPO 模型。
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 根据环境数量和智能体数量计算批次大小和采样步数
    total_envs = env_num * agent_num
    n_steps = agent_config["n_steps"]
    batch_size = agent_config["batch_size"]
    
    # 确保 batch_size 是 total_envs * n_steps 的因数
    min_batch_size = 32
    if (total_envs * n_steps) % batch_size != 0:
        suitable_batch_size = (total_envs * n_steps) // 4
        suitable_batch_size = max(min_batch_size, suitable_batch_size)
        _logger.warning(f"Warning: batch_size ({batch_size}) 应该是 total_envs * n_steps ({total_envs * n_steps}) 的因数。")
        _logger.info(f"自动调整为: {suitable_batch_size}")
        batch_size = suitable_batch_size
        
    # 如果存在模型文件且指定加载现有模型，则加载模型
    if os.path.exists(f"{model_file}"):
        _logger.info(f"加载现有模型：{model_file}")
        model = PPO.load(model_file, env=env, device=device)
    else:
        _logger.info("初始化新模型")
        
        # 处理学习率调度
        if isinstance(agent_config["learning_rate"], dict):
            lr_schedule = get_linear_fn(
                agent_config["learning_rate"]["initial_value"],
                agent_config["learning_rate"]["final_value"],
                agent_config["learning_rate"]["end_fraction"]
            )
            _logger.info(f"学习率调度: {agent_config['learning_rate']}")
        else:
            lr_schedule = agent_config["learning_rate"]
            
        # 处理clip_range调度
        if isinstance(agent_config["clip_range"], dict):
            clip_range_schedule = get_linear_fn(
                agent_config["clip_range"]["initial_value"],
                agent_config["clip_range"]["final_value"],
                agent_config["clip_range"]["end_fraction"]
            )
            _logger.info(f"clip_range调度: {agent_config['clip_range']}")
        else:
            clip_range_schedule = agent_config["clip_range"]
        
        # 定义自定义策略网络
        policy_kwargs = dict(
            net_arch=dict(
                pi=agent_config["pi"],
                vf=agent_config["vf"]
            ),
            ortho_init=True,
            activation_fn=nn.SiLU,  # 改为更合适的激活函数
        )

        model = PPO(
            policy="MultiInputPolicy",
            env=env, 
            verbose=1, 
            learning_rate=lr_schedule, 
            n_steps=n_steps, 
            batch_size=batch_size, 
            gamma=agent_config["gamma"], 
            clip_range=clip_range_schedule, 
            ent_coef=agent_config["ent_coef"], 
            max_grad_norm=agent_config["max_grad_norm"],
            policy_kwargs=policy_kwargs, 
            device=device,
            tensorboard_log="./ppo_tensorboard/",
        )

    # 打印模型摘要
    _logger.info(f"模型已设置：\n- Device: {device}\n- Total Envs: {total_envs}")
    _logger.info(f"- Batch Size: {model.batch_size}\n- n_steps: {model.n_steps}")
    _logger.info(f"- Learning Rate: {agent_config['learning_rate']}")
    _logger.info(f"- Activation Function: {policy_kwargs['activation_fn'].__name__}")
    _logger.info(f"- Policy Network: {agent_config['pi']}")
    _logger.info(f"- Value Network: {agent_config['vf']}")

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
    agent_config: dict,
    task: str,
    entry_point: str,
    time_step: float,
    max_episode_steps: int,
    frame_skip: int,
    action_skip: int,
    total_timesteps: int,
    model_file: str,
    height_map_file: str,
    curriculum_list: list[dict[str, int]],
    render_mode: str,
):
    model = None
    env = None
    try:
        _logger.info(f"simulation running... , orcagym_addresses:  {orcagym_addresses}")

        env_name = "LeggedGym-v0"
        orcagym_addr_list, env_index_list, render_mode_list = generate_env_list(orcagym_addresses, subenv_num)
        env_num = len(orcagym_addr_list)
        _logger.info(f"env num:  {env_num}")
        env_fns = [
            make_env(
                orcagym_addr=orcagym_addr,
                env_name=env_name,
                env_index=env_index,
                agent_num=agent_num,
                agent_name=agent_name,
                task=task,
                run_mode="training",
                entry_point=entry_point,
                time_step=time_step,
                max_episode_steps=max_episode_steps,
                frame_skip=frame_skip,
                action_skip=action_skip,
                is_subenv=is_subenv,
                height_map_file=height_map_file,
                render_mode=render_mode,
            )
            for orcagym_addr, env_index, is_subenv in zip(orcagym_addr_list, env_index_list, render_mode_list)
        ]
        env = OrcaGymAsyncSubprocVecEnv(env_fns, agent_num)
        env.setup_curriculum(curriculum_list[0]["name"])

        _logger.info("Start Simulation!")
        model = setup_model_ppo(
            env=env,
            env_num=env_num,
            agent_num=agent_num,
            agent_config=agent_config,
            model_file=model_file,
        )

        training_model(model, total_timesteps, model_file, curriculum_list)

    finally:
        _logger.info("退出仿真环境")
        if model is not None:
            _logger.info(f"-----------------Save Model-----------------")
            model.save(model_file)
        if env is not None:
            env.close()

def test_model(
    orcagym_addresses: str,
    agent_num: int,
    agent_name: str,
    task: str,
    run_mode: str,
    entry_point: str,
    time_step: float,
    max_episode_steps: int,
    frame_skip: int,
    action_skip: int,
    model_file: str,
    height_map_file: str,
    curriculum_list: list[dict[str, int]],
    render_mode: str,
    ):
    try:
        _logger.info(f"simulation running... , orcagym_addr:  {orcagym_addresses}")

        env_name = "LeggedGym-v0"
        orcagym_addr_list, env_index_list, render_mode_list = generate_env_list(orcagym_addresses, 1)
        env_num = len(orcagym_addr_list)
        _logger.info(f"env num:  {env_num}")
        env_fns = [
            make_env(
                orcagym_addr=orcagym_addr,
                env_name=env_name,
                env_index=env_index,
                agent_num=agent_num,
                agent_name=agent_name,
                task=task,
                run_mode=run_mode,
                entry_point=entry_point,
                time_step=time_step,
                max_episode_steps=max_episode_steps,
                frame_skip=frame_skip,
                action_skip=action_skip,
                is_subenv=is_subenv,
                height_map_file=height_map_file,
                render_mode=render_mode,
            )
            for orcagym_addr, env_index, is_subenv in zip(orcagym_addr_list, env_index_list, render_mode_list)
        ]
        env = OrcaGymAsyncSubprocVecEnv(env_fns, agent_num)
        env.setup_curriculum(curriculum_list[0]["name"])
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        model: PPO = PPO.load(model_file, env=env, device=device)

        testing_model(
            env=env,
            agent_num=agent_num,
            model=model,
            time_step=time_step,
            max_episode_steps=max_episode_steps,
            frame_skip=frame_skip,
            action_skip=action_skip
        )

    except KeyboardInterrupt:
        _logger.info("退出仿真环境")
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
    _logger.info(f"----------------Test: {test}----------------")
    _logger.info(f"Total Reward:  {total_rewards}")
    _logger.info(f"Reward:  {rewards}")
    _logger.info(f"Done:  {dones}")
    _logger.info(f"is_success:  {[agent_info['is_success'] for agent_info in infos]}")
    _logger.info("---------------------------------------")

def testing_model(
    env: OrcaGymAsyncSubprocVecEnv, 
    agent_num: int, 
    model, 
    time_step: float, 
    max_episode_steps: int, 
    frame_skip: int,
    action_skip: int
):
    """
    测试模型。

    参数:
    - env: OrcaGymAsyncSubprocVecEnv
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
    dt = time_step * frame_skip * action_skip
    _logger.info("Start Testing!")
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

            # print("--------------------------------")
            # result = [True if a < 1 and a > -1 else False for a in action]
            # if any(result):
            #     print("action: ", action)
            #     print("observations: ", observations)
            #     print("rewards: ", rewards)
            #     print("dones: ", dones)
            #     print("infos: ", infos)

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
        _logger.info("退出仿真环境")
        env.close()
        