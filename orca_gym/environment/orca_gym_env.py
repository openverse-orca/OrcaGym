from os import path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
from numpy.typing import NDArray

import gymnasium as gym
from gymnasium import error, spaces
from gymnasium.spaces import Space

import asyncio
import sys

from orca_gym.log.orca_log import get_orca_logger
_logger = get_orca_logger()

from orca_gym import OrcaGymLocal
from orca_gym.protos.mjc_message_pb2_grpc import GrpcServiceStub 
from orca_gym.utils.rotations import mat2quat, quat2mat

from orca_gym import OrcaGymModel
from orca_gym import OrcaGymData

class RewardType:
    SPARSE = "sparse"
    DENSE = "dense"


class OrcaGymBaseEnv(gym.Env[NDArray[np.float64], NDArray[np.float32]]):
    """Superclass for all OrcaSim environments."""

    def __init__(
        self,
        frame_skip: int,
        orcagym_addr: str,
        agent_names: list[str],
        time_step: float,
        **kwargs
    ):
        """Base abstract class for OrcaSim based environments.

        Args:
            frame_skip: Number of MuJoCo simulation steps per gym `step()`.
            orcagym_addr: The address of the gRPC server.
            agent_names: The names of the agents in the environment.
            time_step: The time step of the simulation.

        Raises:
        """

        # 初始化GRPC通信管道，采用异步通信
        self.orcagym_addr = orcagym_addr
        self.channel = None
        self.stub = None
        self.gym = None
        self._agent_names = agent_names
        self.seed = 0
        self.loop = asyncio.get_event_loop()
        self.initialize_grpc()
        self.pause_simulation()  # 暂停仿真，Gym 采用被动模式，OrcaSim侧不执行周期循环
        self.frame_skip = frame_skip        
        self.set_time_step(time_step)  # 设置仿真时间步长

        self.model, self.data = self.initialize_simulation()
        
        self.reset_simulation() # 重置仿真环境

        self.init_qpos_qvel()

    # methods to override:
    # ----------------------------
    def step(
        self, action: NDArray[np.float32]
    ) -> Tuple[NDArray[np.float64], np.float64, bool, bool, Dict[str, np.float64]]:
        raise NotImplementedError

    def reset_model(self) -> tuple[dict, dict]:
        """
        Reset the robot degrees of freedom (qpos and qvel).
        Implement this in each subclass.
        """
        raise NotImplementedError

    def initialize_simulation(self) -> Tuple[OrcaGymModel, OrcaGymData]:
        """
        Initialize MuJoCo simulation data structures mjModel and mjData.
        """
        raise NotImplementedError

    def _step_orca_sim_simulation(self, ctrl, n_frames) -> None:
        """
        Step over the MuJoCo simulation.
        """
        raise NotImplementedError

    def render(self) -> Union[NDArray[np.float64], None]:
        """
        Render a frame from the MuJoCo simulation as specified by the render_mode.
        """
        raise NotImplementedError

    def generate_action_space(self, bounds : NDArray[np.float64]) -> Space:
        """
        生成环境的动作空间
        
        术语说明:
            - 动作空间 (Action Space): 强化学习中智能体可以执行的所有动作的集合
            - Box Space: 连续动作空间，每个维度有上下界限制
            - 动作: 发送给执行器的控制命令，通常是扭矩、位置或速度
        
        使用示例:
            ```python
            # 根据执行器控制范围生成动作空间
            ctrlrange = self.model.get_actuator_ctrlrange()
            self.action_space = self.generate_action_space(ctrlrange)
            # 动作空间形状: (nu,)，每个值在 [min, max] 范围内
            ```
        """
        low, high = 0.0, 0.0
        if len(bounds.T) > 0:
            low, high = bounds.T
        action_space = spaces.Box(low=low, high=high, dtype=np.float32)
        return action_space

    def generate_observation_space(self, obs: Union[Dict[str, Any], np.ndarray]) -> spaces.Space:
        """
        生成环境的观测空间
        
        术语说明:
            - 观测空间 (Observation Space): 强化学习中智能体能够观察到的状态信息集合
            - 观测 (Observation): 智能体在每个时间步接收到的状态信息
            - Dict Space: 字典类型的观测空间，包含多个子空间
            - Box Space: 连续观测空间，每个维度有上下界限制
        
        使用示例:
            ```python
            # 根据观测数据生成观测空间
            obs = self._get_obs()  # 获取示例观测
            self.observation_space = self.generate_observation_space(obs)
            # 观测空间可能是 Dict 或 Box，取决于 obs 的类型
            ```
        """
        if obs is None:
            raise ValueError("obs dictionary is None")

        if isinstance(obs, np.ndarray):
            # If obs is a numpy array, create a box space for it
            low = np.full(obs.shape, -np.finfo(np.float32).max, dtype=np.float32)
            high = np.full(obs.shape, np.finfo(np.float32).max, dtype=np.float32)
            return spaces.Box(low=low, high=high, dtype=np.float32)

        obs_space_dict = {}
        for obs_key, obs_data in obs.items():
            if isinstance(obs_data, np.ndarray):
                # print("obs key: ", obs_key, "dtype : ", obs_data.dtype)
                
                # 创建与 obs_data 形状相同的 low 和 high，并确保它们是 float32
                # 使用 float32 的有限边界值
                finite_max = np.finfo(np.float32).max
                low = np.full(obs_data.shape, -finite_max, dtype=np.float32)
                high = np.full(obs_data.shape, finite_max, dtype=np.float32)
                _logger.debug(f"low.dtype: {low.dtype}, high.dtype: {high.dtype}")
                
                obs_space_dict[obs_key] = spaces.Box(
                    low=low,
                    high=high,
                    dtype=obs_data.dtype
                )
            else:
                raise ValueError(f"Unsupported observation type: {type(obs_data)}")
                
        observation_space = spaces.Dict(obs_space_dict)
        return observation_space

    # -----------------------------
    def _get_reset_info(self) -> Dict[str, float]:
        """Function that generates the `info` that is returned during a `reset()`."""
        return {}

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        super().reset(seed=seed)

        if seed is not None:
            self.set_seed_value(seed)

        # mujoco.mj_resetData(self.model, self.data)
        self.reset_simulation()

        obs, info = self.reset_model()

        self.render()
        return obs, info

    def set_seed_value(self, seed=None):
        """设置随机数种子"""
        self.seed_value = seed
        self.np_random = np.random.RandomState(seed)
        return [seed]

    def _name_with_agent0(self, name: str) -> str:
        """为第一个智能体添加名称前缀"""
        if len(self._agent_names) > 0:
            return f"{self._agent_names[0]}_{name}"
        else:
            return name
        
    def _name_with_agent(self, agent_id: int, name: str) -> str:
        """为指定智能体添加名称前缀"""
        if len(self._agent_names) > 0:
            return f"{self._agent_names[agent_id]}_{name}"
        else:
            return name

    def body(self, name: str, agent_id = None) -> str:
        """获取带智能体前缀的 body 名称"""
        if agent_id == None:
            return self._name_with_agent0(name)
        else:
            return self._name_with_agent(agent_id, name)
        
    def joint(self, name: str, agent_id = None) -> str:
        """获取带智能体前缀的关节名称"""
        if agent_id == None:
            return self._name_with_agent0(name)
        else:
            return self._name_with_agent(agent_id, name)
        
    
    def actuator(self, name: str, agent_id = None) -> str:
        """获取带智能体前缀的执行器名称"""
        if agent_id == None:
            return self._name_with_agent0(name)
        else:
            return self._name_with_agent(agent_id, name)
        
    
    def site(self, name: str, agent_id = None) -> str:
        """获取带智能体前缀的 site 名称"""
        if agent_id == None:
            return self._name_with_agent0(name)
        else:
            return self._name_with_agent(agent_id, name)
        
    
    def mocap(self, name: str, agent_id = None) -> str:
        """获取带智能体前缀的 mocap 名称"""
        if agent_id == None:
            return self._name_with_agent0(name)
        else:
            return self._name_with_agent(agent_id, name)

    def sensor(self, name: str, agent_id = None) -> str:
        """获取带智能体前缀的传感器名称"""
        if agent_id == None:
            return self._name_with_agent0(name)
        else:
            return self._name_with_agent(agent_id, name)
    

    @property
    def dt(self) -> float:
        """
        获取环境时间步长（物理时间步长 × frame_skip）
        
        这是 Gym 环境的时间步长，即每次 step() 对应的时间间隔。
        等于物理仿真时间步长乘以 frame_skip。
        
        术语说明:
            - 时间步长 (Time Step): 每次仿真步进对应的时间间隔
            - 物理时间步长 (Timestep): MuJoCo 物理引擎的时间步长，通常很小（如 0.001s）
            - frame_skip: 每次 Gym step() 执行的物理步进次数，用于加速仿真
            - 控制频率: 1/dt，表示每秒执行多少次控制更新
        
        使用示例:
            ```python
            # 计算控制频率
            REALTIME_STEP = TIME_STEP * FRAME_SKIP * ACTION_SKIP
            CONTROL_FREQ = 1 / REALTIME_STEP  # 50 Hz
            
            # 在循环中使用
            dt = env.dt  # 获取环境时间步长
            sim_time += dt  # 累计仿真时间
            ```
        """
        return self.gym.opt.timestep * self.frame_skip
    
    @property
    def agent_num(self) -> int:
        """获取智能体数量"""
        return len(self._agent_names)

    def do_simulation(self, ctrl, n_frames) -> None:
        """
        Step the simulation n number of frames and applying a control action.
        """
        raise NotImplementedError

    def close(self):
        """Close all processes like rendering contexts"""
        raise NotImplementedError

    def initialize_grpc(self):
        """Initialize the GRPC communication channel."""
        raise NotImplementedError
    
    def pause_simulation(self):
        """Pause the simulation."""
        raise NotImplementedError

    async def _resume_simulation(self):
        """Resume the simulation."""
        raise NotImplementedError
    
    def init_qpos_qvel(self):
        """Init qpos and qvel of the model."""
        raise NotImplementedError
    
    async def _reset_simulation(self):
        """Reset the simulation."""
        raise NotImplementedError
    
    def reset_simulation(self):
        """Reset the simulation."""
        raise NotImplementedError

    def set_time_step(self, time_step):
        """Set the time step of the simulation."""
        raise NotImplementedError
