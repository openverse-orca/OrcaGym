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

from orca_gym import OrcaGymRemote, OrcaGymLocal
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
        Generate the action space for the environment.
        """
        low, high = 0.0, 0.0
        if len(bounds.T) > 0:
            low, high = bounds.T
        action_space = spaces.Box(low=low, high=high, dtype=np.float32)
        return action_space

    def generate_observation_space(self, obs: Union[Dict[str, Any], np.ndarray]) -> spaces.Space:
        """
        Generate the observation space for the environment.
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
        self.seed_value = seed
        self.np_random = np.random.RandomState(seed)
        return [seed]

    def _name_with_agent0(self, name: str) -> str:
        if len(self._agent_names) > 0:
            return f"{self._agent_names[0]}_{name}"
        else:
            return name
        
    def _name_with_agent(self, agent_id: int, name: str) -> str:
        if len(self._agent_names) > 0:
            return f"{self._agent_names[agent_id]}_{name}"
        else:
            return name

    def body(self, name: str, agent_id = None) -> str:
        if agent_id == None:
            return self._name_with_agent0(name)
        else:
            return self._name_with_agent(agent_id, name)
        
    def joint(self, name: str, agent_id = None) -> str:
        if agent_id == None:
            return self._name_with_agent0(name)
        else:
            return self._name_with_agent(agent_id, name)
        
    
    def actuator(self, name: str, agent_id = None) -> str:
        if agent_id == None:
            return self._name_with_agent0(name)
        else:
            return self._name_with_agent(agent_id, name)
        
    
    def site(self, name: str, agent_id = None) -> str:
        if agent_id == None:
            return self._name_with_agent0(name)
        else:
            return self._name_with_agent(agent_id, name)
        
    
    def mocap(self, name: str, agent_id = None) -> str:
        if agent_id == None:
            return self._name_with_agent0(name)
        else:
            return self._name_with_agent(agent_id, name)

    def sensor(self, name: str, agent_id = None) -> str:
        if agent_id == None:
            return self._name_with_agent0(name)
        else:
            return self._name_with_agent(agent_id, name)
    

    @property
    def dt(self) -> float:
        return self.gym.opt.timestep * self.frame_skip
    
    @property
    def agent_num(self) -> int:
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
