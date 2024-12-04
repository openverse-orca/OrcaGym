import numpy as np
from gymnasium.core import ObsType
from typing import Optional, Any, SupportsFloat
from gymnasium import spaces
import datetime


class OrcaGymAgent:
    def __init__(self, 
                 env_id: str,
                 agent_name: str, 
                 task: str,
                 max_episode_steps: int,
                 dt: float,
                 **kwargs):
        
        self._env_id = env_id
        self._agent_name = agent_name
        self._task = task
        self._max_episode_steps = max_episode_steps
        self._current_episode_step = 0
        self._dt = dt

        self._joint_names = None
        self._actuator_names = None
        self._site_names = None
        self._sensor_names = None

        self._ctrl = None
        self._nu = None
        self._nq = None
        self._nv = None

    @property
    def dt(self) -> float:
        return self._dt

    @property
    def name(self) -> str:
        return self._agent_name

    def name_space(self, name : str) -> str:
        return f"{self._agent_name}_{name}"
    
    def name_space_list(self, names : list[str]) -> list[str]:
        return [self.name_space(name) for name in names]
    
    @property
    def joint_names(self) -> list[str]:
        return self._joint_names
    
    @property
    def actuator_names(self) -> list[str]:
        return self._actuator_names
    
    @property
    def site_names(self) -> list[str]:
        return self._site_names
    
    @property
    def sensor_names(self) -> list[str]:
        return self._sensor_names

    @property
    def nu(self) -> int:
        return self._nu
    
    @property
    def nq(self) -> int:
        return self._nq
    
    @property
    def nv(self) -> int:
        return self._nv

    @property
    def truncated(self) -> bool:
        return self._current_episode_step >= self._max_episode_steps

    @property
    def ctrl_start(self) -> int:
        return self._ctrl_start

    def get_obs(self, **kwargs):
        raise NotImplementedError

    def init_ctrl_info(self, actuator_dict) -> None:
        """
        Each robot has it's own control method.
        """
        raise NotImplementedError
    
    def set_action_space(self, action_space : spaces) -> None:
        """
        Action space is specific to the agent and is defined in the subclass.
        """
        raise NotImplementedError
    
    def on_step(self, action, **kwargs):
        """
        Called after each step in the environment.
        Implement this method in the subclass to perform additional operations.
        """
        raise NotImplementedError

    def step(self, action, **kwargs):
        self._current_episode_step += 1
        step_info = self.on_step(action, **kwargs)
        return self._ctrl, step_info

    def on_reset(self):
        """
        Called after each reset in the environment.
        Implement this method in the subclass to perform additional operations.
        """
        raise NotImplementedError

    def reset(self, np_random : np.random.Generator):
        self._current_episode_step = 0
        self._np_random = np_random
        reset_info = self.on_reset()
        return reset_info


    def is_success(self, achieved_goal, desired_goal) -> np.float32:
        raise NotImplementedError

    def is_terminated(self, achieved_goal, desired_goal) -> bool:
        raise NotImplementedError
        
    def compute_reward(self, achieved_goal, desired_goal) -> SupportsFloat:
        raise NotImplementedError
        
    def set_init_state(self, joint_qpos: dict, init_site_pos_quat: dict) -> None:
        raise NotImplementedError
    
    def get_action_size(self) -> int:
        """
        Action size can be overridden in the subclass.
        In most of cases, this is the number of actuators in the robot.
        But in some cases, the action size may be different.
        """        
        return self._nu