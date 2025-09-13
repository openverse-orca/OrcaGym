import numpy as np
from gymnasium.core import ObsType
from typing import Optional, Any, SupportsFloat
from gymnasium import spaces
import datetime


class OrcaGymAsyncAgent:
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

        self._actuator_type = None
        self._action_scale = None
        self._action_scale_mask = None
        self._ctrl_range = None
        self._ctrl_delta_range = None
        self._ctrl_range_low = None
        self._ctrl_range_high = None
        self._ctrl_start = None
        self._neutral_joint_values = None

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

    @property
    def action_range(self) -> np.ndarray:
        raise NotImplementedError

    @property
    def kps(self) -> float:
        raise NotImplementedError
    
    @property
    def kds(self) -> float:
        raise NotImplementedError

    def get_obs(self, **kwargs):
        raise NotImplementedError

    def init_ctrl_info(self, actuator_dict, joint_dict) -> None:
        ctrl_range_list = []
        ctrl_delta_range_list = []
        torques_range_list = []
        for i, joint_name in enumerate(self._joint_names):
            if i == 0:
                continue    # pass the first joint, which is the free joint
            # matain the order of actuators
            ctrl_range_list.append(np.array(joint_dict[joint_name]['Range']).flatten())
            ctrl_range_width = ctrl_range_list[-1][1] - ctrl_range_list[-1][0]
            ctrl_delta_range_list.append([-ctrl_range_width/2, ctrl_range_width/2])

        for actuator_name in self._actuator_names:
            torques_range_list.append(np.array(actuator_dict[actuator_name]['CtrlRange']).flatten())


        self._ctrl_start = actuator_dict[self._actuator_names[0]]['ActuatorId']

        self._torques_range = np.array(torques_range_list)
        self._torques_limit = np.array([abs(self._torques_range[i][1]) * self._soft_torque_limit for i in range(len(self._torques_range))])

        self._ctrl_range = np.array(ctrl_range_list)
        self._joint_qpos_limit = np.array([[self._ctrl_range[i][0] * self._soft_joint_qpos_limit, 
                                           self._ctrl_range[i][1] * self._soft_joint_qpos_limit] 
                                           for i in range(len(self._ctrl_range))])

        self._joint_qvel_limit = np.array([self._joint_qvel_range[i] * self._soft_joint_qvel_limit for i in range(len(self._joint_qvel_range))])

        # print("ctrl_range: ", self._ctrl_range)
        # print("soft_joint_qpos_limit: ", self._soft_joint_qpos_limit)
        # print("joint_qpos_limit: ", self._joint_qpos_limit)

        self._ctrl_delta_range = np.array(ctrl_delta_range_list)


        self._ctrl_range_low = np.array([range[0] for range in self._ctrl_range])
        self._ctrl_range_high = np.array([range[1] for range in self._ctrl_range])

    def get_ctrl_info(self) -> dict:
        return {
            "actuator_type": self._actuator_type,
            "action_scale": self._action_scale,
            "action_scale_mask": self._action_scale_mask,
            "ctrl_range": self._ctrl_range,
            "ctrl_delta_range": self._ctrl_delta_range,
            "ctrl_range_low": self._ctrl_range_low,
            "ctrl_range_high": self._ctrl_range_high,
            "ctrl_start": self._ctrl_start,
            "ctrl_end": self._ctrl_start + len(self._actuator_names),
            "neutral_joint_values": self._neutral_joint_values,
        }
    
    def init_joint_index(self, qpos_offset, qvel_offset, qacc_offset, qpos_length, qvel_length, qacc_length):
        """
        Joint index is specific to the agent and is defined in the subclass.
        """
        raise NotImplementedError
    
    def set_action_space(self) -> None:
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

    def on_reset(self, **kwargs):
        """
        Called after each reset in the environment.
        Implement this method in the subclass to perform additional operations.
        """
        raise NotImplementedError

    def reset(self, np_random : np.random.Generator, **kwargs):
        self._current_episode_step = 0
        self._np_random = np_random
        reset_info = self.on_reset(**kwargs)
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

    def compute_torques(self, qpos_buffer : np.ndarray, qvel_buffer : np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def setup_curriculum(self, curriculum : str) -> None:
        raise NotImplementedError