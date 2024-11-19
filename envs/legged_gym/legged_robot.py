import numpy as np
from gymnasium.core import ObsType
from envs import OrcaGymLocalEnv, OrcaGymRemoteEnv
from orca_gym.utils import rotations
from typing import Optional, Any, SupportsFloat
from gymnasium import spaces



class LeggedRobot:
    def __init__(self, 
                 agent_name: str, 
                 task: str,
                 max_episode_steps: int):
        
        self._agent_name = agent_name
        self._task = task
        self._max_episode_steps = max_episode_steps
        self._current_episode_step = 0

        self._joint_names = None
        self._leg_joint_names = None
        self._base_joint_name = None
        self._actuator_names = None
        self._site_names = None
        self._sensor_names = None

        self._neutral_joint_values = None
        self._contact_force_threshold = None

        self._ctrl = None
        self._nu = None
        self._nq = None
        self._nv = None


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
    def neutral_joint_values(self) -> np.ndarray:
        return self._neutral_joint_values
    
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
    def contact_force_threshold(self) -> float:
        return self._contact_force_threshold
    
    @property
    def ctrl_start(self) -> int:
        return self._ctrl_start

    def get_joint_neutral(self) -> dict[str, np.ndarray]:
        joint_qpos = {}
        for name, value in zip(self._leg_joint_names, self.neutral_joint_values):
            joint_qpos[name] = np.array([value])
        return joint_qpos

    def get_body_contact_force(self, sensor_data : dict) -> np.ndarray:
        return NotImplementedError
    
    def get_imu_data(self, sensor_data : dict) -> np.ndarray:
        return NotImplementedError

    def get_obs(self, sensor_data : dict, joint_qpos : dict, dt : float) -> dict:
        leg_joint_qpos = np.array([joint_qpos[joint_name] for joint_name in self._leg_joint_names]).flatten()
        imu_data = self.get_imu_data(sensor_data)
        achieved_goal = self.get_body_contact_force(sensor_data)
        desired_goal = self.contact_force_threshold
        obs = np.concatenate(
                [
                    leg_joint_qpos,
                    imu_data
                ]).copy()                 

        result = {
            "observation": obs,
            "achieved_goal": achieved_goal,
            "desired_goal": desired_goal,
        }

        # print("Agent Obs: ", result)

        return result

    def set_ctrl_info(self, actuator_dict) -> None:
        if not hasattr(self, "_ctrl_range"):
            self._ctrl_range = []
        if not hasattr(self, "_ctrl_offset"):
            self._ctrl_start = 0
            
        for i, actuator_name in enumerate(self._actuator_names):
            # matain the order of actuators
            self._ctrl_range.append(np.array(actuator_dict[actuator_name]['CtrlRange']).flatten())
            if i == 0:
                self._ctrl_start = actuator_dict[actuator_name]['ActuatorId']

    def set_init_state(self, joint_qpos: dict):
        base_joint_qpos = np.array(joint_qpos[self._base_joint_name]).flatten()
        self._init_base_joint_qpos = {self._base_joint_name: base_joint_qpos}

    def set_action(self, action):
        assert len(action) == len(self._ctrl_range)

        for i in range(len(action)):
            # 线性变换到 ctrl range 空间
            # print("action: ", action[i])
            # print("action_space_range: ", self._action_space_range)
            # print("ctrl_range: ", self._ctrl_range[i])
            self._ctrl[i] = np.interp(action[i], self._action_space_range, self._ctrl_range[i])

        # print("Agent: ", self.name, "Ctrl: ", self._ctrl)

        return
    
    def set_action_space(self, action_space : spaces) -> None:
        self._action_space = action_space
        self._action_space_range = [action_space.low[0], action_space.high[0]]
    
    def step(self, action):
        self._current_episode_step += 1
        self.set_action(action)
        return self._ctrl

    def reset(self, np_random) -> dict[str, np.ndarray]:
        self._current_episode_step = 0
        joint_neutral_qpos = self.get_joint_neutral()
        joint_neutral_qpos.update(self._init_base_joint_qpos)
        return joint_neutral_qpos

    def is_truncated(self, achieved_goal, desired_goal) -> bool:
        assert achieved_goal.shape == desired_goal.shape
        return any(achieved_goal > desired_goal)

    def is_success(self, achieved_goal, desired_goal, env_id) -> np.float32:
        if self.is_truncated(achieved_goal, desired_goal):
            # print(f"{env_id} Agent {self.name} Task Failed: achieved goal: ", achieved_goal, "desired goal: ", desired_goal, "steps: ", self._current_episode_step)
            return 0.0
        elif self._current_episode_step >= self._max_episode_steps:
            print(f"{env_id} Agent {self.name} Task Successed!")
            return 1.0
        else:
            return 0.0

        
    def compute_reward(self, achieved_goal, desired_goal) -> SupportsFloat:
        if self.is_truncated(achieved_goal, desired_goal):
            return -1.0
        else:
            return self._current_episode_step * 0.1
        