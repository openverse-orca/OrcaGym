import numpy as np
from gymnasium.core import ObsType
from orca_gym.environment import OrcaGymAgent
from orca_gym.utils import rotations
from typing import Optional, Any, SupportsFloat
from gymnasium import spaces

from .legged_robot_config import LeggedRobotConfig


def get_legged_robot_name(agent_name: str) -> str:
    if agent_name.startswith("go2"):
        return "Go2"
    else:
        raise ValueError(f"Unsupported agent name: {agent_name}")

class LeggedRobot(OrcaGymAgent):
    def __init__(self, 
                 agent_name: str, 
                 task: str,
                 max_episode_steps: int):
        
        super().__init__(agent_name, task, max_episode_steps)

        robot_config = LeggedRobotConfig[get_legged_robot_name(agent_name)]

        self._leg_joint_names = self.name_space_list(robot_config["leg_joint_names"])
        self._base_joint_name = self.name_space(robot_config["base_joint_name"])
        self._joint_names = [self._base_joint_name] + self._leg_joint_names
        
        self._neutral_joint_angles = robot_config["neutral_joint_angles"]
        self._neutral_joint_values = np.array([self._neutral_joint_angles[key] for key in self._neutral_joint_angles]).flatten()
        
        self._actuator_names = self.name_space_list(robot_config["actuator_names"])
        
        self._imu_site_name = self.name_space(robot_config["imu_site_name"])
        self._contact_site_names = self.name_space_list(robot_config["contact_site_names"])
        self._site_names = [self._imu_site_name] + self._contact_site_names


        self._imu_sensor_names = self.name_space_list(robot_config["sensor_imu_names"])
        self._touch_sensor_names = self.name_space_list(robot_config["sensor_base_touch_names"])
        self._sensor_names = self._imu_sensor_names + self._touch_sensor_names

        self._body_contact_force_threshold = np.array(robot_config["body_contact_force_threshold"]).flatten()
        
        self._ctrl = np.zeros(len(self._actuator_names))
        self._nu = len(self._actuator_names)
        self._nq = len(self._leg_joint_names) + (7 * len(self._base_joint_name))
        self._nv = len(self._leg_joint_names) + (6 * len(self._base_joint_name))


    @property
    def neutral_joint_values(self) -> np.ndarray:
        return self._neutral_joint_values
    
    @property
    def body_contact_force_threshold(self) -> float:
        return self._body_contact_force_threshold
    
    def get_joint_neutral(self) -> dict[str, np.ndarray]:
        joint_qpos = {}
        for name, value in zip(self._leg_joint_names, self.neutral_joint_values):
            joint_qpos[name] = np.array([value])
        return joint_qpos

    def _get_body_contact_force(self, sensor_data : dict) -> np.ndarray:
        return NotImplementedError
    
    def _get_imu_data(self, sensor_data : dict) -> np.ndarray:
        return NotImplementedError

    def get_obs(self, sensor_data : dict, joint_qpos : dict, dt : float) -> dict:
        leg_joint_qpos = np.array([joint_qpos[joint_name] for joint_name in self._leg_joint_names]).flatten()
        imu_data = self._get_imu_data(sensor_data)
        achieved_goal = self._get_body_contact_force(sensor_data)
        desired_goal = self.body_contact_force_threshold
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

    def get_action_size(self) -> int:
        return self._nu

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
        

    def _get_body_contact_force(self, sensor_data : dict) -> np.ndarray:
        contact_force = np.zeros(len(self._touch_sensor_names))
        for i, sensor_name in enumerate(self._touch_sensor_names):
            contact_force[i] = sensor_data[sensor_name]['values'][0]
        return contact_force.flatten()
    
    def _get_imu_data(self, sensor_data: dict) -> np.ndarray:
        quat = np.array(sensor_data[self._sensor_imu_quat_name]['values'])
        omega = np.array(sensor_data[self._sensor_imu_omega_name]['values'])
        acc = np.array(sensor_data[self._sensor_imu_acc_name]['values'])
        # print("Quat: ", quat)
        # print("Omega: ", omega)
        # print("Acc: ", acc)
        imu_data = np.concatenate((quat, omega, acc))
        return imu_data.flatten()
