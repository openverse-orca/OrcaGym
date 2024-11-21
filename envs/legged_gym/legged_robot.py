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
    elif agent_name.startswith("A01B"):
        return "A01B"
    else:
        raise ValueError(f"Unsupported agent name: {agent_name}")

class LeggedRobot(OrcaGymAgent):
    def __init__(self, 
                 env_id: str,                 
                 agent_name: str, 
                 task: str,
                 max_episode_steps: int):
        
        super().__init__(env_id, agent_name, task, max_episode_steps)

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

        self._imu_sensor_framequat_name = self.name_space(robot_config["sensor_imu_framequat_name"])
        self._imu_sensor_gyro_name = self.name_space(robot_config["sensor_imu_gyro_name"])
        self._imu_sensor_accelerometer_name = self.name_space(robot_config["sensor_imu_accelerometer_name"])
        self._imu_sensor_names = [self._imu_sensor_framequat_name, self._imu_sensor_gyro_name, self._imu_sensor_accelerometer_name]

        self._foot_touch_sensor_names = self.name_space_list(robot_config["sensor_foot_touch_names"])

        self._sensor_names = self._imu_sensor_names + self._foot_touch_sensor_names

        self._base_contact_body_names = self.name_space_list(robot_config["base_contact_body_names"])
        self._leg_contact_body_names = self.name_space_list(robot_config["leg_contact_body_names"])
        self._contact_body_names = self._base_contact_body_names + self._leg_contact_body_names
        
        self._ctrl = np.zeros(len(self._actuator_names))
        self._nu = len(self._actuator_names)
        self._nq = len(self._leg_joint_names) + (7 * len(self._base_joint_name))
        self._nv = len(self._leg_joint_names) + (6 * len(self._base_joint_name))

        self._is_obs_updated = False


    @property
    def neutral_joint_values(self) -> np.ndarray:
        return self._neutral_joint_values
    
    # @property
    # def body_contact_force_threshold(self) -> float:
    #     return self._body_contact_force_threshold
    
    def get_joint_neutral(self) -> dict[str, np.ndarray]:
        joint_qpos = {}
        for name, value in zip(self._leg_joint_names, self.neutral_joint_values):
            joint_qpos[name] = np.array([value])
        return joint_qpos

    def get_obs(self, sensor_data : dict, joint_qpos : dict, contact_set : set, dt : float) -> dict:
        self._leg_joint_qpos = np.array([joint_qpos[joint_name] for joint_name in self._leg_joint_names]).flatten()
        self._imu_data = self._get_imu_data(sensor_data)
        self._foot_touch_force = self._get_foot_touch_force(sensor_data)  # penality if the foot touch force is too strong
        self._leg_contact = self._get_leg_contact(contact_set)            # penality if the leg is in contact with the ground

        self._achieved_goal = self._get_base_contact(contact_set)         # task failed if the base is in contact with the ground
        self._desired_goal = np.zeros(1)      # 1.0 if the base is in contact with the ground, 0.0 otherwise

        obs = np.concatenate(
                [
                    self._leg_joint_qpos.copy(),
                    self._imu_data.copy(),
                    self._foot_touch_force.copy(),
                    self._leg_contact.copy(),
                ])          

        result = {
            "observation": obs,
            "achieved_goal": self._achieved_goal.copy(),
            "desired_goal": self._desired_goal.copy(),
        }

        self._is_obs_updated = True

        # print("Agent Obs: ", result)

        return result

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

    def is_terminated(self, achieved_goal, desired_goal) -> bool:
        assert achieved_goal.shape == desired_goal.shape
        return any(achieved_goal != desired_goal)

    def is_success(self, achieved_goal, desired_goal) -> np.float32:
        if self.truncated and self.is_terminated(achieved_goal, desired_goal):
            return 1.0
        else:
            return 0.0

        
    def compute_reward(self, achieved_goal, desired_goal) -> SupportsFloat:
        if self._is_obs_updated:
            total_reward = 0.0

            # reward for task success
            if self.is_success(achieved_goal, desired_goal) > 0:
                total_reward += 10.0

            # penality for task failure
            if self.is_terminated(achieved_goal, desired_goal):
                total_reward -= 10.0

            # penality for leg contact with the ground
            total_reward -= 0.1 * np.sum(self._leg_contact)

            # penality for foot touch force
            variance = np.var(abs(self._foot_touch_force))
            total_reward -= 0.1 * variance

            # penality for torques too close to the limits
            for i, torque in enumerate(self._ctrl):
                if abs(torque) > 0.9 * abs(self._ctrl_range[i][0]) and (abs(torque) > 0.9 * abs(self._ctrl_range[i][1])):
                    total_reward -= 0.1

            # penalty for base gyro and accelerometer
            gyro_penalty = np.linalg.norm(self._imu_data_gyro)
            accel_penalty = np.linalg.norm(self._imu_data_accelerometer - np.array([0.0, 0.0, -9.81]))
            total_reward -= 0.1 * gyro_penalty + 0.1 * accel_penalty

            self._is_obs_updated = False
            return total_reward
        else:
            raise ValueError("Observation must be updated before computing reward")

    def _get_base_contact(self, contact_set : set) -> np.ndarray:
        """
        Check if the base is in contact with the ground
        """
        contact_result = False
        for contact_body_name in self._base_contact_body_names:
            if contact_body_name in contact_set:
                contact_result = True
                break
        return np.array([1.0 if contact_result else 0.0])
    
    def _get_leg_contact(self, contact_set : set) -> np.ndarray:
        """
        Check if each of the legs is in contact with the ground
        """
        contact_result = np.zeros(len(self._leg_contact_body_names))
        for i, contact_body_name in enumerate(self._leg_contact_body_names):
            if contact_body_name in contact_set:
                contact_result[i] = 1.0
        return contact_result
    
    def _get_imu_data(self, sensor_data: dict) -> np.ndarray:
        self._imu_data_framequat = sensor_data[self._imu_sensor_framequat_name]
        self._imu_data_gyro = sensor_data[self._imu_sensor_gyro_name]
        self._imu_data_accelerometer = sensor_data[self._imu_sensor_accelerometer_name]
        imu_data = np.concatenate([self._imu_data_framequat, self._imu_data_gyro, self._imu_data_accelerometer])
        return imu_data.flatten()
    
    def _get_foot_touch_force(self, sensor_data: dict) -> np.ndarray:
        contact_force = np.zeros(len(self._foot_touch_sensor_names))
        for i, touch_sensor_name in enumerate(self._foot_touch_sensor_names):
            contact_force[i] = sensor_data[touch_sensor_name]
        return contact_force
