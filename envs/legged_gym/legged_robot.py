import numpy as np
from datetime import datetime
import math
from gymnasium.core import ObsType
from orca_gym.multi_agent import OrcaGymAgent
from orca_gym.utils import rotations
from typing import Optional, Any, SupportsFloat
from gymnasium import spaces
import copy

from .legged_robot_config import LeggedRobotConfig

PRINT_REWARD = True
PRINT_REWARD_INTERVAL = 10

def print_reward(message : str, reward : Optional[float] = None):
    if PRINT_REWARD:
        if print_reward.print:
            if reward is None:
                print(message)
            else:
                print(message, reward)

def print_reward_begin():
    if PRINT_REWARD:
        if not hasattr(print_reward, "last_time"):
            print_reward.last_time = datetime.now()
            print("Current time: ", print_reward.last_time)
            print_reward.print = True

        if (datetime.now() - print_reward.last_time).seconds > PRINT_REWARD_INTERVAL:
            print_reward.print = True
            print_reward.last_time = datetime.now()
        
def print_reward_end():
    if PRINT_REWARD:
        print_reward.print = False

def get_legged_robot_name(agent_name: str) -> str:
    if agent_name.startswith("go2"):
        return "go2"
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

        self._base_neutral_height_offset = robot_config["base_neutral_height_offset"]
        
        self._neutral_joint_angles = robot_config["neutral_joint_angles"]
        self._neutral_joint_values = np.array([self._neutral_joint_angles[key] for key in self._neutral_joint_angles]).flatten()
        
        self._actuator_names = self.name_space_list(robot_config["actuator_names"])
        self._actuator_type = robot_config["actuator_type"]
        self._action_scale = robot_config["action_scale"]
        
        self._imu_site_name = self.name_space(robot_config["imu_site_name"])
        self._contact_site_names = self.name_space_list(robot_config["contact_site_names"])
        self._site_names = [self._imu_site_name] + self._contact_site_names

        self._imu_mocap_name = self.name_space(robot_config["imu_mocap_name"])

        self._imu_sensor_framequat_name = self.name_space(robot_config["sensor_imu_framequat_name"])
        self._imu_sensor_gyro_name = self.name_space(robot_config["sensor_imu_gyro_name"])
        self._imu_sensor_accelerometer_name = self.name_space(robot_config["sensor_imu_accelerometer_name"])
        self._imu_sensor_names = [self._imu_sensor_framequat_name, self._imu_sensor_gyro_name, self._imu_sensor_accelerometer_name]

        self._foot_touch_sensor_names = self.name_space_list(robot_config["sensor_foot_touch_names"])
        self._foot_touch_force_threshold = robot_config["foot_touch_force_threshold"]

        self._sensor_names = self._imu_sensor_names + self._foot_touch_sensor_names

        self._ground_contact_body_names = robot_config["ground_contact_body_names"]  # gound body is not included in the agent's namespace
        self._base_contact_body_names = self.name_space_list(robot_config["base_contact_body_names"])
        self._leg_contact_body_names = self.name_space_list(robot_config["leg_contact_body_names"])
        self._contact_body_names = self._base_contact_body_names + self._leg_contact_body_names
        
        self._ctrl = np.zeros(len(self._actuator_names))
        self._action = np.zeros(len(self._actuator_names))
        self._last_action = np.zeros(len(self._actuator_names))
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

    def get_obs(self, sensor_data : dict, joint_qpos : dict, joint_qacc : dict, joint_qvel : dict, contact_dict : dict, site_pos_quat : dict, dt : float) -> dict:
        self._leg_joint_qpos = np.array([joint_qpos[joint_name] for joint_name in self._leg_joint_names]).flatten()
        self._leg_joint_qacc = np.array([joint_qacc[joint_name] for joint_name in self._leg_joint_names]).flatten()
        self._leg_joint_qvel = np.array([joint_qvel[joint_name] for joint_name in self._leg_joint_names]).flatten()
        self._imu_data = self._get_imu_data(sensor_data)
        self._foot_touch_force = self._get_foot_touch_force(sensor_data)  # Penalty if the foot touch force is too strong
        self._leg_contact = self._get_leg_contact(contact_dict)            # Penalty if the leg is in contact with the ground
        self._imu_site_pos_quat = site_pos_quat[self._imu_site_name].copy()
        imu_site_pos_quat = np.concatenate([self._imu_site_pos_quat["xpos"], self._imu_site_pos_quat["xquat"]]).flatten()
        imu_mocap_pos_quat = np.concatenate([self._imu_mocap_pos_quat["pos"], self._imu_mocap_pos_quat["quat"]]).flatten()

        self._achieved_goal = self._get_base_contact(contact_dict)         # task failed if the base is in contact with the ground
        self._desired_goal = np.zeros(1)      # 1.0 if the base is in contact with the ground, 0.0 otherwise

        obs = np.concatenate(
                [
                    self._leg_joint_qpos.copy(),
                    self._leg_joint_qvel.copy(),
                    self._action.copy(),
                    self._ctrl.copy(),
                    imu_site_pos_quat,
                    imu_mocap_pos_quat,
                ])          

        result = {
            "observation": obs,
            "achieved_goal": self._achieved_goal.copy(),
            "desired_goal": self._desired_goal.copy(),
        }

        self._is_obs_updated = True

        # print("Agent Obs: ", result)

        return result

    def set_init_state(self, joint_qpos: dict, init_site_pos_quat: dict) -> None:
        base_joint_qpos = np.array(joint_qpos[self._base_joint_name]).flatten()
        self._init_base_joint_qpos = {self._base_joint_name: base_joint_qpos}
        self._init_imu_site_pos_quat = init_site_pos_quat[self._imu_site_name].copy()

    def set_action(self, action):
        assert len(action) == len(self._ctrl_range)

        self._last_action = self._action.copy()
        self._action = action.copy()

        # print("Agent: ", self.name, "Orignal action: ", action)

        for i in range(len(action)):
            # 线性变换到 ctrl range 空间
            # print("action: ", action[i])
            # print("action_space_range: ", self._action_space_range)
            # print("ctrl_range: ", self._ctrl_range[i])
            if (self._actuator_type == "position"):
                noise = 0 # np.random.normal(0, 0.01, 1)
                ctrl_delta = np.interp((action[i] + noise) * self._action_scale, self._action_space_range, self._ctrl_delta_range[i])
                # print("ctrl delta : ", ctrl_delta, "action scale: ", self._action_scale, "action: ", action[i], "action space range: ", self._action_space_range, "ctrl delta range: ", self._ctrl_delta_range[i])
                self._ctrl[i] = self._neutral_joint_values[i] + ctrl_delta
            elif (self._actuator_type == "torque"):
                self._ctrl[i] = np.interp(action[i], self._action_space_range, self._ctrl_range[i])

        # print("Agent: ", self.name, "Ctrl: ", self._ctrl)

        return
    
    def set_action_space(self, action_space : spaces) -> None:
        self._action_space = action_space
        self._action_space_range = [action_space.low[0], action_space.high[0]]
    
    def on_step(self, action):
        """
        Update the imu mocap position and quaternion
        """
        # Do mocap update here.
        return {self._imu_mocap_name: self._imu_mocap_pos_quat}

    def on_reset(self) -> dict[str, np.ndarray]:

        if self._actuator_type == "position":
            assert self._ctrl.shape == self._neutral_joint_values.shape
            self._ctrl = self._neutral_joint_values.copy()
            # print("Neutral joint values: ", self._ctrl)
        elif self._actuator_type == "torque":
            self._ctrl = np.zeros(self._nu)
        else:
            raise ValueError(f"Unsupported actuator type: {self._actuator_type}")

        joint_neutral_qpos = self.get_joint_neutral()

        base_neutral_qpos = copy.deepcopy(self._init_base_joint_qpos)
        base_neutral_qpos[self._base_joint_name][2] -= self._base_neutral_height_offset
        # print("Base neutral qpos: ", base_neutral_qpos)
        joint_neutral_qpos.update(base_neutral_qpos)

        self._imu_mocap_pos_quat = {"pos": self._init_imu_site_pos_quat["xpos"].copy(), 
                                    "quat": self._init_imu_site_pos_quat["xquat"].copy()}
        self._imu_mocap_pos_quat["pos"][2] -= self._base_neutral_height_offset

        return joint_neutral_qpos, {self._imu_mocap_name: self._imu_mocap_pos_quat}

    def is_terminated(self, achieved_goal, desired_goal) -> bool:
        assert achieved_goal.shape == desired_goal.shape
        return any(achieved_goal != desired_goal)

    def is_success(self, achieved_goal, desired_goal) -> np.float32:
        if self.truncated and self.is_terminated(achieved_goal, desired_goal):
            return 1.0
        else:
            return 0.0

    def _compute_reward_alive(self, coeff) -> SupportsFloat:
        reward = 1.0 * coeff
        print_reward("Alive reward: ", reward)
        return reward

    def _compute_reward_success(self, achieved_goal, desired_goal, coeff) -> SupportsFloat:
        reward = (1.0 if (self.is_success(achieved_goal, desired_goal) > 0) else 0.0) * coeff
        print_reward("Success reward: ", reward)
        return reward
        
    def _compute_reward_failure(self, achieved_goal, desired_goal, coeff) -> SupportsFloat:
        reward = (-1.0 if self.is_terminated(achieved_goal, desired_goal) else 0.0) * coeff
        print_reward("Failure reward: ", reward)
        return reward
        
    def _compute_reward_contact(self, coeff) -> SupportsFloat:
        reward = (-np.sum(self._leg_contact)) * coeff
        print_reward("Contact reward: ", reward)
        return reward
    
    def _compute_reward_foot_touch(self, coeff) -> SupportsFloat:
        threshold = self._foot_touch_force_threshold
        reward = (-np.sum([(force - threshold) if force > threshold else 0 for force in self._foot_touch_force])) * coeff
        print_reward("Foot touch reward: ", reward)
        return reward
    
    def _compute_reward_joint_angles(self, coeff) -> SupportsFloat:
        base_reward = 0 #1 * len(self._leg_joint_qpos)
        reward = (base_reward - np.sum(np.abs(self._leg_joint_qpos - self._neutral_joint_values))) * coeff
        print_reward("Joint angles reward: ", reward)
        return reward
    
    def _compute_reward_joint_accelerations(self, coeff) -> SupportsFloat:
        reward = (-np.sum(np.abs(self._leg_joint_qacc))) * coeff
        print_reward("Joint accelerations reward: ", reward)
        return reward
    
    def _compute_reward_limit(self, coeff) -> SupportsFloat:
        # limit_threshold = 0.8
        # reward = 0.0

        # for i in range(len(self._ctrl)):
        #     ctrl = max(self._ctrl_range_low[i] + 0.01, min(self._ctrl[i], self._ctrl_range_high[i] - 0.01))
        #     if ctrl < self._ctrl_range_low[i] * limit_threshold:
        #         reward -= 1 / abs(self._ctrl_range_low[i] - ctrl)
        #     elif ctrl > self._ctrl_range_high[i] * limit_threshold:
        #         reward -= 1 / abs(self._ctrl_range_high[i] - ctrl)
                
        reward = -(np.sum(self._action == self._action_space_range[0]) + np.sum(self._action == self._action_space_range[1])) * coeff
        print_reward("Limit over threshold reward: ", reward)
        return reward
    
    def _compute_reward_action_rate(self, coeff) -> SupportsFloat:
        base_reward = 0 # 1 * len(self._action)
        reward = (base_reward - np.sum(np.abs(self._action - self._last_action))) * coeff
        print_reward("Action rate reward: ", reward)
        return reward
    
    def _compute_reward_base_gyro(self, coeff) -> SupportsFloat:
        reward = (-np.sum(np.abs(self._imu_data_gyro))) * coeff
        print_reward("Base gyro reward: ", reward)
        return reward
    
    def _compute_reward_base_accelerometer(self, coeff) -> SupportsFloat:
        reward = (-np.sum(abs(self._imu_data_accelerometer - np.array([0.0, 0.0, -9.81])))) * coeff
        print_reward("Base accelerometer reward: ", reward)
        return reward
    
    def _compute_reward_follow_command(self, coeff) -> SupportsFloat:
        base_reward_distance = 0.1
        base_reward_angle = np.pi / 8
        imu_site_mocap_distance = np.linalg.norm(self._imu_mocap_pos_quat["pos"] - self._imu_site_pos_quat["xpos"])
        imu_site_mocap_angle = math.acos(np.clip(np.dot(self._imu_mocap_pos_quat["quat"], self._imu_site_pos_quat["xquat"]), -1, 1))
        reward = ((base_reward_angle - imu_site_mocap_angle) + (base_reward_distance - imu_site_mocap_distance)) * coeff
        print_reward("Follow command reward: ", reward)
        return reward
        
    def compute_reward(self, achieved_goal, desired_goal) -> SupportsFloat:
        if self._is_obs_updated:
            total_reward = 0.0

            reward_alive_coeff = 0
            reward_success_coeff = 1
            reward_failure_coeff = 1
            reward_contact_coeff = 0.5
            reward_foot_touch_coeff = 0.01
            reward_joint_angles_coeff = 0.1
            reward_joint_accelerations_coeff = 0.00001
            reward_limit_coeff = 0
            reward_action_rate_coeff = 0.01
            reward_base_gyro_coeff = 0.01
            reward_base_accelerometer_coeff = 0.001
            reward_follow_command_coeff = 2

            print_reward_begin()

            # *** Reward for staying alive
            total_reward += self._compute_reward_alive(reward_alive_coeff)

            # *** Reward for task success
            total_reward += self._compute_reward_success(achieved_goal, desired_goal, reward_success_coeff)

            # *** Penalty for task failure
            total_reward += self._compute_reward_failure(achieved_goal, desired_goal, reward_failure_coeff)

            # *** Penalty for leg contact with other bodies
            total_reward +=  self._compute_reward_contact(reward_contact_coeff)

            # *** Penality for foot touch force too strong
            total_reward +=  self._compute_reward_foot_touch(reward_foot_touch_coeff)

            # *** Penalty for joint angles too far from the neutral position
            total_reward +=  self._compute_reward_joint_angles(reward_joint_angles_coeff)

            # *** Penalty for joint accelerations
            total_reward +=  self._compute_reward_joint_accelerations(reward_joint_accelerations_coeff)

            # *** Penalty for torques or angels too close to the limits
            total_reward +=  self._compute_reward_limit(reward_limit_coeff)

            # *** Penalty for action rate
            total_reward +=  self._compute_reward_action_rate(reward_action_rate_coeff)

            # *** Penalty for base gyro and accelerometer
            total_reward +=  self._compute_reward_base_gyro(reward_base_gyro_coeff)
            total_reward +=  self._compute_reward_base_accelerometer(reward_base_accelerometer_coeff)

            # *** Reward for following the command
            total_reward +=  self._compute_reward_follow_command(reward_follow_command_coeff)

            self._is_obs_updated = False
            print_reward("Total Reward: ", total_reward)
            print_reward("-----------------------------")
            print_reward_end()
            return total_reward
        else:
            raise ValueError("Observation must be updated before computing reward")

    def _get_base_contact(self, contact_dict : dict) -> np.ndarray:
        """
        Check if the base is in contact with the ground
        """
        contact_result = False
        for contact_body_name in self._base_contact_body_names:
            if contact_body_name in contact_dict:
                # print("Base contact with: ", contact_dict[contact_body_name])
                for ground_contact_body_name in self._ground_contact_body_names:
                    if ground_contact_body_name in contact_dict[contact_body_name]:
                        contact_result = True
                        break
        return np.array([1.0 if contact_result else 0.0])
    
    def _get_leg_contact(self, contact_dict : dict) -> np.ndarray:
        """
        Check if each of the legs is in contact with the any other body
        """
        contact_result = np.zeros(len(self._leg_contact_body_names))
        for i, contact_body_name in enumerate(self._leg_contact_body_names):
            if contact_body_name in contact_dict:
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

        # print("Foot touch force: ", contact_force)
        return contact_force
