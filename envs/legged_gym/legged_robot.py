import numpy as np
from datetime import datetime
import math
from gymnasium.core import ObsType
from orca_gym.multi_agent import OrcaGymAgent
from orca_gym.utils import rotations
from orca_gym.utils.reward_printer import RewardPrinter
from typing import Optional, Any, SupportsFloat
from gymnasium import spaces
import copy

from .legged_robot_config import LeggedRobotConfig


def get_legged_robot_name(agent_name: str) -> str:
    if agent_name.startswith("go2"):
        return "go2"
    elif agent_name.startswith("A01B"):
        return "A01B"
    else:
        raise ValueError(f"Unsupported agent name: {agent_name}")

def local2global(q_global_to_local, v_local, v_omega_local) -> tuple[np.ndarray, np.ndarray]:
    # Vg = QVlQ*

    # 将速度向量从局部坐标系转换到全局坐标系
    q_v_local = np.array([0, v_local[0], v_local[1], v_local[2]])  # 局部坐标系下的速度向量表示为四元数
    q_v_global = rotations.quat_mul(q_global_to_local, rotations.quat_mul(q_v_local, rotations.quat_conjugate(q_global_to_local)))
    v_global = np.array(q_v_global[1:])  # 提取虚部作为全局坐标系下的线速度

    # 将角速度从局部坐标系转换到全局坐标系
    q_omega_local = np.array([0, v_omega_local[0], v_omega_local[1], v_omega_local[2]])  # 局部坐标系下的角速度向量表示为四元数
    q_omega_global = rotations.quat_mul(q_global_to_local, rotations.quat_mul(q_omega_local, rotations.quat_conjugate(q_global_to_local)))
    v_omega_local = np.array(q_omega_global[1:])  # 提取虚部作为全局坐标系下的角速度

    return v_global, q_omega_global

def global2local(q_global_to_local, v_global, v_omega_global) -> tuple[np.ndarray, np.ndarray]:
    # Vl = Q*VgQ

    # 将速度向量从全局坐标系转换到局部坐标系
    q_v_global = np.array([0, v_global[0], v_global[1], v_global[2]])  # 速度向量表示为四元数
    q_v_local = rotations.quat_mul(rotations.quat_conjugate(q_global_to_local), rotations.quat_mul(q_v_global, q_global_to_local))
    v_local = np.array(q_v_local[1:])  # 提取虚部作为局部坐标系下的线速度

    # 将角速度从全局坐标系转换到局部坐标系
    # print("q_omega_global: ", v_omega_global, "q_global_to_local: ", q_global_to_local)
    q_omega_global = np.array([0, v_omega_global[0], v_omega_global[1], v_omega_global[2]])  # 角速度向量表示为四元数
    q_omega_local = rotations.quat_mul(rotations.quat_conjugate(q_global_to_local), rotations.quat_mul(q_omega_global, q_global_to_local))
    v_omega_local = np.array(q_omega_local[1:])  # 提取虚部作为局部坐标系下的角速度

    return v_local, v_omega_local

class LeggedRobot(OrcaGymAgent):
    def __init__(self, 
                 env_id: str,                 
                 agent_name: str, 
                 task: str,
                 max_episode_steps: int,
                 dt: float,
                 **kwargs):
        
        super().__init__(env_id, agent_name, task, max_episode_steps, dt, **kwargs)

        robot_config = LeggedRobotConfig[get_legged_robot_name(agent_name)]

        self._leg_joint_names = self.name_space_list(robot_config["leg_joint_names"])
        self._base_joint_name = self.name_space(robot_config["base_joint_name"])
        self._joint_names = [self._base_joint_name] + self._leg_joint_names

        self._base_neutral_height_offset = robot_config["base_neutral_height_offset"]
        self._base_born_height_offset = robot_config["base_born_height_offset"]
        self._command_lin_vel_range_x = robot_config["command_lin_vel_range_x"]
        self._command_lin_vel_range_y = robot_config["command_lin_vel_range_y"]
        self._command_ang_vel_range = robot_config["command_ang_vel_range"]
        self._command_ang_rate = robot_config["command_ang_rate"]

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
        self._foot_touch_force_air_threshold = robot_config["foot_touch_force_air_threshold"]
        self._foot_touch_air_time_threshold = robot_config["foot_touch_air_time_threshold"]

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

        self._gravity_quat = rotations.euler2quat([0.0, 0.0, -9.81000042])

        env_idx = int(self._env_id.split("-")[-1])
        # print("agent env_id: ", env_idx, "log_env_ids: ", robot_config["log_env_ids"])
        self._reward_printer = None
        if env_idx in robot_config["log_env_ids"] and self._agent_name in robot_config["log_agent_names"]:
            self._reward_printer = RewardPrinter()

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

        self._body_height, self._body_lin_vel, self._body_ang_vel, \
            self._body_orientation = self._get_body_local(joint_qpos, joint_qvel)

        # self._imu_data = self._get_imu_data(sensor_data)
        self._foot_touch_force = self._get_foot_touch_force(sensor_data)  # Penalty if the foot touch force is too strong
        self._update_foot_touch_air_time(self._foot_touch_force)  # Reward for air time of the feet
        self._leg_contact = self._get_leg_contact(contact_dict)            # Penalty if the leg is in contact with the ground
        # self._imu_site_pos_quat = site_pos_quat[self._imu_site_name].copy()
        # imu_site_pos_quat = np.concatenate([self._imu_site_pos_quat["xpos"], self._imu_site_pos_quat["xquat"]]).flatten()
        # imu_mocap_pos_quat = np.concatenate([self._imu_mocap_pos_quat["pos"], self._imu_mocap_pos_quat["quat"]]).flatten()

        self._achieved_goal = self._get_base_contact(contact_dict)         # task failed if the base is in contact with the ground
        self._desired_goal = np.zeros(1)      # 1.0 if the base is in contact with the ground, 0.0 otherwise

        obs = np.concatenate(
                [
                    self._body_lin_vel.copy(),
                    self._body_ang_vel.copy(),
                    self._body_orientation.copy(),
                    self._command_values.copy(),
                    (self._leg_joint_qpos - self._neutral_joint_values),
                    self._leg_joint_qvel.copy(),
                    self._action.copy(),
                ]).flatten()

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

        # print("agnet ", self._agent_name, " Action: ", action)

        if self._task == "no_action":
            return

        self._last_action = self._action.copy()
        self._action = action.copy()

        # print("Agent: ", self.name, "Orignal action: ", action)

        for i in range(len(action)):
            # 线性变换到 ctrl range 空间
            # print("action: ", action[i])
            # print("action_space_range: ", self._action_space_range)
            # print("ctrl_range: ", self._ctrl_range[i])
            if (self._actuator_type == "position"):
                # ctrl_delta = action[i] * self._action_scale # Use the action to control the joint position directly
                ctrl_delta = np.interp(action[i] * self._action_scale, self._action_space_range, self._ctrl_delta_range[i])
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
        # print("imu mocap: ", self._imu_mocap_pos_quat)
        self._imu_mocap_pos_quat["quat"] = rotations.quat_mul(rotations.euler2quat([0, 0, self._command["ang_vel"] * self.dt]), self._imu_mocap_pos_quat["quat"])
        global_lin_vel, _ = local2global(self._imu_mocap_pos_quat["quat"], self._command["lin_vel"], np.array([0,0,0]))
        self._imu_mocap_pos_quat["pos"] += global_lin_vel * self.dt
        
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

        # Rotate the base body around the Z axis
        z_rotation_quat = rotations.euler2quat([0, 0, np.random.uniform(-np.pi, np.pi)])

        # Get cached default joint values
        joint_neutral_qpos = self.get_joint_neutral()
        base_neutral_qpos = copy.deepcopy(self._init_base_joint_qpos)

        # Move along the Z axis to the born height
        self._base_height_target = base_neutral_qpos[self._base_joint_name][2] - self._base_neutral_height_offset
        base_neutral_qpos[self._base_joint_name][2] = self._base_height_target + self._base_born_height_offset

        # Use the rotate quate
        base_rotate_quat = base_neutral_qpos[self._base_joint_name][3:]
        base_neutral_qpos[self._base_joint_name][3:] = rotations.quat_mul(base_rotate_quat, z_rotation_quat)
        # print("Base neutral qpos: ", base_neutral_qpos)
        joint_neutral_qpos.update(base_neutral_qpos)

        # Get cached default imu mocap position and quaternion
        self._imu_mocap_pos_quat = {"pos": self._init_imu_site_pos_quat["xpos"].copy(), 
                                    "quat": self._init_imu_site_pos_quat["xquat"].copy()}
        # Move along the Z axis to the born height
        self._imu_mocap_pos_quat["pos"][2] -= self._base_neutral_height_offset

        # Use the rotate quate
        self._imu_mocap_pos_quat["quat"] = rotations.quat_mul(self._imu_mocap_pos_quat["quat"], z_rotation_quat)
        

        self._command = self._genarate_command()
        self._command_values = np.concatenate([self._command["lin_vel"], [self._command["ang_vel"]]]).flatten()
        # print("Command: ", self._command)

        return joint_neutral_qpos, {self._imu_mocap_name: self._imu_mocap_pos_quat}

    def is_terminated(self, achieved_goal, desired_goal) -> bool:
        assert achieved_goal.shape == desired_goal.shape
        return any(achieved_goal != desired_goal)

    def is_success(self, achieved_goal, desired_goal) -> np.float32:
        if self.truncated and not self.is_terminated(achieved_goal, desired_goal):
            return 1.0
        else:
            return 0.0

    def _compute_reward_alive(self, coeff) -> SupportsFloat:
        return 0.0
    
        reward = 1.0 * coeff * self.dt
        self._print_reward("Alive reward: ", reward)
        return reward

    def _compute_reward_success(self, achieved_goal, desired_goal, coeff) -> SupportsFloat:
        return 0.0
    
        reward = (1.0 if (self.is_success(achieved_goal, desired_goal) > 0) else 0.0) * coeff * self.dt
        self._print_reward("Success reward: ", reward)
        return reward
        
    def _compute_reward_failure(self, achieved_goal, desired_goal, coeff) -> SupportsFloat:
        reward = (-1.0 if self.is_terminated(achieved_goal, desired_goal) else 0.0) * coeff * self.dt
        self._print_reward("Failure reward: ", reward)
        return reward
        
    def _compute_reward_contact(self, coeff) -> SupportsFloat:
        reward = (-np.sum(self._leg_contact)) * coeff * self.dt
        self._print_reward("Contact reward: ", reward)
        return reward
    
    def _compute_reward_foot_touch(self, coeff) -> SupportsFloat:
        threshold = self._foot_touch_force_threshold
        reward = (-np.sum([(force - threshold) if force > threshold else 0 for force in self._foot_touch_force])) * coeff * self.dt
        self._print_reward("Foot touch reward: ", reward)
        return reward
    
    def _compute_reward_joint_angles(self, coeff) -> SupportsFloat:
        return 0.0
    
        base_reward = 1 * len(self._leg_joint_qpos)
        reward = (base_reward - np.sum(np.abs(self._leg_joint_qpos - self._neutral_joint_values))) * coeff * self.dt
        self._print_reward("Joint angles reward: ", reward)
        return reward
    
    def _compute_reward_joint_accelerations(self, coeff) -> SupportsFloat:
        reward = (-np.sum(np.abs(self._leg_joint_qacc))) * coeff * self.dt
        self._print_reward("Joint accelerations reward: ", reward)
        return reward
    
    def _compute_reward_limit(self, coeff) -> SupportsFloat:
        return 0.0
        # limit_threshold = 0.8
        # reward = 0.0

        # for i in range(len(self._ctrl)):
        #     ctrl = max(self._ctrl_range_low[i] + 0.01, min(self._ctrl[i], self._ctrl_range_high[i] - 0.01))
        #     if ctrl < self._ctrl_range_low[i] * limit_threshold:
        #         reward -= 1 / abs(self._ctrl_range_low[i] - ctrl)
        #     elif ctrl > self._ctrl_range_high[i] * limit_threshold:
        #         reward -= 1 / abs(self._ctrl_range_high[i] - ctrl)
                
        reward = -(np.sum(self._action == self._action_space_range[0]) + np.sum(self._action == self._action_space_range[1])) * coeff * self.dt
        self._print_reward("Limit over threshold reward: ", reward)
        return reward
    
    def _compute_reward_action_rate(self, coeff) -> SupportsFloat:
        reward = -np.sum(np.square(self._last_action - self._action)) * coeff * self.dt
        self._print_reward("Action rate reward: ", reward)
        return reward
    
    def _compute_reward_base_gyro(self, coeff) -> SupportsFloat:
        return 0.0
    
        reward = (-np.sum(np.abs(self._imu_data_gyro))) * coeff * self.dt
        self._print_reward("Base gyro reward: ", reward)
        return reward
    
    def _compute_reward_base_accelerometer(self, coeff) -> SupportsFloat:
        return 0.0
    
        reward = (-np.sum(abs(self._imu_data_accelerometer - np.array([0.0, 0.0, -9.81])))) * coeff * self.dt
        self._print_reward("Base accelerometer reward: ", reward)
        return reward
    
    def _compute_reward_follow_command_linvel(self, coeff) -> SupportsFloat:
        tracking_sigma = 0.25  # 奖励的衰减因子
        # 计算误差 (x 和 y 方向)
        lin_vel_error = np.sum(np.square(self._command["lin_vel"][:2] - self._body_lin_vel[:2]))
        # 计算指数衰减奖励
        reward = np.exp(-lin_vel_error / tracking_sigma) * coeff * self.dt
        self._print_reward("Follow command linvel reward: ", reward)
        return reward
    
    def _compute_reward_follow_command_angvel(self, coeff) -> SupportsFloat:
        tracking_sigma = 0.25  # 奖励的衰减因子
        ang_vel_error = pow(self._command["ang_vel"] - self._body_ang_vel[2], 2)  # 只考虑 Z 轴的角速度
        reward = np.exp(-ang_vel_error / tracking_sigma) * coeff * self.dt
        self._print_reward("Follow command angvel reward: ", reward)
        return reward
    
    def _compute_reward_height(self, coeff) -> SupportsFloat:
        base_reward = 0.1
        reward = (base_reward - abs(self._body_height - self._base_height_target)) * coeff * self.dt
        self._print_reward("Height reward: ", reward)
        return reward
    
    def _compute_reward_body_lin_vel(self, coeff) -> SupportsFloat:
        # Penalty if the body linear velocity in Z axis is too high
        reward = -pow(self._body_lin_vel[2], 2) * coeff * self.dt   
        self._print_reward("Body linear velocity reward: ", reward)
        return reward
    
    def _compute_reward_body_ang_vel(self, coeff) -> SupportsFloat:
        # Penalty if the body angular velocity in XY axis (Yaw) is too high
        reward = -np.sum((self._body_ang_vel[:2]) ** 2) * coeff * self.dt
        self._print_reward("Body angular velocity reward: ", reward)
        return reward
    
    def _compute_reward_body_orientation(self, coeff) -> SupportsFloat:
        # Penalty if the body orientation in XY axis (Yaw) is too high
        reward = -np.sum(np.square(self._body_orientation[:2])) * coeff * self.dt
        self._print_reward("Body orientation reward: ", reward)
        return reward
    
    def _compute_feet_air_time(self, coeff) -> SupportsFloat:
        reward = 0.0
        # print("feet air time: ", self._foot_touch_air_time)
        for i in range(len(self._foot_touch_air_time)):
            if self._foot_touch_air_time[i] > self._foot_touch_air_time_threshold:
                reward += self._foot_touch_air_time[i] * coeff * self.dt
        self._print_reward("Feet air time reward: ", reward)
        return reward
        
    def compute_reward(self, achieved_goal, desired_goal) -> SupportsFloat:
        if self._is_obs_updated:
            total_reward = 0.0

            reward_alive_coeff = 0
            reward_success_coeff = 0
            reward_failure_coeff = 100
            reward_contact_coeff = 1
            reward_foot_touch_coeff = 0.01
            reward_joint_angles_coeff = 0.1
            reward_joint_accelerations_coeff = 0.00001
            reward_limit_coeff = 0
            reward_action_rate_coeff = 0.001
            reward_base_gyro_coeff = 0.01
            reward_base_accelerometer_coeff = 0.001
            reward_follow_command_linvel_coeff = 1
            reward_follow_command_angvel_coeff = 0.5
            reward_height_coeff = 1
            reward_body_lin_vel_coeff = 2
            reward_body_ang_vel_coeff = 0.05
            reward_body_orientation_coeff = 5
            reward_feet_air_time_coeff = 0.1

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
            total_reward +=  self._compute_reward_follow_command_linvel(reward_follow_command_linvel_coeff)

            # *** Reward for following the command
            total_reward +=  self._compute_reward_follow_command_angvel(reward_follow_command_angvel_coeff)

            # *** Reward for maintaining the height
            total_reward +=  self._compute_reward_height(reward_height_coeff)

            # *** Penalty for body linear velocity in Z alxis
            total_reward +=  self._compute_reward_body_lin_vel(reward_body_lin_vel_coeff)

            # *** Penalty for body angular velocity in XY alxis (Yaw)
            total_reward +=  self._compute_reward_body_ang_vel(reward_body_ang_vel_coeff)

            # *** Penalty for body orientation in XY alxis (Yaw)
            total_reward +=  self._compute_reward_body_orientation(reward_body_orientation_coeff)

            # *** Reward for air time of the feet
            total_reward +=  self._compute_feet_air_time(reward_feet_air_time_coeff)

            self._print_reward("Total reward: ", total_reward)
            self._is_obs_updated = False
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
        # print("Contact dict: ", contact_dict)
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
    
    def _update_foot_touch_air_time(self, foot_touch_force: np.ndarray) -> np.ndarray:
        """
        Compute the air time for each foot
        """
        if not hasattr(self, "_foot_touch_air_time"):
            self._foot_touch_air_time = np.zeros(len(foot_touch_force))

        # If the robot is in the air, reset the air time   
        if all(foot_touch_force < self._foot_touch_force_air_threshold):
            self._foot_touch_air_time = np.zeros(len(foot_touch_force))
            return

        for i in range(len(foot_touch_force)):
            if foot_touch_force[i] > self._foot_touch_force_air_threshold:
                self._foot_touch_air_time[i] = 0
            else:
                self._foot_touch_air_time[i] += self.dt

    def _genarate_command(self) -> dict:
        """
        Command is a combination of linear and angular velocity, It's base on the local coordinate system of the robot.
        """

        lin_vel = np.array([self._np_random.uniform(0, self._command_lin_vel_range_x), 
                            self._np_random.uniform(-self._command_lin_vel_range_y, self._command_lin_vel_range_y),
                            0])
        if self._np_random.uniform() < self._command_ang_rate:
            ang_vel = self._np_random.uniform(-self._command_ang_vel_range, self._command_ang_vel_range)
        else:
            ang_vel = 0

        return {"lin_vel": lin_vel, "ang_vel": ang_vel}
    
    def _print_reward(self, message : str, reward : Optional[float] = None):
        if self._reward_printer is not None:
            self._reward_printer.print_reward(message, reward)

    def _get_body_local(self, joint_qpos, joint_qvel) -> tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Robots have a local coordinate system that is defined by the orientation of the base.
        Observations and rewards are given in the local coordinate system. 
        """
        body_joint_qpos = np.array(joint_qpos[self._base_joint_name]).flatten()
        body_joint_qvel = np.array(joint_qvel[self._base_joint_name]).flatten()

        body_height = body_joint_qpos[2]  # 局部坐标高度就是全局坐标高度
        body_orientation_quat = body_joint_qpos[3:7].copy()    # 全局坐标转局部坐标的旋转四元数
        body_lin_vel_vec_global = body_joint_qvel[:3].copy()    # 全局坐标系下的线速度
        body_ang_vel_vec_global = body_joint_qvel[3:6].copy()  # 全局坐标系下的角速度四元数
        # print("body_ang_vel_quat_global: ", body_ang_vel_vec_global, "body_joint_qvel: ", body_joint_qvel)
        # 获取局部坐标系下的线速度和角速度，用向量表示，角速度为 x,y,z 轴分量
        body_lin_vel, body_ang_vel = global2local(body_orientation_quat, body_lin_vel_vec_global, body_ang_vel_vec_global)
        body_orientation = rotations.quat2euler(body_orientation_quat)

        return body_height, body_lin_vel, body_ang_vel, body_orientation