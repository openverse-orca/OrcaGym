import numpy as np
from datetime import datetime
import math
from gymnasium.core import ObsType
from orca_gym.environment.multi_agent import OrcaGymAgent
from orca_gym.utils import rotations
from orca_gym.utils.reward_printer import RewardPrinter
from typing import Optional, Any, SupportsFloat
from gymnasium import spaces
import copy
import sys

from .legged_config import LeggedRobotConfig, LeggedObsConfig


def get_legged_robot_name(agent_name: str) -> str:
    if agent_name.startswith("go2"):
        return "go2"
    elif agent_name.startswith("A01B"):
        return "A01B"
    elif agent_name.startswith("AzureLoong"):
        return "AzureLoong"
    elif agent_name.startswith("Lite3"):
        return "Lite3"
    elif agent_name.startswith("g1"):
        return "g1"
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

def quat_angular_velocity(q1, q2, dt):
    # q1 = q1 / np.linalg.norm(q1)
    # q2 = q2 / np.linalg.norm(q2)
    # 计算四元数的角速度
    q_diff = rotations.quat_mul(q2, rotations.quat_conjugate(q1))
    # print("q_diff: ", q_diff)
    
    if q_diff[0] > 1.0:
        return 0.0
    
    angle = 2 * math.acos(q_diff[0])
    if angle > math.pi:
        angle = 2 * math.pi - angle
    return angle / dt

import math

def smooth_sqr_wave_np(phase, phase_freq, eps):
    """
    生成一个平滑的方波信号。

    参数:
    - phase: 当前相位（0到1之间的值，标量）。
    - phase_freq: 相位频率，用于调整信号的周期（标量）。
    - eps: 一个小值，用于防止分母为零（标量）。

    返回:
    - 平滑的方波信号，值介于0和1之间（标量）。
    """
    p = 2.0 * np.pi * phase * phase_freq
    numerator = np.sin(p)
    denominator = 2.0 * np.sqrt(np.sin(p)**2 + eps**2)
    return numerator / denominator + 0.5

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

        self._base_joint_name = self.name_space(robot_config["base_joint_name"])
        self._leg_joint_names = self.name_space_list(robot_config["leg_joint_names"])
        self._joint_names = [self._base_joint_name] + self._leg_joint_names

        self._base_neutral_height_offset = robot_config["base_neutral_height_offset"]
        self._base_born_height_offset = robot_config["base_born_height_offset"]

        self._neutral_joint_angles = robot_config["neutral_joint_angles"]
        self._neutral_joint_values = np.array([self._neutral_joint_angles[key] for key in self._neutral_joint_angles]).flatten()
        
        if "neutral_joint_angles_coeff" not in robot_config:
            self._neutral_joint_angles_coeff_value = None
        else:
            neutral_joint_angles_coeff = robot_config["neutral_joint_angles_coeff"]
            self._neutral_joint_angles_coeff_value = np.array([neutral_joint_angles_coeff[key] for key in neutral_joint_angles_coeff]).flatten()
        
        self._actuator_names = self.name_space_list(robot_config["actuator_names"])
        self._actuator_type = robot_config["actuator_type"]
        self._action_scale = robot_config["action_scale"]
        
        if "action_scale_mask" in robot_config:
            mask = robot_config["action_scale_mask"]
            self._action_scale_mask = np.array([mask[key] for key in mask]).flatten()
        
        self._imu_site_name = self.name_space(robot_config["imu_site_name"])
        self._contact_site_names = self.name_space_list(robot_config["contact_site_names"])
        self._site_names = [self._imu_site_name] + self._contact_site_names

        self._command_indicator_name = self.name_space(robot_config["command_indicator_name"])

        self._imu_sensor_framequat_name = self.name_space(robot_config["sensor_imu_framequat_name"])
        self._imu_sensor_gyro_name = self.name_space(robot_config["sensor_imu_gyro_name"])
        self._imu_sensor_accelerometer_name = self.name_space(robot_config["sensor_imu_accelerometer_name"])
        self._imu_sensor_names = [self._imu_sensor_framequat_name, self._imu_sensor_gyro_name, self._imu_sensor_accelerometer_name]
        self._use_imu_sensor = robot_config["use_imu_sensor"]

        self._foot_body_names = self.name_space_list(robot_config["foot_body_names"])
        self._foot_fitted_ground_pairs = robot_config["foot_fitted_ground_pairs"]
        self._foot_touch_sensor_names = self.name_space_list(robot_config["sensor_foot_touch_names"])
        self._foot_touch_force_threshold = robot_config["foot_touch_force_threshold"]
        self._foot_touch_force_air_threshold = robot_config["foot_touch_force_air_threshold"]
        self._foot_touch_force_step_threshold = robot_config["foot_touch_force_step_threshold"]
        self._foot_touch_air_time_ideal = robot_config["foot_touch_air_time_ideal"]
        
        self._foot_square_wave = robot_config["foot_square_wave"] if "foot_square_wave" in robot_config else None
        self._square_wave_phase = 0.0
        self._foot_leg_period = robot_config["foot_leg_period"] if "foot_leg_period" in robot_config else None
        

        self._sensor_names = self._imu_sensor_names + self._foot_touch_sensor_names

        self._ground_contact_body_names = robot_config["ground_contact_body_names"]  # gound body is not included in the agent's namespace
        self._base_contact_body_names = self.name_space_list(robot_config["base_contact_body_names"])
        self._leg_contact_body_names = self.name_space_list(robot_config["leg_contact_body_names"])
        self._all_contact_body_names = self._base_contact_body_names + self._leg_contact_body_names + self._foot_body_names


        self._ctrl = np.zeros(len(self._actuator_names))
        self._action = np.zeros(len(self._actuator_names))
        self._last_action = np.zeros(len(self._actuator_names))
        self._nu = len(self._actuator_names)
        self._nq = len(self._leg_joint_names) + (7 * len(self._base_joint_name))
        self._nv = len(self._leg_joint_names) + (6 * len(self._base_joint_name))

        # Scale the observation and noise
        self._gravity_quat = rotations.euler2quat([0.0, 0.0, -9.81000042])
        self._obs_scale_vec = self._get_obs_scale_vec()
        self._noise_scale_vec = self._get_noise_scale_vec()
        assert self._obs_scale_vec.shape == self._noise_scale_vec.shape, "obs_scale_vec and noise_scale_vec should have the same shape"

        # Domain randomization
        self._push_robots = robot_config["push_robots"]
        self._push_interval_s = robot_config["push_interval_s"]
        self._max_push_vel_xy = robot_config["max_push_vel_xy"]
        self._last_push_duration = 0.0

        self._randomize_friction = robot_config["randomize_friction"]
        self._friction_range = robot_config["friction_range"]
        self._randomize_base_mass = robot_config["randomize_base_mass"]
        self._added_mass_range = robot_config["added_mass_range"]
        
        self._pos_random_range = robot_config["pos_random_range"]
        
        # Curriculum learning
        self._curriculum_learning = robot_config["curriculum_learning"]
        self._curriculum_levels = robot_config["curriculum_levels"]
        self._curriculum_commands = robot_config["curriculum_commands"]
        if self._curriculum_learning:
            buffer_size = self._max_episode_steps
            self._curriculum_reward_buffer = {
                "lin_vel" : {
                    "buffer_size": buffer_size,
                    "buffer": np.zeros(buffer_size),
                    "index": 0,
                },
                "ang_vel" : {
                    "buffer_size": buffer_size,
                    "buffer": np.zeros(buffer_size),
                    "index": 0,
                },
            } 
            self._current_level = 0
            self._curriculum_clear_times = 0
            self._max_level_times = 0
        
        self._init_commands_config()
   
        env_idx = int(self._env_id.split("-")[-1])
        # print("agent env_id: ", env_idx, "log_env_ids: ", robot_config["log_env_ids"])
        self._reward_printer = None
        if env_idx in robot_config["log_env_ids"] and self.name in robot_config["log_agent_names"]:
            self._reward_printer = RewardPrinter()

        if self.name in robot_config["visualize_command_agent_names"]:
            self._visualize_command = True
        else:
            self._visualize_command = False
            
        if self.name == robot_config["playable_agent_name"]:
            self._is_playable = True
        else:
            self._is_playable = False
        self._player_control = False

        self._is_obs_updated = False
        self._setup_reward_functions(robot_config)
        self._setup_curriculum_functions()

    @property
    def neutral_joint_values(self) -> np.ndarray:
        return self._neutral_joint_values
    
    @property
    def playable(self) -> bool:
        return self._is_playable
    
    @property
    def player_control(self) -> bool:
        return self._player_control
    
    @player_control.setter
    def player_control(self, value: bool) -> None:
        self._player_control = value
        
    @property
    def added_mass_range(self) -> np.ndarray:
        return self._added_mass_range
    
    @property
    def base_joint_name(self) -> str:
        return self._base_joint_name

    # @property
    # def body_contact_force_threshold(self) -> float:
    #     return self._body_contact_force_threshold
    
    def get_joint_neutral(self) -> dict[str, np.ndarray]:
        joint_qpos = {}
        for name, value in zip(self._leg_joint_names, self.neutral_joint_values):
            joint_qpos[name] = np.array([value])
        return joint_qpos

    def get_obs(self, sensor_data : dict, qpos_buffer : np.ndarray, qvel_buffer : np.ndarray, qacc_buffer : np.ndarray, contact_dict : dict, site_pos_quat : dict, height_map : np.ndarray) -> dict:
        self._leg_joint_qpos[:] = qpos_buffer[self._qpos_index["leg_start"] : self._qpos_index["leg_start"] + self._qpos_index["leg_length"]]
        self._leg_joint_qvel[:] = qvel_buffer[self._qvel_index["leg_start"] : self._qvel_index["leg_start"] + self._qvel_index["leg_length"]]
        self._leg_joint_qacc[:] = qacc_buffer[self._qacc_index["leg_start"] : self._qacc_index["leg_start"] + self._qacc_index["leg_length"]]

        self._last_leg_joint_qpos[:] = self._leg_joint_qpos

        self._body_height, self._body_lin_vel, self._body_ang_vel, \
            self._body_orientation, self._body_pos = self._get_body_local(qpos_buffer, qvel_buffer)

        self._foot_height = self._get_foot_height(site_pos_quat, height_map)  # Foot position in the local frame
        self._foot_touch_force = self._get_foot_touch_force(sensor_data)  # Penalty if the foot touch force is too strong
        self._update_foot_touch_air_time(self._foot_touch_force)  # Reward for air time of the feet
        self._leg_contact = self._get_leg_contact(contact_dict)            # Penalty if the leg is in contact with the ground
        self._feet_contact, self._feet_self_contact = self._get_feet_contact(contact_dict)  # Penalty if the foot is in contact with each other
        self._feet_velp_norm = self._calc_feet_velp_norm(site_pos_quat)  # Penalty if the feet slip on the ground
        self._feet_velr_norm = self._calc_feet_velr_norm(site_pos_quat)  # Penalty if the feet wringing on the ground

        self._achieved_goal = self._get_base_contact(contact_dict).astype(np.float32)         # task failed if the base is in contact with the ground
        self._desired_goal = np.zeros(1).astype(np.float32)      # 1.0 if the base is in contact with the ground, 0.0 otherwise

        square_wave = self._compute_square_wave()
        sin_phase, cos_phase = self._compute_leg_period()
        # obs = np.concatenate(
        #         [
        #             # self._body_lin_vel,
        #             self._body_ang_vel,
        #             self._body_orientation,
        #             self._command_values,
        #             # np.array([square_wave]),
        #             # np.array([sin_phase, cos_phase]),
        #             (self._leg_joint_qpos - self._neutral_joint_values),
        #             self._leg_joint_qvel,
        #             self._action,
        #             # np.array([self._body_height]),
        #         ]).reshape(-1).astype(np.float32)
        obs = np.concatenate(
                [
                    self._body_lin_vel,
                    self._body_ang_vel,
                    self._body_orientation,
                    self._command_values,
                    np.array([square_wave]),
                    np.array([sin_phase, cos_phase]),
                    (self._leg_joint_qpos - self._neutral_joint_values),
                    self._leg_joint_qvel,
                    self._action,
                    # np.array([self._body_height]),
                ]).reshape(-1).astype(np.float32)
        
        obs *= self._obs_scale_vec

        noise_vec = ((self._np_random.random(len(self._noise_scale_vec)) * 2) - 1) * self._noise_scale_vec
        # print("obs: ", obs, "Noise vec: ", noise_vec)
        obs += noise_vec

        result = {
            "observation": obs,
            "achieved_goal": self._achieved_goal,
            "desired_goal": self._desired_goal,
        }

        self._is_obs_updated = True
        # print("Agent Obs: ", result)

        return result

    def set_init_state(self, joint_qpos: dict, init_site_pos_quat: dict) -> None:
        base_joint_qpos = np.array(joint_qpos[self._base_joint_name]).flatten()
        self._init_base_joint_qpos = {self._base_joint_name: base_joint_qpos}

    def _set_action(self, action) -> None:
        self._last_action[:] = self._action
        self._action[:] = action
    
    def set_action_space(self, action_space : spaces) -> None:
        self._action_space_range = np.array([action_space.low[0], action_space.high[0]]) 
        
    def init_joint_index(self, qpos_offset, qvel_offset, qacc_offset, qpos_length, qvel_length, qacc_length) -> None:
        self._qpos_index = {joint_name: {"offset": qpos_offset[i], "len": qpos_length[i]} for i, joint_name in enumerate(self.joint_names)}
        assert "leg_start" not in self._qpos_index and "leg_length" not in self._qpos_index, "qpos_index: joint_name 'leg_start' or 'leg_length' already exists"
        self._qpos_index["leg_start"], self._qpos_index["leg_length"] = self._calc_agent_leg_buffer_index(self._qpos_index)
        self._leg_joint_qpos = np.zeros(self._qpos_index["leg_length"])
        self._last_leg_joint_qpos = np.zeros(self._qpos_index["leg_length"])

        self._qvel_index = {joint_name: {"offset": qvel_offset[i], "len": qvel_length[i]} for i, joint_name in enumerate(self.joint_names)}
        assert "leg_start" not in self._qvel_index and "leg_length" not in self._qvel_index, "qvel_index: joint_name 'leg_start' or 'leg_length' already exists"
        self._qvel_index["leg_start"], self._qvel_index["leg_length"] = self._calc_agent_leg_buffer_index(self._qvel_index)
        self._leg_joint_qvel = np.zeros(self._qvel_index["leg_length"])

        self._qacc_index = {joint_name: {"offset": qacc_offset[i], "len": qacc_length[i]} for i, joint_name in enumerate(self.joint_names)}
        assert "leg_start" not in self._qacc_index and "leg_length" not in self._qacc_index, "qacc_index: joint_name 'leg_start' or 'leg_length' already exists"
        self._qacc_index["leg_start"], self._qacc_index["leg_length"] = self._calc_agent_leg_buffer_index(self._qacc_index)   
        self._leg_joint_qacc = np.zeros(self._qacc_index["leg_length"])
    
    def on_step(self, action, update_mocap: bool = False) -> dict:
        """
        Called after each step in the environment.
        """
        self._set_action(action)
        # self._set_action_ctrl(action_ctrl)

        visualized_command = {}
        if update_mocap and self._visualize_command:        
            # Do mocap update here.
            # print("Agent: ", self.name, "command indicator mocap: ", self._cmd_mocap_pos_quat, "command: ", self._command)
            
            # self._cmd_mocap_pos_quat["quat"] = rotations.quat_mul(rotations.euler2quat([0, 0, self._command["ang_vel"] * self.dt]), self._cmd_mocap_pos_quat["quat"])
            global_lin_vel, _ = local2global(self._cmd_mocap_pos_quat["quat"], self._command["lin_vel"], np.array([0,0,0]))
            self._cmd_mocap_pos_quat["pos"] += global_lin_vel * self.dt
            visualized_command = {self._command_indicator_name: self._cmd_mocap_pos_quat}
        
        return visualized_command

    def step(self, action, **kwargs):
        self._current_episode_step += 1
        step_info = self.on_step(action, **kwargs)
        return self._ctrl, step_info

    def _set_action_ctrl(self, action_ctrl):
        self._ctrl = action_ctrl

    def on_reset(self, height_map : np.ndarray) -> dict[str, np.ndarray]:
        # Reset ctrl to default values.
        if self._actuator_type == "position":
            assert self._ctrl.shape == self._neutral_joint_values.shape
            self._ctrl = self._neutral_joint_values.copy()
            # print("Neutral joint values: ", self._ctrl)
        elif self._actuator_type == "torque":
            self._ctrl = np.zeros(self._nu)
        else:
            raise ValueError(f"Unsupported actuator type: {self._actuator_type}")
        
        
        # Use curriculum learning to set the initial position
        if self._curriculum_learning:
            # print("Curriculum level: ", curriculum_level)
            level_offset = np.array([self._curriculum_levels[self._current_level]["offset"]]).flatten()
        else:
            level_offset = np.zeros(3)
            
        # Randomize the initial position x,y offset
        pos_noise = self._np_random.uniform(-self._pos_random_range, self._pos_random_range, 3)
        pos_noise[2] = 0.0
        pos_offset = pos_noise + level_offset

        # Rotate the base body around the Z axis
        z_rotation_angle = self._np_random.uniform(-np.pi, np.pi)
        z_rotation_quat = rotations.euler2quat([0, 0, z_rotation_angle])

        # Get cached default joint values
        joint_neutral_qpos = self.get_joint_neutral()
        self._base_neutral_qpos = copy.deepcopy(self._init_base_joint_qpos)
        
        
        # Use curriculum learning to set the initial position
        self._base_neutral_qpos[self._base_joint_name][:3] += pos_offset

        # Move along the Z axis to the born height
        self._base_height_target = self._base_neutral_qpos[self._base_joint_name][2] - self._base_neutral_height_offset
        self._base_neutral_qpos[self._base_joint_name][2] = self._base_height_target + self._base_born_height_offset
        
        # Use the rotate quate
        base_rotate_quat = self._base_neutral_qpos[self._base_joint_name][3:]
        self._base_neutral_qpos[self._base_joint_name][3:] = rotations.quat_mul(base_rotate_quat, z_rotation_quat)
        
        # Update the height of the base body
        self._base_neutral_qpos[self._base_joint_name][2] += self._compute_base_height(height_map)
        
        # print("Base neutral qpos: ", self._base_neutral_qpos)
        joint_neutral_qpos.update(self._base_neutral_qpos)

        self._command = self._genarate_command(z_rotation_angle)
        self._command_values = np.concatenate([self._command["lin_vel"], [self._command["ang_vel"]]]).flatten()
        self._command_resample_duration = 0
        # print("Command: ", self._command)
        
        self._last_push_duration = 0.0
        self._last_contact_site_xpos = None
        self._last_contact_site_xquat = None

        # Reset the velocity of the joints
        joint_zero_qvel = {joint_name: np.zeros(self._qvel_index[joint_name]["len"]) for joint_name in self.joint_names}
        
        return joint_neutral_qpos, joint_zero_qvel

    def _compute_base_height(self, height_map : np.ndarray) -> float:
        height_map_x = int(self._base_neutral_qpos[self._base_joint_name][0] * 10 + height_map.shape[0] / 2)
        height_map_y = int(self._base_neutral_qpos[self._base_joint_name][1] * 10 + height_map.shape[1] / 2)
        
        height_map_x_start = max(0, int(height_map_x - 5))
        height_map_x_end = min(height_map.shape[0], int(height_map_x + 5))
        height_map_y_start = max(0, int(height_map_y - 5))
        height_map_y_end = min(height_map.shape[1], int(height_map_y + 5))
        
        height_map_cliped = height_map[height_map_x_start:height_map_x_end, height_map_y_start:height_map_y_end]
        height = height_map_cliped.max()
        # print("Move up : ", height, "Height map: ", height_map_x, height_map_y)
        return height
        
    def reset_command_indicator(self, qpos_buffer : np.ndarray) -> dict[str, np.ndarray]:
        if not hasattr(self, "_cmd_mocap_pos_quat"):
            self._cmd_mocap_pos_quat = {
                "pos": np.array([0.0, 0.0, -10000.0]),  # Move the indicator to a far away place
                "quat": np.array([1.0, 0.0, 0.0, 0.0]),
            }
            
        if self._visualize_command:
            # print("Agent: ", self.name, "reset command indicator")
            base_qpos = qpos_buffer[self._qpos_index[self._base_joint_name]["offset"] : self._qpos_index[self._base_joint_name]["offset"] + self._qpos_index[self._base_joint_name]["len"]]
            self._cmd_mocap_pos_quat["pos"] = base_qpos[:3].copy()
            
            heading_angle = rotations.euler2quat([0, 0, self._command["heading_angle"]])
            self._cmd_mocap_pos_quat["quat"] = rotations.quat_mul(np.array([1.0, 0.0, 0.0, 0.0]), heading_angle)
            
        return {self._command_indicator_name: self._cmd_mocap_pos_quat}

    def is_terminated(self, achieved_goal, desired_goal) -> bool:
        assert achieved_goal.shape == desired_goal.shape
        return any(achieved_goal != desired_goal)

    def is_success(self, achieved_goal, desired_goal) -> np.float32:
        if self.truncated and not self.is_terminated(achieved_goal, desired_goal):
            return 1.0
        else:
            return 0.0
        
    def push_robot(self, qvel_buffer : np.ndarray) -> dict[str, np.ndarray]:
        if not self._push_robots:
            return {}
        
        if self._last_push_duration > self._push_interval_s:
            self._last_push_duration = 0.0
            push_vel = self._np_random.uniform(-self._max_push_vel_xy, self._max_push_vel_xy, 2)
            push_vel = np.concatenate([push_vel, [0.0, 0, 0, 0]])
            
            offset = self._qvel_index[self._base_joint_name]["offset"]
            len = self._qvel_index[self._base_joint_name]["len"]
            current_base_qvel = qvel_buffer[offset : offset + len]
            
            # print("Push robot: ", push_vel, "Current base qvel: ", current_base_qvel)
                    
            return {self._base_joint_name: current_base_qvel + push_vel}
        else:
            self._last_push_duration += self.dt
            return {}
        
    def randomize_foot_friction(self, random_friction : float, geom_dict : dict) -> dict:
        if not self._randomize_friction:
            return {}
        
        # 缩放到指定范围
        min_friction, max_friction = self._friction_range
        assert min_friction < max_friction, "min_friction should be less than max_friction"
        random_friction = max(min(random_friction, 1.0), 0.0)
        scaled_random_friction = min_friction + random_friction * (max_friction - min_friction)

        geom_friction_dict : dict[str, np.ndarray] = {}
        for name, geom in geom_dict.items():
            if geom["BodyName"] in self._foot_body_names:
                friction = geom["Friction"]
                friction[0] = scaled_random_friction
                geom_friction_dict[name] = friction

        return geom_friction_dict
    
    def generate_randomized_weight_on_base(self, random_weight, random_weight_pos, joint_dict):
        randomized_body_weight = {}
        for joint_name, joint in joint_dict.items():
            if joint_name == self._base_joint_name:
                body_id = joint["BodyID"]
                randomized_body_weight[body_id] = {"weight": random_weight, "pos": random_weight_pos}
        return randomized_body_weight

    def _compute_reward_alive(self, coeff) -> SupportsFloat:
        reward = 1.0 * coeff * self.dt
        self._print_reward("Alive reward: ", reward, coeff * self.dt)
        return reward

    def _compute_reward_success(self, coeff) -> SupportsFloat:
        reward = (1.0 if (self.is_success(self._achieved_goal, self._desired_goal) > 0) else 0.0) * coeff * self.dt
        self._print_reward("Success reward: ", reward, coeff * self.dt)
        return reward
        
    def _compute_reward_failure(self, coeff) -> SupportsFloat:
        reward = (-1.0 if self.is_terminated(self._achieved_goal, self._desired_goal) else 0.0) * coeff * self.dt
        self._print_reward("Failure reward: ", reward, coeff * self.dt)
        return reward
        
    def _compute_reward_contact(self, coeff) -> SupportsFloat:
        reward = (-np.sum(self._leg_contact)) * coeff * self.dt
        self._print_reward("Contact reward: ", reward, coeff * self.dt)
        return reward
    
    def _compute_reward_foot_touch(self, coeff) -> SupportsFloat:
        threshold = self._foot_touch_force_threshold
        reward = (-np.sum([(force - threshold) if force > threshold else 0 for force in self._foot_touch_force])) * coeff * self.dt
        self._print_reward("Foot touch reward: ", reward, coeff * self.dt)
        return reward
    
    def _compute_reward_joint_angles(self, coeff) -> SupportsFloat:
        # print("Joint angles: ", self._leg_joint_qpos, "Neutral joint values: ", self._neutral_joint_values, "Joint angles coeff: ", self._neutral_joint_angles_coeff_value)
        if self._neutral_joint_angles_coeff_value is None:
            joint_angles_diff = np.abs(self._leg_joint_qpos - self._neutral_joint_values)
        else:
            joint_angles_diff = np.abs(self._leg_joint_qpos - self._neutral_joint_values) * self._neutral_joint_angles_coeff_value
        # print("Joint angles diff: ", joint_angles_diff)
        reward = -np.sum(joint_angles_diff) * coeff * self.dt
        self._print_reward("Joint angles reward: ", reward, coeff * self.dt)
        return reward
    
    def _compute_reward_joint_accelerations(self, coeff) -> SupportsFloat:
        reward = (-np.sum(np.abs(self._leg_joint_qacc))) * coeff * self.dt
        self._print_reward("Joint accelerations reward: ", reward, coeff * self.dt)
        return reward
    
    def _compute_reward_limit(self, coeff) -> SupportsFloat:     
        reward = -(np.sum(self._action == self._action_space_range[0]) + np.sum(self._action == self._action_space_range[1])) * coeff * self.dt
        self._print_reward("Limit over threshold reward: ", reward, coeff * self.dt)
        return reward
    
    def _compute_reward_action_rate(self, coeff) -> SupportsFloat:
        reward = -np.sum(np.square(self._last_action - self._action)) * coeff * self.dt
        self._print_reward("Action rate reward: ", reward, coeff * self.dt)
        return reward
    
    def _compute_reward_base_gyro(self, coeff) -> SupportsFloat:
        reward = (-np.sum(np.abs(self._imu_data_gyro))) * coeff * self.dt
        self._print_reward("Base gyro reward: ", reward, coeff * self.dt)
        return reward
    
    def _compute_reward_base_accelerometer(self, coeff) -> SupportsFloat:
        reward = (-np.sum(abs(self._imu_data_accelerometer - np.array([0.0, 0.0, -9.81])))) * coeff * self.dt
        self._print_reward("Base accelerometer reward: ", reward, coeff * self.dt)
        return reward
    
    def _compute_reward_follow_command_linvel(self, coeff) -> SupportsFloat:
        tracking_sigma = 0.25  # 奖励的衰减因子
        # 计算误差 (x 和 y 方向)
        lin_vel_error = np.sum(np.square(self._command["lin_vel"][:2] - self._body_lin_vel[:2]))
        # 计算指数衰减奖励
        reward = np.exp(-lin_vel_error / tracking_sigma) * coeff * self.dt
        self._print_reward("Follow command linvel reward: ", reward, coeff * self.dt)
        
        # Curiiculum learning
        if self._curriculum_learning:
            buffer = self._curriculum_reward_buffer["lin_vel"]
            if buffer["index"] < buffer["buffer_size"]:
                buffer["buffer"][buffer["index"]] = reward
                buffer["index"] += 1

        return reward
    
    def _compute_reward_follow_command_angvel(self, coeff) -> SupportsFloat:
        tracking_sigma = 0.25  # 奖励的衰减因子
        ang_vel_error = pow(self._command["ang_vel"] - self._body_ang_vel[2], 2)  # 只考虑 Z 轴的角速度
        reward = np.exp(-ang_vel_error / tracking_sigma) * coeff * self.dt
        self._print_reward("Follow command angvel reward: ", reward, coeff * self.dt)
        # print("command ang vel: ", self._command["ang_vel"], "body ang vel: ", self._body_ang_vel[2])    
        
        # Curiiculum learning
        if self._curriculum_learning:
            buffer = self._curriculum_reward_buffer["ang_vel"]
            if buffer["index"] < buffer["buffer_size"]:
                buffer["buffer"][buffer["index"]] = reward
                buffer["index"] += 1
            
        return reward
    
    def _compute_reward_height(self, coeff) -> SupportsFloat:
        base_reward = 0.1
        reward = (base_reward - abs(self._body_height - self._base_height_target)) * coeff * self.dt
        self._print_reward("Height reward: ", reward, coeff * self.dt)
        return reward
    
    def _compute_reward_body_lin_vel(self, coeff) -> SupportsFloat:
        # Penalty if the body linear velocity in Z axis is too high
        reward = -pow(self._body_lin_vel[2], 2) * coeff * self.dt   
        self._print_reward("Body linear velocity reward: ", reward, coeff * self.dt)
        return reward
    
    def _compute_reward_body_ang_vel(self, coeff) -> SupportsFloat:
        # Penalty if the body angular velocity in XY axis (Yaw) is too high
        reward = -np.sum((self._body_ang_vel[:2]) ** 2) * coeff * self.dt
        self._print_reward("Body angular velocity reward: ", reward, coeff * self.dt)
        return reward
    
    def _compute_reward_body_orientation(self, coeff) -> SupportsFloat:
        # Penalty if the body orientation in XY axis (Yaw) is too high
        reward = -np.sum(np.square(self._body_orientation[:2])) * coeff * self.dt
        self._print_reward("Body orientation reward: ", reward, coeff * self.dt)
        return reward
    
    def _compute_feet_air_time(self, coeff) -> SupportsFloat:
        if self._command["lin_vel"][0] == 0.0 and self._command["lin_vel"][1] == 0.0:
            return 0.0
        
        reward = 0.0
        # Penalty if the feet touch the ground too fast.
        # If air time is longer than the ideal air time, reward is still 0. 
        # The agent need to put it's feet on the ground to get the reward. 
        # So the agent will learn to keep the feet in the air for the ideal air time, go get the maximum reward in one episode.
        
        # print("feet air time: ", self._foot_touch_air_time)
        for i in range(len(self._foot_touch_air_time)):
            if self._foot_in_air_time[i] > self._foot_touch_air_time_ideal:
                reward -= (self._foot_in_air_time[i] - self._foot_touch_air_time_ideal) * coeff * self.dt
            if self._foot_touch_air_time[i] > 0:
                # reward += min((self._foot_touch_air_time[i] - self._foot_touch_air_time_ideal), 0)  * coeff * self.dt
                reward += (self._foot_touch_air_time[i] - self._foot_touch_air_time_ideal) * coeff * self.dt
                # reward += -(self._foot_touch_air_time[i] - self._foot_touch_air_time_ideal)**2  * coeff * self.dt

        self._print_reward("Feet air time reward: ", reward, coeff * self.dt)
        return reward

    def _compute_reward_feet_self_contact(self, coeff) -> SupportsFloat:
        reward = (-np.sum(self._feet_self_contact)) * coeff * self.dtcompute_reward_symmetry
        self._print_reward("Foot self contact reward: ", reward, coeff * self.dt)
        return reward
    
    def _compute_reward_feet_slip(self, coeff) -> SupportsFloat:
        reward = -np.sum(self._feet_velp_norm * self._feet_contact) * coeff * self.dt
        self._print_reward("Foot slip reward: ", reward, coeff * self.dt)
        return reward
    
    def _compute_reward_feet_wringing(self, coeff) -> SupportsFloat:
        reward = -np.sum(self._feet_velr_norm * self._feet_contact) * coeff * self.dt
        self._print_reward("Foot wringing reward: ", reward, coeff * self.dt)
        return reward
    
    def _compute_reward_feet_fitted_ground(self, coeff) -> SupportsFloat:
        # avoid stepping when no command is given
        if self._command["lin_vel"][0] == 0.0 and self._command["lin_vel"][1] == 0.0:
            return 0.0
        
        reward = 0.0
        for i in range(len(self._foot_fitted_ground_pairs)):
            foot_front_index = self._foot_fitted_ground_pairs[i][0]
            foot_back_index = self._foot_fitted_ground_pairs[i][1]
            reward += -(1.0 if bool(self._feet_contact[foot_front_index]) ^ bool(self._feet_contact[foot_back_index]) else 0.0) * coeff * self.dt
        self._print_reward("Foot fitted ground reward: ", reward, coeff * self.dt)
        # print("Foot fitted ground reward: ", reward)
        return reward
    
    def _compute_reward_fly(self, coeff) -> SupportsFloat:
        reward = -1.0 * (np.sum(self._feet_contact) == 0) * coeff * self.dt
        self._print_reward("Fly reward: ", reward, coeff * self.dt)
        return reward
    
    def _compute_reward_stepping(self, coeff) -> SupportsFloat:
        reward = 0.0
        
        if self._command["lin_vel"][0] == 0.0 and self._command["lin_vel"][1] == 0.0:
        # if self._command["lin_vel"][0] == 0.0:
            # Penalty if the robot is stepping when no command is given
            reward = np.sum(self._feet_contact - np.ones(len(self._feet_contact))) * coeff * self.dt
        
        self._print_reward("Stepping reward: ", reward, coeff * self.dt)
        return reward

    def _compute_reward_feet_contact(self, coeff) -> SupportsFloat:
        # Reward if the feet are on the ground while the leg is in stance phase
        is_stance = self._leg_phase[:] < self._foot_leg_period["stance_threshold"]
        contact = self._feet_contact[:] > 0
                
        # print("Robot: ", self.name, "Feet contact: ", contact, "Stance: ", is_stance, "command: ", self._command["lin_vel"][0])
                
        reward = np.sum(~(contact ^ is_stance)) * coeff * self.dt
        self._print_reward("Feet contact reward: ", reward, coeff * self.dt)
        return reward
    
    def _compute_reward_feet_swing_height(self, coeff) -> SupportsFloat:
        # avoid stepping when no command is given
        if self._command["lin_vel"][0] == 0.0 and self._command["lin_vel"][1] == 0.0:
            return 0.0
        # Penalize the feet swing in error height
        swing_height = self._foot_leg_period["swing_height"]
        contact = self._feet_contact[:] > 0
        pos_error = np.square(self._foot_height[:] - swing_height) * ~contact
        
        # print("Feet swing height: ", self._foot_height, "Swing height: ", swing_height, "Contact: ", contact, "Pos error: ", pos_error)
        
        reward = -np.sum(pos_error) * coeff * self.dt
        self._print_reward("Feet swing height reward: ", reward, coeff * self.dt)
        return reward

    def _compute_reward_feet_swing_height_v2(self, coeff) -> SupportsFloat:
        # avoid stepping when no command is given
        if self._command["lin_vel"][0] == 0.0 and self._command["lin_vel"][1] == 0.0:
            return 0.0

        # Get target swing height from config
        swing_height = self._foot_leg_period["swing_height"]
        
        # Get foot positions in world frame
        foot_site_pos = [self._last_contact_site_xpos[foot_site_name] for foot_site_name in self._contact_site_names]
        
        # Get base position and orientation
        # body_qpos_index = self._qpos_index[self._base_joint_name]
        # body_joint_qpos = self._qpos_buffer[body_qpos_index["offset"] : body_qpos_index["offset"] + body_qpos_index["len"]]
        body_pos = self._body_pos[0:3]
        body_quat = self._body_pos[3:7]
        
        # Transform foot positions to body frame using global2local
        foot_pos_local = []
        for foot_pos in foot_site_pos:
            # Get relative position in global frame
            pos_relative = foot_pos - body_pos
            # Convert to local frame using global2local
            # We pass zero angular velocity since we only care about position
            local_pos, _ = global2local(body_quat, pos_relative, np.zeros(3))
            foot_pos_local.append(local_pos)
        
        # Calculate height error in body frame for each foot
        contact = self._feet_contact[:] > 0
        pos_error = np.zeros(len(foot_pos_local))
        
        for i, local_pos in enumerate(foot_pos_local):
            if not contact[i]:  # Only penalize feet in swing phase
                # Use z-coordinate in body frame
                current_height = local_pos[2]
                # Square error from desired swing height
                pos_error[i] = (current_height - swing_height) ** 2
                
                # # Additional penalty for negative height (foot above body)
                # if current_height < 0:
                #     pos_error[i] *= 2.0
        
        reward = -np.sum(pos_error) * coeff * self.dt
        self._print_reward("Feet swing height v2 reward: ", reward, coeff * self.dt)
        return reward
    
    def _compute_reward_contact_no_vel(self, coeff) -> SupportsFloat:
        # Penalize contact with no velocity
        contact = self._feet_contact[:] > 0
        contact_feet_vel = self._feet_velp_norm * contact
        
        # print("Contact feet vel: ", contact_feet_vel)
        
        reward = -np.sum(np.square(contact_feet_vel)) * coeff * self.dt
        self._print_reward("Contact no velocity reward: ", reward, coeff * self.dt)
        return reward
    
    ## useless
    # def _compute_reward_action_obs_diff(self, coeff) -> SupportsFloat:
    #     reward = -np.sum(np.square(self._last_leg_joint_qpos - self._ctrl)) * coeff * self.dt
    #     self._print_reward("Action obs diff reward: ", reward, coeff * self.dt)
    #     return reward
    
    # def compute_reward_symmetry(self, coeff) -> SupportsFloat:
    #     front_left_joint_qpos = self._leg_joint_qpos[0:3] - self._neutral_joint_values[0:3]
    #     front_right_joint_qpos = self._leg_joint_qpos[3:6] - self._neutral_joint_values[3:6]
    #     back_left_joint_qpos = self._leg_joint_qpos[6:9] - self._neutral_joint_values[6:9]
    #     back_right_joint_qpos = self._leg_joint_qpos[9:12] - self._neutral_joint_values[9:12]

    #     symmetry_diagonal_1 = np.square(front_left_joint_qpos - back_right_joint_qpos)
    #     symmetry_diagonal_2 = np.square(front_right_joint_qpos - back_left_joint_qpos)
    #     symmetry_front = np.square(front_left_joint_qpos + front_right_joint_qpos)
    #     symmetry_vertical = np.square(back_left_joint_qpos + back_right_joint_qpos)
        
    #     reward = -np.sum([symmetry_diagonal_1, symmetry_diagonal_2, symmetry_front, symmetry_vertical]) * coeff * self.dt
    #     self._print_reward("Symmetry reward: ", reward, coeff * self.dt)
    #     return reward
    
    def compute_reward(self, achieved_goal, desired_goal) -> SupportsFloat:
        if self._is_obs_updated:
            total_reward = 0.0
            self._achieved_goal = achieved_goal
            self._desired_goal = desired_goal

            for reward_function in self._reward_functions:
                if reward_function["coeff"] == 0:
                    continue
                else:    
                    total_reward += reward_function["function"](reward_function["coeff"])

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
                    
        if self.player_control:
            contact_result = False
            
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
    
    def _get_feet_contact(self, contact_dict : dict) -> np.ndarray:
        """
        Check if the feet are in contact with each other
        """
        feet_contact_result = np.zeros(len(self._foot_body_names))
        feet_self_contact_result = np.zeros(len(self._foot_body_names))
        for i, foot_body_name in enumerate(self._foot_body_names):
            if self._foot_touch_force[i] > self._foot_touch_force_step_threshold and foot_body_name in contact_dict:
                feet_contact_result[i] = 1.0
                foot_contact_list = contact_dict[foot_body_name]
                if any([contact_body_name in self._foot_body_names for contact_body_name in foot_contact_list]):
                    feet_self_contact_result[i] = 1.0
        return feet_contact_result, feet_self_contact_result
    
    def _calc_feet_velp_norm(self, site_pos_quat : dict) -> tuple[np.ndarray, np.ndarray]:
        feet_velp_norm = np.zeros(len(self._contact_site_names))
        
        if self._last_contact_site_xpos is None:
            self._last_contact_site_xpos = {contact_site_name: site_pos_quat[contact_site_name]["xpos"] for contact_site_name in self._contact_site_names}
            
        for i, contact_site_name in enumerate(self._contact_site_names):
            feet_velp_norm[i] = np.linalg.norm((site_pos_quat[contact_site_name]["xpos"] - self._last_contact_site_xpos[contact_site_name])[:2]) / self.dt
            self._last_contact_site_xpos[contact_site_name] = site_pos_quat[contact_site_name]["xpos"]

        return feet_velp_norm
    
    def _calc_feet_velr_norm(self, site_pos_quat : dict) -> tuple[np.ndarray, np.ndarray]:
        feet_velr_norm = np.zeros(len(self._contact_site_names))
        
        if self._last_contact_site_xquat is None:
            self._last_contact_site_xquat = {contact_site_name: site_pos_quat[contact_site_name]["xquat"] for contact_site_name in self._contact_site_names}

        for i, contact_site_name in enumerate(self._contact_site_names):
            feet_velr_norm[i] = abs(quat_angular_velocity(self._last_contact_site_xquat[contact_site_name], site_pos_quat[contact_site_name]["xquat"], self.dt))
            self._last_contact_site_xquat[contact_site_name] = site_pos_quat[contact_site_name]["xquat"]

        return feet_velr_norm
    
    def _get_imu_data(self, sensor_data: dict) -> np.ndarray:
        self._imu_data_framequat = sensor_data[self._imu_sensor_framequat_name]
        self._imu_data_gyro = sensor_data[self._imu_sensor_gyro_name]
        self._imu_data_accelerometer = sensor_data[self._imu_sensor_accelerometer_name]
        imu_data = np.concatenate([self._imu_data_framequat, self._imu_data_gyro, self._imu_data_accelerometer])
        return imu_data.flatten()
    
    def _get_foot_height(self, site_pos_quat: dict, height_map : np.ndarray) -> np.ndarray:
        foot_site_pos = [site_pos_quat[foot_site_name]["xpos"] for foot_site_name in self._contact_site_names]
        foot_height = np.zeros(len(foot_site_pos))
        for i in range(len(foot_site_pos)):
            foot_height[i] = foot_site_pos[i][2] - height_map[int(foot_site_pos[i][0] * 10 + height_map.shape[0] / 2), 
                                                              int(foot_site_pos[i][1] * 10 + height_map.shape[1] / 2)]

        # print("Foot site pos: ", foot_site_pos, "Foot height: ", foot_height)
        return foot_height
    
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
            self._foot_in_air_time = np.zeros(len(foot_touch_force))

        # If the robot is in the air, reset the air time   
        # if all(foot_touch_force < self._foot_touch_force_air_threshold):
        #     self._foot_touch_air_time = np.zeros(len(foot_touch_force))
        #     return

        for i in range(len(foot_touch_force)):
            if foot_touch_force[i] > self._foot_touch_force_air_threshold:
                self._foot_touch_air_time[i] = self._foot_in_air_time[i]
                self._foot_in_air_time[i] = 0
            else:
                self._foot_in_air_time[i] += self.dt
                self._foot_touch_air_time[i] = 0

    def _genarate_command(self, heading_angle : float) -> dict:
        """
        Command is a combination of linear and angular velocity, It's base on the local coordinate system of the robot.
        """
        if self._np_random.random() < 1.0:            
            # Forward
            lin_vel = np.array([self._np_random.uniform(0, self._command_lin_vel_range_x[1]), 
                                self._np_random.uniform(self._command_lin_vel_range_y[0], self._command_lin_vel_range_y[1]),
                                0])
        else:
            # Backward
            lin_vel = np.array([self._np_random.uniform(self._command_lin_vel_range_x[0], 0), 
                                self._np_random.uniform(self._command_lin_vel_range_y[0], self._command_lin_vel_range_y[1]),
                                0])
        
        # Avoiding the robot to move tremble when the linear velocity is too small
        if lin_vel[0] < self._command_lin_vel_threshold[1] and lin_vel[0] > self._command_lin_vel_threshold[0]:
            lin_vel = np.array([0, 0, 0])
        # if lin_vel[0] < self._command_lin_vel_threshold[1] and lin_vel[0] > self._command_lin_vel_threshold[0]:
        #     # lin_vel = np.array([0, 0, 0])
        #     lin_vel = np.array([0, lin_vel[1], 0])
        # if lin_vel[1] < self._command_lin_vel_threshold[1] and lin_vel[1] > -self._command_lin_vel_threshold[1]:
        #     lin_vel = np.array([lin_vel[0], 0, 0])

        return {"lin_vel": lin_vel, "ang_vel": 0, "heading_angle": heading_angle}
    
    def update_command(self, qpos_buffer : np.ndarray) -> None:
        """
        Update the z ang_vel by heading 
        """
        self._resample_command()

        # Get the body heading quaternion in global coordinate system
        body_qpos_index = self._qpos_index[self._base_joint_name]
        body_joint_qpos = qpos_buffer[body_qpos_index["offset"] : body_qpos_index["offset"] + body_qpos_index["len"]]
        body_orientation_quat = body_joint_qpos[3:7].copy()
        body_heading_angle = rotations.quat2euler(body_orientation_quat)[2]
        angle_error = self._command["heading_angle"] - body_heading_angle
        angle_error = (angle_error + np.pi) % (2 * np.pi) - np.pi
        self._command["ang_vel"] = min(max(angle_error, -self._command_ang_vel_range), self._command_ang_vel_range)
        self._command_values[3] = self._command["ang_vel"]

        # print("Command: ", self._command, "Body heading angle: ", body_heading_angle, "Angle error: ", angle_error)


    def _resample_command(self) -> None:
        self._command_resample_duration += self.dt
        if self._command_resample_duration > self._command_resample_interval:
            self._command_resample_duration = 0
            turn_angle = self._np_random.uniform(-self._command_ang_vel_range, self._command_ang_vel_range)
            self._command = self._genarate_command(self._command["heading_angle"] + turn_angle)
            self._command_values[:3] = self._command["lin_vel"]
            
            turn_quat = rotations.euler2quat([0, 0, turn_angle])
            self._cmd_mocap_pos_quat["quat"] = rotations.quat_mul(self._cmd_mocap_pos_quat["quat"], turn_quat)
    
    def _print_reward(self, message : str, reward : Optional[float] = 0, coeff : Optional[float] = 1) -> None:
        if self._reward_printer is not None:
            self._reward_printer.print_reward(message, reward, coeff)

    def _get_body_local(self, qpos_buffer : np.ndarray, qvel_buffer : np.ndarray) -> tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Robots have a local coordinate system that is defined by the orientation of the base.
        Observations and rewards are given in the local coordinate system. 
        """
        body_qpos_index = self._qpos_index[self._base_joint_name]
        body_joint_qpos = qpos_buffer[body_qpos_index["offset"] : body_qpos_index["offset"] + body_qpos_index["len"]]
        body_qvel_index = self._qvel_index[self._base_joint_name]
        body_joint_qvel = qvel_buffer[body_qvel_index["offset"] : body_qvel_index["offset"] + body_qvel_index["len"]]

        body_height = body_joint_qpos[2]  # 局部坐标高度就是全局坐标高度
        body_orientation_quat = body_joint_qpos[3:7].copy()    # 全局坐标转局部坐标的旋转四元数
        body_lin_vel_vec_global = body_joint_qvel[:3].copy()    # 全局坐标系下的线速度
        body_ang_vel_vec_global = body_joint_qvel[3:6].copy()  # 全局坐标系下的角速度四元数
        # print("body_ang_vel_quat_global: ", body_ang_vel_vec_global, "body_joint_qvel: ", body_joint_qvel)
        # 获取局部坐标系下的线速度和角速度，用向量表示，角速度为 x,y,z 轴分量
        body_lin_vel, body_ang_vel = global2local(body_orientation_quat, body_lin_vel_vec_global, body_ang_vel_vec_global)
        body_orientation = rotations.quat2euler(body_orientation_quat)
        body_orientation[2] = 0

        return body_height, body_lin_vel, body_ang_vel, body_orientation, body_joint_qpos
    
    def _get_obs_scale_vec(self):
        """ Sets a vector used to scale the observations.
            [NOTE]: Must be adapted when changing the observations structure

        Args:
            cfg (Dict): Environment config file

        Returns:
            [torch.Tensor]: Vector of scales used to normalize the observations
        """
        scale_lin_vel = np.array([1, 1, 1]) * LeggedObsConfig["scale"]["lin_vel"]
        scale_ang_vel = np.array([1, 1, 1]) * LeggedObsConfig["scale"]["ang_vel"]
        scale_orientation = np.array([1, 1, 1])  # No scaling on the orientation
        scale_command = np.array([LeggedObsConfig["scale"]["lin_vel"], LeggedObsConfig["scale"]["lin_vel"], LeggedObsConfig["scale"]["lin_vel"], LeggedObsConfig["scale"]["ang_vel"]])
        scale_square_wave = np.ones(1)  # No scaling on the square wave
        scale_leg_phase = np.ones(2)  # No scaling on the leg phase
        scale_leg_joint_qpos = np.array([1] * len(self._leg_joint_names)) * LeggedObsConfig["scale"]["qpos"]
        scale_leg_joint_qvel = np.array([1] * len(self._leg_joint_names)) * LeggedObsConfig["scale"]["qvel"]
        scale_action = np.array([1] * len(self._actuator_names)) # No scaling on the action
        scale_height = np.array([1]) * LeggedObsConfig["scale"]["height"]

        # scale_vec = np.concatenate([
        #     # scale_lin_vel, 
        #     scale_ang_vel, 
        #     scale_orientation, 
        #     scale_command, 
        #     # scale_square_wave,
        #     # scale_leg_phase,
        #     scale_leg_joint_qpos, 
        #     scale_leg_joint_qvel, 
        #     scale_action, 
        #     # scale_height
        #     ]).flatten()
        scale_vec = np.concatenate([
            scale_lin_vel, 
            scale_ang_vel, 
            scale_orientation, 
            scale_command, 
            scale_square_wave,
            scale_leg_phase,
            scale_leg_joint_qpos, 
            scale_leg_joint_qvel, 
            scale_action, 
            # scale_height
            ]).flatten()
        
        return scale_vec.astype(np.float32)

    def _get_noise_scale_vec(self):
        """ Sets a vector used to scale the noise added to the observations.
            [NOTE]: Must be adapted when changing the observations structure

        Args:
            cfg (Dict): Environment config file

        Returns:
            [torch.Tensor]: Vector of scales used to multiply a uniform distribution in [-1, 1]
        """
        noise_level = LeggedObsConfig["noise"]["noise_level"]
        noise_lin_vel = np.array([1, 1, 1]) * noise_level * LeggedObsConfig["noise"]["lin_vel"] * LeggedObsConfig["scale"]["lin_vel"]
        noise_ang_vel = np.array([1, 1, 1]) * noise_level * LeggedObsConfig["noise"]["ang_vel"] * LeggedObsConfig["scale"]["ang_vel"]
        noise_orientation = np.array([1, 1, 1]) * noise_level * LeggedObsConfig["noise"]["orientation"]
        noise_command = np.zeros(4)  # No noise on the command
        noise_square_wave = np.zeros(1)  # No noise on the square wave
        scale_leg_phase = np.zeros(2)  # No noise on the leg phase
        noise_leg_joint_qpos = np.array([1] * len(self._leg_joint_names)) * noise_level * LeggedObsConfig["noise"]["qpos"] * LeggedObsConfig["scale"]["qpos"]
        noise_leg_joint_qvel = np.array([1] * len(self._leg_joint_names)) * noise_level * LeggedObsConfig["noise"]["qvel"] * LeggedObsConfig["scale"]["qvel"]
        noise_action = np.zeros(len(self._actuator_names))  # No noise on the action
        noise_height = np.array([1]) * noise_level * LeggedObsConfig["noise"]["height"] * LeggedObsConfig["scale"]["height"]

        # noise_vec = np.concatenate([
        #     # noise_lin_vel, 
        #     noise_ang_vel, 
        #     noise_orientation, 
        #     noise_command, 
        #     # noise_square_wave,
        #     # scale_leg_phase,
        #     noise_leg_joint_qpos, 
        #     noise_leg_joint_qvel, 
        #     noise_action, 
        #     # noise_height
        #     ]).flatten()
        noise_vec = np.concatenate([
            noise_lin_vel, 
            noise_ang_vel, 
            noise_orientation, 
            noise_command, 
            noise_square_wave,
            scale_leg_phase,
            noise_leg_joint_qpos, 
            noise_leg_joint_qvel, 
            noise_action, 
            # noise_height
            ]).flatten()
        
        # print("noise vec: ", noise_vec)

        return noise_vec.astype(np.float32)
    
    
    def _calc_agent_leg_buffer_index(self, joint_index: dict) -> np.ndarray:
        """
        Calculate the start index and length of the agent buffer.
        The order of the joint names (defined in the config file) should be the same as they have been defined in the xml file.
        If the joint index overlap or gap, check the config file for details.
        """
        # for joint_name in self._leg_joint_names:
        #     if joint_name not in joint_index:
        #         raise ValueError(f"Joint name {joint_name} not found in the joint index")
        #     index = joint_index[joint_name]
        #     print("Joint index: ", joint_name, index)
        #     print("Joint index offset: ", index["offset"], "Joint index len: ", index["len"])
            
        index_array = np.array([np.array([joint_index[joint_name]["offset"], joint_index[joint_name]["len"]]) for joint_name in self._leg_joint_names])
        for i in range(len(index_array) - 1):
            if index_array[i, 0] + index_array[i, 1] > index_array[i + 1, 0]:
                raise ValueError("Joint index overlap")
            elif index_array[i, 0] + index_array[i, 1] < index_array[i + 1, 0]:
                raise ValueError("Joint index gap")
            
        index_start = index_array[0, 0]
        index_len = index_array[-1, 0] + index_array[-1, 1] - index_start
        return index_start, index_len    
    
    def _setup_reward_functions(self, robot_config : dict) -> None:
        reward_coeff = robot_config["reward_coeff"]
        self._reward_functions = [
            {"function": self._compute_reward_alive, "coeff": reward_coeff["alive"] if "alive" in reward_coeff else 0},
            {"function": self._compute_reward_success, "coeff": reward_coeff["success"] if "success" in reward_coeff else 0},
            {"function": self._compute_reward_failure, "coeff": reward_coeff["failure"] if "failure" in reward_coeff else 0},
            {"function": self._compute_reward_contact, "coeff": reward_coeff["contact"] if "contact" in reward_coeff else 0},
            {"function": self._compute_reward_foot_touch, "coeff": reward_coeff["foot_touch"] if "foot_touch" in reward_coeff else 0},
            {"function": self._compute_reward_joint_angles, "coeff": reward_coeff["joint_angles"] if "joint_angles" in reward_coeff else 0},
            {"function": self._compute_reward_joint_accelerations, "coeff": reward_coeff["joint_accelerations"] if "joint_accelerations" in reward_coeff else 0},
            {"function": self._compute_reward_limit, "coeff": reward_coeff["limit"] if "limit" in reward_coeff else 0},
            {"function": self._compute_reward_action_rate, "coeff": reward_coeff["action_rate"] if "action_rate" in reward_coeff else 0},
            {"function": self._compute_reward_base_gyro, "coeff": reward_coeff["base_gyro"] if "base_gyro" in reward_coeff else 0},
            {"function": self._compute_reward_base_accelerometer, "coeff": reward_coeff["base_accelerometer"] if "base_accelerometer" in reward_coeff else 0},
            {"function": self._compute_reward_follow_command_linvel, "coeff": reward_coeff["follow_command_linvel"] if "follow_command_linvel" in reward_coeff else 0},
            {"function": self._compute_reward_follow_command_angvel, "coeff": reward_coeff["follow_command_angvel"] if "follow_command_angvel" in reward_coeff else 0},
            {"function": self._compute_reward_height, "coeff": reward_coeff["height"] if "height" in reward_coeff else 0},
            {"function": self._compute_reward_body_lin_vel, "coeff": reward_coeff["body_lin_vel"] if "body_lin_vel" in reward_coeff else 0},
            {"function": self._compute_reward_body_ang_vel, "coeff": reward_coeff["body_ang_vel"] if "body_ang_vel" in reward_coeff else 0},
            {"function": self._compute_reward_body_orientation, "coeff": reward_coeff["body_orientation"] if "body_orientation" in reward_coeff else 0},
            {"function": self._compute_feet_air_time, "coeff": reward_coeff["feet_air_time"] if "feet_air_time" in reward_coeff else 0},
            {"function": self._compute_reward_feet_self_contact, "coeff": reward_coeff["feet_self_contact"] if "feet_self_contact" in reward_coeff else 0},
            {"function": self._compute_reward_feet_slip, "coeff": reward_coeff["feet_slip"] if "feet_slip" in reward_coeff else 0},
            {"function": self._compute_reward_feet_wringing, "coeff": reward_coeff["feet_wringing"] if "feet_wringing" in reward_coeff else 0},
            {"function": self._compute_reward_feet_fitted_ground, "coeff": reward_coeff["feet_fitted_ground"] if "feet_fitted_ground" in reward_coeff else 0},
            {"function": self._compute_reward_fly, "coeff": reward_coeff["fly"] if "fly" in reward_coeff else 0},
            {"function": self._compute_reward_stepping, "coeff": reward_coeff["stepping"] if "stepping" in reward_coeff else 0},
            {"function": self._compute_reward_feet_contact, "coeff": reward_coeff["feet_contact"] if "feet_contact" in reward_coeff else 0},
            {"function": self._compute_reward_feet_swing_height_v2, "coeff": reward_coeff["feet_swing_height"] if "feet_swing_height" in reward_coeff else 0},
            {"function": self._compute_reward_contact_no_vel, "coeff": reward_coeff["contact_no_vel"] if "contact_no_vel" in reward_coeff else 0},
            {"function": self._compute_reward_feet_contact, "coeff": reward_coeff["phase_contact"] if "phase_contact" in reward_coeff else 0},  
        ]
        
    def _setup_curriculum_functions(self):
        self._curriculum_functions = [
            {"function": self._rating_curriculum_follow_command_linvel, "coeff": 1},
            {"function": self._rating_curriculum_follow_command_angvel, "coeff": 0.5},
        ]
        
    def _rating_curriculum_follow_command_linvel(self, coeff) -> float:
        buffer_index = self._curriculum_reward_buffer["lin_vel"]["index"]
        rating = np.mean(self._curriculum_reward_buffer["lin_vel"]["buffer"][:buffer_index+1]) / (coeff * self.dt)
        return rating
    
    
    def _rating_curriculum_follow_command_angvel(self, coeff) -> float:
        buffer_index = self._curriculum_reward_buffer["ang_vel"]["index"]
        rating = np.mean(self._curriculum_reward_buffer["ang_vel"]["buffer"][:buffer_index+1]) / (coeff * self.dt)
        return rating
    
    def update_curriculum_level(self, qpos_buffer : np.ndarray) -> None:    
        if not self._curriculum_learning or self._player_control:
            return
            
        ratings = [curriculum_function["function"](curriculum_function["coeff"]) for curriculum_function in self._curriculum_functions]
        mean_rating = np.mean(ratings)
        
        # 低于奖励阈值，或者摔倒，降级
        # 高于奖励阈值，并达到行走距离，升级
        if mean_rating < self._curriculum_levels[self._current_level]["rating"] + self._curriculum_clear_times * 0.01:
            self._current_level = max(self._current_level - 1, 0)
            # print("Agent: ", self._env_id + self.name, "Level Downgrade! Curriculum level: ", self._curriculum_current_level, "mena rating: ", mean_rating)
        elif hasattr(self, "_base_neutral_qpos") and not self.is_terminated(self._achieved_goal, self._desired_goal):
            start_pos = self._base_neutral_qpos[self._base_joint_name][:3]
            current_pos = qpos_buffer[self._qpos_index[self._base_joint_name]["offset"] : self._qpos_index[self._base_joint_name]["offset"] + self._qpos_index[self._base_joint_name]["len"]][:3]
            move_distance = np.linalg.norm(start_pos - current_pos)
            # print("Agent: ", self._env_id + self.name, "Move distance: ", move_distance)
            if move_distance > self._curriculum_levels[self._current_level]["distance"] + self._curriculum_clear_times * 0.5:
                # print("Agent: ", self._env_id + self.name, "Level Upgrade! Curriculum level: ", self._curriculum_current_level, "mena rating: ", mean_rating, "Move distance: ", move_distance)
                if self._current_level == len(self._curriculum_levels) - 1:
                    self._curriculum_clear_times += 1
                    self._current_level = 0
                    if self._curriculum_clear_times > 10:
                        self._curriculum_clear_times = 0
                        self._max_level_times += 1
                    print("Agent: ", self._env_id + self.name, "Curriculum cleared! mean rating: ", mean_rating, "Move distance: ", move_distance, "Clear times: ", self._curriculum_clear_times, "Max level times: ", self._max_level_times)
                else:
                    self._current_level = min(self._current_level + 1, len(self._curriculum_levels) - 1)

        
        for buffer in self._curriculum_reward_buffer.values():
            buffer["index"] = 0
        # print("Curriculum reward buffer: ", self._curriculum_reward_buffer)
        
        # Update the command config for different levels
        self._reset_commands_config(self._current_level)
                    
    def init_playable(self):
        self._command_resample_interval = sys.maxsize
        self._max_episode_steps = sys.maxsize
        
    def update_playable(self, lin_vel : np.ndarray, turn_angle : float, rebone : bool = False):
        # print("Agent: ", self.name, "Update playable: ", lin_vel, turn_angle, rebone)
        
        self._command["lin_vel"] = lin_vel
        self._command["heading_angle"] += turn_angle
        self._command_values[:3] = self._command["lin_vel"]
        if rebone:
            self._max_episode_steps = self._current_episode_step  # 在下一步重新出生
        else:
            self._max_episode_steps = sys.maxsize
            
    def _reset_commands_config(self, current_level : int) -> None:
        command_type = self._curriculum_levels[current_level]["command_type"]
        self._command_lin_vel_range_x = self._curriculum_commands[command_type]["command_lin_vel_range_x"]
        self._command_lin_vel_range_y = self._curriculum_commands[command_type]["command_lin_vel_range_y"]
        self._command_lin_vel_threshold = self._curriculum_commands[command_type]["command_lin_vel_threshold"]
        self._command_ang_vel_range = self._curriculum_commands[command_type]["command_ang_vel_range"]
   
        
    def _init_commands_config(self) -> None:
        self._reset_commands_config(0)
        command_type = self._curriculum_levels[0]["command_type"]
        self._command_resample_interval = self._curriculum_commands[command_type]["command_resample_interval"]              

    def _compute_square_wave(self):
        """
        计算 square_wave 信号，条件不满足时使用平滑方波。

        参数:
        - time_to_stand_still: 当前时间到静止的时间（标量）。
        - static_delay: 静止延迟时间（标量）。
        - p5: 当条件满足时的常数值（标量）。
        - phase: 当前相位（0到1之间的值，标量）。
        - phase_freq: 相位频率，用于调整信号的周期（标量）。
        - eps: 平滑方波的 epsilon 值（标量）。

        返回:
        - square_wave 信号（标量）。
        """
        if self._foot_square_wave is None:
            return 0.0
        
        self._square_wave_phase = np.fmod(self._square_wave_phase + self.dt, 1.0)
        
        if self._command["lin_vel"][0] == 0.0:
            return self._foot_square_wave["p5"]
        else:
            return smooth_sqr_wave_np(self._square_wave_phase, self._foot_square_wave["phase_freq"], self._foot_square_wave["eps"])
        
    def _compute_leg_period(self) -> tuple:
        """
        Compute the leg period and phase for the robot.
        The leg period is the time it takes for one complete cycle of the leg movement.
        The phase is the current position of the leg in the cycle, ranging from 0 to 1.
        """
        if self._foot_leg_period is None:
            return 0.0, 0.0
        
        #if self._command["lin_vel"][0] == 0.0:
            # If the robot is not moving, return a constant value for the leg period and phase
            # print("Robot: ", self.name, " is not moving, return constant leg period and phase")
          #  self._leg_phase = np.zeros(4)
          #  return 0.0, 1.0
            
        period = self._foot_leg_period["period"]
        offset = self._foot_leg_period["offset"]
        leg_period_phase = (self._current_episode_step * self.dt) % period / period
        leg_period_phase_left = leg_period_phase
        leg_period_phase_right = (leg_period_phase + offset) % 1
        self._leg_phase = np.concatenate([[leg_period_phase_left, leg_period_phase_right], [leg_period_phase_right, leg_period_phase_left]]).flatten()
        
        # print("Leg period phase: ", leg_period_phase, "Leg period phase left: ", leg_period_phase_left, "Leg period phase right: ", leg_period_phase_right, " _leg_phase: ", self._leg_phase)
        
        sin_phase = np.sin(leg_period_phase * 2 * np.pi)
        cos_phase = np.cos(leg_period_phase * 2 * np.pi)
        
        # print("Robot: ", self.name, " Sin phase: ", sin_phase, "Cos phase: ", cos_phase)
        
        return sin_phase, cos_phase