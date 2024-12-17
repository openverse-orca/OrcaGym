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

from .legged_robot_config import LeggedRobotConfig, LeggedObsConfig


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

        self._base_joint_name = self.name_space(robot_config["base_joint_name"])
        self._leg_joint_names = self.name_space_list(robot_config["leg_joint_names"])
        self._joint_names = [self._base_joint_name] + self._leg_joint_names

        self._base_neutral_height_offset = robot_config["base_neutral_height_offset"]
        self._base_born_height_offset = robot_config["base_born_height_offset"]
        self._command_lin_vel_range_x = robot_config["command_lin_vel_range_x"]
        self._command_lin_vel_range_y = robot_config["command_lin_vel_range_y"]
        self._command_ang_vel_range = robot_config["command_ang_vel_range"]
        self._command_resample_interval = robot_config["command_resample_interval"]

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

        self._foot_body_names = self.name_space_list(robot_config["foot_body_names"])
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
        
        # Curriculum learning
        self._curriculum_learning = robot_config["curriculum_learning"]
        self._curriculum_levels = robot_config["curriculum_levels"]
        self._agent_pos_offset = robot_config["agent_pos_offset"]
        if self._curriculum_learning:
            self._curriculum_reward_buffer_size = robot_config["curriculum_reward_buffer_size"]
            self._curriculum_reward_buffer = np.zeros(self._curriculum_reward_buffer_size)
            self._curriculum_reward_buffer_index = 0
            self._curriculum_current_level = 0
        
        env_idx = int(self._env_id.split("-")[-1])
        # print("agent env_id: ", env_idx, "log_env_ids: ", robot_config["log_env_ids"])
        self._reward_printer = None
        if env_idx in robot_config["log_env_ids"] and self.name in robot_config["log_agent_names"]:
            self._reward_printer = RewardPrinter()

        if self.name in robot_config["visualize_command_agent_names"]:
            self._visualize_command = True
        else:
            self._visualize_command = False

        self._is_obs_updated = False
        self._setup_reward_functions()

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

    def get_obs(self, sensor_data : dict, qpos_buffer : np.ndarray, qvel_buffer : np.ndarray, qacc_buffer : np.ndarray, contact_dict : dict) -> dict:
        self._leg_joint_qpos[:] = qpos_buffer[self._qpos_index["leg_start"] : self._qpos_index["leg_start"] + self._qpos_index["leg_length"]]
        self._leg_joint_qvel[:] = qvel_buffer[self._qvel_index["leg_start"] : self._qvel_index["leg_start"] + self._qvel_index["leg_length"]]
        self._leg_joint_qacc[:] = qacc_buffer[self._qacc_index["leg_start"] : self._qacc_index["leg_start"] + self._qacc_index["leg_length"]]

        self._body_height, self._body_lin_vel, self._body_ang_vel, \
            self._body_orientation = self._get_body_local(qpos_buffer, qvel_buffer)

        self._foot_touch_force = self._get_foot_touch_force(sensor_data)  # Penalty if the foot touch force is too strong
        self._update_foot_touch_air_time(self._foot_touch_force)  # Reward for air time of the feet
        self._leg_contact = self._get_leg_contact(contact_dict)            # Penalty if the leg is in contact with the ground

        self._achieved_goal = self._get_base_contact(contact_dict).astype(np.float32)         # task failed if the base is in contact with the ground
        self._desired_goal = np.zeros(1).astype(np.float32)      # 1.0 if the base is in contact with the ground, 0.0 otherwise

        obs = np.concatenate(
                [
                    self._body_lin_vel,
                    self._body_ang_vel,
                    self._body_orientation,
                    self._command_values,
                    (self._leg_joint_qpos - self._neutral_joint_values),
                    self._leg_joint_qvel,
                    self._action,
                    np.array([self._body_height]),
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
        self._init_imu_site_pos_quat = init_site_pos_quat[self._imu_site_name].copy()

    def _set_action(self, action) -> None:
        self._last_action[:] = self._action
        self._action[:] = action
    
    def set_action_space(self, action_space : spaces) -> None:
        self._action_space = action_space
        self._action_space_range = np.array([action_space.low[0], action_space.high[0]]) 
        
    def init_joint_index(self, qpos_offset, qvel_offset, qacc_offset, qpos_length, qvel_length, qacc_length) -> None:
        self._qpos_index = {joint_name: {"offset": qpos_offset[i], "len": qpos_length[i]} for i, joint_name in enumerate(self.joint_names)}
        assert "leg_start" not in self._qpos_index and "leg_length" not in self._qpos_index, "qpos_index: joint_name 'leg_start' or 'leg_length' already exists"
        self._qpos_index["leg_start"], self._qpos_index["leg_length"] = self._calc_agent_leg_buffer_index(self._qpos_index)
        self._leg_joint_qpos = np.zeros(self._qpos_index["leg_length"])

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

        visualized_command = {}
        if update_mocap and self._visualize_command:        
            # Do mocap update here.
            # print("imu mocap: ", self._imu_mocap_pos_quat)
            
            # self._imu_mocap_pos_quat["quat"] = rotations.quat_mul(rotations.euler2quat([0, 0, self._command["ang_vel"] * self.dt]), self._imu_mocap_pos_quat["quat"])
            global_lin_vel, _ = local2global(self._imu_mocap_pos_quat["quat"], self._command["lin_vel"], np.array([0,0,0]))
            self._imu_mocap_pos_quat["pos"] += global_lin_vel * self.dt
            visualized_command = {self._imu_mocap_name: self._imu_mocap_pos_quat}
        
        return visualized_command


    def on_reset(self) -> dict[str, np.ndarray]:
        if self._actuator_type == "position":
            assert self._ctrl.shape == self._neutral_joint_values.shape
            self._ctrl = self._neutral_joint_values.copy()
            # print("Neutral joint values: ", self._ctrl)
        elif self._actuator_type == "torque":
            self._ctrl = np.zeros(self._nu)
        else:
            raise ValueError(f"Unsupported actuator type: {self._actuator_type}")
        
        if self._curriculum_learning:
            curriculum_level = self._curriculum_levels[self._curriculum_current_level]
            # print("Curriculum level: ", curriculum_level)
            pos_offset = np.array([self._agent_pos_offset[curriculum_level][self.name]]).flatten()
        else:
            pos_offset = np.zeros(3)

        # Rotate the base body around the Z axis
        z_rotation_angle = self._np_random.uniform(-np.pi, np.pi)
        z_rotation_quat = rotations.euler2quat([0, 0, z_rotation_angle])

        # Get cached default joint values
        joint_neutral_qpos = self.get_joint_neutral()
        base_neutral_qpos = copy.deepcopy(self._init_base_joint_qpos)
        
        # Use curriculum learning to set the initial position
        base_neutral_qpos[self._base_joint_name][:3] += pos_offset

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

        # Use curriculum learning to set the initial position
        self._imu_mocap_pos_quat["pos"][:3] += pos_offset        
        
        # Move along the Z axis to the born height
        if self._visualize_command:
            self._imu_mocap_pos_quat["pos"][2] -= self._base_neutral_height_offset
        else:
            self._imu_mocap_pos_quat["pos"][2] -= 1000.0  # Move the mocap to a far away place

        # Use the rotate quate
        self._imu_mocap_pos_quat["quat"] = rotations.quat_mul(self._imu_mocap_pos_quat["quat"], z_rotation_quat)
        

        self._command = self._genarate_command(z_rotation_angle)
        self._command_values = np.concatenate([self._command["lin_vel"], [self._command["ang_vel"]]]).flatten()
        self._command_resample_duration = 0
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

    def _compute_reward_alive(self, coeff) -> SupportsFloat:
        reward = 1.0 * coeff * self.dt
        self._print_reward("Alive reward: ", reward)
        return reward

    def _compute_reward_success(self, coeff) -> SupportsFloat:
        reward = (1.0 if (self.is_success(self._achieved_goal, self._desired_goal) > 0) else 0.0) * coeff * self.dt
        self._print_reward("Success reward: ", reward)
        return reward
        
    def _compute_reward_failure(self, coeff) -> SupportsFloat:
        reward = (-1.0 if self.is_terminated(self._achieved_goal, self._desired_goal) else 0.0) * coeff * self.dt
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
        base_reward = 1 * len(self._leg_joint_qpos)
        reward = (base_reward - np.sum(np.abs(self._leg_joint_qpos - self._neutral_joint_values))) * coeff * self.dt
        self._print_reward("Joint angles reward: ", reward)
        return reward
    
    def _compute_reward_joint_accelerations(self, coeff) -> SupportsFloat:
        reward = (-np.sum(np.abs(self._leg_joint_qacc))) * coeff * self.dt
        self._print_reward("Joint accelerations reward: ", reward)
        return reward
    
    def _compute_reward_limit(self, coeff) -> SupportsFloat:     
        reward = -(np.sum(self._action == self._action_space_range[0]) + np.sum(self._action == self._action_space_range[1])) * coeff * self.dt
        self._print_reward("Limit over threshold reward: ", reward)
        return reward
    
    def _compute_reward_action_rate(self, coeff) -> SupportsFloat:
        reward = -np.sum(np.square(self._last_action - self._action)) * coeff * self.dt
        self._print_reward("Action rate reward: ", reward)
        return reward
    
    def _compute_reward_base_gyro(self, coeff) -> SupportsFloat:
        reward = (-np.sum(np.abs(self._imu_data_gyro))) * coeff * self.dt
        self._print_reward("Base gyro reward: ", reward)
        return reward
    
    def _compute_reward_base_accelerometer(self, coeff) -> SupportsFloat:
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
        
        # Curiiculum learning
        if self._curriculum_learning:
            self._curriculum_reward_buffer[self._curriculum_reward_buffer_index] = reward
            self._curriculum_reward_buffer_index += 1
            if self._curriculum_reward_buffer_index >= self._curriculum_reward_buffer_size:
                mean_reward = np.mean(self._curriculum_reward_buffer) / (coeff * self.dt)
                # 达到奖励阈值，升级。低于奖励阈值，降级
                if mean_reward > 0.8:
                    self._curriculum_current_level = min(self._curriculum_current_level + 1, len(self._curriculum_levels) - 1)
                    print("Agent: ", self._env_id + self.name, "Level Upgrade! Curriculum level: ", self._curriculum_current_level, "mena reward: ", mean_reward)
                elif mean_reward < 0.6:
                    self._curriculum_current_level = max(self._curriculum_current_level - 1, 0)
                    print("Agent: ", self._env_id + self.name, "Level Downgrade! Curriculum level: ", self._curriculum_current_level, "mena reward: ", mean_reward)
                self._curriculum_reward_buffer_index = 0
            # print("Curriculum reward buffer: ", self._curriculum_reward_buffer)
            
        return reward
    
    def _compute_reward_follow_command_angvel(self, coeff) -> SupportsFloat:
        tracking_sigma = 0.25  # 奖励的衰减因子
        ang_vel_error = pow(self._command["ang_vel"] - self._body_ang_vel[2], 2)  # 只考虑 Z 轴的角速度
        reward = np.exp(-ang_vel_error / tracking_sigma) * coeff * self.dt
        self._print_reward("Follow command angvel reward: ", reward)
        # print("command ang vel: ", self._command["ang_vel"], "body ang vel: ", self._body_ang_vel[2])        
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
        # no reward if the command is not to move
        if np.linalg.norm(self._command["lin_vel"]) > 0.1:
            # print("feet air time: ", self._foot_touch_air_time)
            for i in range(len(self._foot_touch_air_time)):
                if self._foot_touch_air_time[i] > 0:
                    reward += (self._foot_touch_air_time[i] - self._foot_touch_air_time_threshold)  * coeff * self.dt
                    
        self._print_reward("Feet air time reward: ", reward)
        return reward
        
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
        lin_vel = np.array([self._np_random.uniform(0, self._command_lin_vel_range_x), 
                            self._np_random.uniform(-self._command_lin_vel_range_y, self._command_lin_vel_range_y),
                            0])
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


    def _resample_command(self) -> None:
        self._command_resample_duration += self.dt
        if self._command_resample_duration > self._command_resample_interval:
            self._command_resample_duration = 0
            turn_angle = self._np_random.uniform(-self._command_ang_vel_range, self._command_ang_vel_range)
            self._command = self._genarate_command(self._command["heading_angle"] + turn_angle)
            self._command_values[:3] = self._command["lin_vel"]
            
            turn_quat = rotations.euler2quat([0, 0, turn_angle])
            self._imu_mocap_pos_quat["quat"] = rotations.quat_mul(self._imu_mocap_pos_quat["quat"], turn_quat)
    
    def _print_reward(self, message : str, reward : Optional[float] = None):
        if self._reward_printer is not None:
            self._reward_printer.print_reward(message, reward)

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

        return body_height, body_lin_vel, body_ang_vel, body_orientation
    
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
        scale_leg_joint_qpos = np.array([1] * len(self._leg_joint_names)) * LeggedObsConfig["scale"]["qpos"]
        scale_leg_joint_qvel = np.array([1] * len(self._leg_joint_names)) * LeggedObsConfig["scale"]["qvel"]
        scale_action = np.array([1] * len(self._actuator_names)) # No scaling on the action
        scale_height = np.array([1]) * LeggedObsConfig["scale"]["height"]

        scale_vec = np.concatenate([scale_lin_vel, 
                                    scale_ang_vel, 
                                    scale_orientation, 
                                    scale_command, 
                                    scale_leg_joint_qpos, 
                                    scale_leg_joint_qvel, 
                                    scale_action, 
                                    scale_height]).flatten()
        
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
        noise_leg_joint_qpos = np.array([1] * len(self._leg_joint_names)) * noise_level * LeggedObsConfig["noise"]["qpos"] * LeggedObsConfig["scale"]["qpos"]
        noise_leg_joint_qvel = np.array([1] * len(self._leg_joint_names)) * noise_level * LeggedObsConfig["noise"]["qvel"] * LeggedObsConfig["scale"]["qvel"]
        noise_action = np.zeros(len(self._actuator_names))  # No noise on the action
        noise_height = np.array([1]) * noise_level * LeggedObsConfig["noise"]["height"] * LeggedObsConfig["scale"]["height"]

        noise_vec = np.concatenate([noise_lin_vel, 
                                    noise_ang_vel, 
                                    noise_orientation, 
                                    noise_command, 
                                    noise_leg_joint_qpos, 
                                    noise_leg_joint_qvel, 
                                    noise_action, 
                                    noise_height]).flatten()
        
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
    
    def _setup_reward_functions(self):
        self._reward_functions = [
            {"function": self._compute_reward_alive, "coeff": 0},
            {"function": self._compute_reward_success, "coeff": 0},
            {"function": self._compute_reward_failure, "coeff": 0},
            {"function": self._compute_reward_contact, "coeff": 1},
            {"function": self._compute_reward_foot_touch, "coeff": 0},
            {"function": self._compute_reward_joint_angles, "coeff": 0},
            {"function": self._compute_reward_joint_accelerations, "coeff": 2.5e-7},
            {"function": self._compute_reward_limit, "coeff": 0},
            {"function": self._compute_reward_action_rate, "coeff": 0.01},
            {"function": self._compute_reward_base_gyro, "coeff": 0},
            {"function": self._compute_reward_base_accelerometer, "coeff": 0},
            {"function": self._compute_reward_follow_command_linvel, "coeff": 1},
            {"function": self._compute_reward_follow_command_angvel, "coeff": 0.5},
            {"function": self._compute_reward_height, "coeff": 0},
            {"function": self._compute_reward_body_lin_vel, "coeff": 2},
            {"function": self._compute_reward_body_ang_vel, "coeff": 0.05},
            {"function": self._compute_reward_body_orientation, "coeff": 0},
            {"function": self._compute_feet_air_time, "coeff": 1},
        ]