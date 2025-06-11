import os
import sys
import argparse
import time
import csv
import numpy as np
from datetime import datetime
import gymnasium as gym
from stable_baselines3 import PPO, SAC, DDPG
from gymnasium.envs.registration import register
from datetime import datetime
import torch
import torch.nn as nn
import pickle
from orca_gym.devices.keyboard import KeyboardInput

Degree2Rad = np.pi / 180.0


current_file_path = os.path.abspath('')
project_root = os.path.dirname(os.path.dirname(current_file_path))

# 将项目根目录添加到 PYTHONPATH
if project_root not in sys.path:
    sys.path.append(project_root)

from envs.legged_gym.legged_config import LeggedObsConfig, LeggedRobotConfig

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

class Lite3RealAgent:
    def __init__(self,):
        # self.model_file = "/home/superfhwl/repo/yaoxiang/dog_sim2real_0418/trained_models_tmp/Lite3_ppo_2304-agents_200-episodes_2025-04-28_18-35-03/Lite3_follow_command.zip_iteration_500.zip"
        # self.model_file = "/home/superfhwl/repo/yaoxiang/dog_sim2real_0418/trained_models_tmp/Lite3_ppo_2304-agents_200-episodes_2025-04-30_11-04-17/Lite3_follow_command.zip"
        self.model_file = "/home/superfhwl/repo/yaoxiang/dog_sim2real_0418/trained_models_tmp/Lite3_ppo_288-agents_200-episodes_2025-06-11_11-28-43/Lite3_follow_command.zip"
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = PPO.load(self.model_file, device=device)

        robot_config = LeggedRobotConfig["Lite3"]

        self._obs_scale_vec = self._get_obs_scale_vec()
        self._action = np.array([0.0] * 12) # arbitrary
        self._foot_square_wave = robot_config["foot_square_wave"]
        self.dt = 0.1 # arbitrary
        self._phase = 0.0 # arbitrary
        self._neutral_joint_values = self._get_neutral_joint_values()

        self._command = np.array([0.0] * 4)
        self._init_playable()

        self.generate_action_scale_array()

        self._init_log()


    def _init_log(self):
        # Create log directory if it doesn't exist
        log_dir = "logs"
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
            
        # Create filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.filename = os.path.join(log_dir, f"robot_log_{timestamp}.csv")


    def get_action(self, raw_obs):
        """
        Get the action from real observation of Lite3.
        """
        self._update_playable()
        obs = self.restructure_obs(raw_obs)
        # print("obs: ", obs)
        action, _states = self.model.predict(obs, deterministic=True)
        # return restructure_action(action)
        self._set_action(action)
        self.log_observation(raw_obs, action)
        # print(raw_obs)
        return self._action2ctrl(action)

    
    
    def _action2ctrl(self, action: np.ndarray) -> np.ndarray:
        # 缩放后的 action
        scaled_action = action * self._action_scale

        # 限制 scaled_action 在有效范围内
        clipped_action = np.clip(scaled_action, self._action_space_range[0], self._action_space_range[1])

        # 批量计算插值
        ctrl_delta = (
            self._ctrl_delta_range[:, 0] +  # fp1
            (self._ctrl_delta_range[:, 1] - self._ctrl_delta_range[:, 0]) *  # (fp2 - fp1)
            (clipped_action - self._action_space_range[0]) /  # (x - xp1)
            (self._action_space_range[1] - self._action_space_range[0])  # (xp2 - xp1)
        )

        actuator_ctrl = self._neutral_joint_values + ctrl_delta
        
        return actuator_ctrl  

    def generate_action_scale_array(self):
        with open("ctrl_info.pkl", "rb") as f:
            ctrl_info = pickle.load(f)["Lite3_000"]
        # print("ctrl_info: ", ctrl_info)
        self._action_scale = ctrl_info["action_scale"]             # shape = (1)
        self._action_space_range = ctrl_info["action_space_range"] # shape = (2)

        self._ctrl_delta_range = ctrl_info["ctrl_delta_range"]  # shape = (agent_num x actor_num, 2)
        # self._neutral_joint_values = np.array([ctrl["neutral_joint_values"] for key, ctrl in ctrl_info.items()]).reshape(-1) # shape = (agent_num x actor_num)


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
        # if self._command["lin_vel"][0] == 0.0:
        if self._command[0] == 0.0:
            return self._foot_square_wave["p5"]
        else:
            return smooth_sqr_wave_np(self._phase, self._foot_square_wave["phase_freq"], self._foot_square_wave["eps"])

    def _set_action(self, action):
        self._action = action

    def _get_neutral_joint_values(self):
        _neutral_joint_angles = LeggedRobotConfig["Lite3"]["neutral_joint_angles"]
        _neutral_joint_values = np.array([_neutral_joint_angles[key] for key in _neutral_joint_angles]).flatten()
        return _neutral_joint_values



    def _get_obs_scale_vec(self):
        """ Sets a vector used to scale the observations.
            [NOTE]: Must be adapted when changing the observations structure

        Args:
            cfg (Dict): Environment config file

        Returns:
            [torch.Tensor]: Vector of scales used to normalize the observations
        """
        # scale_lin_vel = np.array([1, 1, 1]) * LeggedObsConfig["scale"]["lin_vel"]
        scale_ang_vel = np.array([1, 1, 1]) * LeggedObsConfig["scale"]["ang_vel"]
        scale_orientation = np.array([1, 1, 1])  # No scaling on the orientation
        scale_command = np.array([LeggedObsConfig["scale"]["lin_vel"]]*3 + [LeggedObsConfig["scale"]["ang_vel"]]*1)
        scale_square_wave = np.ones(1)  # No scaling on the square wave
        scale_leg_joint_qpos = np.array([1] * 12) * LeggedObsConfig["scale"]["qpos"]
        scale_leg_joint_qvel = np.array([1] * 12) * LeggedObsConfig["scale"]["qvel"]
        scale_action = np.array([1] * 12) # No scaling on the action
        # scale_height = np.array([1]) * LeggedObsConfig["scale"]["height"]

        scale_vec = np.concatenate([
            # scale_lin_vel, 
            scale_ang_vel, 
            scale_orientation, 
            scale_command, 
            # scale_square_wave,
            scale_leg_joint_qpos, 
            scale_leg_joint_qvel, 
            scale_action, 
            # scale_height
                                    ]).flatten()
        
        return scale_vec.astype(np.float32)



    def restructure_obs(self, real_format_obs):
        """
        Restructure the observation dictionary to match the simulation format.
        """
        # print("real_format_obs: ", real_format_obs)
        imu_data = real_format_obs["imu"]
        joint_data = []
        for leg_name in ["flLeg", "frLeg", "hlLeg", "hrLeg"]:
            joint_data += real_format_obs["jointData"][leg_name]

        # body_lin_vel = np.array([real_format_obs.body_lin_vel[0], real_format_obs.body_lin_vel[1], real_format_obs.body_lin_vel[2]])
        body_ang_vel = np.array([imu_data["angularVelocityRoll"], imu_data["angularVelocityPitch"], imu_data["angularVelocityYaw"]]) * Degree2Rad
        body_orientation = np.array([imu_data["angleRoll"], imu_data["anglePitch"], imu_data["angleYaw"]]) * Degree2Rad
        leg_joint_qpos = [t["position"] for t in joint_data]
        leg_joint_qvel = [t["velocity"] for t in joint_data]
        # body_height = real_format_obs.body_height  # Uncomment if needed

        self._phase = np.fmod(self._phase + self.dt, 1.0)
        square_wave = self._compute_square_wave()


        # Combine into the desired format
        obs = np.concatenate([
            # body_lin_vel
            body_ang_vel,
            body_orientation,
            self._command,
            # np.array([square_wave]),
            leg_joint_qpos - self._neutral_joint_values,
            leg_joint_qvel,
            self._action,
            # np.array([body_height]),  # Uncomment if needed
        ])
        # print("obs: ", obs.shape)
        obs *= self._obs_scale_vec

        # no noise when inference
        # noise_vec = ((self._np_random.random(len(self._noise_scale_vec)) * 2) - 1) * self._noise_scale_vec
        # # print("obs: ", obs, "Noise vec: ", noise_vec)
        # obs += noise_vec

        return {"observation": obs, "achieved_goal": np.array([0.0]), "desired_goal": np.array([0.0])}


    def get_ctrl_info(self) -> dict:
        return {
            "actuator_type": self._actuator_type,
            "action_scale": self._action_scale,
            "action_space_range": self._action_space_range,
            "ctrl_range": self._ctrl_range,
            "ctrl_delta_range": self._ctrl_delta_range,
            "ctrl_range_low": self._ctrl_range_low,
            "ctrl_range_high": self._ctrl_range_high,
            "ctrl_start": self._ctrl_start,
            "ctrl_end": self._ctrl_start + len(self._actuator_names),
            "neutral_joint_values": self._neutral_joint_values,
        }

    def _action2ctrl(self, action: np.ndarray) -> np.ndarray:
        # 缩放后的 action
        scaled_action = action * self._action_scale
        # 限制 scaled_action 在有效范围内
        clipped_action = np.clip(scaled_action, self._action_space_range[0], self._action_space_range[1])

        # 批量计算插值
        ctrl_delta = (
            self._ctrl_delta_range[:, 0] +  # fp1
            (self._ctrl_delta_range[:, 1] - self._ctrl_delta_range[:, 0]) *  # (fp2 - fp1)
            (clipped_action - self._action_space_range[0]) /  # (x - xp1)
            (self._action_space_range[1] - self._action_space_range[0])  # (xp2 - xp1)
        )

        actuator_ctrl = self._neutral_joint_values + ctrl_delta
        
        return actuator_ctrl    

    def _init_playable(self) -> None:
        self._keyboard_controller = KeyboardInput()
        self._key_status = {"W": 0, "A": 0, "S": 0, "D": 0, "Space": 0, "Up": 0, "Down": 0, "LShift": 0, "RShift": 0}   
        
        # self._player_agent = self.agent
        # self.agent.init_playable()
        # self.agent.player_control = True
            
        robot_config = LeggedRobotConfig["Lite3"]

            
        self._player_agent_lin_vel_x = np.array(robot_config["curriculum_commands"]["flat_plane"]["command_lin_vel_range_x"]) / 2
        self._player_agent_lin_vel_y = np.array(robot_config["curriculum_commands"]["flat_plane"]["command_lin_vel_range_y"]) / 2
    
    def _update_playable(self) -> None:
        lin_vel, turn_angel, reborn = self._update_keyboard_control()
        # self._player_agent.update_playable(lin_vel, turn_angel)
        self._command[0:3] = lin_vel
        self._command[3] = turn_angel * 3   # the code of turning is different from the simulation

        # agent_cmd_mocap = self._player_agent.reset_command_indicator(env.data.qpos)
        # env.set_mocap_pos_and_quat(agent_cmd_mocap)      
    
    def _update_keyboard_control(self) -> tuple[np.ndarray, float, bool]:
        self._keyboard_controller.update()
        key_status = self._keyboard_controller.get_state()
        lin_vel = np.zeros(3)
        turn_angel = 0.0
        reborn = False
        # print(key_status)
        if key_status["W"] == 1:
            lin_vel[0] = self._player_agent_lin_vel_x[1]
        if key_status["S"] == 1:
            lin_vel[0] = self._player_agent_lin_vel_x[0]
        if key_status["Q"] == 1:
            lin_vel[1] = self._player_agent_lin_vel_y[1]
        if key_status["E"] == 1:
            lin_vel[1] = self._player_agent_lin_vel_y[0]
        if key_status["A"] == 1:
            turn_angel += np.pi / 2 * self.dt
        if key_status["D"] == 1:
            turn_angel += -np.pi / 2 * self.dt
        if self._key_status["Space"] == 0 and key_status["Space"] == 1:
            reborn = True
        if key_status["LShift"] == 1:
            lin_vel[:2] *= 2

        self._key_status = key_status.copy()
        # print("Lin vel: ", lin_vel, "Turn angel: ", turn_angel, "Reborn: ", reborn)
        
        return lin_vel, turn_angel, reborn    

    def log_observation(self, obs, action):
        """
        Log observations and actions to a CSV file.
        
        Args:
            obs (dict): Observation dictionary containing IMU and joint data
            action (np.ndarray): Action array
        """        # Define CSV headers
        headers = [
            "timestamp",
            # IMU data
            "imu_angle_roll", "imu_angle_pitch", "imu_angle_yaw",
            "imu_angular_velocity_roll", "imu_angular_velocity_pitch", "imu_angular_velocity_yaw",
            "imu_acc_x", "imu_acc_y", "imu_acc_z",
            # Joint data - Front Left Leg
            "fl_joint1_pos", "fl_joint1_vel", "fl_joint1_torque",
            "fl_joint2_pos", "fl_joint2_vel", "fl_joint2_torque",
            "fl_joint3_pos", "fl_joint3_vel", "fl_joint3_torque",
            # Joint data - Front Right Leg
            "fr_joint1_pos", "fr_joint1_vel", "fr_joint1_torque",
            "fr_joint2_pos", "fr_joint2_vel", "fr_joint2_torque",
            "fr_joint3_pos", "fr_joint3_vel", "fr_joint3_torque",
            # Joint data - Hind Left Leg
            "hl_joint1_pos", "hl_joint1_vel", "hl_joint1_torque",
            "hl_joint2_pos", "hl_joint2_vel", "hl_joint2_torque",
            "hl_joint3_pos", "hl_joint3_vel", "hl_joint3_torque",
            # Joint data - Hind Right Leg
            "hr_joint1_pos", "hr_joint1_vel", "hr_joint1_torque",
            "hr_joint2_pos", "hr_joint2_vel", "hr_joint2_torque",
            "hr_joint3_pos", "hr_joint3_vel", "hr_joint3_torque",
            # Contact forces
            "fl_force_x", "fl_force_y", "fl_force_z",
            "fr_force_x", "fr_force_y", "fr_force_z",
            "hl_force_x", "hl_force_y", "hl_force_z",
            "hr_force_x", "hr_force_y", "hr_force_z",
            # Actions
            "fl_hip_action", "fl_thigh_action", "fl_calf_action",
            "fr_hip_action", "fr_thigh_action", "fr_calf_action",
            "hl_hip_action", "hl_thigh_action", "hl_calf_action",
            "hr_hip_action", "hr_thigh_action", "hr_calf_action",
            # command
            "command_lin_vel_x", "command_lin_vel_y", "command_lin_vel_z", "command_ang_vel"
        ]
        
        # Create file and write headers if it doesn't exist
        file_exists = os.path.exists(self.filename)
        
        with open(self.filename, 'a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(headers)
            
            # Prepare data row
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            
            # Extract IMU data
            imu_data = obs["imu"]
            imu_values = [
                imu_data["angleRoll"], imu_data["anglePitch"], imu_data["angleYaw"],
                imu_data["angularVelocityRoll"], imu_data["angularVelocityPitch"], imu_data["angularVelocityYaw"],
                imu_data["accX"], imu_data["accY"], imu_data["accZ"]
            ]
            
            # Extract joint data
            joint_data = obs["jointData"]
            joint_values = []
            for leg in ["flLeg", "frLeg", "hlLeg", "hrLeg"]:
                for joint in joint_data[leg]:
                    joint_values.extend([
                        joint["position"],
                        joint["velocity"],
                        joint["torque"],
                        # joint["temperature"]
                    ])
            
            # Extract contact forces
            contact_forces = obs.get("contactForce", {})
            force_values = []
            for leg in ["flLeg", "frLeg", "hlLeg", "hrLeg"]:
                forces = contact_forces.get(leg, [0.0, 0.0, 0.0])
                force_values.extend(forces)
            
            # Combine all data
            row = [current_time] + imu_values + joint_values + force_values + list(action) + list(self._command)
            
            writer.writerow(row)