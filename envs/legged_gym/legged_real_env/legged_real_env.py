import os
import sys
import argparse
import time
import numpy as np
from datetime import datetime
import gymnasium as gym
from stable_baselines3 import PPO, SAC, DDPG
from gymnasium.envs.registration import register
from datetime import datetime
import torch
import torch.nn as nn
import pickle

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
        self.model_file = "/home/superfhwl/桌面/yaoxiang/orcagym/trained_models/trained_models_tmp/Lite3_ppo_1152-agents_200-episodes_2025-04-14_12-34-09_ckp100000000.zip"
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = PPO.load(self.model_file, device=device)

        robot_config = LeggedRobotConfig["Lite3"]

        self._obs_scale_vec = self._get_obs_scale_vec()
        self._action = np.array([0.0] * 12) # arbitrary
        self._foot_square_wave = robot_config["foot_square_wave"]
        self.dt = 0.1 # arbitrary
        self._phase = 0.0 # arbitrary
        self._neutral_joint_values = self._get_neutral_joint_values()

        self.generate_action_scale_array()

    def get_action(self, obs):
        """
        Get the action from real observation of Lite3.
        """
        obs = self.restructure_obs(obs)
        # print("obs: ", obs)
        action, _states = self.model.predict(obs, deterministic=True)
        # return restructure_action(action)
        self._set_action(action)
        return action
    
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
            ctrl_info = pickle.load(f)["Lite3"]
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
        # scale_orientation = np.array([1, 1, 1])  # No scaling on the orientation
        scale_command = np.array([LeggedObsConfig["scale"]["lin_vel"]]*3 + [LeggedObsConfig["scale"]["ang_vel"]]*1)
        scale_square_wave = np.ones(1)  # No scaling on the square wave
        scale_leg_joint_qpos = np.array([1] * 12) * LeggedObsConfig["scale"]["qpos"]
        scale_leg_joint_qvel = np.array([1] * 12) * LeggedObsConfig["scale"]["qvel"]
        scale_action = np.array([1] * 12) # No scaling on the action
        # scale_height = np.array([1]) * LeggedObsConfig["scale"]["height"]

        scale_vec = np.concatenate([
            # scale_lin_vel, 
            scale_ang_vel, 
            # scale_orientation, 
            scale_command, 
            scale_square_wave,
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
        body_ang_vel = np.array([imu_data["angularVelocityRoll"], imu_data["angularVelocityPitch"], imu_data["angularVelocityYaw"]])
        # body_orientation = np.array(real_format_obs.body_orientation)  # Assuming it's a quaternion or similar
        self._command = np.array([0] * 4)
        leg_joint_qpos = [t["position"] for t in joint_data]
        leg_joint_qvel = [t["velocity"] for t in joint_data]
        # body_height = real_format_obs.body_height  # Uncomment if needed

        self._phase = np.fmod(self._phase + self.dt, 1.0)
        square_wave = self._compute_square_wave()


        # Combine into the desired format
        obs = np.concatenate([
            # body_lin_vel
            body_ang_vel,
            # body_orientation,
            self._command,
            np.array([square_wave]),
            leg_joint_qpos - self._neutral_joint_values,
            leg_joint_qvel,
            self._action,
            # np.array([body_height]),  # Uncomment if needed
        ])
        print("obs: ", obs.shape)
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



