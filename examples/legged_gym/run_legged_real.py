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

current_file_path = os.path.abspath('')
project_root = os.path.dirname(os.path.dirname(current_file_path))

# 将项目根目录添加到 PYTHONPATH
if project_root not in sys.path:
    sys.path.append(project_root)


from envs.legged_gym.legged_config import LeggedEnvConfig, LeggedObsConfig
import orca_gym.scripts.multi_agent_rl as rl
from orca_gym.utils.dir_utils import create_tmp_dir

TIME_STEP = LeggedEnvConfig["TIME_STEP"]

FRAME_SKIP_REALTIME = LeggedEnvConfig["FRAME_SKIP_REALTIME"]
FRAME_SKIP_SHORT = LeggedEnvConfig["FRAME_SKIP_SHORT"]
FRAME_SKIP_LONG = LeggedEnvConfig["FRAME_SKIP_LONG"]

EPISODE_TIME_VERY_SHORT = LeggedEnvConfig["EPISODE_TIME_VERY_SHORT"]
EPISODE_TIME_SHORT = LeggedEnvConfig["EPISODE_TIME_SHORT"]
EPISODE_TIME_LONG = LeggedEnvConfig["EPISODE_TIME_LONG"]



FRAME_SKIP = FRAME_SKIP_SHORT # FRAME_SKIP_REALTIME

TIME_STEP = 0.005                       # 200 Hz for physics simulation
FRAME_SKIP = 4
REALTIME_STEP = TIME_STEP * FRAME_SKIP  # 50 Hz for rendering
CONTROL_FREQ = 1 / REALTIME_STEP        # 50 Hz for ppo policy

MAX_EPISODE_STEPS = int(EPISODE_TIME_SHORT / REALTIME_STEP)  # 10 seconds



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
    scale_command = np.array([LeggedObsConfig["scale"]["lin_vel"], LeggedObsConfig["scale"]["lin_vel"], LeggedObsConfig["scale"]["lin_vel"], LeggedObsConfig["scale"]["ang_vel"]])
    scale_square_wave = np.ones(1)  # No scaling on the square wave
    scale_leg_joint_qpos = np.array([1] * len(self._leg_joint_names)) * LeggedObsConfig["scale"]["qpos"]
    scale_leg_joint_qvel = np.array([1] * len(self._leg_joint_names)) * LeggedObsConfig["scale"]["qvel"]
    scale_action = np.array([1] * len(self._actuator_names)) # No scaling on the action
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

_obs_scale_vec = _get_obs_scale_vec()

def restructure_obs(real_format_obs):
    """
    Restructure the observation dictionary to match the simulation format.
    """
    body_lin_vel = np.array([real_format_obs.body_lin_vel[0], real_format_obs.body_lin_vel[1], real_format_obs.body_lin_vel[2]])
    body_ang_vel = np.array([real_format_obs.body_ang_vel[0], real_format_obs.body_ang_vel[1], real_format_obs.body_ang_vel[2]])
    body_orientation = np.array(real_format_obs.body_orientation)  # Assuming it's a quaternion or similar
    command_values = np.array(real_format_obs.command_values)
    square_wave = real_format_obs.square_wave
    leg_joint_qpos = np.array(real_format_obs.leg_joint_qpos)
    leg_joint_qvel = np.array(real_format_obs.leg_joint_qvel)
    action = np.array(real_format_obs.action)
    # body_height = real_format_obs.body_height  # Uncomment if needed

    # Combine into the desired format
    obs = np.concatenate([
        # body_lin_vel,
        body_ang_vel,
        # body_orientation,
        command_values,
        np.array([square_wave]),
        leg_joint_qpos,
        leg_joint_qvel,
        action,
        # np.array([body_height]),  # Uncomment if needed
    ])

    obs *= _obs_scale_vec

    # no noise when inference
    # noise_vec = ((self._np_random.random(len(self._noise_scale_vec)) * 2) - 1) * self._noise_scale_vec
    # # print("obs: ", obs, "Noise vec: ", noise_vec)
    # obs += noise_vec

    return obs

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

def get_action(obs, model):
    """
    Get the action from real observation of Lite3.
    """
    obs = restructure_obs(obs)
    # print("obs: ", obs)
    action, _states = model.predict(obs, deterministic=True)
    


def main(orcagym_addr: str,
        agent_names: str,
        model_file: str,
        ctrl_device: str = "keyboard",):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PPO.load(model_file, device=device)

    action, _states = model.predict(obs, deterministic=True)

    

def run_simulation(env: gym.Env, 
                 agent_name_list: list[str],
                 model: nn.Module, 
                 time_step: float, 
                 frame_skip: int):
    obs, info = env.reset()
    # print("obs: ", obs)
    dt = time_step * frame_skip
    try:
        while True:
            start_time = datetime.now()

            segmented_obs = segment_obs(obs, agent_name_list)
            # print("segmented_obs: ", segmented_obs)
            action_list = []
            for agent_obs in segmented_obs.values():
                # print("agent_obs: ", agent_obs)
                # predict_start = datetime.now()
                action, _states = model.predict(agent_obs, deterministic=True)
                action_list.append(action)
                # predict_time = datetime.now() - predict_start
                # print("Predict Time: ", predict_time.total_seconds(), flush=True)

            action = np.concatenate(action_list).flatten()
            # print("action: ", action)
            # setp_start = datetime.now()
            obs, reward, terminated, truncated, info = env.step(action)

            # print("obs, reward, terminated, truncated, info: ", observation, reward, terminated, truncated, info)


            env.render()
            # step_time = datetime.now() - setp_start
            # print("Step Time: ", step_time.total_seconds(), flush=True)

            elapsed_time = datetime.now() - start_time
            if elapsed_time.total_seconds() < dt:
                # print("Sleep for ", dt - elapsed_time.total_seconds())
                time.sleep(dt - elapsed_time.total_seconds())
            
    finally:
        print("退出仿真环境")
        env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run multiple instances of the script with different gRPC addresses.')
    parser.add_argument('--orcagym_addr', type=str, default='localhost:50051', help='The gRPC addresses to connect to')
    parser.add_argument('--agent_names', type=str, default='go2', help='The name list of the agent to control, separated by space')
    parser.add_argument('--model_file', type=str, help='The model file to load')
    parser.add_argument('--ctrl_device', type=str, default='keyboard', help='The control device to use ')
    args = parser.parse_args()

    main(
        orcagym_addr=args.orcagym_addr, 
        agent_names=args.agent_names, 
        model_file=args.model_file,
        ctrl_device=args.ctrl_device
    )    


