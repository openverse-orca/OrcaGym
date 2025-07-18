import os
import sys
import argparse
import time
import numpy as np
from datetime import datetime
import gymnasium as gym
from stable_baselines3 import PPO
from gymnasium.envs.registration import register
from datetime import datetime
import torch
import torch.nn as nn
import csv

current_file_path = os.path.abspath('')
project_root = os.path.dirname(os.path.dirname(current_file_path))

# 将项目根目录添加到 PYTHONPATH
if project_root not in sys.path:
    sys.path.append(project_root)


from envs.legged_gym.legged_config import LeggedEnvConfig

TIME_STEP = LeggedEnvConfig["TIME_STEP"]

FRAME_SKIP_REALTIME = LeggedEnvConfig["FRAME_SKIP_REALTIME"]
FRAME_SKIP_SHORT = LeggedEnvConfig["FRAME_SKIP_SHORT"]
FRAME_SKIP_LONG = LeggedEnvConfig["FRAME_SKIP_LONG"]

EPISODE_TIME_VERY_SHORT = LeggedEnvConfig["EPISODE_TIME_VERY_SHORT"]
EPISODE_TIME_SHORT = LeggedEnvConfig["EPISODE_TIME_SHORT"]
EPISODE_TIME_LONG = LeggedEnvConfig["EPISODE_TIME_LONG"]



FRAME_SKIP = FRAME_SKIP_SHORT # FRAME_SKIP_REALTIME

TIME_STEP = 0.001                       # 1000 Hz for physics simulation
FRAME_SKIP = 20
REALTIME_STEP = TIME_STEP * FRAME_SKIP  # 50 Hz for rendering
CONTROL_FREQ = 1 / REALTIME_STEP        # 50 Hz for ppo policy

MAX_EPISODE_STEPS = int(EPISODE_TIME_SHORT / REALTIME_STEP)  # 10 seconds

def register_env(orcagym_addr : str, 
                 env_name : str, 
                 env_index : int, 
                 agent_names : str,
                 ctrl_device : str,
                 max_episode_steps : int,) -> str:
    orcagym_addr_str = orcagym_addr.replace(":", "-")
    env_id = env_name + "-OrcaGym-" + orcagym_addr_str + f"-{env_index:03d}"
    agent_names_list = agent_names.split(" ")
    print("Agent names: ", agent_names_list)
    kwargs = {'frame_skip': FRAME_SKIP,   
                'orcagym_addr': orcagym_addr, 
                'env_id': env_id,
                'agent_names': agent_names_list,
                'time_step': TIME_STEP,
                'max_episode_steps': max_episode_steps,
                'ctrl_device': ctrl_device,
                'control_freq': CONTROL_FREQ,}
    gym.register(
        id=env_id,
        entry_point='envs.legged_gym.legged_sim_env:LeggedSimEnv',
        kwargs=kwargs,
        max_episode_steps= max_episode_steps,
        reward_threshold=0.0,
    )
    return env_id, kwargs


def main(orcagym_addr: str,
                   agent_names: str,
                   model_file: str,
                   ctrl_device: str = "keyboard",):
    try:
        print("simulation running... , orcagym_addr: ", orcagym_addr)
        env_name = "LeggedSim-v0"
        env_id, kwargs = register_env(
            orcagym_addr=orcagym_addr, 
            env_name=env_name, 
            env_index=0, 
            agent_names=agent_names, 
            ctrl_device=ctrl_device, 
            max_episode_steps=MAX_EPISODE_STEPS,
        )
        print("Registered environment: ", env_id)

        env = gym.make(env_id)
        print("Starting simulation...")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = PPO.load(model_file, device=device)

        agent_name_list = agent_names.split(" ")
        run_simulation(
            env=env,
            agent_name_list=agent_name_list,
            model=model,
            time_step=TIME_STEP,
            frame_skip=FRAME_SKIP
        )
    finally:
        print("退出仿真环境")
        env.close()

def segment_obs(obs: dict[str, np.ndarray], agent_name_list: list[str]) -> dict[str, dict[str, np.ndarray]]:
    if len(agent_name_list) == 1:
        return {agent_name_list[0]: obs}
    
    segmented_obs = {}
    for agent_name in agent_name_list:
        segmented_obs[agent_name] = {}
        for key in obs.keys():
            if key.startswith(agent_name):
                new_key = key.replace(f"{agent_name}_", "")
                segmented_obs[agent_name][new_key] = obs[key]
    return segmented_obs
    

def log_observation(obs: dict, action: np.ndarray, filename: str, physics_step: int, control_step: int, sim_time: float):
    """
    Log observations and actions to a CSV file.
    
    Args:
        obs (dict): Observation dictionary containing IMU and joint data
        action (np.ndarray): Action array
        filename (str): Path to the CSV file
        physics_step (int): Current physics simulation step count
        control_step (int): Current control/policy step count
        sim_time (float): Current simulation time in seconds
    """
    # Define CSV headers
    headers = [
        "timestamp",
        "sim_time",
        "physics_step",
        "control_step",
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
        "hr_hip_action", "hr_thigh_action", "hr_calf_action"
    ]
    
    # Create file and write headers if it doesn't exist
    file_exists = os.path.exists(filename)
    
    with open(filename, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(headers)
        
        # Prepare data row
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        
        # Combine all data
        row = [current_time, sim_time, physics_step, control_step] + list(obs["observation"]) + list(action)
        
        writer.writerow(row)

def run_simulation(env: gym.Env, 
                 agent_name_list: list[str],
                 model: nn.Module, 
                 time_step: float, 
                 frame_skip: int):
    obs, info = env.reset()
    dt = time_step * frame_skip
    if not os.path.exists("./log"):
        os.makedirs("./log")
    log_file = f"./log/simulation_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    
    # Add step counting
    physics_step = 0
    control_step = 0
    sim_time = 0.0
    
    try:
        while True:
            start_time = datetime.now()

            segmented_obs = segment_obs(obs, agent_name_list)
            action_list = []
            for agent_obs in segmented_obs.values():
                action, _states = model.predict(agent_obs, deterministic=True)
                action_list.append(action)

            action = np.concatenate(action_list).flatten()
            
            # Log with step information
            log_observation(obs, action, log_file, physics_step, control_step, sim_time)
            
            # Update step counters before next step
            physics_step += frame_skip  # Each control step includes frame_skip physics steps
            control_step += 1
            sim_time += dt
            
            obs, reward, terminated, truncated, info = env.step(action)
            env.render()

            elapsed_time = datetime.now() - start_time
            if elapsed_time.total_seconds() < dt:
                time.sleep(dt - elapsed_time.total_seconds())
            
    finally:
        print("退出仿真环境")
        env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run multiple instances of the script with different gRPC addresses.')
    parser.add_argument('--orcagym_addr', type=str, default='localhost:50051', help='The gRPC addresses to connect to')
    parser.add_argument('--agent_name', type=str, default='go2', help='The name list of the agent to control, separated by space')
    parser.add_argument('--model_file', type=str, help='The model file to load')
    parser.add_argument('--ctrl_device', type=str, default='keyboard', help='The control device to use ')
    args = parser.parse_args()

    main(
        orcagym_addr=args.orcagym_addr, 
        agent_names=args.agent_name, 
        model_file=args.model_file,
        ctrl_device=args.ctrl_device
    )    


