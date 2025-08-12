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
import yaml
from scene_util import clear_scene, publish_terrain, generate_height_map_file, publish_scene
from orca_gym.devices.keyboard import KeyboardInput, KeyboardInputSourceType
from envs.legged_gym.legged_sim_env import LeggedSimEnv
from envs.legged_gym.legged_config import LeggedRobotConfig

current_file_path = os.path.abspath('')
project_root = os.path.dirname(os.path.dirname(current_file_path))

# 将项目根目录添加到 PYTHONPATH
if project_root not in sys.path:
    sys.path.append(project_root)


from envs.legged_gym.legged_config import LeggedEnvConfig

EPISODE_TIME_VERY_SHORT = LeggedEnvConfig["EPISODE_TIME_VERY_SHORT"]
EPISODE_TIME_SHORT = LeggedEnvConfig["EPISODE_TIME_SHORT"]
EPISODE_TIME_LONG = LeggedEnvConfig["EPISODE_TIME_LONG"]

TIME_STEP = 0.001                       # 1000 Hz for physics simulation
FRAME_SKIP = 5
ACTION_SKIP = 4

REALTIME_STEP = TIME_STEP * FRAME_SKIP * ACTION_SKIP  # 50 Hz for policy
CONTROL_FREQ = 1 / REALTIME_STEP        # 50 Hz for ppo policy

MAX_EPISODE_STEPS = int(EPISODE_TIME_SHORT / REALTIME_STEP)  # 10 seconds


class KeyboardControl:
    def __init__(self, orcagym_addr: str, env: LeggedSimEnv, robot_config: dict):
        self.keyboard_controller = KeyboardInput(KeyboardInputSourceType.ORCASTUDIO, orcagym_addr)
        self._last_key_status = {"W": 0, "A": 0, "S": 0, "D": 0, "Space": 0, "Up": 0, "Down": 0, "LShift": 0, "RShift": 0, "R": 0, "F": 0}   
        self.env = env
        self.player_agent_lin_vel_x = np.array(robot_config["curriculum_commands"]["move_fast"]["command_lin_vel_range_x"]) / 2
        self.player_agent_lin_vel_y = np.array(robot_config["curriculum_commands"]["move_fast"]["command_lin_vel_range_y"]) / 2
        self.player_agent_turn_angel = np.array(robot_config["curriculum_commands"]["move_fast"]["command_ang_vel_range"])
        self.terrain_type = "flat_terrain"

    def update(self):
        self.keyboard_controller.update()
        key_status = self.keyboard_controller.get_state()
        lin_vel = np.zeros(3)
        ang_vel = 0.0
        reborn = False
        
        if key_status["W"] == 1:
            lin_vel[0] = self.player_agent_lin_vel_x[1]
        if key_status["S"] == 1:
            lin_vel[0] = self.player_agent_lin_vel_x[0]
        if key_status["Q"] == 1:
            lin_vel[1] = self.player_agent_lin_vel_y[1]
        if key_status["E"] == 1:
            lin_vel[1] = self.player_agent_lin_vel_y[0]
        if key_status["A"] == 1:
            if lin_vel[0] > 0:
                ang_vel = self.player_agent_turn_angel
            elif lin_vel[0] < 0:
                ang_vel = -self.player_agent_turn_angel
        if key_status["D"] == 1:
            if lin_vel[0] > 0:
                ang_vel = -self.player_agent_turn_angel
            elif lin_vel[0] < 0:
                ang_vel = self.player_agent_turn_angel
        if self._last_key_status["R"] == 0 and key_status["R"] == 1:
            reborn = True
        if key_status["LShift"] == 1:
            lin_vel[:2] *= 2
        if key_status["Space"] == 0 and self._last_key_status["Space"] == 1:
            if self.terrain_type == "flat_terrain":
                self.terrain_type = "rough_terrain"
                print("Switch to rough terrain")
            else:
                self.terrain_type = "flat_terrain"
                print("Switch to flat terrain")

        self._last_key_status = key_status.copy()
        # print("Lin vel: ", lin_vel, "Turn angel: ", ang_vel, "Reborn: ", reborn, "Terrain type: ", self.terrain_type)
        return lin_vel, ang_vel, reborn, self.terrain_type

    def get_state(self):
        return self.key_status

    def get_terrain_type(self):
        return self.terrain_type


def register_env(orcagym_addr : str, 
                 env_name : str, 
                 env_index : int, 
                 agent_names : str,
                 ctrl_device : str,
                 max_episode_steps : int,
                 height_map : str,
                 ) -> str:
    orcagym_addr_str = orcagym_addr.replace(":", "-")
    env_id = env_name + "-OrcaGym-" + orcagym_addr_str + f"-{env_index:03d}"
    agent_names_list = agent_names.split(" ")
    print("Agent names: ", agent_names_list)
    kwargs = {'frame_skip': FRAME_SKIP,   
                'orcagym_addr': orcagym_addr, 
                'env_id': env_id,
                'agent_names': agent_names_list,
                'time_step': TIME_STEP,
                'action_skip': ACTION_SKIP,
                'max_episode_steps': max_episode_steps,
                'ctrl_device': ctrl_device,
                'control_freq': CONTROL_FREQ,
                'height_map': height_map,
                }
    gym.register(
        id=env_id,
        entry_point='envs.legged_gym.legged_sim_env:LeggedSimEnv',
        kwargs=kwargs,
        max_episode_steps= max_episode_steps,
        reward_threshold=0.0,
    )
    return env_id, kwargs


def main(
    config: dict,
    remote: str,
    ):
    try:
        if remote is not None:
            orcagym_addresses = [remote]
        else:
            orcagym_addresses = config['orcagym_addresses']

        agent_name = config['agent_name']
        model_file = config['model_file']
        ctrl_device = config['ctrl_device']
        terrain_spawnable_names = config['terrain_spawnable_names']
        agent_spawnable_name = config['agent_spawnable_name']

        model_dir = os.path.dirname(list(model_file.values())[0])
        command_model = config['command_model']

        # 清空场景
        clear_scene(
            orcagym_addresses=orcagym_addresses,
        )

        # 发布地形
        publish_terrain(
            orcagym_addresses=orcagym_addresses,
            terrain_spawnable_names=terrain_spawnable_names,
        )

        # 空场景生成高度图
        height_map_file = generate_height_map_file(
            orcagym_addresses=orcagym_addresses,
            model_dir=model_dir,
        )

        # 放置机器人
        publish_scene(
            orcagym_addresses=orcagym_addresses,
            agent_name=agent_name,
            agent_spawnable_name=agent_spawnable_name,
            agent_num=1,
            terrain_spawnable_names=terrain_spawnable_names,
        )

        agent_names = f"{agent_name}_000"

        print("simulation running... , orcagym_addr: ", orcagym_addresses)
        env_name = "LeggedSim-v0"
        env_id, kwargs = register_env(
            orcagym_addr=orcagym_addresses[0], 
            env_name=env_name, 
            env_index=0, 
            agent_names=agent_names, 
            ctrl_device=ctrl_device, 
            max_episode_steps=MAX_EPISODE_STEPS,
            height_map=height_map_file,
        )
        print("Registered environment: ", env_id)

        env = gym.make(env_id)
        print("Starting simulation...")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        models = {}
        for key, value in model_file.items():
            models[key] = PPO.load(value, device=device)

        keyboard_control = KeyboardControl(orcagym_addresses[0], env, LeggedRobotConfig[agent_name])

        agent_name_list = [agent_name]
        run_simulation(
            env=env,
            agent_name_list=agent_name_list,
            models=models,
            time_step=TIME_STEP,
            frame_skip=FRAME_SKIP,
            action_skip=ACTION_SKIP,
            keyboard_control=keyboard_control,
            command_model=command_model,
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
                 models: dict[str, nn.Module], 
                 time_step: float, 
                 frame_skip: int,
                 action_skip: int,
                 keyboard_control: KeyboardControl,
                 command_model: dict[str, str]):
    obs, info = env.reset()
    dt = time_step * frame_skip * action_skip
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

            lin_vel, ang_vel, reborn, terrain_type = keyboard_control.update()
            if reborn:
                obs, info = env.reset()
                continue

            if np.linalg.norm(lin_vel) == 0.0:
                if ang_vel != 0.0:
                    model = models[command_model[terrain_type]["trun"]]
                else:
                    model = models[command_model[terrain_type]["stand_still"]]
            else:
                if lin_vel[0] >= 0:
                    model = models[command_model[terrain_type]["forward"]]
                elif lin_vel[0] < 0:
                    model = models[command_model[terrain_type]["backward"]]

            command_dict = {"lin_vel": lin_vel, "ang_vel": ang_vel}
            if hasattr(env, "setup_command"):
                env.setup_command(command_dict)
            else:
                env.unwrapped.setup_command(command_dict)

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

            # no action testing
            # action = np.zeros(env.action_space.shape[0])
            
            obs, reward, terminated, truncated, info = env.step(action)
            env.render()

            # print("--------------------------------")
            # print("action: ", action)
            # print("obs: ", obs)

            elapsed_time = datetime.now() - start_time
            if elapsed_time.total_seconds() < dt:
                time.sleep(dt - elapsed_time.total_seconds())
            
    finally:
        print("退出仿真环境")
        env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run multiple instances of the script with different gRPC addresses.')
    parser.add_argument('--config', type=str, default='go2_sim_config.yaml', help='The path of the config file')
    parser.add_argument('--remote', type=str, help='The remote address of the orca studio')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    main(
        config=config,
        remote=args.remote,
    )


