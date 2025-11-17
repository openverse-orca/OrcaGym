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
from scripts.scene_util import clear_scene, publish_terrain, generate_height_map_file, publish_scene
from orca_gym.devices.keyboard import KeyboardInput, KeyboardInputSourceType
from envs.legged_gym.legged_sim_env import LeggedSimEnv
from envs.legged_gym.legged_config import LeggedRobotConfig, LeggedObsConfig, CurriculumConfig, LeggedEnvConfig
from orca_gym.utils.device_utils import get_torch_device, get_gpu_info, print_gpu_info

from scripts.grpc_client import GrpcInferenceClient, create_grpc_client


current_file_path = os.path.abspath('')
project_root = os.path.dirname(os.path.dirname(current_file_path))

# 将项目根目录添加到 PYTHONPATH
if project_root not in sys.path:
    sys.path.append(project_root)


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
    def __init__(self, orcagym_addr: str, env: LeggedSimEnv, command_model: dict, model_type: str):
        self.keyboard_controller = KeyboardInput(KeyboardInputSourceType.ORCASTUDIO, orcagym_addr)
        self._last_key_status = {"W": 0, "A": 0, "S": 0, "D": 0, "Space": 0, "Up": 0, "Down": 0, "LShift": 0, "RShift": 0, "R": 0, "F": 0, "M": 0}   
        self.env = env
        self.player_agent_lin_vel_x = {terrain_type: np.array(command_model[terrain_type]["forward_speed"]) for terrain_type in command_model.keys()}
        self.player_agent_lin_vel_y = {terrain_type: np.array(command_model[terrain_type]["left_speed"]) for terrain_type in command_model.keys()}
        self.player_agent_turn_angel = {terrain_type: np.array(command_model[terrain_type]["turn_speed"]) for terrain_type in command_model.keys()}
        self.player_agent_turbo_scale = {terrain_type: np.array(command_model[terrain_type]["turbo_scale"]) for terrain_type in command_model.keys()}
        self.terrain_type = "flat_terrain"
        self.model_type = model_type

    def update(self):
        self.keyboard_controller.update()
        key_status = self.keyboard_controller.get_state()
        lin_vel = np.zeros(3)
        ang_vel = 0.0
        reborn = False
        
        if key_status["W"] == 1:
            lin_vel[0] = self.player_agent_lin_vel_x[self.terrain_type][1]
        if key_status["S"] == 1:
            lin_vel[0] = self.player_agent_lin_vel_x[self.terrain_type][0]
        if key_status["Q"] == 1:
            lin_vel[1] = self.player_agent_lin_vel_y[self.terrain_type][0]
        if key_status["E"] == 1:
            lin_vel[1] = self.player_agent_lin_vel_y[self.terrain_type][1]
        if key_status["A"] == 1:
            if lin_vel[0] >= 0:
                ang_vel = self.player_agent_turn_angel[self.terrain_type]
            elif lin_vel[0] < 0:
                ang_vel = -self.player_agent_turn_angel[self.terrain_type]
        if key_status["D"] == 1:
            if lin_vel[0] >= 0:
                ang_vel = -self.player_agent_turn_angel[self.terrain_type]
            elif lin_vel[0] < 0:
                ang_vel = self.player_agent_turn_angel[self.terrain_type]
        if self._last_key_status["R"] == 0 and key_status["R"] == 1:
            reborn = True
        if key_status["LShift"] == 1:
            lin_vel[:2] *= self.player_agent_turbo_scale[self.terrain_type]
        if key_status["Space"] == 0 and self._last_key_status["Space"] == 1:
            if self.terrain_type == "flat_terrain":
                self.terrain_type = "rough_terrain"
                print("Switch to rough terrain")
            else:
                self.terrain_type = "flat_terrain"
                print("Switch to flat terrain")
        if key_status["M"] == 0 and self._last_key_status["M"] == 1:
            supported_model_types = ["sb3", "onnx", "grpc", "rllib"]
            if self.model_type in supported_model_types:
                current_index = supported_model_types.index(self.model_type)
                self.model_type = supported_model_types[(current_index + 1) % len(supported_model_types)]
                print(f"Switch to {self.model_type} model")

        self._last_key_status = key_status.copy()
        # print("Lin vel: ", lin_vel, "Turn angel: ", ang_vel, "Reborn: ", reborn, "Terrain type: ", self.terrain_type)
        return lin_vel, ang_vel, reborn, self.terrain_type, self.model_type

    def get_state(self):
        return self.key_status

    def get_terrain_type(self):
        return self.terrain_type


def register_env(orcagym_addr : str, 
                 env_name : str, 
                 env_index : int, 
                 agent_name : str,
                 ctrl_device : str,
                 max_episode_steps : int,
                 height_map : str,
                 ) -> str:
    orcagym_addr_str = orcagym_addr.replace(":", "-")
    env_id = env_name + "-OrcaGym-" + orcagym_addr_str + f"-{env_index:03d}"
    agent_names = [f"{agent_name}_000"]
    print("Agent names: ", agent_names)
    kwargs = {'frame_skip': FRAME_SKIP,   
                'orcagym_addr': orcagym_addr, 
                'env_id': env_id,
                'agent_names': agent_names,
                'time_step': TIME_STEP,
                'action_skip': ACTION_SKIP,
                'max_episode_steps': max_episode_steps,
                'ctrl_device': ctrl_device,
                'control_freq': CONTROL_FREQ,
                'height_map': height_map,
                'robot_config': LeggedRobotConfig[agent_name],
                'legged_obs_config': LeggedObsConfig,
                'curriculum_config': CurriculumConfig,
                'legged_env_config': LeggedEnvConfig,
                }
    gym.register(
        id=env_id,
        entry_point='envs.legged_gym.legged_sim_env:LeggedSimEnv',
        kwargs=kwargs,
        max_episode_steps= max_episode_steps,
        reward_threshold=0.0,
    )
    return env_id, kwargs

def load_sb3_model(model_file: dict):
    """加载 SB3 模型（支持 MUSA 和 CUDA）"""
    device = get_torch_device(try_to_use_gpu=True)
    print(f"加载 SB3 模型，使用设备: {device}")
    models = {}
    for key, value in model_file.items(): 
        models[key] = PPO.load(value, device=device)
    return models

def load_onnx_model(model_file: dict):
    import onnxruntime as ort
    models = {}
    for key, value in model_file.items():
        # 显式指定GPU优先
        providers = [
            'CUDAExecutionProvider',  # 优先尝试GPU
            'CPUExecutionProvider'    # GPU不可用时回退到CPU
        ]
        models[key] = ort.InferenceSession(value, providers=providers)
    return models


def load_grpc_model(model_file: dict):
    """加载gRPC模型客户端"""
    
    models = {}
    for key, value in model_file.items():
        # value应该是服务器地址，格式为 "host:port"
        if isinstance(value, str):
            server_address = value
            timeout = 5.0
            max_retries = 3
        else:
            # 如果value是字典，提取服务器地址
            server_address = value.get("server_address", "localhost:50051")
            timeout = value.get("timeout", 5.0)
            max_retries = value.get("max_retries", 3)
        
        models[key] = create_grpc_client(
            server_address=server_address,
            timeout=timeout,
            max_retries=max_retries
        )
    return models

def load_rllib_model(model_file: dict):
    """加载 RLLib 模型（支持 MUSA 和 CUDA）"""
    from ray.rllib.core.rl_module.rl_module import RLModule
    from ray.rllib.core import DEFAULT_MODULE_ID
    import examples.legged_gym.scripts.rllib_appo_rl as rllib_appo_rl

    # 验证 GPU 环境（支持 CUDA 和 MUSA）
    gpu_info = get_gpu_info()
    if gpu_info['available']:
        print(f"GPU 环境验证通过: {gpu_info['device_type']}")
        print_gpu_info()
    else:
        print("GPU 环境设置失败，将使用 CPU")

    models = {}
    for key, value in model_file.items():
        print("Loading rllib model: ", value)
        # 从字符串中提取绝对路径
        checkpoint_path = os.path.abspath(value)
        checkpoint_path = os.path.join(
            checkpoint_path,
            "learner_group",
            "learner",
            "rl_module",
            DEFAULT_MODULE_ID,
        )
        print("Checkpoint path: ", checkpoint_path)
        rl_module = RLModule.from_checkpoint(checkpoint_path)

        models[key] = rl_module
    return models


def main(
    config: dict,
    remote: str,
    ):
    try:
        # 打印 GPU 信息
        print("="*50)
        print("推理环境 GPU 信息")
        print_gpu_info()
        print("="*50)
        
        if remote is not None:
            orcagym_addresses = [remote]
        else:
            orcagym_addresses = config['orcagym_addresses']

        agent_name = config['agent_name']
        model_file = config['model_file']
        model_type = config['model_type']
        ctrl_device = config['ctrl_device']
        terrain_asset_paths = config['terrain_asset_paths']
        agent_asset_path = config['agent_asset_path']

        height_map_dir = "./height_map"
        command_model = config['command_model']

        assert model_type in ["sb3", "onnx", "torch", "grpc", "rllib"], f"Invalid model type: {model_type}"

        models = {}
        if "sb3" in model_file:
            models["sb3"] = load_sb3_model(model_file["sb3"])
        if "onnx" in model_file:
            models["onnx"] = load_onnx_model(model_file["onnx"])
        if "grpc" in model_file:
            models["grpc"] = load_grpc_model(model_file["grpc"])
        if "rllib" in model_file:
            models["rllib"] = load_rllib_model(model_file["rllib"])

        # 清空场景
        clear_scene(
            orcagym_addresses=orcagym_addresses,
        )

        # 发布地形
        publish_terrain(
            orcagym_addresses=orcagym_addresses,
            terrain_asset_paths=terrain_asset_paths,
        )

        # 空场景生成高度图
        height_map_file = generate_height_map_file(
            orcagym_addresses=orcagym_addresses,
        )

        # 放置机器人
        publish_scene(
            orcagym_addresses=orcagym_addresses,
            agent_name=agent_name,
            agent_asset_path=agent_asset_path,
            agent_num=1,
            terrain_asset_paths=terrain_asset_paths,
        )

        print("simulation running... , orcagym_addr: ", orcagym_addresses)
        env_name = "LeggedSim-v0"
        env_id, kwargs = register_env(
            orcagym_addr=orcagym_addresses[0], 
            env_name=env_name, 
            env_index=0, 
            agent_name=agent_name, 
            ctrl_device=ctrl_device, 
            max_episode_steps=MAX_EPISODE_STEPS,
            height_map=height_map_file,
        )
        print("Registered environment: ", env_id)

        env = gym.make(env_id)
        print("Starting simulation...")

        friction_scale = config['friction_scale']
        if friction_scale is not None:
            env.unwrapped.setup_base_friction(friction_scale)

        keyboard_control = KeyboardControl(orcagym_addresses[0], env, command_model, model_type)

        agent_name_list = [agent_name]
        run_simulation(
            env=env,
            agent_name_list=agent_name_list,
            models=models,
            model_type=model_type,
            time_step=TIME_STEP,
            frame_skip=FRAME_SKIP,
            action_skip=ACTION_SKIP,
            keyboard_control=keyboard_control,
            command_model=command_model,
        )
    finally:
        print("退出仿真环境")
        # 清理gRPC客户端连接
        if model_type == "grpc" and models and "grpc" in models:
            for client in models["grpc"].values():
                if hasattr(client, 'close'):
                    client.close()
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
    Log observations and actions to CSV files in the format matching the reference files.
    
    Args:
        obs (dict): Observation dictionary containing IMU and joint data
        action (np.ndarray): Action array
        filename (str): Base path for the CSV files (without extension)
        physics_step (int): Current physics simulation step count
        control_step (int): Current control/policy step count
        sim_time (float): Current simulation time in seconds
    """
    import os
    import csv
    from datetime import datetime
    
    # Create base directory if it doesn't exist
    base_dir = os.path.dirname(filename)
    if base_dir and not os.path.exists(base_dir):
        os.makedirs(base_dir)
    
    # Generate timestamp for this step (using physics_step as timestamp)
    timestamp = sim_time
    
    # Extract base filename without extension
    base_filename = os.path.splitext(filename)[0]
    
    # 1. Log observation data
    obs_filename = f"{base_filename}_observation.csv"
    obs_file_exists = os.path.exists(obs_filename)
    
    with open(obs_filename, 'a', newline='') as f:
        writer = csv.writer(f)
        if not obs_file_exists:
            # Write header: timestamp, obs_0, obs_1, ..., obs_64
            obs_headers = ["timestamp"] + [f"obs_{i}" for i in range(len(obs["observation"]))]
            writer.writerow(obs_headers)
        
        # Write observation data
        obs_data = [timestamp] + list(obs["observation"])
        writer.writerow(obs_data)
    
    # 2. Log action data (processed actions)
    action_filename = f"{base_filename}_action.csv"
    action_file_exists = os.path.exists(action_filename)
    
    with open(action_filename, 'a', newline='') as f:
        writer = csv.writer(f)
        if not action_file_exists:
            # Write header: timestamp, action_0, action_1, ..., action_11
            action_headers = ["timestamp"] + [f"action_{i}" for i in range(len(action))]
            writer.writerow(action_headers)
        
        # Write action data
        action_data = [timestamp] + list(action)
        writer.writerow(action_data)
    
    # 3. Log raw action data (same as action for now, but could be different if needed)
    raw_action_filename = f"{base_filename}_raw_action.csv"
    raw_action_file_exists = os.path.exists(raw_action_filename)
    
    with open(raw_action_filename, 'a', newline='') as f:
        writer = csv.writer(f)
        if not raw_action_file_exists:
            # Write header: timestamp, raw_action_0, raw_action_1, ..., raw_action_11
            raw_action_headers = ["timestamp"] + [f"raw_action_{i}" for i in range(len(action))]
            writer.writerow(raw_action_headers)
        
        # Write raw action data (currently same as processed action)
        raw_action_data = [timestamp] + list(action)
        writer.writerow(raw_action_data)

def run_simulation(env: gym.Env, 
                 agent_name_list: list[str],
                 models: dict, 
                 model_type: str,
                 time_step: float, 
                 frame_skip: int,
                 action_skip: int,
                 keyboard_control: KeyboardControl,
                 command_model: dict[str, str]):
    obs, info = env.reset()

    dt = time_step * frame_skip * action_skip
    if not os.path.exists("./log"):
        os.makedirs("./log")
    # Generate base filename for robot data files
    timestamp_str = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    log_file = f"./log/robot_data_{timestamp_str}"
    
    # Add step counting
    physics_step = 0
    control_step = 0
    sim_time = 0.0
    brake_time = 0.0
        
    try:
        while True:
            start_time = datetime.now()


            lin_vel, ang_vel, reborn, terrain_type, model_type = keyboard_control.update()
            if reborn:
                obs, info = env.reset()
                continue

            if np.linalg.norm(lin_vel) == 0.0:
                brake_time += dt
            else:
                brake_time = 0.0

            model = models[model_type][terrain_type]

            command_dict = {"lin_vel": lin_vel, "ang_vel": ang_vel}
            if hasattr(env, "setup_command"):
                env.setup_command(command_dict)
            else:
                env.unwrapped.setup_command(command_dict)

            segmented_obs = segment_obs(obs, agent_name_list)
            action_list = []
            for agent_obs in segmented_obs.values():
                if model_type == "sb3":
                    # print("sb3 obs: ", agent_obs)
                    sb3_action, _states = model.predict(agent_obs, deterministic=True)
                    # print("sb3 action: ", sb3_action)
                    # print("--------------------------------")
                    action = sb3_action

                elif model_type == "onnx":
                    agent_obs = {
                        "observation_achieved_goal": np.array([agent_obs["achieved_goal"]], dtype=np.float32),
                        "observation_desired_goal": np.array([agent_obs["desired_goal"]], dtype=np.float32),
                        "observation_observation": np.array([agent_obs["observation"]], dtype=np.float32)
                    }
                    # print("onnx obs: ", agent_obs)
                    onnx_actions = model.run(None, agent_obs)[0]
                    onnx_action = onnx_actions[0]
                    onnx_action = np.clip(onnx_action, -100, 100)
                    # print("onnx action: ", onnx_action)
                    # print("--------------------------------")
                    action = onnx_action

                elif model_type == "grpc":
                    # 准备gRPC请求的观察数据
                    grpc_obs = {
                        "observation": agent_obs["observation"].astype(np.float32),
                        "desired_goal": agent_obs["desired_goal"].astype(np.float32),
                        "achieved_goal": agent_obs["achieved_goal"].astype(np.float32)
                    }
                    # print("grpc obs: ", grpc_obs)
                    grpc_action, _states = model.predict(grpc_obs, model_type=terrain_type, deterministic=True)
                    if grpc_action is None:
                        grpc_action = np.zeros(env.action_space.shape[0])
                    # print("grpc action: ", grpc_action)
                    # print("--------------------------------")
                    action = grpc_action

                elif model_type == "rllib":
                    from ray.rllib.core.columns import Columns
                    from torch import torch
                    from ray.rllib.utils.numpy import convert_to_numpy
                    # 使用正确的设备（支持 MUSA 和 CUDA）
                    device = get_torch_device(try_to_use_gpu=True)
                    input_dict = {Columns.OBS: torch.from_numpy(agent_obs["observation"]).unsqueeze(0).to(device)}
                    rl_module_out = model.forward_inference(input_dict)
                    logits = convert_to_numpy(rl_module_out[Columns.ACTION_DIST_INPUTS])
                    mu = logits[:, :env.action_space.shape[0]]
                    action = np.clip(mu[0], env.action_space.low, env.action_space.high)
                    # print("rllib action: ", action)
                    # print("--------------------------------")
                else:
                    raise ValueError(f"Invalid model type: {model_type}")

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
    parser.add_argument('--config', type=str, help='The path of the config file')
    parser.add_argument('--remote', type=str, help='The remote address of the orca studio')
    args = parser.parse_args()

    if args.config is None:
        raise ValueError("Config file is required")
    
    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    main(
        config=config,
        remote=args.remote,
    )


