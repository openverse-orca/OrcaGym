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


from envs.legged_gym.legged_config import LeggedEnvConfig
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


