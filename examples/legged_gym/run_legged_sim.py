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

TIME_STEP = 0.005                       # 200 Hz for physics simulation
FRAME_SKIP = 4
REALTIME_STEP = TIME_STEP * FRAME_SKIP  # 50 Hz for rendering
CONTROL_FREQ = 1 / REALTIME_STEP        # 50 Hz for ppo policy

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
    pico_ports = pico_ports.split(" ")
    print("Pico ports: ", pico_ports)
    kwargs = {'frame_skip': FRAME_SKIP,   
                'orcagym_addr': orcagym_addr, 
                'agent_names': agent_names_list,
                'time_step': TIME_STEP,
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


def run_sim(orcagym_addresses, 
               agent_name, 
               model_file,
               ctrl_device,):
    try:
        print("simulation running... , orcagym_addr: ", orcagym_addresses)
        env_name = "LeggedSim-v0"
        env_id, kwargs = register_env(orcagym_addresses, 
                                      env_name, 
                                      0, 
                                      [agent_name], 
                                      ctrl_device, 
                                      sys.maxsize, 
                                      )
        print("Registered environment: ", env_id)

        env = gym.make(env_id)
        print("Starting simulation...")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = PPO.load(model_file, env=env, device=device)

        testing_model(env, 1, model, time_step, max_episode_steps, frame_skip)
    except KeyboardInterrupt:
        print("退出仿真环境")
        env.close()

def testing_model(env : SubprocVecEnvMA, agent_num, model, time_step, max_episode_steps, frame_skip):
    # 测试模型
    observations = env.reset()
    test = 0
    total_rewards = np.zeros(agent_num)
    step = 0
    dt = time_step * frame_skip
    print("Start Testing!")
    try:
        while True:
            step += 1
            start_time = datetime.now()

            obs_list = _segment_observation(observations, agent_num)
            action_list = []
            for agent_obs in obs_list:
                # predict_start = datetime.now()
                action, _states = model.predict(agent_obs, deterministic=True)
                action_list.append(action)
                # predict_time = datetime.now() - predict_start
                # print("Predict Time: ", predict_time.total_seconds(), flush=True)

            action = np.concatenate(action_list).flatten()
            # print("action: ", action)
            # setp_start = datetime.now()
            observations, rewards, dones, infos = env.step(action)

            # print("obs, reward, terminated, truncated, info: ", observation, reward, terminated, truncated, info)


            env.render()
            # step_time = datetime.now() - setp_start
            # print("Step Time: ", step_time.total_seconds(), flush=True)

            total_rewards += rewards

            # 
            elapsed_time = datetime.now() - start_time
            if elapsed_time.total_seconds() < dt:
                # print("Sleep for ", dt - elapsed_time.total_seconds())
                time.sleep(dt - elapsed_time.total_seconds())

            if step == max_episode_steps:
                _output_test_info(test, total_rewards, rewards, dones, infos)
                step = 0
                test += 1
                total_rewards = np.zeros(agent_num)
            
    finally:
        print("退出仿真环境")
        env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run multiple instances of the script with different gRPC addresses.')
    parser.add_argument('--orcagym_addresses', type=str, nargs='+', default=['localhost:50051'], help='The gRPC addresses to connect to')
    parser.add_argument('--agent_name', type=str, default='go2', help='The name of the agent')
    parser.add_argument('--model_file', type=str, help='The model file to save/load. If not provided, a new model file will be created while training')
    parser.add_argument('--ctrl_device', type=str, default='keyboard', help='The control device to use ')
    args = parser.parse_args()



    orcagym_addresses = args.orcagym_addresses
    model_file = args.model_file
    agent_name = args.agent_name
    ctrl_device = args.ctrl_device

    run_sim(orcagym_addresses, 
               agent_name, 
               model_file,
               ctrl_device)    


