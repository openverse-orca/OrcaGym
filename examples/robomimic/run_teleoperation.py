import os
import sys
import time

current_file_path = os.path.abspath('')
project_root = os.path.dirname(os.path.dirname(current_file_path))

if project_root not in sys.path:
    sys.path.append(project_root)


import gymnasium as gym
from gymnasium.envs.registration import register
from datetime import datetime
from orca_gym.environment.orca_gym_env import RewardType
from envs.robomimic.robomimic_env import ControlType
from envs.robomimic.dataset_util import DatasetWriter

import numpy as np
import argparse



def register_env(orcagym_addr, env_name, env_index, agent_name, max_episode_steps) -> str:
    orcagym_addr_str = orcagym_addr.replace(":", "-")
    env_id = env_name + "-OrcaGym-" + orcagym_addr_str + f"-{env_index:03d}"
    agent_names = [f"{agent_name}"]
    kwargs = {'frame_skip': 1,   
                'reward_type': RewardType.SPARSE,
                'orcagym_addr': orcagym_addr, 
                'agent_names': agent_names, 
                'time_step': TIME_STEP,
                'control_type': ControlType.TELEOPERATION,
                'control_freq': 20}           
    gym.register(
        id=env_id,
        entry_point="envs.franka_control.franka_teleoperation_env:FrankaTeleoperationEnv",
        kwargs=kwargs,
        max_episode_steps= max_episode_steps,  # 10 seconds
        reward_threshold=0.0,
    )
    return env_id, kwargs

def run_episode(env, dataset_writer):
    obs, info = env.reset(seed=42)
    obs_list = {obs_key: list([]) for obs_key, obs_data in obs.items()}
    reward_list = []
    done_list = []
    info_list = []    
    terminated_times = 0
    while True:
        start_time = datetime.now()

        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        env.render()
        
        for obs_key, obs_data in obs.items():
            obs_list[obs_key].append(obs_data)
            
        reward_list.append(reward)
        done_list.append(0 if not terminated else 1)
        info_list.append(info)
        terminated_times = terminated_times + 1 if terminated else 0

        if terminated_times >= 5 or truncated:
            return obs_list, reward_list, done_list, info_list

        elapsed_time = datetime.now() - start_time
        if elapsed_time.total_seconds() < TIME_STEP:
            time.sleep(TIME_STEP - elapsed_time.total_seconds())
        else:
            print("Over time! elapsed_time (ms): ", elapsed_time.total_seconds() * 1000)

def user_comfirm_save_record(task_result):
    while True:
        user_input = input(f"Task is {task_result}! Do you want to save the record? (y/n): ")
        if user_input == 'y':
            return True
        elif user_input == 'n':
            return False
        else:
            print("Invalid input! Please input 'y' or 'n'.")

def do_teleoperation(env, dataset_writer):
    while True:
        obs_list, reward_list, done_list, info_list = run_episode(env, dataset_writer)
        task_result = "Success" if done_list[-1] == 1 else "Failed"
        save_record = user_comfirm_save_record(task_result)
        if save_record:
            dataset_writer.add_demo({
                'states': np.array([np.concatenate([info["state"]["qpos"], info["state"]["qvel"]]) for info in info_list]),
                'actions': np.array([info["action"] for info in info_list]),
                'rewards': np.array(reward_list),
                'dones': np.array(done_list),
                'obs': obs_list
            })

def run_example(orcagym_addr : str, agent_name : str, max_episode_steps : int):
    try:
        print("simulation running... , orcagym_addr: ", orcagym_addr)
        env_name = "Franka-Teleoperation-v0"
        env_index = 0
        env_id, kwargs = register_env(orcagym_addr, env_name, env_index, agent_name, max_episode_steps)
        print("Registered environment: ", env_id)

        env = gym.make(env_id)        
        print("Starting simulation...")

        formatted_now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        dataset_writer = DatasetWriter(file_path=f"teleoperation_dataset_{formatted_now}.hdf5",
                                       env_name=env_id,
                                       env_version=env.unwrapped.get_env_version(),
                                       env_kwargs=kwargs)

        do_teleoperation(env, dataset_writer)
        dataset_writer.finalize()
    except KeyboardInterrupt:
        print("Simulation stopped")        
        dataset_writer.finalize()
        env.close()
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run multiple instances of the script with different gRPC addresses.')
    parser.add_argument('--orcagym_address', type=str, default='localhost:50051', help='The gRPC addresses to connect to')
    parser.add_argument('--agent_name', type=str, default='Panda', help='The agent name to control')
    parser.add_argument('--record_time', type=int, default=20, help='The time to record the teleoperation in 1 episode')
    args = parser.parse_args()
    
    orcagym_addr = args.orcagym_address
    agent_name = args.agent_name
    record_time = args.record_time

    TIME_STEP = 0.01
    max_episode_steps = int(record_time / TIME_STEP)

    run_example(orcagym_addr, agent_name, max_episode_steps)
