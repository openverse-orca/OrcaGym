import os
import sys
import time

current_file_path = os.path.abspath('')
project_root = os.path.dirname(current_file_path)

if project_root not in sys.path:
    sys.path.append(project_root)


import gymnasium as gym
from gymnasium.envs.registration import register
from datetime import datetime
from envs.orca_gym_env import RewardType
from envs.robomimic.robomimic_env import ControlType
from envs.robomimic.dataset_util import DatasetWriter

import numpy as np


TIME_STEP = 0.01
MAX_EPISODE_STEPS = 10 / TIME_STEP # 10 seconds in normal speed.

def register_env(orcagym_addr, env_name, env_index, **kwargs) -> str:
    orcagym_addr_str = orcagym_addr.replace(":", "-")
    env_id = env_name + "-OrcaGym-" + orcagym_addr_str + f"-{env_index:03d}"
    gym.register(
        id=env_id,
        entry_point="envs.franka_control.franka_teleoperation_env:FrankaTeleoperationEnv",
        kwargs=kwargs,
        max_episode_steps= MAX_EPISODE_STEPS,  # 10 seconds
        reward_threshold=0.0,
    )
    return env_id

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

def run_example():
    try:
        orcagym_addr = "localhost:50051"
        print("simulation running... , orcagym_addr: ", orcagym_addr)

        env_name = "Franka-Teleoperation-v0"
        env_index = 0
        kwargs = {'frame_skip': 1,   
                    'reward_type': RewardType.SPARSE,
                    'orcagym_addr': orcagym_addr, 
                    'agent_names': ['Panda'], 
                    'time_step': TIME_STEP,
                    'control_type': ControlType.TELEOPERATION,
                    'control_freq': 20}        
        env_id = register_env(orcagym_addr, env_name, env_index, **kwargs)
        print("Registered environment: ", env_id)

        env = gym.make(env_id)        
        print("Starting simulation...")

        formatted_now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        dataset_writer = DatasetWriter(file_path=f"teleoperation_dataset_{formatted_now}.hdf5",
                                       env_name=env_id,
                                       env_version=env.get_env_version(),
                                       env_kwargs=kwargs)

        do_teleoperation(env, dataset_writer)
        dataset_writer.finalize()
    except KeyboardInterrupt:
        print("Simulation stopped")        
        dataset_writer.finalize()
        env.close()
    

if __name__ == "__main__":
    """
    An example of an OSC (Operational Space Control) motion algorithm controlling a Franka robotic arm.
    Level: Franka_Joystick
    Differences from the mocap version:
    1. Motor control uses torque output (moto) instead of setting joint angles.
    2. Torque calculation is based on the OSC algorithm.
    3. The mocap point can move freely and is not welded to the site; the pulling method is not used.
    """
    run_example()
