import os
import sys
import time

import numpy as np



current_file_path = os.path.abspath('./../../')

if current_file_path not in sys.path:
    print("add path: ", current_file_path)
    sys.path.append(current_file_path)
from envs.robomimic.dataset_util import DatasetWriter

import gymnasium as gym
from gymnasium.envs.registration import register
from datetime import datetime

# 
TIME_STEP = 0.01

def register_env(orcagym_addr, env_name, env_index, control_freq=20) -> str:
    orcagym_addr_str = orcagym_addr.replace(":", "-")
    env_id = env_name + "-OrcaGym-" + orcagym_addr_str + f"-{env_index:03d}"
    gym.register(
        id=env_id,
        entry_point="envs.openloong.Openloong_arm_vr_env:OpenloongArmEnv",
        kwargs={'frame_skip': 1,   
                'reward_type': "dense",
                'orcagym_addr': orcagym_addr, 
                'agent_names': ['AzureLoong'], 
                'time_step': TIME_STEP,
                'control_freq': control_freq},
        max_episode_steps=sys.maxsize,
        reward_threshold=0.0,
    )
    return env_id

def continue_training(env):
    observation, info = env.reset(seed=42)
    print("""
        To use VR controllers, please press left joystick to connect / disconnect to the simulator.
        And then press right joystick to reset the robot's hands to the initial position.
        """)  
    while True:
        start_time = datetime.now()

        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
        # Collect data after each step
        obs_list = [observation]  # Collect observations for this step
        action_list = [action]  # Collect actions
        reward_list = [reward]  # Collect rewards
        done_list = [1 if terminated else 0]  # Collect done flags
        info_list = [info]  # Collect additional info (like state)

        # Save data to .h5 using DatasetWriter
        dataset_writer.add_demo({
            'states': np.array([np.concatenate([info["state"]["qpos"], info["state"]["qvel"]]) for info in info_list]),
            'actions': np.array(action_list),
            'rewards': np.array(reward_list),
            'dones': np.array(done_list),
            'obs': obs_list
        })
        elapsed_time = datetime.now() - start_time
        if elapsed_time.total_seconds() < TIME_STEP:
            time.sleep(TIME_STEP - elapsed_time.total_seconds())

    

if __name__ == "__main__":
    """
    OSC运动算法控制青龙机器人机械臂的示例
    """
    try:
        orcagym_addr = "localhost:50051"
        print("simulation running... , orcagym_addr: ", orcagym_addr)

        env_name = "OpenloongArmVR-v0"
        env_index = 0
        env_id = register_env(orcagym_addr, env_name, env_index, 20)
        print("Registering environment with id: ", env_id)
        # Create a new DatasetWriter instance to save the data
        formatted_now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        dataset_writer = DatasetWriter(file_path=f"teleoperation_dataset_{formatted_now}.hdf5",
                                       env_name=env_name,
                                       env_version="v1",
                                       env_kwargs={'orcagym_addr': orcagym_addr})
        env = gym.make(env_id)        
        print("Starting simulation...")

        continue_training(env)
    except KeyboardInterrupt:
        print("Simulation stopped")        
        env.close()