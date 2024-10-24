import os
import sys
import time

current_file_path = os.path.abspath('')
project_root = os.path.dirname(os.path.dirname(current_file_path))

# 将项目根目录添加到 PYTHONPATH
if project_root not in sys.path:
    sys.path.append(project_root)


import gymnasium as gym
from gymnasium.envs.registration import register
from datetime import datetime
from envs.orca_gym_env import ActionSpaceType

# 
TIME_STEP = 0.005

def register_env(grpc_address, control_freq=20):
    print("register_env: ", grpc_address)
    gym.register(
        id=f"XboxControl-v0-OrcaGym-{grpc_address[-2:]}",
        entry_point="envs.realman.rm65b_joystick_env:RM65BJoystickEnv",
        kwargs={'frame_skip': 1,   
                'reward_type': "dense",
                'action_space_type': ActionSpaceType.CONTINUOUS,
                'action_step_count': 0,
                'grpc_address': grpc_address, 
                'agent_names': ['RM65B'], 
                'time_step': TIME_STEP,
                'control_freq': control_freq},
        max_episode_steps=sys.maxsize,
        reward_threshold=0.0,
    )

def continue_training(env):
    observation, info = env.reset(seed=42)
    while True:
        start_time = datetime.now()

        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)

        # 
        elapsed_time = datetime.now() - start_time
        if elapsed_time.total_seconds() < TIME_STEP:
            time.sleep(TIME_STEP - elapsed_time.total_seconds())

    

if __name__ == "__main__":
    try:
        grpc_address = "localhost:50051"
        print("simulation running... , grpc_address: ", grpc_address)
        env_id = f"XboxControl-v0-OrcaGym-{grpc_address[-2:]}"

        register_env(grpc_address, 20)

        env = gym.make(env_id)        
        print("Start Simulation!")

        continue_training(env)
    except KeyboardInterrupt:
        print("Exit Simulation!")        
        env.close()