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
#from envs.orca_gym_env import ActionSpaceType

# 
#TIME_STEP = 0.005
TIME_STEP = 0.01

def register_env(orcagym_addr, env_name, env_index, control_freq=20) -> str:
    print("register_env: ", orcagym_addr)
    orcagym_addr_str = orcagym_addr.replace(":", "-")
    env_id = env_name + "-OrcaGym-" + orcagym_addr_str + f"-{env_index:03d}"
    gym.register(
        id=env_id,
        entry_point="envs.realman.rm75bv_vr_env:RM75BVVREnv",
        kwargs={'frame_skip': 1,   
                'reward_type': "dense",
               # 'action_space_type': ActionSpaceType.CONTINUOUS,
              #  'action_step_count': 0,
                'grpc_address': orcagym_addr, 
                'agent_names': ['RM75_B_V'], 
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

        elapsed_time = datetime.now() - start_time
        if elapsed_time.total_seconds() < TIME_STEP:
            time.sleep(TIME_STEP - elapsed_time.total_seconds())
    

if __name__ == "__main__":
    try:
        orcagym_addr = "localhost:50051"
        print("simulation running... , grpc_address: ", orcagym_addr)
       # env_id = f"XboxControl-v0-OrcaGym-{grpc_address[-2:]}"
                
        env_name = "RM75VRControl-v0"
        env_index = 0
        env_id = register_env(orcagym_addr, env_name, env_index, 20)
       # register_env(grpc_address, 20)

        env = gym.make(env_id)        
        print("Start Simulation!")

        continue_training(env)
    except KeyboardInterrupt:
        print("Exit Simulation!")        
        env.close()