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

from orca_gym.log.orca_log import get_orca_logger
_logger = get_orca_logger()


# 
TIME_STEP = 0.005

def register_env(orcagym_addr, env_name, env_index, control_freq=20) -> str:
    orcagym_addr_str = orcagym_addr.replace(":", "-")
    env_id = env_name + "-OrcaGym-" + orcagym_addr_str + f"-{env_index:03d}"
    gym.register(
        id=env_id,
        entry_point="envs.realman.rm65b_joystick_env:RM65BJoystickEnv",
        kwargs={'frame_skip': 1,   
                'reward_type': "dense",
                'orcagym_addr': orcagym_addr, 
                'agent_names': ['RM65B'], 
                'time_step': TIME_STEP,
                'control_freq': control_freq},
        max_episode_steps=sys.maxsize,
        reward_threshold=0.0,
    )
    return env_id

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
        orcagym_addr = "localhost:50051"
        _logger.info(f"simulation running... , orcagym_addr:  {orcagym_addr}")

        env_name = "RM65BXboxControl-v0"
        env_index = 0
        env_id = register_env(orcagym_addr, env_name, env_index, 20)
        _logger.info(f"Registering environment with id:  {env_id}")

        env = gym.make(env_id)        
        _logger.info("Start Simulation!")

        continue_training(env)
    except KeyboardInterrupt:
        print("Exit Simulation!")        
        env.close()