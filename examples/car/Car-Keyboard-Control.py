# Car-Keyboard-Control.py

import os
import sys
import gymnasium as gym
import asyncio
import nest_asyncio
from datetime import datetime
current_file_path = os.path.abspath('')
project_root = os.path.dirname(os.path.dirname(current_file_path))
# 将项目根目录添加到 PYTHONPATH
if project_root not in sys.path:
    sys.path.append(project_root)


nest_asyncio.apply()

# TIME_STEP = 0.016666666666666
TIME_STEP = 0.005

def register_env(orcagym_addr, env_name, env_index) -> str:
    orcagym_addr_str = orcagym_addr.replace(":", "-")
    env_id = env_name + "-OrcaGym-" + orcagym_addr_str + f"-{env_index:03d}"
    gym.register(
        id=env_id,
        entry_point="envs.car_keyboard_env.CarKeyboardEnv",
        kwargs={
            'frame_skip': 1,   # 1 action per frame
            'orcagym_addr': orcagym_addr,
            'agent_names': ['Agent0'],
            'time_step': TIME_STEP,
        },
        max_episode_steps=60 * 60 * 60,  # 60fps @ 1 hour
        reward_threshold=0.0,
    )
    return env_id

async def continue_training(env):
    observation, info = env.reset(seed=42)
    while True:
        start_time = datetime.now()

        # 采样一个随机动作，可以用键盘输入替代此处
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)

        elapsed_time = datetime.now() - start_time
        if elapsed_time.total_seconds() < TIME_STEP:
            await asyncio.sleep(TIME_STEP - elapsed_time.total_seconds())

if __name__ == "__main__":
    try:
        orcagym_addr = "localhost:50051"
        print("simulation running... , orcagym_addr: ", orcagym_addr)

        env_name = "CarKeyboardControl-v0"
        env_index = 0
        env_id = register_env(orcagym_addr, env_name, env_index)
        print("Registering environment with id: ", env_id)

        env = gym.make(env_id)
        print("Starting simulation...")

        asyncio.run(continue_training(env))
    except KeyboardInterrupt:
        print("Simulation stopped")
        env.close()
