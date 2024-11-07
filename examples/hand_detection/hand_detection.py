import os
import sys

current_file_path = os.path.abspath('./')

# 将项目根目录添加到 PYTHONPATH
print(current_file_path)
if current_file_path not in sys.path:
    sys.path.append(current_file_path)


import gymnasium as gym
import asyncio
import nest_asyncio
from gymnasium.envs.registration import register
from datetime import datetime


nest_asyncio.apply()

# TIME_STEP = 0.016666666666666
TIME_STEP = 0.005

def register_env(orcagym_addr, env_name, env_index) -> str:
    env_id = env_name + "-OrcaGym-" + orcagym_addr + f"-{env_index}"
    gym.register(
        id=env_id,
        entry_point="envs.hand_detection.hand_detection_env:HandDetectionEnv",
        kwargs={'frame_skip': 1,   # 1 action per frame
                'reward_type': "dense",
                'orcagym_addr': orcagym_addr, 
                'agent_names': [''], 
                'time_step': TIME_STEP,},
        max_episode_steps=60 * 60 * 60,  # 60fps @ 1 hour
        reward_threshold=0.0,
    )
    return env_id

async def continue_training(env):
    observation, info = env.reset(seed=42)
    while True:
        start_time = datetime.now()

        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)

        # 帧率为 60fps ，为显示为正常速度，每次渲染间隔 16ms
        elapsed_time = datetime.now() - start_time
        if elapsed_time.total_seconds() < TIME_STEP:
            await asyncio.sleep(TIME_STEP - elapsed_time.total_seconds())

    

if __name__ == "__main__":
    try:
        orcagym_addr = "localhost:50051"
        print("simulation running... , orcagym_addr: ", orcagym_addr)

        env_name = "HandDetection-v0"
        env_index = 0
        env_id = register_env(orcagym_addr, env_name, env_index)
        print("Registering environment with id: ", env_id)

        env = gym.make(env_id)        
        print("启动仿真环境")

        asyncio.run(continue_training(env))
    except KeyboardInterrupt:
        print("关闭仿真环境")        
        env.close()