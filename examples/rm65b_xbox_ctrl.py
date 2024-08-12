import os
import sys

current_file_path = os.path.abspath('')
project_root = os.path.dirname(current_file_path)

# 将项目根目录添加到 PYTHONPATH
if project_root not in sys.path:
    sys.path.append(project_root)


import gymnasium as gym
import asyncio
import nest_asyncio
from gymnasium.envs.registration import register
from datetime import datetime
from envs.orca_gym_env import ActionSpaceType
from envs.franka_control.franka_joystick_env import RecordState


nest_asyncio.apply()

# TIME_STEP = 0.016666666666666
TIME_STEP = 0.005

def register_env(grpc_address, record_state, record_file):
    print("register_env: ", grpc_address)
    gym.register(
        id=f"XboxControl-v0-OrcaGym-{grpc_address[-2:]}",
        entry_point="envs.realman_rm65b.rm65b_joystick_env:RM65BJoystickEnv",
        kwargs={'frame_skip': 1,   # 1 action per frame
                'reward_type': "dense",
                'action_space_type': ActionSpaceType.CONTINUOUS,
                'action_step_count': 0,
                'grpc_address': grpc_address, 
                'agent_names': ['RM65B'], 
                'time_step': TIME_STEP,
                'record_state': record_state,
                'record_file': record_file},
        max_episode_steps=60 * 60 * 60,  # 60fps @ 1 hour
        reward_threshold=0.0,
    )

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
        grpc_address = "localhost:50051"
        print("simulation running... , grpc_address: ", grpc_address)
        env_id = f"XboxControl-v0-OrcaGym-{grpc_address[-2:]}"

        # RecordState 控制录制和回放状态
        register_env(grpc_address, RecordState.NONE, 'xbox_control_record.h5')

        env = gym.make(env_id)        
        print("启动仿真环境")

        asyncio.run(continue_training(env))
    except KeyboardInterrupt:
        print("关闭仿真环境")        
        env.save_record()
        env.close()