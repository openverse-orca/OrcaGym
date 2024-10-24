import os
import sys
import gymnasium as gym
import asyncio
import nest_asyncio
from gymnasium.envs.registration import register
from datetime import datetime
current_file_path = os.path.abspath('')
project_root = os.path.dirname(os.path.dirname(current_file_path))

# 将项目根目录添加到 PYTHONPATH
if project_root not in sys.path:
    sys.path.append(project_root)
from envs.orca_gym_env import ActionSpaceType
from envs.car_control.car_joystick_env import CarEnv, RecordState  # Replace with actual path to your modules

nest_asyncio.apply()

# TIME_STEP = 0.016666666666666
TIME_STEP = 0.005

def register_env(grpc_address, record_state, record_file):
    print("register_env: ", grpc_address)
    gym.register(
        id=f"XboxControl-v0-OrcaGym-{grpc_address[-2:]}",
        entry_point=CarEnv,
        kwargs={
            'frame_skip': 1,   # 1 action per frame
            'grpc_address': grpc_address, 
            'agent_names': ['Agent0'], 
            'time_step': TIME_STEP,
            'record_state': record_state,
            'record_file': record_file,
            'action_space_type': ActionSpaceType.CONTINUOUS,  # Example value
            'action_step_count': 0,  # Example value, adjust as needed
        },
        max_episode_steps=60 * 60 * 60,  # 60fps @ 1 hour
        reward_threshold=0.0,
    )

async def continue_training(env):
    observation, info = env.reset(seed=42)
    while True:
        start_time = datetime.now()

        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)

        elapsed_time = datetime.now() - start_time
        if elapsed_time.total_seconds() < TIME_STEP:
            await asyncio.sleep(TIME_STEP - elapsed_time.total_seconds())

if __name__ == "__main__":
    try:
        grpc_address = "localhost:50051"
        print("simulation running... , grpc_address: ", grpc_address)
        env_id = f"XboxControl-v0-OrcaGym-{grpc_address[-2:]}"

        # RecordState controls the recording of the simulation data
        register_env(grpc_address, RecordState.RECORD, 'xbox_control_record.h5')

        env = gym.make(env_id)        
        print("Starting simulation...")

        asyncio.run(continue_training(env))
    except KeyboardInterrupt:
        print("Simulation stopped")        
        env.close()
