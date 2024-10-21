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
from envs.orca_gym_env import ActionSpaceType

TIME_STEP = 0.01

def register_env(grpc_address, control_type = "None", control_freq=20):
    print("register_env: ", grpc_address)
    gym.register(
        id=f"Franka-Control-v0-OrcaGym-{grpc_address[-2:]}",
        entry_point="envs.franka_control.franka_teleoperation_env:FrankaTeleoperationEnv",
        kwargs={'frame_skip': 1,   
                'reward_type': "sparse",
                'action_space_type': ActionSpaceType.CONTINUOUS,
                'action_step_count': 0,
                'grpc_address': grpc_address, 
                'agent_names': ['Panda'], 
                'time_step': TIME_STEP,
                'control_type': control_type,
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

        elapsed_time = datetime.now() - start_time
        if elapsed_time.total_seconds() < TIME_STEP:
            time.sleep(TIME_STEP - elapsed_time.total_seconds())

    

if __name__ == "__main__":
    """
    An example of an OSC (Operational Space Control) motion algorithm controlling a Franka robotic arm.
    Level: Franka_Teleoperation
    Differences from Franka_Joystick:
    1. Motor control uses torque output (moto) instead of setting joint angles.
    2. Torque calculation is based on the OSC algorithm.
    3. The mocap point can move freely and is not welded to the site; the pulling method is not used.
    """
    try:
        grpc_address = "localhost:50051"
        print("simulation running... , grpc_address: ", grpc_address)
        env_id = f"Franka-Control-v0-OrcaGym-{grpc_address[-2:]}"

        # RecordState controls the recording of the simulation data
        register_env(grpc_address, "Xbox", 20)

        env = gym.make(env_id)        
        print("Starting simulation...")

        continue_training(env)
    except KeyboardInterrupt:
        print("Simulation stopped")        
        env.close()