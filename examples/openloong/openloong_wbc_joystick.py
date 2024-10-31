import os
import subprocess
import argparse
import sys
from datetime import datetime
import time

current_file_path = os.path.abspath('')
project_root = os.path.dirname(os.path.dirname(current_file_path))

# 将项目根目录添加到 PYTHONPATH
if project_root not in sys.path:
    sys.path.append(project_root)

import gymnasium as gym

def register_env(grpc_address, agent_name, time_step, urdf_path, json_path, log_path, individual_control):
    print("register_env: ", grpc_address)
    gym.register(
        id=f"Openloong-v0-OrcaGym-{grpc_address[-2:]}",
        entry_point="envs.openloong.openloong_env:OpenLoongEnv",
        kwargs={'frame_skip': 1,   
                'reward_type': "dense",
                'grpc_address': grpc_address, 
                'agent_names': [agent_name], 
                'time_step': time_step,
                'urdf_path': urdf_path,
                'json_path': json_path,
                'log_path': log_path,
                'individual_control': individual_control
                },
        max_episode_steps=sys.maxsize,  # never stop
        reward_threshold=0.0,
    )

def run_simulation(env, time_step):
    observation, info = env.reset(seed=42)
    while True:
        start_time = datetime.now()

        action = env.action_space.sample()

        observation, reward, terminated, truncated, info = env.step(action)

        elapsed_time_1 = datetime.now() - start_time

        env.render()

        # 
        elapsed_time = datetime.now() - start_time

        # print(f"elapsed_time_1 (ms): {elapsed_time_1.total_seconds() * 1000}")
        # print(f"elapsed_time (ms): {elapsed_time.total_seconds() * 1000}")

        if elapsed_time.total_seconds() < time_step:
            time.sleep(time_step - elapsed_time.total_seconds())


if __name__ == '__main__':
    """
    The startup script for the openloong walking wbc control using joystick.
    """

    parser = argparse.ArgumentParser(description='Simulation Configuration')
    parser.add_argument('--grpc_address', type=str, default="localhost:50051", help='The gRPC address for the simulation')
    parser.add_argument('--agent_name', type=str, default="AzureLoong", help='The agent name for the simulation')
    parser.add_argument('--individual_control', type=str, default="True", help='Control the robots individually')
    args = parser.parse_args()

    grpc_address = f"{args.grpc_address}"
    agent_name = f"{args.agent_name}"
    individual_control = True if f"{args.individual_control}" == "True" else False

    simulation_frequency = 1000
    time_step = 1.0 / simulation_frequency

    urdf_path = project_root + "/envs/openloong/external/openloong-dyn-control/models/AzureLoong.urdf"
    json_path = project_root + "/envs/openloong/external/openloong-dyn-control/common/joint_ctrl_config.json"
    log_path = project_root + "/envs/openloong/records/datalog.log"

    if not os.path.exists(project_root + "/envs/openloong/records"):
        os.makedirs(project_root + "/envs/openloong/records")

    print("simulation running... , grpc_address: ", grpc_address, ", agent_name: ", agent_name)
    env_id = f"Openloong-v0-OrcaGym-{grpc_address[-2:]}"

    register_env(grpc_address, agent_name, time_step, urdf_path, json_path, log_path, individual_control)

    env = gym.make(env_id)        
    print("Start Simulation!")    

    try:
        run_simulation(env, time_step)
    finally:
        print("Exit Simulation!")        
        env.close()