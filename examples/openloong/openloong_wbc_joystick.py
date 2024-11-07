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

def register_env(orcagym_addr, env_name, env_index, agent_names, time_step, render_mode, urdf_path, json_path, log_path, individual_control):
    env_id = env_name + "-OrcaGym-" + orcagym_addr + f"-{env_index}"
    gym.register(
        id=env_id,
        entry_point="envs.openloong.openloong_env:OpenLoongEnv",
        kwargs={'frame_skip': 1,   
                'reward_type': "dense",
                'orcagym_addr': orcagym_addr, 
                'agent_names': agent_names, 
                'time_step': time_step,
                'render_mode': render_mode,
                'urdf_path': urdf_path,
                'json_path': json_path,
                'log_path': log_path,
                'individual_control': individual_control
                },
        max_episode_steps=sys.maxsize,  # never stop
        reward_threshold=0.0,
    )
    return env_id



def run_simulation(env, time_step):
    observation, info = env.reset(seed=42)
    time_counter = 0
    while True:
        start_time = time.perf_counter()

        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
        env.render()

        end_time = time.perf_counter()
        elapsed_time = end_time - start_time

        time_counter += 1
        if time_counter % 1000 == 0:
            print(f"elapsed_time (ms): {elapsed_time * 1000}")

        # if elapsed_time < time_step:
        #     time.sleep(time_step - elapsed_time)


if __name__ == '__main__':
    """
    The startup script for the openloong walking wbc control using joystick.
    """

    parser = argparse.ArgumentParser(description='Simulation Configuration')
    parser.add_argument('--orcagym_addr', type=str, default="localhost:50051", help='The gRPC address for the simulation')
    parser.add_argument('--agent_name', type=str, default="AzureLoong", help='The agent name for the simulation')
    parser.add_argument('--individual_control', type=str, default="True", help='Control the robots individually')
    parser.add_argument('--render_mode' , type=str, default="human", help='The render mode for the simulation')
    args = parser.parse_args()

    orcagym_addr = f"{args.orcagym_addr}"
    agent_name = f"{args.agent_name}"
    individual_control = True if f"{args.individual_control}" == "True" else False
    render_mode = f"{args.render_mode}"
    print("___render_mode: ", render_mode)

    simulation_frequency = 1000
    time_step = 1.0 / simulation_frequency

    urdf_path = project_root + "/envs/openloong/external/openloong-dyn-control/models/AzureLoong.urdf"
    json_path = project_root + "/envs/openloong/external/openloong-dyn-control/common/joint_ctrl_config.json"
    log_path = project_root + "/envs/openloong/records/datalog.log"

    if not os.path.exists(project_root + "/envs/openloong/records"):
        os.makedirs(project_root + "/envs/openloong/records")

    print("simulation running... , orcagym_addr: ", orcagym_addr, ", agent_name: ", agent_name)

    env_name = "Openloong-v0"
    env_index = 0
    # register_env(orcagym_addr, env_name, env_index, [agent_name, f"{agent_name}_01", f"{agent_name}_02"], time_step, urdf_path, json_path, log_path, individual_control)
    env_id = register_env(orcagym_addr, env_name, env_index, [agent_name], time_step, render_mode, urdf_path, json_path, log_path, individual_control)
    env = gym.make(env_id)        
    print("Start Simulation!")    

    try:
        run_simulation(env, time_step)
    finally:
        print("Exit Simulation!")        
        env.close()