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

from envs.orca_gym_env import ActionSpaceType
import gymnasium as gym
from envs.franka_control.franka_joystick_env import RecordState


def register_env(grpc_address, record_state, record_file, agent_name, time_step, urdf_path, json_path, log_path):
    print("register_env: ", grpc_address)
    gym.register(
        id=f"Openloong-v0-OrcaGym-{grpc_address[-2:]}",
        entry_point="envs.openloong.openloong_env:OpenLoongEnv",
        kwargs={'frame_skip': 1,   # 1 action per frame
                'reward_type': "dense",
                'action_space_type': ActionSpaceType.CONTINUOUS,
                'action_step_count': 0,
                'grpc_address': grpc_address, 
                'agent_names': [agent_name], 
                'time_step': time_step,
                'record_state': record_state,
                'record_file': record_file,
                'urdf_path': urdf_path,
                'json_path': json_path,
                'log_path': log_path,
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

        # 帧率为 60fps ，为显示为正常速度，每次渲染间隔 16ms
        elapsed_time = datetime.now() - start_time

        print(f"elapsed_time (ms): {elapsed_time.total_seconds() * 1000}")

        if elapsed_time.total_seconds() < time_step:
            time.sleep(time_step - elapsed_time.total_seconds())


if __name__ == '__main__':
    """
    The startup script for the openloong walking wbc control using joystick.
    """

    parser = argparse.ArgumentParser(description='Simulation Configuration')
    parser.add_argument('--grpc_address', type=str, required=True, help='The gRPC address for the simulation')
    args = parser.parse_args()

    grpc_address = f"{args.grpc_address}"

    simulation_frequency = 500
    time_step = 1.0 / simulation_frequency

    urdf_path = project_root + "/envs/openloong/external/openloong-dyn-control/models/AzureLoong.urdf"
    json_path = project_root + "/envs/openloong/external/openloong-dyn-control/common/joint_ctrl_config.json"
    log_path = project_root + "/envs/openloong/records/datalog.log"

    if not os.path.exists(project_root + "/envs/openloong/records"):
        os.makedirs(project_root + "/envs/openloong/records")

    print("simulation running... , grpc_address: ", grpc_address)
    env_id = f"Openloong-v0-OrcaGym-{grpc_address[-2:]}"

    register_env(grpc_address, RecordState.NONE, 'openloong_ctrl.h5', "AzureLoong", time_step, urdf_path, json_path, log_path)

    env = gym.make(env_id)        
    print("启动仿真环境")    

    try:
        run_simulation(env, time_step)
    except KeyboardInterrupt:
        print("关闭仿真环境")        
        env.save_record()
        env.close()