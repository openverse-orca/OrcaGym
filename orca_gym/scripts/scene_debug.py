import os
import sys
import time
import subprocess
import signal


# if project_root not in sys.path:
#     sys.path.append(project_root)

from typing import Any, Dict
import gymnasium as gym
from gymnasium.envs.registration import register
from datetime import datetime
from orca_gym.environment.orca_gym_env import RewardType
from orca_gym.sensor.rgbd_camera import Monitor, CameraWrapper
from envs.manipulation.gripper_env import GripperEnv, ControlDevice
import orca_gym.utils.rotations as rotations

import numpy as np
import argparse

import os
import sys
import time
import subprocess
import signal

import numpy as np
import camera_monitor


ENV_ENTRY_POINT = {
    "Gripper": "envs.manipulation.gripper_env:GripperEnv"
}

TIME_STEP = 0.001
FRAME_SKIP = 20
REALTIME_STEP = TIME_STEP * FRAME_SKIP
CONTROL_FREQ = 1 / REALTIME_STEP

def register_env(orcagym_addr : str, 
                 env_name : str, 
                 env_index : int, 
                 agent_name : str, 
                 ctrl_device : str, 
                 max_episode_steps : int) -> str:
    orcagym_addr_str = orcagym_addr.replace(":", "-")
    env_id = env_name + "-OrcaGym-" + orcagym_addr_str + f"-{env_index:03d}"
    agent_names = [f"{agent_name}"]
    kwargs = {'frame_skip': FRAME_SKIP,   
                'reward_type': RewardType.SPARSE,
                'orcagym_addr': orcagym_addr, 
                'agent_names': agent_names, 
                'time_step': TIME_STEP,
                'ctrl_device': ctrl_device}           
    gym.register(
        id=env_id,
        entry_point=ENV_ENTRY_POINT[env_name],
        kwargs=kwargs,
        max_episode_steps= max_episode_steps,
        reward_threshold=0.0,
    )
    return env_id, kwargs



def do_teleoperation(env : GripperEnv):    
    obs = env.reset()
    while True:
        start_time = datetime.now()

        action = env.action_space.sample()
  
        obs, reward, terminated, truncated, info = env.step(action)
        
        env.render()

        elapsed_time = datetime.now() - start_time
        if elapsed_time.total_seconds() < REALTIME_STEP:
            time.sleep(REALTIME_STEP - elapsed_time.total_seconds())
    


def run_example(orcagym_addr : str, 
                agent_name : str, 
                ctrl_device : str):
    try:
        print("simulation running... , orcagym_addr: ", orcagym_addr)

        env_name = "Gripper"
        env_index = 0
        env_id, kwargs = register_env(orcagym_addr, 
                                      env_name, 
                                      env_index, 
                                      agent_name, 
                                      ctrl_device, 
                                      sys.maxsize)
        print("Registered environment: ", env_id)

        env = gym.make(env_id)        
        print("Starting simulation...")

        do_teleoperation(env)


    except KeyboardInterrupt:
        print("Simulation stopped")        
        env.close()
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run multiple instances of the script with different gRPC addresses.')
    parser.add_argument('--orcagym_address', type=str, default='localhost:50051', help='The gRPC addresses to connect to')
    parser.add_argument('--agent_name', type=str, default='gripper_mocap_usda', help='The agent name to control')
    parser.add_argument('--ctrl_device', type=str, default='xbox', help='The control device to use (xbox or keyboard)')

    
    args = parser.parse_args()
    
    orcagym_addr = args.orcagym_address
    agent_name = args.agent_name
    
    if args.ctrl_device == 'xbox':
        ctrl_device = ControlDevice.XBOX
    elif args.ctrl_device == 'keyboard':
        ctrl_device = ControlDevice.KEYBOARD
    else:
        print("Invalid control device! Please input 'xbox' or 'keyboard'.")
        sys.exit(1)

    # 启动 Monitor 子进程
    ports = [7090]
    monitor_processes = []
    for port in ports:
        process = camera_monitor.start_monitor(port=port)
        monitor_processes.append(process)

    run_example(orcagym_addr, 
                agent_name, 
                ctrl_device)

    # 终止 Monitor 子进程
    for process in monitor_processes:
        camera_monitor.terminate_monitor(process)