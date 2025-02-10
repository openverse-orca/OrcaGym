import os
import sys
import time
import subprocess
import signal

current_file_path = os.path.abspath('')
project_root = os.path.dirname(os.path.dirname(current_file_path))

# if project_root not in sys.path:
#     sys.path.append(project_root)


import gymnasium as gym
from gymnasium.envs.registration import register
from datetime import datetime
from orca_gym.environment.orca_gym_env import RewardType
from orca_gym.robomimic.dataset_util import DatasetWriter, DatasetReader
from orca_gym.sensor.rgbd_camera import Monitor
from envs.franka.franka_env import FrankaEnv, RunMode, ControlDevice
from examples.imitation.train_policy import train_policy
from examples.imitation.test_policy import create_env, rollout
from orca_gym.utils.dir_utils import create_tmp_dir
from robomimic.utils.file_utils import maybe_dict_from_checkpoint
import orca_gym.utils.rotations as rotations

import numpy as np
import argparse

from transformers import AutoModelForVision2Seq, AutoProcessor
from PIL import Image

import torch

from orca_gym.sensor.rgbd_camera import CameraWrapper
import cv2

import matplotlib.pyplot as plt
import matplotlib
import matplotlib.animation as animation

ENV_ENTRY_POINT = {
    "Franka": "envs.franka.franka_env:FrankaEnv"
}

TIME_STEP = 0.005
FRAME_SKIP = 8
REALTIME_STEP = TIME_STEP * FRAME_SKIP
CONTROL_FREQ = 1 / REALTIME_STEP
RECORD_TIME = 20

def register_env(orcagym_addr, env_name, env_index, agent_name, run_mode : str, task : str, ctrl_device : str, max_episode_steps) -> str:
    orcagym_addr_str = orcagym_addr.replace(":", "-")
    env_id = env_name + "-OrcaGym-" + orcagym_addr_str + f"-{env_index:03d}"
    agent_names = [f"{agent_name}"]
    kwargs = {'frame_skip': FRAME_SKIP,   
                'reward_type': RewardType.SPARSE,
                'orcagym_addr': orcagym_addr, 
                'agent_names': agent_names, 
                'time_step': TIME_STEP,
                'run_mode': run_mode,
                'task': task,
                'ctrl_device': ctrl_device,
                'control_freq': CONTROL_FREQ}           
    gym.register(
        id=env_id,
        entry_point=ENV_ENTRY_POINT[env_name],
        kwargs=kwargs,
        max_episode_steps= max_episode_steps,
        reward_threshold=0.0,
    )
    return env_id, kwargs

def run_episode(env : FrankaEnv, processor, vla, camera_arm : CameraWrapper):
    obs, info = env.reset(seed=42)
    env.render()
    obs_list = {obs_key: list([]) for obs_key, obs_data in obs.items()}
    reward_list = []
    done_list = []
    info_list = []    
    terminated_times = 0
    INSTRUCTION = 'pick up brown box'

    while True:
        start_time = datetime.now()

        # Grab image input & format prompt

        frame = camera_arm.get_frame(format='rgb24')
        print("frame shape = ", frame.shape, "frame type = ", frame.dtype)
        
        image: Image.Image = Image.fromarray(frame)
        # image = Image.fromarray(np.asarray(np.random.rand(256, 256, 3) * 255, dtype=np.uint8))
        
        # image.show()
        prompt = f"In: What action should the robot take to {INSTRUCTION}?\nOut:"
        # prompt = "xxxxx"
        
        print("The prompt is: ", prompt)

        # Predict Action (7-DoF; un-normalize for BridgeData V2)
        inputs = processor(prompt, image).to("cuda:0", dtype=torch.bfloat16)
        action = vla.predict_action(**inputs, unnorm_key="libero_object", do_sample=False)

        # Execute...
        # robot.act(action, ...)    

        # action = env.action_space.sample()
        gripper_width = action[6]
        action = np.concatenate([action[:7], [gripper_width]]).flatten()
        
        action[:3] *= 0.1
        
        action[0] += 0.6
        action[2] += 0.1
        
        # Rotate around x axis for 180 degrees
        action[3] += np.pi
        
        
        # print("action = ", action)
        formatted_action = ['{:.2f}'.format(num) for num in action]
        print(formatted_action)
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        env.render()
        
        for obs_key, obs_data in obs.items():
            obs_list[obs_key].append(obs_data)
            
        reward_list.append(reward)
        done_list.append(0 if not terminated else 1)
        info_list.append(info)
        terminated_times = terminated_times + 1 if terminated else 0

        if terminated_times >= 5 or truncated:
            return obs_list, reward_list, done_list, info_list

        elapsed_time = datetime.now() - start_time
        if elapsed_time.total_seconds() < REALTIME_STEP:
            time.sleep(REALTIME_STEP - elapsed_time.total_seconds())
            

def run_policy(env):    
    # Load Processor & VLA
    # processor = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)
    processor = AutoProcessor.from_pretrained("openvla/openvla-7b-finetuned-libero-object", trust_remote_code=True)
    
    vla = AutoModelForVision2Seq.from_pretrained(
        "openvla/openvla-7b-finetuned-libero-object", 
        attn_implementation="flash_attention_2",  # [Optional] Requires `flash_attn`
        torch_dtype=torch.bfloat16, 
        low_cpu_mem_usage=True, 
        trust_remote_code=True
    ).to("cuda:0")
    
    camera_arm = CameraWrapper("camera_arm", 7080)
    camera_arm.start()
    
    current_round = 1
    success_count = 0
    while True:
        obs_list, reward_list, done_list, info_list = run_episode(env, processor=processor, vla=vla, camera_arm=camera_arm)
        success_count += 1 if done_list[-1] == 1 else 0
        print(f"Round {current_round}: success = {True if done_list[-1] == 1 else False}, total success = {success_count}")


def run_example(orcagym_addr : str, 
                agent_name : str, 
                task : str,
                max_episode_steps : int, 
                ckpt_path : str,
                ):
    try:
        print("simulation running... , orcagym_addr: ", orcagym_addr)

        env_name = "Franka"
        env_index = 0
        env_id, kwargs = register_env(orcagym_addr, env_name, env_index, agent_name, RunMode.POLICY_RAW, task, ControlDevice.XBOX, max_episode_steps)
        print("Registered environment: ", env_id)

        env = gym.make(env_id)        
        print("Starting simulation...")

        run_policy(env)


    except KeyboardInterrupt:
        print("Simulation stopped")        
        env.close()
    

def start_monitor():
    """
    启动 monitor.py 作为子进程。
    """
    # 获取当前脚本所在的目录
    current_file_path = os.path.abspath('')
    project_root = os.path.dirname(os.path.dirname(current_file_path))    
    monitor_script = f"{project_root}/orca_gym/scripts/camera_monitor.py"




    print("Monitor 脚本路径: ", monitor_script)    

    # 启动 monitor.py
    # 使用 sys.executable 确保使用相同的 Python 解释器
    process = subprocess.Popen(
        [sys.executable, monitor_script, "--port", "7080"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    return process

def terminate_monitor(process):
    """
    终止子进程。
    """
    try:
        if os.name != 'nt':
            # Unix/Linux: 发送 SIGTERM 给整个进程组
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
        else:
            # Windows: 使用 terminate 方法
            process.terminate()
    except Exception as e:
        print(f"终止子进程时发生错误: {e}")
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run multiple instances of the script with different gRPC addresses.')
    parser.add_argument('--orcagym_address', type=str, default='localhost:50051', help='The gRPC addresses to connect to')
    parser.add_argument('--agent_name', type=str, default='panda_mocap_moto_usda', help='The agent name to control')
    parser.add_argument('--task', type=str, default='lift', help='The task to do in the environment (pick_and_place / push / lift)')
    parser.add_argument('--model_file', type=str, help='The model file to load for rollout the policy')
    
    args = parser.parse_args()
    
    orcagym_addr = args.orcagym_address
    agent_name = args.agent_name
    task = args.task
    ckpt_path = args.model_file
        

    max_episode_steps = int(RECORD_TIME / REALTIME_STEP)
    print(f"Run episode in {max_episode_steps} steps as {RECORD_TIME} seconds.")

    # 启动 Monitor 子进程
    monitor_process = start_monitor()
    print(f"Monitor 进程已启动，PID: {monitor_process.pid}")

    run_example(orcagym_addr, 
                agent_name, 
                task,
                max_episode_steps, 
                ckpt_path,
                )

    # 终止 Monitor 子进程
    terminate_monitor(monitor_process)