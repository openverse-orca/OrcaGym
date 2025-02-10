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
from orca_gym.sensor.rgbd_camera import Monitor, CameraWrapper
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

import orca_gym.scripts.franka_manipulation as franka_manipulation

RGB_SIZE = (256, 256)

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
        if elapsed_time.total_seconds() < franka_manipulation.REALTIME_STEP:
            time.sleep(franka_manipulation.REALTIME_STEP - elapsed_time.total_seconds())
            

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
                run_mode : str,
                task : str,
                max_episode_steps : int, 
                ckpt_path : str,
                teleoperation_rounds : int,
                sample_range : float):
    try:
        print("simulation running... , orcagym_addr: ", orcagym_addr)
        if run_mode == "rollout":        
            env_name = "Franka"
            env_index = 0
            env_id, kwargs = franka_manipulation.register_env(orcagym_addr, env_name, env_index, agent_name, RunMode.POLICY_RAW, task, ControlDevice.XBOX, max_episode_steps, sample_range)
            print("Registered environment: ", env_id)

            env = gym.make(env_id)        
            print("Starting simulation...")

            run_policy(env)
            
        elif run_mode == "teleoperation":
            env_name = "Franka"
            env_index = 0
            env_id, kwargs = franka_manipulation.register_env(orcagym_addr, env_name, env_index, agent_name, RunMode.TELEOPERATION, task, ControlDevice.XBOX, max_episode_steps, sample_range)
            print("Registered environment: ", env_id)

            env = gym.make(env_id)        
            print("Starting simulation...")
            kwargs["run_mode"] = RunMode.POLICY_NORMALIZED  # 此处用于训练的时候读取
            dataset_writer = DatasetWriter(file_path=record_path,
                                        env_name=env_id,
                                        env_version=env.unwrapped.get_env_version(),
                                        env_kwargs=kwargs)

            cameras = [CameraWrapper(name="camera_primary", port=7090),
                    #    CameraWrapper(name="camera_secondary", port=7080),
                    #    CameraWrapper(name="camera_wrist", port=7070),
                       ]

            franka_manipulation.do_teleoperation(env, dataset_writer, teleoperation_rounds, 
                                                 cameras=cameras, rgb_size=RGB_SIZE, action_step=5,
                                                 language_instruction="pick up brown box, lift it up for 10cm.")
            dataset_writer.shuffle_demos()
            dataset_writer.finalize()


    except KeyboardInterrupt:
        print("Simulation stopped")        
        env.close()
    

        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run multiple instances of the script with different gRPC addresses.')
    parser.add_argument('--orcagym_address', type=str, default='localhost:50051', help='The gRPC addresses to connect to')
    parser.add_argument('--agent_name', type=str, default='panda_mocap_moto_usda', help='The agent name to control')
    parser.add_argument('--task', type=str, default='lift', help='The task to do in the environment (pick_and_place / push / lift)')
    parser.add_argument('--model_file', type=str, help='The model file to load for rollout the policy')
    parser.add_argument('--record_length', type=int, default=20, help='The time length in seconds to record the teleoperation in 1 episode')
    parser.add_argument('--teleoperation_rounds', type=int, default=20, help='The rounds to do teleoperation')
    parser.add_argument('--sample_range', type=float, default=0.1, help='The area range to sample the object and goal position')
    parser.add_argument('--run_mode', type=str, default='teleoperation', help='The run mode of the environment (teleoperation / playback / rollout / augmentation)')
    parser.add_argument('--dataset', type=str, help='The file path to save the record')
    parser.add_argument('--augmented_sacle', type=float, default=0.01, help='The scale to augment the dataset')
    parser.add_argument('--augmented_times', type=int, default=2, help='The times to augment the dataset')
    
    args = parser.parse_args()
    
    orcagym_addr = args.orcagym_address
    agent_name = args.agent_name
    task = args.task
    ckpt_path = args.model_file
    record_time = args.record_length        
    teleoperation_rounds = args.teleoperation_rounds
    sample_range = args.sample_range
    run_mode = args.run_mode
    record_path = args.dataset
    augmented_sacle = args.augmented_sacle
    augmented_times = args.augmented_times    

    assert record_time > 0, "The record time should be greater than 0."
    assert teleoperation_rounds > 0, "The teleoperation rounds should be greater than 0."
    assert sample_range >= 0.0, "The sample range should be greater than or equal to 0."
    assert augmented_sacle >= 0.0, "The augmented scale should be greater than or equal to 0."
    assert augmented_times > 0, "The augmented times should be greater than 0."
        
    create_tmp_dir("records_tmp")
    create_tmp_dir("augmented_datasets_tmp")

    if run_mode == "teleoperation":
        if task is None:
            task = "lift"
            
        if record_path is None:
            now = datetime.now()
            formatted_now = now.strftime("%Y-%m-%d_%H-%M-%S")
            record_path = f"./records_tmp/Franka_{task}_{formatted_now}.hdf5"
    if run_mode == "playback" or run_mode == "augmentation":
        if record_path is None:
            print("Please input the record file path.")
            sys.exit(1)
    if run_mode == "rollout":
        if ckpt_path is None:
            print("Please input the model file path.")
            sys.exit(1) 
    if run_mode not in ["teleoperation", "playback", "rollout", "augmentation"]:
        print("Invalid run mode! Please input 'teleoperation', 'playback', 'rollout' or 'augmentation'.")
        sys.exit(1)
        
    max_episode_steps = int(record_time / franka_manipulation.REALTIME_STEP)
    print(f"Run episode in {max_episode_steps} steps as {record_time} seconds.")

    # 启动 Monitor 子进程
    ports = [7070, 7080, 7090]
    monitor_processes = []
    for port in ports:
        process = franka_manipulation.start_monitor(port=port, project_root=project_root)
        monitor_processes.append(process)

    run_example(orcagym_addr, 
                agent_name, 
                run_mode,
                task,
                max_episode_steps, 
                ckpt_path,
                teleoperation_rounds,
                sample_range)

    # 终止 Monitor 子进程
    for process in monitor_processes:
        franka_manipulation.terminate_monitor(process)