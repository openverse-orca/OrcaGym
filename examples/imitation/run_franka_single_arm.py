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
import orca_gym.scripts.franka_manipulation as franka_manipulation

import numpy as np
import argparse


RGB_SIZE = (128, 128)

def run_example(orcagym_addr : str, 
                agent_name : str, 
                record_path : str, 
                run_mode : str, 
                task : str,
                algo_config : str,
                ctrl_device : str, 
                max_episode_steps : int, 
                playback_mode : str,
                rollout_times : int,
                ckpt_path : str, 
                augmented_sacle : float,
                augmented_times : int,
                teleoperation_rounds : int,
                sample_range : float):
    try:
        print("simulation running... , orcagym_addr: ", orcagym_addr)
        if run_mode == "playback":
            dataset_reader = DatasetReader(file_path=record_path)
            task = dataset_reader.get_env_kwargs()["task"]            
            env_name = dataset_reader.get_env_name()
            env_name = env_name.split("-OrcaGym-")[0]
            env_index = 0
            env_id, kwargs = franka_manipulation.register_env(orcagym_addr, env_name, env_index, agent_name, RunMode.POLICY_NORMALIZED, task, ctrl_device, max_episode_steps, sample_range)
            print("Registered environment: ", env_id)

            env = gym.make(env_id)
            print("Starting simulation...")
            franka_manipulation.do_playback(env, dataset_reader, playback_mode)

        elif run_mode == "teleoperation":
            env_name = "Franka"
            env_index = 0
            env_id, kwargs = franka_manipulation.register_env(orcagym_addr, env_name, env_index, agent_name, RunMode.TELEOPERATION, task, ctrl_device, max_episode_steps, sample_range)
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
                       CameraWrapper(name="camera_wrist", port=7070)]

            franka_manipulation.do_teleoperation(env, dataset_writer, teleoperation_rounds, cameras=cameras, rgb_size=RGB_SIZE)
            dataset_writer.shuffle_demos()
            dataset_writer.finalize()
            
        elif run_mode == "imitation":
            dataset_reader = DatasetReader(file_path=record_path)
            env_name = dataset_reader.get_env_name()
            task = dataset_reader.get_env_kwargs()["task"]
            env_name = env_name.split("-OrcaGym-")[0]
            env_index = 0
            env_id, kwargs = franka_manipulation.register_env(orcagym_addr, env_name, env_index, agent_name, RunMode.POLICY_NORMALIZED, task, ctrl_device, max_episode_steps, sample_range)
            print("Registered environment: ", env_id)

            env = gym.make(env_id)
            print("Starting simulation...")
            
            now = datetime.now()
            formatted_now = now.strftime("%Y-%m-%d_%H-%M-%S")
            output_dir = f"{current_file_path}/trained_models_tmp/train_temp_dir_{formatted_now}"
            train_policy(config=algo_config, algo=None, dataset=record_path, name=None, output_dir=output_dir, debug=False)
            
        elif run_mode == "rollout":
            ckpt_dict = maybe_dict_from_checkpoint(ckpt_path=ckpt_path)

            # metadata from model dict to get info needed to create environment
            env_meta = ckpt_dict["env_metadata"]    
            env_name = env_meta["env_name"]
            env_name = env_name.split("-OrcaGym-")[0]
            env_index = 0
            
            env_kwargs = env_meta["env_kwargs"]
            task = env_kwargs["task"]  
            
            env_id, kwargs = franka_manipulation.register_env(orcagym_addr, env_name, env_index, agent_name, RunMode.POLICY_NORMALIZED, task, ctrl_device, max_episode_steps, sample_range)
            print("Registered environment: ", env_id)
            
            env, policy = create_env(ckpt_path)

            for i in range(rollout_times):
                stats = rollout(
                    policy=policy, 
                    env=env, 
                    horizon=max_episode_steps, 
                    render=True, 
                    video_writer=None, 
                    video_skip=5, 
                    camera_names=["agentview"],
                    realtime_step=franka_manipulation.REALTIME_STEP
                )
                print(stats)
        elif run_mode == "augmentation":
            dataset_reader = DatasetReader(file_path=record_path)
            env_name = dataset_reader.get_env_name()
            task = dataset_reader.get_env_kwargs()["task"]
            env_name = env_name.split("-OrcaGym-")[0]
            env_index = 0
            env_id, kwargs = franka_manipulation.register_env(orcagym_addr, env_name, env_index, agent_name, RunMode.POLICY_NORMALIZED, task, ctrl_device, max_episode_steps, sample_range)
            print("Registered environment: ", env_id)

            env = gym.make(env_id)
            print("Starting simulation...")
            
            now = datetime.now()
            formatted_now = now.strftime("%Y-%m-%d_%H-%M-%S")
            agumented_dataset_file_path = f"{current_file_path}/augmented_datasets_tmp/augmented_dataset_{formatted_now}.hdf5"
            franka_manipulation.do_augmentation(env, record_path, agumented_dataset_file_path, augmented_sacle, augmented_times)
            print("Augmentation done! The augmented dataset is saved to: ", agumented_dataset_file_path)
        else:
            print("Invalid run mode! Please input 'teleoperation' or 'playback'.")

    except KeyboardInterrupt:
        print("Simulation stopped")        
        if run_mode == "teleoperation":
            dataset_writer.finalize()
        env.close()
    
        
def _get_algo_config(algo_name):
    if algo_name == "bc":
        return ["config/bc.json"]
    elif algo_name == "td3_bc":
        return ["config/td3_bc.json"]
    elif algo_name == "cql":
        return ["config/cql.json"]
    elif algo_name == "iris":
        return ["config/iris.json"]
    elif algo_name == "hbc":
        return ["config/hbc.json"]
    elif algo_name == "bcq":
        return ["config/bcq.json"]
    elif algo_name == "iql":
        return ["config/iql.json"]
    elif algo_name == "all":
        return ["config/bc.json", 
                "config/td3_bc.json", 
                "config/cql.json", 
                "config/iris.json", 
                "config/hbc.json", 
                "config/bcq.json", 
                "config/iql.json"]
    else:
        raise ValueError(f"Invalid algorithm name: {algo_name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run multiple instances of the script with different gRPC addresses.')
    parser.add_argument('--orcagym_address', type=str, default='localhost:50051', help='The gRPC addresses to connect to')
    parser.add_argument('--agent_name', type=str, default='panda_mocap_moto_usda', help='The agent name to control')
    parser.add_argument('--run_mode', type=str, default='teleoperation', help='The run mode of the environment (teleoperation / playback / imitation / rollout / augmentation)')
    parser.add_argument('--task', type=str, help='The task to do in the environment (pick_and_place / push / lift)')
    parser.add_argument('--algo', type=str, default='bc', help='The algorithm to use for training the policy')
    parser.add_argument('--dataset', type=str, help='The file path to save the record')
    parser.add_argument('--model_file', type=str, help='The model file to load for rollout the policy')
    parser.add_argument('--record_length', type=int, default=20, help='The time length in seconds to record the teleoperation in 1 episode')
    parser.add_argument('--ctrl_device', type=str, default='xbox', help='The control device to use (xbox or keyboard)')
    parser.add_argument('--playback_mode', type=str, default='random', help='The playback mode of the environment (loop or random)')
    parser.add_argument('--rollout_times', type=int, default=10, help='The times to rollout the policy')
    parser.add_argument('--augmented_sacle', type=float, default=0.01, help='The scale to augment the dataset')
    parser.add_argument('--augmented_times', type=int, default=2, help='The times to augment the dataset')
    parser.add_argument('--teleoperation_rounds', type=int, default=20, help='The rounds to do teleoperation')
    parser.add_argument('--sample_range', type=float, default=0.2, help='The area range to sample the object and goal position')
    
    args = parser.parse_args()
    
    orcagym_addr = args.orcagym_address
    agent_name = args.agent_name
    record_time = args.record_length
    record_path = args.dataset
    playback_mode = args.playback_mode
    run_mode = args.run_mode
    task = args.task
    algo = args.algo
    rollout_times = args.rollout_times
    ckpt_path = args.model_file
    augmented_sacle = args.augmented_sacle
    augmented_times = args.augmented_times
    teleoperation_rounds = args.teleoperation_rounds
    sample_range = args.sample_range
    
    assert record_time > 0, "The record time should be greater than 0."
    assert teleoperation_rounds > 0, "The teleoperation rounds should be greater than 0."
    assert sample_range >= 0.0, "The sample range should be greater than or equal to 0."
    assert augmented_sacle >= 0.0, "The augmented scale should be greater than or equal to 0."
    assert augmented_times > 0, "The augmented times should be greater than 0."
        
    create_tmp_dir("records_tmp")
    create_tmp_dir("trained_models_tmp")
    create_tmp_dir("augmented_datasets_tmp")
    
    algo_config = _get_algo_config(algo) if run_mode == "imitation" else ["none_algorithm"]
    
    if run_mode == "teleoperation":
        if task is None:
            task = "lift"
            
        if record_path is None:
            now = datetime.now()
            formatted_now = now.strftime("%Y-%m-%d_%H-%M-%S")
            record_path = f"./records_tmp/Franka_{task}_{formatted_now}.hdf5"
    if run_mode == "imitation" or run_mode == "playback" or run_mode == "augmentation":
        if record_path is None:
            print("Please input the record file path.")
            sys.exit(1)
    if run_mode == "rollout":
        if ckpt_path is None:
            print("Please input the model file path.")
            sys.exit(1) 
    if run_mode not in ["teleoperation", "playback", "imitation", "rollout", "augmentation"]:
        print("Invalid run mode! Please input 'teleoperation', 'playback', 'imitation', 'rollout' or 'augmentation'.")
        sys.exit(1)

    if args.ctrl_device == 'xbox':
        ctrl_device = ControlDevice.XBOX
    elif args.ctrl_device == 'keyboard':
        ctrl_device = ControlDevice.KEYBOARD
    else:
        print("Invalid control device! Please input 'xbox' or 'keyboard'.")
        sys.exit(1)

    max_episode_steps = int(record_time / franka_manipulation.REALTIME_STEP)
    print(f"Run episode in {max_episode_steps} steps as {record_time} seconds.")

    # 启动 Monitor 子进程
    ports = [7070, 7080, 7090]
    monitor_processes = []
    for port in ports:
        process = franka_manipulation.start_monitor(port=port, project_root=project_root)
        monitor_processes.append(process)

    for config in algo_config:
        run_example(orcagym_addr, 
                    agent_name, 
                    record_path, 
                    run_mode, 
                    task,
                    config,
                    ctrl_device, 
                    max_episode_steps, 
                    playback_mode, 
                    rollout_times, 
                    ckpt_path, 
                    augmented_sacle,
                    augmented_times,
                    teleoperation_rounds,
                    sample_range)

    # 终止 Monitor 子进程
    for process in monitor_processes:
        franka_manipulation.terminate_monitor(process)