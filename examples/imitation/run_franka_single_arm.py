import os
import sys
import time

current_file_path = os.path.abspath('')
project_root = os.path.dirname(os.path.dirname(current_file_path))

if project_root not in sys.path:
    sys.path.append(project_root)


import gymnasium as gym
from gymnasium.envs.registration import register
from datetime import datetime
from orca_gym.environment.orca_gym_env import RewardType
from orca_gym.robomimic.robomimic_env import RunMode, ControlDevice
from orca_gym.robomimic.dataset_util import DatasetWriter, DatasetReader
from orca_gym.sensor.rgbd_camera import Monitor
from envs.imitation.franka_env import FrankaEnv
from examples.imitation.train_bc_rnn import run_train_bc_rnn
from orca_gym.utils.dir_utils import create_tmp_dir

import numpy as np
import argparse

def register_env(orcagym_addr, env_name, env_index, agent_name, run_mode : str, ctrl_device : str, max_episode_steps) -> str:
    orcagym_addr_str = orcagym_addr.replace(":", "-")
    env_id = env_name + "-OrcaGym-" + orcagym_addr_str + f"-{env_index:03d}"
    agent_names = [f"{agent_name}"]
    kwargs = {'frame_skip': 1,   
                'reward_type': RewardType.SPARSE,
                'orcagym_addr': orcagym_addr, 
                'agent_names': agent_names, 
                'time_step': TIME_STEP,
                'run_mode': run_mode,
                'ctrl_device': ctrl_device,
                'control_freq': 20}           
    gym.register(
        id=env_id,
        entry_point="envs.imitation.franka_env:FrankaEnv",
        kwargs=kwargs,
        max_episode_steps= max_episode_steps,  # 10 seconds
        reward_threshold=0.0,
    )
    return env_id, kwargs

def run_episode(env : FrankaEnv, dataset_writer):
    obs, info = env.reset(seed=42)
    obs_list = {obs_key: list([]) for obs_key, obs_data in obs.items()}
    reward_list = []
    done_list = []
    info_list = []    
    terminated_times = 0
    while True:
        start_time = datetime.now()

        action = env.action_space.sample()
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
        if elapsed_time.total_seconds() < TIME_STEP:
            time.sleep(TIME_STEP - elapsed_time.total_seconds())
        else:
            print("Over time! elapsed_time (ms): ", elapsed_time.total_seconds() * 1000)

def user_comfirm_save_record(task_result):
    while True:
        user_input = input(f"Task is {task_result}! Do you want to save the record? y(save), n(ignore), e(ignore & exit): ")
        if user_input == 'y':
            return True, False
        elif user_input == 'n':
            return False, False
        elif user_input == 'e':
            return False, True
        else:
            print("Invalid input! Please input 'y', 'n' or 'e'.")

def do_teleoperation(env, dataset_writer):
    while True:
        obs_list, reward_list, done_list, info_list = run_episode(env, dataset_writer)
        task_result = "Success" if done_list[-1] == 1 else "Failed"
        save_record, exit_program = user_comfirm_save_record(task_result)
        if save_record:
            dataset_writer.add_demo({
                'states': np.array([np.concatenate([info["state"]["qpos"], info["state"]["qvel"]]) for info in info_list]),
                'actions': np.array([info["action"] for info in info_list]),
                'rewards': np.array(reward_list),
                'dones': np.array(done_list),
                'obs': obs_list
            })
        if exit_program:
            break

def playback_episode(env : FrankaEnv, action_list, done_list):
    for i in range(len(action_list)):
        start_time = datetime.now()

        action = action_list[i]
        done = done_list[i]
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()

        elapsed_time = datetime.now() - start_time
        if elapsed_time.total_seconds() < TIME_STEP:
            time.sleep(TIME_STEP - elapsed_time.total_seconds())

        if done:
            print("Episode done!")
            return

        if terminated or truncated:
            print("Episode terminated!")
            return
    
    print("Episode tunkated!")

def reset_playback_env(env : FrankaEnv, demo_data):
    obs, info = env.reset(seed=42)
    
    object_data = demo_data['obs']['object']
    obj_xpos = object_data[0][0:3]
    obj_xquat = object_data[0][3:7]
    # print("Resetting object position: ", obj_xpos, obj_xquat)
    env.unwrapped.replace_object(obj_xpos, obj_xquat)
    
def do_playback(env : FrankaEnv, dataset_reader : DatasetReader):
    demo_names = dataset_reader.get_demo_names()
    for demo_name in demo_names:
        demo_data = dataset_reader.get_demo_data(demo_name)
        action_list = demo_data['actions']
        done_list = demo_data['dones']
        print("Playing back episode: ", demo_name, " with ", len(action_list), " steps.")
        # for i, action in enumerate(action_list):
        #     print(f"Playback Action ({i}): ", action)
        reset_playback_env(env, demo_data)
        playback_episode(env, action_list, done_list)
        time.sleep(1)



def run_example(orcagym_addr : str, agent_name : str, record_file_path : str, run_mode : str, ctrl_device : str, max_episode_steps : int):
    try:
        print("simulation running... , orcagym_addr: ", orcagym_addr)
        if run_mode == RunMode.PLAYBACK:
            dataset_reader = DatasetReader(file_path=record_file_path)
            env_name = dataset_reader.get_env_name()
            env_index = 0
            env_id, kwargs = register_env(orcagym_addr, env_name, env_index, agent_name, run_mode, ctrl_device, max_episode_steps)
            print("Registered environment: ", env_id)

            env = gym.make(env_id)
            print("Starting simulation...")
            do_playback(env, dataset_reader)

        elif run_mode == RunMode.TELEOPERATION:
            env_name = "Franka-Teleoperation-v0"
            env_index = 0
            env_id, kwargs = register_env(orcagym_addr, env_name, env_index, agent_name, run_mode, ctrl_device, max_episode_steps)
            print("Registered environment: ", env_id)

            env = gym.make(env_id)        
            print("Starting simulation...")
            kwargs["run_mode"] = RunMode.IMITATION  # 此处用于训练的时候读取
            dataset_writer = DatasetWriter(file_path=record_file_path,
                                        env_name=env_id,
                                        env_version=env.unwrapped.get_env_version(),
                                        env_kwargs=kwargs)

            do_teleoperation(env, dataset_writer)
            dataset_writer.shuffle_demos()
            dataset_writer.finalize()
            
        elif run_mode == RunMode.IMITATION:
            dataset_reader = DatasetReader(file_path=record_file_path)
            env_name = dataset_reader.get_env_name()
            env_name = env_name.split("-OrcaGym-")[0]
            env_index = 0
            env_id, kwargs = register_env(orcagym_addr, env_name, env_index, agent_name, run_mode, ctrl_device, max_episode_steps)
            print("Registered environment: ", env_id)

            env = gym.make(env_id)
            print("Starting simulation...")
            
            now = datetime.now()
            formatted_now = now.strftime("%Y-%m-%d_%H-%M-%S")
            output_dir = f"{current_file_path}/trained_models_tmp/train_temp_dir_{formatted_now}"
            run_train_bc_rnn(dataset_type="orca_gym", dataset=record_file_path, output=output_dir, debug=False)
        else:
            print("Invalid run mode! Please input 'teleoperation' or 'playback'.")

    except KeyboardInterrupt:
        print("Simulation stopped")        
        dataset_writer.finalize()
        env.close()
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run multiple instances of the script with different gRPC addresses.')
    parser.add_argument('--orcagym_address', type=str, default='localhost:50051', help='The gRPC addresses to connect to')
    parser.add_argument('--agent_name', type=str, default='panda_mocap_moto_usda', help='The agent name to control')
    parser.add_argument('--record_time', type=int, default=20, help='The time to record the teleoperation in 1 episode')
    parser.add_argument('--record_file_path', type=str, help='The file path to save the record')
    parser.add_argument('--run_mode', type=str, default='teleoperation', help='The run mode of the environment (teleoperation or imitation or playback)')
    parser.add_argument('--ctrl_device', type=str, default='xbox', help='The control device to use (xbox or keyboard)')
    args = parser.parse_args()
    
    orcagym_addr = args.orcagym_address
    agent_name = args.agent_name
    record_time = args.record_time
    record_file_path = args.record_file_path
    
    create_tmp_dir("records_tmp")
    create_tmp_dir("trained_models_tmp")
    
    if args.run_mode == 'teleoperation':
        run_mode = RunMode.TELEOPERATION
        if record_file_path is None:
            now = datetime.now()
            formatted_now = now.strftime("%Y-%m-%d_%H-%M-%S")
            record_file_path = f"./records_tmp/franka_singel_arm_{formatted_now}.hdf5"
    elif args.run_mode == 'imitation':
        run_mode = RunMode.IMITATION
    elif args.run_mode == 'playback':
        run_mode = RunMode.PLAYBACK
    else:
        print("Invalid run mode! Please input 'teleoperation', 'playback' or 'imitation'.")
        sys.exit(1)

    if record_file_path is None:
        print("Please input the record file path.")
        sys.exit(1)

    if args.ctrl_device == 'xbox':
        ctrl_device = ControlDevice.XBOX
    elif args.ctrl_device == 'keyboard':
        ctrl_device = ControlDevice.KEYBOARD
    else:
        print("Invalid control device! Please input 'xbox' or 'keyboard'.")
        sys.exit(1)


    TIME_STEP = 0.01
    max_episode_steps = int(record_time / TIME_STEP)

    run_example(orcagym_addr, agent_name, record_file_path, run_mode, ctrl_device, max_episode_steps)
