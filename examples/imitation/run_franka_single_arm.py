import os
import sys
import time
import subprocess
import signal

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
from examples.imitation.test_policy import create_env, rollout
from orca_gym.utils.dir_utils import create_tmp_dir
from robomimic.utils.file_utils import maybe_dict_from_checkpoint
import orca_gym.utils.rotations as rotations

import numpy as np
import argparse

ENV_ENTRY_POINT = {
    "Franka": "envs.imitation.franka_env:FrankaEnv"
}

TIME_STEP = 0.005
FRAME_SKIP = 4
REALTIME_STEP = TIME_STEP * FRAME_SKIP
CONTROL_FREQ = 1 / REALTIME_STEP

def register_env(orcagym_addr, env_name, env_index, agent_name, run_mode : str, ctrl_device : str, max_episode_steps) -> str:
    orcagym_addr_str = orcagym_addr.replace(":", "-")
    env_id = env_name + "-OrcaGym-" + orcagym_addr_str + f"-{env_index:03d}"
    agent_names = [f"{agent_name}"]
    kwargs = {'frame_skip': FRAME_SKIP,   
                'reward_type': RewardType.SPARSE,
                'orcagym_addr': orcagym_addr, 
                'agent_names': agent_names, 
                'time_step': TIME_STEP,
                'run_mode': run_mode,
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

def run_episode(env : FrankaEnv):
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
        if elapsed_time.total_seconds() < REALTIME_STEP:
            time.sleep(REALTIME_STEP - elapsed_time.total_seconds())
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

def do_teleoperation(env, dataset_writer : DatasetWriter):    
    while True:
        obs_list, reward_list, done_list, info_list = run_episode(env)
        task_result = "Success" if done_list[-1] == 1 else "Failed"
        save_record, exit_program = user_comfirm_save_record(task_result)
        if save_record:
            dataset_writer.add_demo_data({
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
        if elapsed_time.total_seconds() < REALTIME_STEP:
            time.sleep(REALTIME_STEP - elapsed_time.total_seconds())

        if done:
            print("Episode done!")
            return

        if terminated or truncated:
            print("Episode terminated!")
            return
    
    print("Episode tunkated!")

def reset_playback_env(env : FrankaEnv, demo_data, noise_scale=0.0):
    obs, info = env.reset(seed=42)
    
    object_data = demo_data['obs']['object']
    obj_xpos = object_data[0][0:3]
    obj_xquat = object_data[0][3:7]
    
    goal_data = demo_data['obs']['goal']
    goal_xpos = goal_data[0][0:3]
    goal_xquat = goal_data[0][3:7]
    
    if noise_scale > 0.0:
        obj_xpos += np.random.normal(0, noise_scale, len(obj_xpos))
        obj_euler = rotations.quat2euler(obj_xquat)
        obj_euler += np.random.normal(0, noise_scale, len(obj_euler))
        obj_xquat = rotations.euler2quat(obj_euler)
        
        goal_xpos += np.random.normal(0, noise_scale, len(goal_xpos))
        goal_euler = rotations.quat2euler(goal_xquat)
        goal_euler += np.random.normal(0, noise_scale, len(goal_euler))
        goal_xquat = rotations.euler2quat(goal_euler)
    
    # print("Resetting object position: ", obj_xpos, obj_xquat)
    env.unwrapped.replace_obj_goal(obj_xpos, obj_xquat, goal_xpos, goal_xquat)
    
def do_playback(env : FrankaEnv, dataset_reader : DatasetReader, playback_mode : str):
    demo_names = dataset_reader.get_demo_names()
    if playback_mode == "loop":
        demo_name_index_list = list(range(len(demo_names)))
    elif playback_mode == "random":
        demo_name_index_list = np.random.permutation(list(range(len(demo_names))))
    else:
        print("Invalid playback mode! Please input 'loop' or 'random'.")
        return
    
    for i in demo_name_index_list:
        demo_data = dataset_reader.get_demo_data(demo_names[i])
        action_list = demo_data['actions']
        done_list = demo_data['dones']
        print("Playing back episode: ", demo_names[i], " with ", len(action_list), " steps.")
        # for i, action in enumerate(action_list):
        #     print(f"Playback Action ({i}): ", action)
        reset_playback_env(env, demo_data)
        playback_episode(env, action_list, done_list)
        time.sleep(1)

def autment_episode(env : FrankaEnv, demo_data, noise_scale, realtime=False):
    obs, info = env.reset(seed=42)    
    obs_list = {obs_key: list([]) for obs_key, obs_data in obs.items()}
    reward_list = []
    done_list = []
    info_list = []    
    terminated_times = 0    
    
    reset_playback_env(env, demo_data, noise_scale)    
    action_list = demo_data['actions']
    action_index_list = list(range(len(action_list)))
    holdon_action_index_list = action_index_list[-1] * np.ones(20, dtype=int)
    action_index_list = np.concatenate([action_index_list, holdon_action_index_list]).flatten()
    
    for i in action_index_list:
        action = action_list[i]
        start_time = datetime.now()

        if noise_scale > 0.0:
            action += np.random.normal(0, noise_scale, len(action))
        
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
        if elapsed_time.total_seconds() < REALTIME_STEP and realtime:
            time.sleep(REALTIME_STEP - elapsed_time.total_seconds())

    return obs_list, reward_list, done_list, info_list
 
    

def do_augmentation(env : FrankaEnv, 
                    original_dataset_path : str, 
                    augmented_dataset_path : str, 
                    augmented_scale : float, 
                    augmented_times : int):
    
    REALTIME = False
    
    dataset_reader = DatasetReader(file_path=original_dataset_path)
    dataset_writer = DatasetWriter(file_path=augmented_dataset_path,
                                    env_name=dataset_reader.get_env_name(),
                                    env_version=dataset_reader.get_env_version(),
                                    env_kwargs=dataset_reader.get_env_kwargs())
    
    demo_names = dataset_reader.get_demo_names()
    need_demo_count = len(demo_names) * augmented_times
    done_demo_count = 0
    while done_demo_count < need_demo_count:
        original_demo_name = np.random.choice(demo_names)

        demo_data = dataset_reader.get_demo_data(original_demo_name)
        print("Augmenting original demo: ", original_demo_name)
        
        obs_list, reward_list, done_list, info_list = autment_episode(env, demo_data, noise_scale=augmented_scale, realtime=REALTIME)
        if done_list[-1] == 1:
            dataset_writer.add_demo_data({
                'states': np.array([np.concatenate([info["state"]["qpos"], info["state"]["qvel"]]) for info in info_list]),
                'actions': np.array([info["action"] for info in info_list]),
                'rewards': np.array(reward_list),
                'dones': np.array(done_list),
                'obs': obs_list
            })
            
            done_demo_count += 1
            print(f"Episode done! {done_demo_count} / {need_demo_count}")
        else:
            print("Episode failed!")
    
    if REALTIME:
        time.sleep(1)

    dataset_writer.shuffle_demos()
    dataset_writer.finalize()          

def run_example(orcagym_addr : str, 
                agent_name : str, 
                record_path : str, 
                run_mode : str, 
                ctrl_device : str, 
                max_episode_steps : int, 
                playback_mode : str,
                rollout_times : int,
                ckpt_path : str, 
                augmented_sacle : float,
                augmented_times : int):
    try:
        print("simulation running... , orcagym_addr: ", orcagym_addr)
        if run_mode == RunMode.PLAYBACK:
            dataset_reader = DatasetReader(file_path=record_path)
            env_name = dataset_reader.get_env_name()
            env_name = env_name.split("-OrcaGym-")[0]
            env_index = 0
            env_id, kwargs = register_env(orcagym_addr, env_name, env_index, agent_name, run_mode, ctrl_device, max_episode_steps)
            print("Registered environment: ", env_id)

            env = gym.make(env_id)
            print("Starting simulation...")
            do_playback(env, dataset_reader, playback_mode)

        elif run_mode == RunMode.TELEOPERATION:
            env_name = "Franka"
            env_index = 0
            env_id, kwargs = register_env(orcagym_addr, env_name, env_index, agent_name, run_mode, ctrl_device, max_episode_steps)
            print("Registered environment: ", env_id)

            env = gym.make(env_id)        
            print("Starting simulation...")
            kwargs["run_mode"] = RunMode.IMITATION  # 此处用于训练的时候读取
            dataset_writer = DatasetWriter(file_path=record_path,
                                        env_name=env_id,
                                        env_version=env.unwrapped.get_env_version(),
                                        env_kwargs=kwargs)

            do_teleoperation(env, dataset_writer)
            dataset_writer.shuffle_demos()
            dataset_writer.finalize()
            
        elif run_mode == RunMode.IMITATION:
            dataset_reader = DatasetReader(file_path=record_path)
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
            run_train_bc_rnn(dataset_type="orca_gym", dataset=record_path, output=output_dir, debug=False)
            
        elif run_mode == RunMode.ROLLOUT:
            ckpt_dict = maybe_dict_from_checkpoint(ckpt_path=ckpt_path)

            # metadata from model dict to get info needed to create environment
            env_meta = ckpt_dict["env_metadata"]    
            env_name = env_meta["env_name"]
            env_name = env_name.split("-OrcaGym-")[0]
            env_index = 0
            env_id, kwargs = register_env(orcagym_addr, env_name, env_index, agent_name, run_mode, ctrl_device, max_episode_steps)
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
                    camera_names=["agentview"]
                )
                print(stats)
        elif run_mode == RunMode.AUGMENTATION:
            dataset_reader = DatasetReader(file_path=record_path)
            env_name = dataset_reader.get_env_name()
            env_name = env_name.split("-OrcaGym-")[0]
            env_index = 0
            env_id, kwargs = register_env(orcagym_addr, env_name, env_index, agent_name, run_mode, ctrl_device, max_episode_steps)
            print("Registered environment: ", env_id)

            env = gym.make(env_id)
            print("Starting simulation...")
            
            now = datetime.now()
            formatted_now = now.strftime("%Y-%m-%d_%H-%M-%S")
            agumented_dataset_file_path = f"{current_file_path}/augmented_datasets_tmp/augmented_dataset_{formatted_now}.hdf5"
            do_augmentation(env, record_path, agumented_dataset_file_path, augmented_sacle, augmented_times)
            print("Augmentation done! The augmented dataset is saved to: ", agumented_dataset_file_path)
        else:
            print("Invalid run mode! Please input 'teleoperation' or 'playback'.")

    except KeyboardInterrupt:
        print("Simulation stopped")        
        if run_mode == RunMode.TELEOPERATION:
            dataset_writer.finalize()
        env.close()
    

def start_monitor():
    """
    启动 monitor.py 作为子进程。
    """
    # 获取当前脚本所在的目录
    monitor_script = f"{project_root}/orca_gym/scripts/camera_monitor.py"

    # 启动 monitor.py
    # 使用 sys.executable 确保使用相同的 Python 解释器
    process = subprocess.Popen(
        [sys.executable, monitor_script],
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
    parser.add_argument('--run_mode', type=str, default='teleoperation', help='The run mode of the environment (teleoperation / playback / imitation / rollout / augmentation)')
    parser.add_argument('--record_path', type=str, help='The file path to save the record')
    parser.add_argument('--ckpt_path', type=str, help='The model file to load for rollout the policy')
    parser.add_argument('--record_time', type=int, default=30, help='The time to record the teleoperation in 1 episode')
    parser.add_argument('--ctrl_device', type=str, default='xbox', help='The control device to use (xbox or keyboard)')
    parser.add_argument('--playback_mode', type=str, default='random', help='The playback mode of the environment (loop or random)')
    parser.add_argument('--rollout_times', type=int, default=10, help='The times to rollout the policy')
    parser.add_argument('--augmented_sacle', type=float, default=0.01, help='The scale to augment the dataset')
    parser.add_argument('--augmented_times', type=int, default=5, help='The times to augment the dataset')
    args = parser.parse_args()
    
    orcagym_addr = args.orcagym_address
    agent_name = args.agent_name
    record_time = args.record_time
    record_path = args.record_path
    playback_mode = args.playback_mode
    run_mode = args.run_mode
    rollout_times = args.rollout_times
    ckpt_path = args.ckpt_path
    augmented_sacle = args.augmented_sacle
    augmented_times = args.augmented_times
        
    create_tmp_dir("records_tmp")
    create_tmp_dir("trained_models_tmp")
    create_tmp_dir("augmented_datasets_tmp")
    
    if run_mode == RunMode.TELEOPERATION:
        if record_path is None:
            now = datetime.now()
            formatted_now = now.strftime("%Y-%m-%d_%H-%M-%S")
            record_path = f"./records_tmp/franka_singel_arm_{formatted_now}.hdf5"
    if run_mode == RunMode.IMITATION or run_mode == RunMode.PLAYBACK or run_mode == RunMode.AUGMENTATION:
        if record_path is None:
            print("Please input the record file path.")
            sys.exit(1)
    if run_mode == RunMode.ROLLOUT:
        if ckpt_path is None:
            print("Please input the model file path.")
            sys.exit(1) 
    if run_mode not in [RunMode.TELEOPERATION, RunMode.PLAYBACK, RunMode.IMITATION, RunMode.ROLLOUT, RunMode.AUGMENTATION]:
        print("Invalid run mode! Please input 'teleoperation', 'playback', 'imitation', 'rollout' or 'augmentation'.")
        sys.exit(1)

    if args.ctrl_device == 'xbox':
        ctrl_device = ControlDevice.XBOX
    elif args.ctrl_device == 'keyboard':
        ctrl_device = ControlDevice.KEYBOARD
    else:
        print("Invalid control device! Please input 'xbox' or 'keyboard'.")
        sys.exit(1)

    max_episode_steps = int(record_time / REALTIME_STEP)

    # 启动 Monitor 子进程
    monitor_process = start_monitor()
    print(f"Monitor 进程已启动，PID: {monitor_process.pid}")

    run_example(orcagym_addr, 
                agent_name, 
                record_path, 
                run_mode, 
                ctrl_device, 
                max_episode_steps, 
                playback_mode, 
                rollout_times, 
                ckpt_path, 
                augmented_sacle,
                augmented_times)

    # 终止 Monitor 子进程
    terminate_monitor(monitor_process)