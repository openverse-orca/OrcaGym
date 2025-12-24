import os
import sys
import time
import subprocess
import signal


from typing import Any, Dict
import gymnasium as gym
from gymnasium.envs.registration import register
from datetime import datetime
from orca_gym.environment.orca_gym_env import RewardType
from orca_gym.adapters.robomimic.dataset_util import DatasetWriter, DatasetReader
from orca_gym.sensor.rgbd_camera import Monitor, CameraWrapper
from envs.manipulation.single_arm_env import SingleArmEnv, RunMode, ControlDevice
from examples.imitation.train_policy import train_policy
from examples.imitation.test_policy import create_env, rollout
from orca_gym.utils.dir_utils import create_tmp_dir
from robomimic.utils.file_utils import maybe_dict_from_checkpoint
import orca_gym.utils.rotations as rotations

import numpy as np

from orca_gym.log.orca_log import get_orca_logger
_logger = get_orca_logger()



ENV_ENTRY_POINT = {
    "Franka": "envs.manipulation.franka_env:FrankaEnv"
}

TIME_STEP = 0.005
FRAME_SKIP = 8
REALTIME_STEP = TIME_STEP * FRAME_SKIP
CONTROL_FREQ = 1 / REALTIME_STEP

def register_env(orcagym_addr : str, 
                 env_name : str, 
                 env_index : int, 
                 agent_name : str, 
                 run_mode : str, 
                 task : str, 
                 ctrl_device : str, 
                 max_episode_steps : int,
                 sample_range : float,
                 action_step : int,
                 camera_config : Dict[str, Any]) -> str:
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
                'control_freq': CONTROL_FREQ,
                'sample_range': sample_range,
                'action_step': action_step,
                'camera_config': camera_config}           
    gym.register(
        id=env_id,
        entry_point=ENV_ENTRY_POINT[env_name],
        kwargs=kwargs,
        max_episode_steps= max_episode_steps,
        reward_threshold=0.0,
    )
    return env_id, kwargs


def teleoperation_episode(env : SingleArmEnv, cameras : list[CameraWrapper], rgb_size : tuple = (256, 256), action_step : int = 1):
    """_summary_

    Args:
        env (FrankaEnv): envairoment instance for franka robot manipulation task
        cameras (list[CameraWrapper]): list of camera instances, (primary, secondary, wrist)
        rgb_size (tuple, optional): image size for rgb camera. Defaults to (256, 256).
        action_step (int, optional): 
            How many physics steps to take in each time the model pridiects an action. 
            
            - For the robomimic model, the model pridiects an action for every physics step.
            Witch means the model can run in realtime speed.
            
            - For the VLA model, the model pridiects an action for every 5 physics steps. 
            Means the model can only run in 1/5 realtime speed. (about 5 Hz on Nvidia 3090 GPU, as the paper reported.)

    Returns:
        _type_: _description_
    """
    obs, info = env.reset(seed=42)
    obs_list = {obs_key: list([]) for obs_key, obs_data in obs.items()}
    reward_list = []
    done_list = []
    info_list = []    
    terminated_times = 0
    camera_frames = {camera.name: [] for camera in cameras}
    timestep_list = []
    action_step_taken = 0
    while True:
        start_time = datetime.now()

        action = env.action_space.sample()
  
        obs, reward, terminated, truncated, info = env.step(action)
        
        env.render()
        
        action_step_taken += 1
        if action_step_taken >= action_step:        
            action_step_taken = 0
            for obs_key, obs_data in obs.items():
                obs_list[obs_key].append(obs_data)
                
            reward_list.append(reward)
            done_list.append(0 if not terminated else 1)
            info_list.append(info)
            terminated_times = terminated_times + 1 if terminated else 0
            
            for camera in cameras:
                camera_frame, _ = camera.get_frame(format='rgb24', size=rgb_size)
                camera_frames[camera.name].append(camera_frame)
                
            # print("Timestep: ", env.unwrapped.gym.data.time)
            timestep_list.append(env.unwrapped.gym.data.time)

        if terminated_times >= 5 or truncated:
            return obs_list, reward_list, done_list, info_list, camera_frames, timestep_list

        elapsed_time = datetime.now() - start_time
        if elapsed_time.total_seconds() < REALTIME_STEP:
            time.sleep(REALTIME_STEP - elapsed_time.total_seconds())
            
def user_comfirm_save_record(task_result, currnet_round, teleoperation_rounds):
    while True:
        user_input = input(f"Round {currnet_round} / {teleoperation_rounds}, Task is {task_result}! Do you want to save the record? y(save), n(ignore), e(ignore & exit): ")
        if user_input == 'y':
            return True, False
        elif user_input == 'n':
            return False, False
        elif user_input == 'e':
            return False, True
        else:
            _logger.error(f"Invalid input! Please input 'y', 'n' or 'e'.")

def add_demo_to_dataset(dataset_writer : DatasetWriter,
                        obs_list, 
                        reward_list, 
                        done_list, 
                        info_list, 
                        camera_frames, 
                        timestep_list, 
                        language_instruction):
        
    dataset_writer.add_demo_data({
        'states': np.array([np.concatenate([info["state"]["qpos"], info["state"]["qvel"]]) for info in info_list], dtype=np.float32),
        'actions': np.array([info["action"] for info in info_list], dtype=np.float32),
        'objects': np.array([info["object"] for info in info_list], dtype=np.float32),
        'goals': np.array([info["goal"] for info in info_list], dtype=np.float32),
        'rewards': np.array(reward_list, dtype=np.float32),
        'dones': np.array(done_list, dtype=np.int32),
        'obs': obs_list,
        'camera_frames': camera_frames,
        'timesteps': np.array(timestep_list, dtype=np.float32),
        'language_instruction': language_instruction
    })

def do_teleoperation(env, 
                     dataset_writer : DatasetWriter, 
                     teleoperation_rounds : int, 
                     cameras : list[CameraWrapper], 
                     obs_camera : bool,
                     rgb_size : tuple = (256, 256),
                     action_step : int = 1,
                     language_instruction : str = None):    
    
    current_round = 1
    
    for camera in cameras:
        camera.start()
    
    while True:
        obs_list, reward_list, done_list, info_list, camera_frames, timestep_list = teleoperation_episode(env, cameras, rgb_size, action_step)
        task_result = "Success" if done_list[-1] == 1 else "Failed"
        save_record, exit_program = user_comfirm_save_record(task_result, current_round, teleoperation_rounds)
        if save_record:
            current_round += 1
            
            if obs_camera:
                for camera in cameras:
                    obs_list[camera.name] = camera_frames[camera.name]
                    camera_frames[camera.name] = []         
                empty_camera_frames = {}
                add_demo_to_dataset(dataset_writer, obs_list, reward_list, done_list, info_list, empty_camera_frames, timestep_list, language_instruction)
            else:
                add_demo_to_dataset(dataset_writer, obs_list, reward_list, done_list, info_list, camera_frames, timestep_list, language_instruction)
        if exit_program or current_round > teleoperation_rounds:
            break
        
    for camera in cameras:
        camera.stop()

def playback_episode(env : SingleArmEnv, 
                     action_list : list[np.ndarray], 
                     done_list : list[int],
                     action_step : int = 1,
                     realtime : bool = False):
    for i in range(len(action_list)):


        action = action_list[i]
        done = done_list[i]
        
        for _ in range(action_step):
            if realtime:
                start_time = datetime.now()
                        
            obs, reward, terminated, truncated, info = env.step(action)
            env.render()

            if realtime:
                elapsed_time = datetime.now() - start_time
                if elapsed_time.total_seconds() < REALTIME_STEP:
                    time.sleep(REALTIME_STEP - elapsed_time.total_seconds())

        if done:
            _logger.info("Episode done!")
            return

        if terminated or truncated:
            _logger.info("Episode terminated!")
            return
    
    _logger.info("Episode tunkated!")

def reset_playback_env(env : SingleArmEnv, demo_data, sample_range=0.0):
    obs, info = env.reset(seed=42)
    
    object_data = demo_data['objects']
    init_obj_xpos = object_data[0][0:3]
    init_obj_xquat = object_data[0][3:7]
    
    goal_data = demo_data['goals']
    init_goal_xpos = goal_data[0][0:3]
    init_goal_xquat = goal_data[0][3:7]
    
    obj_xpos, obj_xquat, goal_xpos, goal_xquat = env.unwrapped.sample_obj_goal(init_obj_xpos, init_obj_xquat, init_goal_xpos, init_goal_xquat, sample_range)
    
    # print("Resetting object position: ", obj_xpos, obj_xquat, goal_xquat)
    
    env.unwrapped.replace_obj_goal(obj_xpos, obj_xquat, goal_xpos, goal_xquat)
    
    # print("Resetting object position: ", obj_xpos, obj_xquat)
    
def do_playback(env : SingleArmEnv, 
                dataset_reader : DatasetReader, 
                playback_mode : str,
                action_step : int = 1,
                realtime : bool = False):
    demo_names = dataset_reader.get_demo_names()
    if playback_mode == "loop":
        demo_name_index_list = list(range(len(demo_names)))
    elif playback_mode == "random":
        demo_name_index_list = np.random.permutation(list(range(len(demo_names))))
    else:
        _logger.error("Invalid playback mode! Please input 'loop' or 'random'.")
        return
    
    for i in demo_name_index_list:
        demo_data = dataset_reader.get_demo_data(demo_names[i])
        action_list = demo_data['actions']
        done_list = demo_data['dones']
        _logger.info(f"Playing back episode: {demo_names[i]} with {len(action_list)} steps.")
        # for i, action in enumerate(action_list):
        #     print(f"Playback Action ({i}): ", action)
        reset_playback_env(env, demo_data)
        playback_episode(env, action_list, done_list, action_step, realtime)
        time.sleep(1)

def augment_episode(env : SingleArmEnv, 
                    cameras : list[CameraWrapper], 
                    rgb_size : tuple,
                    demo_data : dict, 
                    noise_scale : float, 
                    sample_range : float, 
                    realtime : bool = False,
                    action_step : int = 1):
    obs, info = env.reset(seed=42)    
    obs_list = {obs_key: list([]) for obs_key, obs_data in obs.items()}
    reward_list = []
    done_list = []
    info_list = []    
    terminated_times = 0    
    camera_frames = {camera.name: [] for camera in cameras}
    timestep_list = []
    
    reset_playback_env(env, demo_data, sample_range)    
    action_list = demo_data['actions']
    action_index_list = list(range(len(action_list)))
    holdon_action_index_list = action_index_list[-1] * np.ones(20, dtype=int)
    action_index_list = np.concatenate([action_index_list, holdon_action_index_list]).flatten()
    
    for i in action_index_list:
        action = action_list[i]


        if noise_scale > 0.0:
            noise = np.random.normal(0, noise_scale, len(action))
            action += noise * np.abs(action)
            action = np.clip(action, -1.0, 1.0)
        
        for _ in range(action_step):
            if realtime:
                start_time = datetime.now()
                
            obs, reward, terminated, truncated, info = env.step(action)
            
            if realtime:
                env.render()
                elapsed_time = datetime.now() - start_time
                if elapsed_time.total_seconds() < REALTIME_STEP:
                    time.sleep(REALTIME_STEP - elapsed_time.total_seconds())
                    
        env.render()
                    
        if len(cameras) > 0:
            time.sleep(0.03)    # wait for camera to get new frame
            for camera in cameras:
                camera_frame, _ = camera.get_frame(format='rgb24', size=rgb_size)
                camera_frames[camera.name].append(camera_frame)

        for obs_key, obs_data in obs.items():
            obs_list[obs_key].append(obs_data)
            
        reward_list.append(reward)
        done_list.append(0 if not terminated else 1)
        info_list.append(info)
        terminated_times = terminated_times + 1 if terminated else 0
        timestep_list.append(env.unwrapped.gym.data.time)
        

        if terminated_times >= 5 or truncated:
            return obs_list, reward_list, done_list, info_list, camera_frames, timestep_list
        
    return obs_list, reward_list, done_list, info_list, camera_frames, timestep_list
 
    

def do_augmentation(env : SingleArmEnv, 
                    cameras : list[CameraWrapper], 
                    obs_camera : bool,
                    rgb_size : tuple,                    
                    original_dataset_path : str, 
                    augmented_dataset_path : str, 
                    augmented_noise : float, 
                    sample_range : float,
                    augmented_rounds : int,
                    action_step : int = 1):
    
    realtime = False
    
    # Copy the original dataset to the augmented dataset
    dataset_reader = DatasetReader(file_path=original_dataset_path)
    dataset_writer = DatasetWriter(file_path=augmented_dataset_path,
                                    env_name=dataset_reader.get_env_name(),
                                    env_version=dataset_reader.get_env_version(),
                                    env_kwargs=dataset_reader.get_env_kwargs())
    demo_names = dataset_reader.get_demo_names()
    for demo_name in demo_names:
        demo_data = dataset_reader.get_demo_data(demo_name)
        dataset_writer.add_demo_data(demo_data)
        
    # Use the augmented dataset as the new dataset reader
    dataset_reader = DatasetReader(file_path=augmented_dataset_path)

    for camera in cameras:
        camera.start()
    
    for round in range(augmented_rounds):    
        need_demo_count = dataset_reader.get_demo_count()
        done_demo_count = 0
            
        demo_names = dataset_reader.get_demo_names()
        for original_demo_name in demo_names:
            done = False
            while not done:
                demo_data = dataset_reader.get_demo_data(original_demo_name)
                _logger.info(f"Augmenting original demo:  {original_demo_name}")
                language_instruction = demo_data['language_instruction']
                
                obs_list, reward_list, done_list, info_list\
                    , camera_frames, timestep_list = augment_episode(env, cameras,rgb_size,
                                                                    demo_data, noise_scale=augmented_noise, 
                                                                    sample_range=sample_range, realtime=realtime, 
                                                                    action_step=action_step)
                if done_list[-1] == 1:
                    if obs_camera:
                        for camera in cameras:
                            obs_list[camera.name] = camera_frames[camera.name]
                        empty_camera_frames = {}
                        add_demo_to_dataset(dataset_writer, obs_list, reward_list, done_list, info_list, empty_camera_frames, timestep_list, language_instruction)
                    else:
                        add_demo_to_dataset(dataset_writer, obs_list, reward_list, done_list, info_list, camera_frames, timestep_list, language_instruction)
                    
                    done_demo_count += 1
                    _logger.info(f"Episode done! {done_demo_count} / {need_demo_count} for round {round + 1}")
                    done = True
                else:
                    _logger.error("Episode failed!")
    

    dataset_writer.shuffle_demos()
    dataset_writer.finalize()       
    
    for camera in cameras:
        camera.stop()
    
def start_monitor(port=7070, project_root : str = None):
    """
    启动 monitor.py 作为子进程。
    """
    monitor_script = f"{project_root}/orca_gym/scripts/camera_monitor.py"

    # 启动 monitor.py
    # 使用 sys.executable 确保使用相同的 Python 解释器
    process = subprocess.Popen(
        [sys.executable, monitor_script, "--port", f"{port}"],
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