import os
import sys
import time
import subprocess
import signal


from typing import Any, Dict
from flask import g
import gymnasium as gym
from gymnasium.envs.registration import register
from datetime import datetime

import h5py
from orca_gym.environment.orca_gym_env import RewardType
from orca_gym.robomimic.dataset_util import DatasetWriter, DatasetReader
from orca_gym.sensor.rgbd_camera import Monitor, CameraWrapper
from envs.manipulation.openloong_env import OpenLoongEnv, ControlDevice
from examples.imitation.train_policy import train_policy
from examples.imitation.test_policy import create_env, rollout
from orca_gym.utils.dir_utils import create_tmp_dir
from robomimic.utils.file_utils import maybe_dict_from_checkpoint
from robomimic.utils.train_utils import run_rollout
import orca_gym.utils.rotations as rotations
from envs.manipulation.openloong_env import TaskStatus
from envs.manipulation.openloong_env import ActionType
import re
import gymnasium as gym
from envs.manipulation.openloong_env import ControlDevice, RunMode, OpenLoongEnv
import numpy as np
import yaml

ENV_ENTRY_POINT = {
    "OpenLoong": "envs.manipulation.openloong_env:OpenLoongEnv"
}

TIME_STEP = 0.005                       # 200 Hz for physics simulation
FRAME_SKIP = 4
REALTIME_STEP = TIME_STEP * FRAME_SKIP  # 50 Hz for rendering
CONTROL_FREQ = 1 / REALTIME_STEP        # 50 Hz for OSC controller computation

RGB_SIZE = (256, 256)
CAMERA_CONFIG = {
    "camera_head": 7070,
    # "camera_wrist_r": 7080,
    # "camera_wrist_l": 7090,
}


def register_env(orcagym_addr : str, 
                 env_name : str, 
                 env_index : int, 
                 agent_names : str,
                 pico_ports : str,
                 run_mode : str, 
                 action_type : str,
                 ctrl_device : str,
                 max_episode_steps : int,
                 sample_range : float,
                 action_step : int,
                 camera_config : Dict[str, Any],
                 task_config_dict: Dict[str, Any] = None) -> str:
    orcagym_addr_str = orcagym_addr.replace(":", "-")
    env_id = env_name + "-OrcaGym-" + orcagym_addr_str + f"-{env_index:03d}"
    agent_names_list = agent_names.split(" ")
    print("Agent names: ", agent_names_list)
    pico_ports = pico_ports.split(" ")
    print("Pico ports: ", pico_ports)
    kwargs = {'frame_skip': FRAME_SKIP,   
                'reward_type': RewardType.SPARSE,
                'orcagym_addr': orcagym_addr, 
                'agent_names': agent_names_list,
                'pico_ports': pico_ports,
                'time_step': TIME_STEP,
                'run_mode': run_mode,
                'action_type': action_type,
                'ctrl_device': ctrl_device,
                'control_freq': CONTROL_FREQ,
                'sample_range': sample_range,
                'action_step': action_step,
                'camera_config': camera_config,
                'task_config_dict': task_config_dict}
    gym.register(
        id=env_id,
        entry_point=ENV_ENTRY_POINT[env_name],
        kwargs=kwargs,
        max_episode_steps= max_episode_steps,
        reward_threshold=0.0,
    )
    return env_id, kwargs


def teleoperation_episode(env : OpenLoongEnv, cameras : list[CameraWrapper], rgb_size : tuple = (256, 256), action_step : int = 1):
    """_summary_

    Args:
        env (OpenLoongEnv): envairoment instance for franka robot manipulation task
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
    in_goal_list = [] 
    camera_frames = {camera.name: [] for camera in cameras}
    timestep_list = []
    action_step_taken = 0
    while True:
        start_time = datetime.now()

        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        lang_instr = info.get('language_instruction', '')
        obj_match = re.search(r'object:\s*([^\s]+)', lang_instr)
        goal_match = re.search(r'goal:\s*([^\s]+)', lang_instr)
        target_obj = obj_match.group(1) if obj_match else None
        target_goal = goal_match.group(1) if goal_match else None
        if not (target_obj and target_goal):
            print(f"[Warning] 无法从 language_instruction 中解析 object/goal: '{lang_instr}'")
        env.render()
        task_status = info['task_status']
        
        action_step_taken += 1
        if action_step_taken >= action_step:        
            action_step_taken = 0
            if task_status in [TaskStatus.SUCCESS, TaskStatus.FAILURE, TaskStatus.BEGIN]:
                if task_status == TaskStatus.SUCCESS:
                    goal_info = info["goal"]
                    object_info = info["object"]
                    object_joint_name = object_info["joint_name"][0]
                    object_body_name = object_joint_name.replace("_joint", "")
                    print("object_body_name:",object_body_name)
                    obj_joints = info['object']['joint_name']
                    goal_joints = info['goal']['joint_name']

                    # 查找 obj_idx
                    obj_idx = next((i for i, jn in enumerate(obj_joints) if target_obj in jn), None)
                    # 查找 goal_idx
                    goal_idx = next((i for i, gn in enumerate(goal_joints) if target_goal in gn), None)
                    if obj_idx is None or goal_idx is None:
                        print(f"[Error] 未找到匹配: object='{target_obj}'，goal='{target_goal}'")
                    else:
                        # 获取物体位置
                        joint_name = obj_joints[obj_idx]
                        body_name = joint_name.replace('_joint', '')
                        pos, _, _ = env.unwrapped.get_body_xpos_xmat_xquat([body_name])
                        pos_vec = pos[0] if hasattr(pos, 'ndim') and pos.ndim > 1 else pos
                        xy = pos_vec[:2]
                        # 获取目标区域边界
                        gmin = info['goal']['min'][goal_idx][:2]
                        gmax = info['goal']['max'][goal_idx][:2]
                        in_goal = gmin[0] <= xy[0] <= gmax[0] and gmin[1] <= xy[1] <= gmax[1]
                        in_goal_list.append(in_goal)
                        print(f"检查 {target_obj} 在 {target_goal} 区域: {in_goal}")
                        if not in_goal:
                            info['task_status'] = TaskStatus.FAILURE
                            print("⚠️ 标记为 FAILURE: 物体未进入目标区域。")
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
            return obs_list, reward_list, done_list, info_list, camera_frames, timestep_list,in_goal_list

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
            print("Invalid input! Please input 'y', 'n' or 'e'.")

def add_demo_to_dataset(dataset_writer : DatasetWriter,
                        obs_list, 
                        reward_list, 
                        done_list, 
                        info_list, 
                        camera_frames, 
                        timestep_list, 
                        language_instruction):
        
    # 只处理第一个info对象（初始状态）
    first_info = info_list[0]
    
    dtype = np.dtype([
        ('joint_name', h5py.special_dtype(vlen=str)),
        ('position', 'f4', (3,)),
        ('orientation', 'f4', (4,))
    ])
    gdtype = np.dtype([
        ('joint_name', h5py.special_dtype(vlen=str)),
        ('position', 'f4', (3,)),
        ('orientation', 'f4', (4,)),
        ('min', 'f4', (3,)),
        ('max', 'f4', (3,)),
        ('size', 'f4', (3,))
    ])
    
    # 只提取第一个对象的关节信息
    objects_array = np.array([
        (
            joint_name,
            position,
            orientation
        )
        for joint_name, position, orientation in zip(
            first_info['object']['joint_name'],
            first_info['object']['position'],
            first_info['object']['orientation']
        )
    ], dtype=dtype)
    goals_array = np.array([
        (
            joint_name,
            position,
            orientation,
            min,
            max,
            size
        )
        for joint_name, position, orientation,max,min,size in zip(
            first_info['goal']['joint_name'],
            first_info['goal']['position'],
            first_info['goal']['orientation'],
            first_info['goal']['min'],
            first_info['goal']['max'],
            first_info['goal']['size']
        )
    ], dtype=gdtype)

        
    dataset_writer.add_demo_data({
        'states': np.array([np.concatenate([info["state"]["qpos"], info["state"]["qvel"]]) for info in info_list], dtype=np.float32),
        'actions': np.array([info["action"] for info in info_list], dtype=np.float32),
        'objects': objects_array,
        'goals': goals_array,
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
                     action_step : int = 1,):    
    
    current_round = 1
    
    for camera in cameras:
        camera.start()
    
    while True:
        obs_list, reward_list, done_list, info_list, camera_frames, timestep_list,in_goal_list = teleoperation_episode(env, cameras, rgb_size, action_step)
        last_done = (len(done_list)>0 and done_list[-1]==1)
        last_in_goal = (len(in_goal_list)>0 and in_goal_list[-1])
        save_record = last_done and last_in_goal
        task_result = "Success" if save_record else "Failed"
        # save_record, exit_program = user_comfirm_save_record(task_result, current_round, teleoperation_rounds)
        save_record = task_result == "Success"
        exit_program = False
        if save_record:
            print(f"Round {current_round} / {teleoperation_rounds}, Task is {task_result}!")
            current_round += 1
            
            if obs_camera:
                for camera in cameras:
                    obs_list[camera.name] = camera_frames[camera.name]
                    camera_frames[camera.name] = []         
                empty_camera_frames = {}
                add_demo_to_dataset(dataset_writer, obs_list, reward_list, done_list, info_list, empty_camera_frames, timestep_list, info_list[0]["language_instruction"])
            else:
                add_demo_to_dataset(dataset_writer, obs_list, reward_list, done_list, info_list, camera_frames, timestep_list, info_list[0]["language_instruction"])
        if exit_program or current_round > teleoperation_rounds:
            break
        
    for camera in cameras:
        camera.stop()

def playback_episode(env : OpenLoongEnv, 
                     action_list : list[np.ndarray], 
                     done_list : list[int],
                     action_step : int = 1,
                     realtime : bool = False):
    for i in range(len(action_list)):

        action = action_list[i]
        last_action = action_list[i - 1] if i > 0 else action
        # 插值生成一组动作序列，长度为 action_step
        action_chunk = np.array([last_action + (action - last_action) * (i / action_step) for i in range(action_step)])
        # print("Playback Action: ", action_chunk)

        done = done_list[i]
        
        for i in range(action_step):
            if realtime:
                start_time = datetime.now()

            obs, reward, terminated, truncated, info = env.step(action_chunk[i])
            env.render()

            if realtime:
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


def reset_playback_env(env: OpenLoongEnv, demo_data, sample_range=0.0):
    """
    Reset 环境，然后把 demo_data['objects'] 和 demo_data['goals'] 恢复到场景里。
    """
    # 1) 先走一次基本 reset，让 self.objects 和 self.goals（如果存在）都被清空
    obs, info = env.reset(seed=42)

    # 2) 恢复 objects
    recorded_objs = demo_data['objects']
    env.unwrapped.replace_objects(recorded_objs)

    # 3) 恢复 goals
    recorded_goals = demo_data.get('goals')
    if recorded_goals is not None:
        # 调用你之前写的 replace_goals，会将结构化数组或纯数值数组都转换并赋给 self.goals
        env.unwrapped.replace_goals(recorded_goals)
    else:
        print("[Warning] demo_data 中没有 'goals'，无法恢复目标区域信息")

    # 4) 推一步仿真，让新的 objects/goals 生效
    env.unwrapped.mj_forward()

    # 5) 再取一次 obs，组装 info
    obs = env.unwrapped._get_obs().copy()
    info = {
        "object": recorded_objs,
        "goal":   recorded_goals
    }
    return obs, info



    
def do_playback(env : OpenLoongEnv, 
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
        print("Invalid playback mode! Please input 'loop' or 'random'.")
        return
    
    for i in demo_name_index_list:
        demo_data = dataset_reader.get_demo_data(demo_names[i])
        action_list = demo_data['actions']
        done_list = demo_data['dones']
        print("Playing back episode: ", demo_names[i], " with ", len(action_list), " steps.")
        # for i, action in enumerate(action_list):
        #     print(f"Playback Action ({i}): ", action)
        env.unwrapped.objects = demo_data['objects']
        reset_playback_env(env, demo_data)
        playback_episode(env, action_list, done_list, action_step, realtime)
        time.sleep(1)

def augment_episode(env : OpenLoongEnv, 
                    cameras : list[CameraWrapper], 
                    rgb_size : tuple,
                    demo_data : dict, 
                    noise_scale : float, 
                    sample_range : float, 
                    realtime : bool = False,
                    action_step : int = 1):
    obs, info = reset_playback_env(env, demo_data, sample_range)
    obs_list    = {k: [] for k in obs}
    reward_list = []
    done_list = []
    info_list = []    
    terminated_times = 0    
    camera_frames = {camera.name: [] for camera in cameras}
    timestep_list = []
    lang_instr = demo_data.get("language_instruction", b"")
    if isinstance(lang_instr, (bytes, bytearray)):
        lang_instr = lang_instr.decode("utf-8")
    obj_match = re.search(r'object:\s*([^\s]+)', lang_instr)
    goal_match = re.search(r'goal:\s*([^\s]+)', lang_instr)
    target_obj = obj_match.group(1) if obj_match else None
    target_goal = goal_match.group(1) if goal_match else None

    in_goal=False
  
    action_list = demo_data['actions']
    action_index_list = list(range(len(action_list)))
    holdon_action_index_list = action_index_list[-1] * np.ones(20, dtype=int)
    action_index_list = np.concatenate([action_index_list, holdon_action_index_list]).flatten()
    
    for i in action_index_list:
        action = action_list[i]
        last_action = action_list[i - 1] if i > 0 else action
        # 插值生成一组动作序列，长度为 action_step
        action_chunk = np.array([last_action + (action - last_action) * (i / action_step) for i in range(action_step)])
        # print("Playback Action: ", action_chunk)

        if noise_scale > 0.0:
            noise = np.random.normal(0, noise_scale, len(action))
            action += noise * np.abs(action)
            action = np.clip(action, -1.0, 1.0)
        
        for i in range(action_step):
            if realtime:
                start_time = datetime.now()
                
            obs, reward, terminated, truncated, info = env.step(action_chunk[i])
            terminated_times = terminated_times + 1 if terminated else 0
            timestep_list.append(env.unwrapped.gym.data.time)
            
            if realtime:
                env.render()
                elapsed_time = datetime.now() - start_time
                if elapsed_time.total_seconds() < REALTIME_STEP:
                    time.sleep(REALTIME_STEP - elapsed_time.total_seconds())
        if target_obj and target_goal:
            raw_obj_joints  = info['object']['joint_name']
            obj_joints = [jn.decode('utf-8') if isinstance(jn, (bytes, bytearray)) else jn
                       for jn in raw_obj_joints]
            raw_goal_joints = info['goal']['joint_name']
            goal_joints = [gn.decode('utf-8') if isinstance(gn, (bytes, bytearray)) else gn
                       for gn in raw_goal_joints]
            obj_idx  = next((i for i, jn in enumerate(obj_joints)  if target_obj  in jn), None)
            goal_idx = next((i for i, gn in enumerate(goal_joints) if target_goal in gn), None)

            if obj_idx is not None and goal_idx is not None:
                joint_name = obj_joints[obj_idx]
                body_name = joint_name.replace('_joint', '')
                pos, _, _ = env.unwrapped.get_body_xpos_xmat_xquat([body_name])
                pos_vec = pos[0] if hasattr(pos, 'ndim') and pos.ndim > 1 else pos
                xy = pos_vec[:2]
                # 读取原始值
                gmin_raw = info['goal']['min'][goal_idx][:2]
                gmax_raw = info['goal']['max'][goal_idx][:2]
                # 排序：
                gmin = np.minimum(gmin_raw, gmax_raw)
                gmax = np.maximum(gmin_raw, gmax_raw)
                # 然后再判定
                in_goal = (gmin[0] <= xy[0] <= gmax[0]) and (gmin[1] <= xy[1] <= gmax[1])
            else:
                print(f"[Aug Error] 目标匹配失败：{target_obj}, {target_goal}")
                    
        env.render()
                    
        if len(cameras) > 0:
            time.sleep(0.01)    # wait for camera to get new frame
            for camera in cameras:
                camera_frame, _ = camera.get_frame(format='rgb24', size=rgb_size)
                camera_frames[camera.name].append(camera_frame)

        for obs_key, obs_data in obs.items():
            obs_list[obs_key].append(obs_data)
            
        reward_list.append(reward)
        done_list.append(1)
        info_list.append(info)

        

        if terminated_times >= 5 or truncated:
            return obs_list, reward_list, done_list, info_list, camera_frames, timestep_list,in_goal
        
    return obs_list, reward_list, done_list, info_list, camera_frames, timestep_list,in_goal
 
    

def do_augmentation(env : OpenLoongEnv, 
                    cameras : list[CameraWrapper], 
                    obs_camera : bool,
                    rgb_size : tuple,                    
                    original_dataset_path : str, 
                    augmented_dataset_path : str, 
                    augmented_scale : float, 
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


    for camera in cameras:
        camera.start()
    
    for round in range(augmented_rounds):    
        need_demo_count = dataset_reader.get_demo_count()
        done_demo_count = 0
            
        demo_names = dataset_reader.get_demo_names()

        for original_demo_name in demo_names:
            done = False
            trial_count = 0
            max_trials = 5
            while not done and trial_count < max_trials:
                demo_data = dataset_reader.get_demo_data(original_demo_name)
                env.unwrapped.objects = demo_data['objects']
                obs, info = reset_playback_env(env, demo_data, sample_range)
                print("Augmenting original demo: ", original_demo_name)
                language_instruction = demo_data['language_instruction']
                
                obs_list, reward_list, done_list, info_list\
                    , camera_frames, timestep_list,in_goal = augment_episode(env, cameras,rgb_size,
                                                                    demo_data, noise_scale=augmented_scale, 
                                                                    sample_range=sample_range, realtime=realtime, 
                                                                    action_step=action_step)
                if  in_goal:
                    
                    if obs_camera:
                        for camera in cameras:
                            obs_list[camera.name] = camera_frames[camera.name]
                        empty_camera_frames = {}
                        add_demo_to_dataset(dataset_writer, obs_list, reward_list, done_list, info_list, empty_camera_frames, timestep_list, language_instruction)
                    else:
                        add_demo_to_dataset(dataset_writer, obs_list, reward_list, done_list, info_list, camera_frames, timestep_list, language_instruction)
                    
                    done_demo_count += 1
                    print(f"Episode done! {done_demo_count} / {need_demo_count} for round {round + 1}")
                    done = True
                else:
                    print("Episode failed! Retrying...")
                    trial_count += 1
            if not done:
                print(f"Failed to augment demo {original_demo_name} after {max_trials} tries. Skipping.")
    

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



def run_example(orcagym_addr : str,
                agent_names : str,
                pico_ports : str,
                record_path : str,
                run_mode : str,
                action_type : str,
                action_step : int,
                algo_config : str,
                ctrl_device : str,
                max_episode_steps : int,
                playback_mode : str,
                rollout_times : int,
                ckpt_path : str,
                augmented_sacle : float,
                augmented_rounds : int,
                teleoperation_rounds : int,
                sample_range : float,
                realtime_playback : bool,
                current_file_path : str,
                task_config : str,):
    try:
        print("simulation running... , orcagym_addr: ", orcagym_addr)
        if run_mode == "playback":
            dataset_reader = DatasetReader(file_path=record_path)
            print("kwargs: ", dataset_reader.get_env_kwargs())
            camera_config = dataset_reader.get_env_kwargs()["camera_config"]
            action_step = dataset_reader.get_env_kwargs()["action_step"]
            action_type = dataset_reader.get_env_kwargs()["action_type"]
            env_name = dataset_reader.get_env_name()
            env_name = env_name.split("-OrcaGym-")[0]
            env_index = 0
            env_id, kwargs = register_env(orcagym_addr, env_name, env_index, agent_names, pico_ports, RunMode.POLICY_NORMALIZED, action_type, ctrl_device, max_episode_steps, sample_range, action_step, camera_config)
            print("Registered environment: ", env_id)

            env = gym.make(env_id)
            print("Starting simulation...")
            do_playback(env, dataset_reader, playback_mode, action_step, realtime_playback)

        elif run_mode == "teleoperation":
            env_name = "OpenLoong"
            env_index = 0
            camera_config = CAMERA_CONFIG

            if task_config is None:
                task_config_dict = {}
            else:
                with open(task_config, 'r') as f:
                    task_config_dict = yaml.safe_load(f)

            env_id, kwargs = register_env(orcagym_addr, env_name, env_index, agent_names, pico_ports, RunMode.TELEOPERATION, action_type, ctrl_device, max_episode_steps, sample_range, action_step, camera_config, task_config_dict)
            print("Registered environment: ", env_id)

            env = gym.make(env_id)
            print("Starting simulation...")
            kwargs["run_mode"] = RunMode.POLICY_NORMALIZED  # 此处用于训练的时候读取

            if RGB_SIZE is None:
                cameras = []
            else:
                cameras = [CameraWrapper(name=camera_name, port=camera_port) for camera_name, camera_port in camera_config.items()]

            dataset_writer = DatasetWriter(file_path=record_path,
                                        env_name=env_id,
                                        env_version=env.unwrapped.get_env_version(),
                                        env_kwargs=kwargs)


            do_teleoperation(env, dataset_writer, teleoperation_rounds,
                                                 cameras=cameras, obs_camera=True, rgb_size=RGB_SIZE, action_step=action_step,)
            dataset_writer.shuffle_demos()
            dataset_writer.finalize()

        elif run_mode == "imitation":
            dataset_reader = DatasetReader(file_path=record_path)
            env_name = dataset_reader.get_env_name()
            camera_config = dataset_reader.get_env_kwargs()["camera_config"]
            action_step = dataset_reader.get_env_kwargs()["action_step"]
            action_type = dataset_reader.get_env_kwargs()["action_type"]
            env_name = env_name.split("-OrcaGym-")[0]
            env_index = 0
            env_id, kwargs = register_env(orcagym_addr, env_name, env_index, agent_names, pico_ports, RunMode.POLICY_NORMALIZED, action_type, ctrl_device, max_episode_steps, sample_range, action_step, camera_config)
            print("Registered environment: ", env_id)

            # env = gym.make(env_id)
            # print("Starting simulation...")

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
            camera_config = env_kwargs["camera_config"]
            sample_range = env_kwargs["sample_range"]
            action_step = env_kwargs["action_step"]
            action_type = env_kwargs["action_type"]

            env_id, kwargs = register_env(orcagym_addr, env_name, env_index, agent_names, pico_ports, RunMode.POLICY_NORMALIZED, action_type, ctrl_device, max_episode_steps, sample_range, action_step, camera_config)
            print("Registered environment: ", env_id)

            env, policy = create_env(ckpt_path)

            for i in range(rollout_times):
                stats = run_rollout(
                    policy=policy,
                    env=env,
                    horizon=int(max_episode_steps / action_step),
                    render=True,
                    realtime_step=REALTIME_STEP
                )
                print(stats)
        elif run_mode == "augmentation":
            dataset_reader = DatasetReader(file_path=record_path)
            env_name = dataset_reader.get_env_name()
            camera_config = dataset_reader.get_env_kwargs()["camera_config"]
            action_step = dataset_reader.get_env_kwargs()["action_step"]
            action_type = dataset_reader.get_env_kwargs()["action_type"]
            env_name = env_name.split("-OrcaGym-")[0]
            env_index = 0
            env_id, kwargs = register_env(orcagym_addr, env_name, env_index, agent_names, pico_ports, RunMode.POLICY_NORMALIZED, action_type, ctrl_device, max_episode_steps, sample_range, action_step, camera_config)
            print("Registered environment: ", env_id)

            env = gym.make(env_id)
            print("Starting simulation...")

            now = datetime.now()
            formatted_now = now.strftime("%Y-%m-%d_%H-%M-%S")
            agumented_dataset_file_path = f"{current_file_path}/augmented_datasets_tmp/augmented_dataset_{formatted_now}.hdf5"

            if RGB_SIZE is None:
                cameras = []
            else:
                cameras = [CameraWrapper(name=camera_name, port=camera_port) for camera_name, camera_port in camera_config.items()]

            do_augmentation(env, cameras, False, RGB_SIZE, record_path, agumented_dataset_file_path, augmented_sacle, sample_range, augmented_rounds, action_step)
            print("Augmentation done! The augmented dataset is saved to: ", agumented_dataset_file_path)
        else:
            print("Invalid run mode! Please input 'teleoperation' or 'playback'.")

    finally:
        print("Simulation stopped")
        if run_mode == "teleoperation":
            dataset_writer.finalize()
        env.close()


def _get_algo_config(algo_name):
    if algo_name == "bc":
        return ["config/bc.json"]
    elif algo_name == "bc_transformer":
        return ["config/bc_transformer.json"]
    elif algo_name == "openpi":
        return ["openpi"]

    elif algo_name == "all":
        return ["config/bc.json",

                "config/bc_transformer.json"]
    else:
        raise ValueError(f"Invalid algorithm name: {algo_name}")


def run_openloong_sim(args, project_root : str = None, current_file_path : str = None):
    orcagym_addr = args.orcagym_address
    agent_names = args.agent_names
    pico_ports = args.pico_ports
    record_time = args.record_length
    record_path = args.dataset
    playback_mode = args.playback_mode
    run_mode = args.run_mode
    action_type = args.action_type
    action_step = args.action_step
    task_config = args.task_config
    algo = args.algo
    rollout_times = args.rollout_times
    ckpt_path = args.model_file
    augmented_sacle = args.augmented_sacle
    augmented_rounds = args.augmented_rounds
    teleoperation_rounds = args.teleoperation_rounds
    sample_range = args.sample_range
    realtime_playback = args.realtime_playback

    assert record_time > 0, "The record time should be greater than 0."
    assert teleoperation_rounds > 0, "The teleoperation rounds should be greater than 0."
    assert sample_range >= 0.0, "The sample range should be greater than or equal to 0."
    assert augmented_sacle >= 0.0, "The augmented scale should be greater than or equal to 0."
    assert augmented_rounds > 0, "The augmented times should be greater than 0."

    create_tmp_dir("records_tmp")
    create_tmp_dir("trained_models_tmp")
    create_tmp_dir("augmented_datasets_tmp")

    algo_config = _get_algo_config(algo) if run_mode == "imitation" else ["none_algorithm"]

    if run_mode == "teleoperation":
        if record_path is None:
            now = datetime.now()
            formatted_now = now.strftime("%Y-%m-%d_%H-%M-%S")
            record_path = f"{current_file_path}/records_tmp/OpenLoong_{formatted_now}.hdf5"
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

    if args.ctrl_device == 'vr':
        ctrl_device = ControlDevice.VR
    else:
        print("Invalid control device! Please input 'xbox' or 'keyboard'.")
        sys.exit(1)

    max_episode_steps = int(record_time / REALTIME_STEP)
    print(f"Run episode in {max_episode_steps} steps as {record_time} seconds.")

    # 启动 Monitor 子进程
    ports = [
        7070, 7080, 7090,        # Agent1
        # 8070, 8080, 8090,        # Agent2
    ]
    monitor_processes = []
    for port in ports:
        process = start_monitor(port=port, project_root=project_root)
        monitor_processes.append(process)

    for config in algo_config:
        run_example(orcagym_addr,
                    agent_names,
                    pico_ports,
                    record_path,
                    run_mode,
                    action_type,
                    action_step,
                    config,
                    ctrl_device,
                    max_episode_steps,
                    playback_mode,
                    rollout_times,
                    ckpt_path,
                    augmented_sacle,
                    augmented_rounds,
                    teleoperation_rounds,
                    sample_range,
                    realtime_playback,
                    current_file_path,
                    task_config=task_config)

    # 终止 Monitor 子进程
    for process in monitor_processes:
        terminate_monitor(process)