import math
import os
import shutil
import sys
import time
import subprocess
import signal


from typing import Any, Dict
import uuid
from flask import g, json
import gymnasium as gym
from gymnasium.envs.registration import register
from datetime import datetime, timedelta, timezone

import h5py
from orca_gym.environment.orca_gym_env import RewardType
from orca_gym.robomimic.dataset_util import DatasetWriter, DatasetReader
from orca_gym.sensor.rgbd_camera import Monitor, CameraWrapper
from envs.manipulation.dual_arm_env import DualArmEnv, ControlDevice
from examples.imitation.train_policy import train_policy
from examples.imitation.test_policy import create_env, rollout
from orca_gym.utils.dir_utils import create_tmp_dir
from robomimic.utils.file_utils import maybe_dict_from_checkpoint
from robomimic.utils.train_utils import run_rollout
import orca_gym.utils.rotations as rotations
from envs.manipulation.dual_arm_env import TaskStatus
from envs.manipulation.dual_arm_env import ActionType
import re
import gymnasium as gym
from envs.manipulation.dual_arm_env import ControlDevice, RunMode, DualArmEnv
from envs.manipulation.dual_arm_robot import DualArmRobot
import numpy as np
import yaml

ENV_NAME = "DualArmEnv"
ENV_ENTRY_POINT = {
    "DualArmEnv": "envs.manipulation.dual_arm_env:DualArmEnv"
}

TIME_STEP = 0.001                       # 1000 Hz for physics simulation
FRAME_SKIP = 20                         
REALTIME_STEP = TIME_STEP * FRAME_SKIP  # 50 Hz for python program loop
CONTROL_FREQ = 1 / REALTIME_STEP        # 50 Hz for OSC controller computation

RGB_SIZE = (1280, 720)
CAMERA_CONFIG = {
    "camera_head": 7070,
    # "camera_wrist_r": 7080,
    # "camera_wrist_l": 7090,
}
_light_counter = 0
_LIGHT_SWITCH_PERIOD = 20  # 每 20 次 reset 才切一次光
INIT_SCENE_TEXT = {
    "shop":    ("一个机器人站在柜台前",  "A robot stands in front of a counter."),
    "yaodian": ("一个机器人站在药柜前",  "A robot stands in front of a medicine cabinet."),
    "kitchen": ("一个机器人站在灶台前",  "A robot stands in front of a stove."),
    "jiazi":   ("一个机器人站在货架前",  "A robot stands in front of a shelf."),
}
_light_counter = 0
_LIGHT_SWITCH_PERIOD = 20  # 每 20 次 reset 才切一次光

# 顶部：维护一个简单的中英映射


OBJ_CN = {
    "can": "罐子",
    "bottle_red":"红色瓶子",
    "bottle_blue":"蓝色瓶子",
    "jar_01": "罐子",
    "salt": "盐罐",
    "basket": "篮子",
    "medicinechest": "药柜",
    "niuhuangwan": "牛黄丸",
    "qinghuopian": "清火片",
    "orangebox" : "橙色药盒",
    "pipalu": "枇杷露",
    "xiaokepian": "消咳片",
    "bow_yellow_kps": "黄色碗",
    "coffeecup_white_kps": "白色咖啡杯",
    "basket_kitchen_01" : "篮子",
    "shoppingtrolley_01" : "购物手推车"
}

with open("camera_config.yaml", "r") as f:
    cam_cfg_all = yaml.safe_load(f)
default_cfg = cam_cfg_all["default"]
scene_cfgs = cam_cfg_all["scenes"]

# —— 生成对应场景下的 CAMERA_STATIC_CFG ——
def get_camera_static_cfg(level_name: str):
    scene = scene_cfgs.get(level_name)
    if scene is None:
        raise KeyError(f"No camera config for scene '{level_name}'")
    cfg = {}
    for cam_name, base in default_cfg.items():
        extr = scene.get(cam_name)
        if extr is None:
            raise KeyError(f"No scene extrinsics for camera '{cam_name}' in scene '{level_name}'")
        cfg[cam_name] = {
            **base,
            "translation": base["translation"],       # 原始外参（机器人坐标系）
            "rotation":    base["rotation"],          # 原始外参（机器人坐标系）
            "translation_world": extr["translation"], # 对齐外参（世界坐标系）
            "rotation_world":    extr["rotation"],    # 对齐外参（世界坐标系）
        }
    return cfg

# 3) dump_static_camera_params，写出三种 JSON

def dump_static_camera_params(uuid_dir: str,
                              level_name: str
                              ):
    CAMERA_STATIC_CFG = get_camera_static_cfg(level_name)
    out_dir = os.path.join(uuid_dir, "parameters", "camera")
    os.makedirs(out_dir, exist_ok=True)

    for name, cfg in CAMERA_STATIC_CFG.items():
        # —— 内参 ——
        intrinsic = {
            "manufacturer": "openverse",
            "mode":         "Mujoco Camera Capture",
            "SN":           "CA85BDD5-C631-4C52-853C-06D654AE7E4D",
            "fps":          30,
            "width":        cfg["width"],
            "height":       cfg["height"],
            "intrinsic": {
                "fx": (cfg["height"]/2) / math.tan(math.radians(cfg["fovy"]/2)),
                "fy": (cfg["height"]/2) / math.tan(math.radians(cfg["fovy"]/2)),
                "ppx": cfg["width"]/2,
                "ppy": cfg["height"]/2,
                "distortion_model": "plumb_bob",
                "k1": 0, "k2": 0, "k3": 0, "p1": 0, "p2": 0
            }
        }
        # —— 原始 extrinsic（机器人坐标系） ——
        extrinsic = {
            "translation_vector": cfg["translation"],
            "rotation_euler":     cfg["rotation"],
        }
        # —— 对齐 extrinsic（世界坐标系） ——
        extrinsic_aligned = {
            "translation_vector": cfg["translation_world"],
            "rotation_euler":     cfg["rotation_world"],
        }

        # 写文件
        with open(os.path.join(out_dir, f"{name}_intrinsic_params.json"),   "w", encoding="utf-8") as f:
            json.dump(intrinsic, f, ensure_ascii=False, indent=2)
        with open(os.path.join(out_dir, f"{name}_extrinsic_params.json"),   "w", encoding="utf-8") as f:
            json.dump(extrinsic, f, ensure_ascii=False, indent=2)
        with open(os.path.join(out_dir, f"{name}_extrinsic_params_aligned.json"), "w", encoding="utf-8") as f:
            json.dump(extrinsic_aligned, f, ensure_ascii=False, indent=2)


def normalize_key(key: str) -> str:
    """去掉多余标点，统一小写"""
    return re.sub(r'[^\w]', '', key).lower()

def eng2cn(instruction_en: str) -> str:
    """
    把类似 "level: shop object: bottle_blue to goal: basket" 或
    "put bottle_blue into basket" 之类的英文指令，翻成中文。
    """
    text = instruction_en.strip().lower()
    
    # 1) 先试“object … to goal …” 的结构
    m = re.search(r'object[:\s]+([\w_]+)\s+to\s+goal[:\s]+([\w_]+)', text)
    if m:
        obj_key  = normalize_key(m.group(1))
        goal_key = normalize_key(m.group(2))
        obj_cn   = OBJ_CN.get(obj_key, obj_key)
        goal_cn  = OBJ_CN.get(goal_key, goal_key)
        return f"将{obj_cn}放入{goal_cn}中"
    
    # 2) 再试 put … into … 结构
    m = re.search(r'put\s+([\w_]+)\s+into\s+([\w_]+)', text)
    if m:
        obj_key  = normalize_key(m.group(1))
        goal_key = normalize_key(m.group(2))
        obj_cn   = OBJ_CN.get(obj_key, obj_key)
        goal_cn  = OBJ_CN.get(goal_key, goal_key)
        return f"将{obj_cn}放入{goal_cn}中"
    
    # 3) 再试 move … to … 结构
    m = re.search(r'move\s+([\w_]+)\s+to\s+([\w_]+)', text)
    if m:
        obj_key  = normalize_key(m.group(1))
        goal_key = normalize_key(m.group(2))
        obj_cn   = OBJ_CN.get(obj_key, obj_key)
        goal_cn  = OBJ_CN.get(goal_key, goal_key)
        return f"将{obj_cn}移动到{goal_cn}前"
    
    # 4) 回退：保留原文
    return instruction_en

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


# def teleoperation_episode(env : DualArmEnv, cameras : list[CameraWrapper], rgb_size : tuple = (256, 256), action_step : int = 1):
def teleoperation_episode(env : DualArmEnv, cameras : list[CameraWrapper], dataset_writer : DatasetWriter,rgb_size : tuple = (256, 256), action_step : int = 1):
    """_summary_

    Args:
        env (DualArmEnv): envairoment instance for franka robot manipulation task
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
    spawn_scene(env, env._config)  # 初始化场景
    obs, info = env.reset(seed=42)
    obs_list = {obs_key: list([]) for obs_key, obs_data in obs.items()}
    reward_list = []
    done_list = []
    info_list = []    
    terminated_times = 0
    in_goal_list = [] 
    # camera_frames = {camera.name: [] for camera in cameras}
    camera_frame_index = []
    timestep_list = []
    action_step_taken = 0
    saving_mp4 = False
    while True:
        start_time = datetime.now()

        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        target_obj, target_goal = get_target_object_and_goal(info)
    
        env.render()
        task_status = info['task_status']
        
        action_step_taken += 1
        if action_step_taken >= action_step:        
            action_step_taken = 0
            if task_status in [TaskStatus.SUCCESS, TaskStatus.FAILURE, TaskStatus.BEGIN]:
                if task_status == TaskStatus.BEGIN:
                    if saving_mp4 == False:
                        dataset_writer.set_UUIDPATH()
                        mp4_save_path = dataset_writer.get_mp4_save_path()

                        env.begin_save_video(mp4_save_path)
#                       #  print("get_mp4_save_path..........................:",mp4_save_path)
#                         env.unwrapped.begin_save_video(mp4_save_path)

                        saving_mp4 = True
                        camera_frame_index.append(env.get_current_frame())
                        
                    else:
                        camera_frame_index.append(env.get_current_frame())
                
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
                        pos, _, _ = env.get_body_xpos_xmat_xquat([body_name])
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
                    env.stop_save_video()
                    saving_mp4 = False
                elif task_status == TaskStatus.FAILURE:

#                     env.stop_save_video()
#                     saving_mp4 = False

                    # env.unwrapped.stop_save_video()
                    env.stop_save_video()
                    #通过os接口删除路径

                    rmpath = dataset_writer.get_UUIDPath()
                    print(f"删除目录: {rmpath}")
                    if os.path.exists(rmpath) and os.path.isdir(rmpath):
                        shutil.rmtree(rmpath)
                  #      print(f"目录及其内容已删除: {rmpath}")
           
                    saving_mp4 = False              

                for obs_key, obs_data in obs.items():
                    obs_list[obs_key].append(obs_data)
                    
                reward_list.append(reward)
                done_list.append(0 if not terminated else 1)
                info_list.append(info)
                terminated_times = terminated_times + 1 if terminated else 0
                timestep_list.append(info['time_step'])
                

        if terminated_times >= 5 or truncated:
            # return obs_list, reward_list, done_list, info_list, camera_frames, timestep_list,in_goal_list
            return obs_list, reward_list, done_list, info_list, camera_frame_index, timestep_list,in_goal_list

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

def append_task_info_json(
        json_path,
        episode_id: str,
        level_name: str,
        sub_scene_name: str,
        language_instruction_cn: str,
        language_instruction_en: str,
        action_config: list[dict],
        data_gen_mode: str = "simulation",
        sn_code: str = "A2D0001AB00029",
        sn_name: str = "青龙"):
    """把一条 episode 的元信息追加到同一个 task_info.json 里。"""
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            all_eps = json.load(f)
    except FileNotFoundError:
        all_eps = []

    scene_key = "shop" if level_name == "jiazi" else level_name
    init_cn, init_en = INIT_SCENE_TEXT.get(scene_key, ("", ""))

    episode = {
        "episode_id": episode_id,
        "scene_name": scene_key.title(),
        "sub_scene_name": sub_scene_name,
        "init_scene_text": init_cn,
        "english_init_scene_text": init_en,
        "task_name": language_instruction_cn,
        "english_task_name": language_instruction_en,
        "data_type": "常规",
        "episode_status": "approved",
        "data_gen_mode": data_gen_mode,
        "sn_code": sn_code,
        "sn_name": sn_name,
        "label_info": {
            "action_config": action_config,
            "key_frame": []
        }
    }

    all_eps.append(episode)
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(all_eps, f, ensure_ascii=False, indent=4)

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
    # ----------------- 新增 JSON 写入 -----------------
    # 1) HDF5 文件路径 & 场景目录
    h5_path   = dataset_writer.basedir                      # e.g. "yaodian/dual_arm_XXXX.hdf5"
    scene_dir = os.path.dirname(h5_path)
    start_frame = 0
    max_len = max(len(frames) for frames in (camera_frames.values() if isinstance(camera_frames, dict) else [camera_frames]))

    end_frame = max_len - 1

    CST = timezone(timedelta(hours=8))
    timestamp_cst = datetime.now(CST).isoformat()

    # 2) 公开接口拿注册进来的 task_config
    task_cfg   = dataset_writer._env_args.get("task_config", {})
    text = info_list[0]["language_instruction"]
    m = re.search(r'level[:\s]+([\w_]+)', text, re.IGNORECASE)
    level_name = m.group(1).lower() if m else "unknown"
    # 然后同上，用 level_name 去拿描述、拼文件名……
    episode_id = dataset_writer.experiment_id

    # 3) 场景名／子场景／初始文本
    scene_name     = "shop" if level_name == "jiazi" else level_name
    sub_scene_name = scene_name

    # 4) JSON 文件固定名
    json_path = os.path.join(scene_dir, "task_info.json")

    # 5) 英文任务名：如果 info_list 里有 en 字段就用它
    language_instruction = info_list[0].get("language_instruction")
    lang_cn = eng2cn(language_instruction)

    action_config = [{
        "start_frame": start_frame,
        "end_frame": end_frame,
        "timestamp_cst": timestamp_cst,
        "skill": "pick and place",                # 或者其他你想用的内部标识
        "action_text": lang_cn,
        "english_action_text": language_instruction
    }]
    
    append_task_info_json(
        json_path=json_path,
        episode_id=episode_id,
        level_name=level_name,
        sub_scene_name=sub_scene_name,
        language_instruction_cn=lang_cn,
        language_instruction_en=language_instruction,
        action_config=action_config,
        data_gen_mode="simulation",
    )
    uuid_dir   = dataset_writer.get_UUIDPath()
    dump_static_camera_params(uuid_dir, level_name)



def do_teleoperation(env, 
                     dataset_writer : DatasetWriter, 
                     teleoperation_rounds : int, 
                     cameras : list[CameraWrapper], 
                     rgb_size : tuple = (256, 256),
                     action_step : int = 1,):    
    
    current_round = 1
    
    for camera in cameras:
        camera.start()
    
    # while True:
    #     # obs_list, reward_list, done_list, info_list, camera_frames, timestep_list,in_goal_list = teleoperation_episode(env, cameras, rgb_size, action_step)
    #     obs_list, reward_list, done_list, info_list, camera_frame_index, timestep_list,in_goal_list = teleoperation_episode(env, cameras, dataset_writer, rgb_size, action_step)
    #     last_done = (len(done_list)>0 and done_list[-1]==1)
    #     last_in_goal = (len(in_goal_list)>0 and in_goal_list[-1])
    #     save_record = last_done and last_in_goal
    #     task_result = "Success" if save_record else "Failed"
    #     # save_record, exit_program = user_comfirm_save_record(task_result, current_round, teleoperation_rounds)
    #     save_record = task_result == "Success"
    #     exit_program = False
    #     if save_record:
    #         print(f"Round {current_round} / {teleoperation_rounds}, Task is {task_result}!")
    #         current_round += 1
            
    #         # if obs_camera:
    #         #     for camera in cameras:
    #         #         obs_list[camera.name] = camera_frames[camera.name]
    #         #         camera_frames[camera.name] = []         
    #         #     empty_camera_frames = {}
    #         #     add_demo_to_dataset(dataset_writer, obs_list, reward_list, done_list, info_list, empty_camera_frames, timestep_list, info_list[0]["language_instruction"])
    #         # else:
    #             # add_demo_to_dataset(dataset_writer, obs_list, reward_list, done_list, info_list, camera_frames, timestep_list, info_list[0]["language_instruction"])
    #         add_demo_to_dataset(dataset_writer, obs_list, reward_list, done_list, info_list, camera_frame_index, timestep_list, info_list[0]["language_instruction"])
    #         # dataset_writer.set_file_path()
    #     if exit_program or current_round > teleoperation_rounds:
    #         break
        # 初始化基础DatasetWriter（仅一次）
   # lldir = "records_tmp/" level
    # base_writer = DatasetWriter(
    #     base_dir= "records_tmp/shoprpp/", # type: ignore
    #     env_name="shop",
    #     env_version="1.0"
    # )
    while True:
        obs_list, reward_list, done_list, info_list, camera_frame_index, timestep_list, in_goal_list = teleoperation_episode(
        env, cameras, dataset_writer, rgb_size, action_step
    )
    
        last_done = (len(done_list) > 0 and done_list[-1] == 1)
        last_in_goal = (len(in_goal_list) > 0 and in_goal_list[-1])
        save_record = last_done and last_in_goal
        task_result = "Success" if save_record else "Failed"
        save_record = task_result == "Success"
        exit_program = False
    
        if save_record:
            print(f"Round {current_round} / {teleoperation_rounds}, Task is {task_result}!")
            current_round += 1
            
            # # 关键修改: 创建全新的DatasetWriter实例(而不是修改现有实例)
            # dataset_writer = DatasetWriter(
            #     base_dir="records_tmp/shoprpp",
            #     env_name="shop",
            #     env_version="1.0"
            # )
            #dataset_writer.set_UUIDPATH()
            # 以下是您原有的保存代码(保持不变)
            add_demo_to_dataset(dataset_writer, obs_list, reward_list, done_list, info_list, camera_frame_index, timestep_list, info_list[0]["language_instruction"])
            #dataset_writer.set_()
        if exit_program or current_round > teleoperation_rounds:
            break
        
    for camera in cameras:
        camera.stop()

def playback_episode(env : DualArmEnv, 
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

def all_object_joints_exist(cfg, model) -> bool:
    """
    检查 cfg 中定义的 object_joints 是否都存在于 model 的 joint 字典中。
    """
    expected_joints = cfg.get("object_joints", [])
    model_joints = model.get_joint_dict().keys()

    missing = [j for j in expected_joints if not any(j in mj for mj in model_joints)]
    
    if missing:
        print("[Check] Missing joints:")
        for j in missing:
            print(f"  - {j} (expected but not found in model)")
        return False

    return True

def spawn_scene(env: DualArmEnv, task_config: Dict[str, Any]) -> None:
    global _light_counter

    env._task.load_config(task_config)

    # 初次初始化，什么都没有，先把object创建出来
    if not all_object_joints_exist(task_config, env.model):
        env._task.publish_scene()   # 清空当前缓存
        env._task.generate_actors(random_actor=False)
        env._task.publish_scene()
        time.sleep(1)  # 等待场景加载完成


    # 满足light切换条件，从新创建一次场景
    if env._config.get("random_light", False) and (_light_counter % _LIGHT_SWITCH_PERIOD == 0):
        actor_idxs = env._task.generate_actors(random_actor=False)
        light_idxs = env._task.generate_lights()
        env._task.publish_scene()

        # 先publish scene，确保所有actor和light都被创建，然后再设置
        if env._task.random_actor_color:
            for idx in actor_idxs:
                env._task.set_actor_material(env._task.actors[idx])

        for idx in light_idxs:
            env._task.set_light_info(env._task.lights[idx])
        env.mj_forward()
        env.render()
        time.sleep(1)  # 等待场景加载完成

    _light_counter += 1

def reset_playback_env(env: DualArmEnv, demo_data, sample_range=0.0):
    if "objects" in demo_data:
        env.objects = demo_data["objects"]        # 结构化物体信息，reset_model会用

    # 场景初始化
    spawn_scene(env, env._config)
    obs, info = env.reset(seed=42)

    # 如果需要重新采样目标物体位置
    if sample_range > 0.0:
        replace_objects(env, demo_data) # 先还原，获得物品的原始旋转值
        resample_objects(env, demo_data, sample_range)
    else:
        replace_objects(env, demo_data)

    obs = env._get_obs().copy()
    info = {
        "object": demo_data['objects'],
        "goal":   demo_data.get('goals')
    }
    return obs, info


def get_target_object_and_goal(demo: dict) -> tuple:
    lang_instr = demo.get("language_instruction", b"")
    if isinstance(lang_instr, (bytes, bytearray)):
        lang_instr = lang_instr.decode("utf-8")
    obj_match = re.search(r'object:\s*([^\s]+)', lang_instr)
    goal_match = re.search(r'goal:\s*([^\s]+)', lang_instr)
    target_obj = obj_match.group(1) if obj_match else None
    target_goal = goal_match.group(1) if goal_match else None

    return target_obj, target_goal

def replace_objects(env: DualArmEnv, demo_data: dict) -> None:
    env.replace_objects(demo_data['objects'])
    if 'goals' in demo_data and demo_data['goals'] is not None:
        env.replace_goals(demo_data['goals'])
    env.mj_forward()

def resample_objects(env: DualArmEnv, demo: dict, sample_range: float) -> None:
    env._task.load_config(env._config)
    target_obj, _ = get_target_object_and_goal(demo)
    if target_obj:
        target_obj_joint_name = ""
        target_obj_position = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        demo_objects = demo.get("objects", [])
        for obj in demo_objects:
            joint_name = obj["joint_name"].decode('utf-8') if isinstance(obj["joint_name"], (bytes, bytearray)) else obj["joint_name"]
            if target_obj in joint_name:
                target_obj_joint_name = joint_name
                target_obj_position = np.array(obj["position"], dtype=np.float32)
                break

        # print("env._task.object_joints:", env._task.object_joints, "env._task.goal_joints:", env._task.goal_joints)
        # env._task.random_objs_and_goals(env)
        # print("env._task.object_joints:", env._task.object_joints, "env._task.goal_joints:", env._task.goal_joints)

        resample_success = False
        target_obj_position_delta = 0
        for _ in range(100):  # 尝试100次
            env._task.random_objs_and_goals(env, random_rotation=False, target_obj_joint_name=target_obj_joint_name)
            target_obj_joint_qpos = env.query_joint_qpos([target_obj_joint_name])[target_obj_joint_name]
            target_obj_position_delta = np.linalg.norm(target_obj_joint_qpos[:2] - target_obj_position[:2])
            # print(f"[Info] Resampling target object {target_obj} position: {target_obj_joint_qpos[:2]} (delta: {target_obj_position_delta})")
            if target_obj_position_delta < sample_range:
                resample_success = True
                break

        if not resample_success:
            print(f"Warning: Failed to resample target object {target_obj} within range {sample_range}. Using original position.")
        
        env.update_objects_goals(env._task.randomized_object_positions, env._task.randomized_goal_positions)

        print("[Info] Resampling target object", target_obj, "to delta:", target_obj_position_delta)
        pass


def calculate_transform_matrix(a, b, a1, b1):
    """
    计算从向量[a, b]到[a1, b1]的变换矩阵
    返回一个2x2 NumPy数组
    
    注意: 当输入向量为零向量时处理特殊情况
    """
    # 计算分母 D = a² + b²
    D = a**2 + b**2
    
    if D == 0:  # 输入向量是零向量
        if a1 == 0 and b1 == 0:
            # 返回单位矩阵 (但这不是唯一解)
            return np.eye(2)
        else:
            # 线性变换必须保持零向量不变
            raise ValueError("输入向量是零向量但输出向量非零 - 无效线性变换")
    
    # 使用公式计算矩阵元素
    m11 = (a*a1 + b*b1) / D
    m12 = (b*a1 - a*b1) / D  # 注意这等同于公式中的 -(a*b1 - b*a1)/D
    
    # 简化形式创建矩阵 [m11, m12; -m12, m11]
    return np.array([[m11, m12], [-m12, m11]])

def apply_transform(matrix, x, y):
    """
    应用变换矩阵到向量[x, y]
    返回变换后的向量[x1, y1]
    """
    # 将输入向量转换为列向量
    vector = np.array([[x], [y]])
    
    # 应用矩阵乘法
    transformed = matrix @ vector
    
    # 返回结果为一维数组
    return transformed.flatten()

def _get_noscale_action_list(env: DualArmEnv, action_list: np.ndarray) -> np.ndarray:
    noscale_action_list = []
    for action in action_list:
        noscale_action = env.denormalize_action(action, env.env_action_range_min, env.env_action_range_max)
        noscale_action_list.append(noscale_action)
    return np.array(noscale_action_list)

def _get_scaled_action_list(env: DualArmEnv, noscale_action_list: np.ndarray) -> np.ndarray:
    scaled_action_list = []
    for noscale_action in noscale_action_list:
        scaled_action = env.normalize_action(noscale_action, env.env_action_range_min, env.env_action_range_max)
        scaled_action_list.append(scaled_action)
    return np.array(scaled_action_list)

def _get_eef_xy_from_action(env: DualArmEnv, noscale_action_list: np.ndarray) -> np.ndarray:
    xy_list = np.zeros((0, 4), dtype=np.float32)
    robot: DualArmRobot = env._agents[env._agent_names[0]]
    for noscale_action in noscale_action_list:
        left_eef_action = robot._action_B_to_action(noscale_action[:6])
        right_eef_action = robot._action_B_to_action(noscale_action[14:20])
        xy = np.concatenate([left_eef_action[:2], right_eef_action[:2]]).flatten()
        xy_list = np.concatenate([xy_list, [xy]])

    return xy_list

def _set_eef_xy_to_action(env: DualArmEnv, noscale_action_list: np.ndarray, xy_list: np.ndarray) -> np.ndarray:
    robot: DualArmRobot = env._agents[env._agent_names[0]]
    
    for i, noscale_action in enumerate(noscale_action_list):
        left_eef_action = robot._action_B_to_action(noscale_action[:6])
        right_eef_action = robot._action_B_to_action(noscale_action[14:20])
        left_eef_action[:2] = xy_list[i][:2]
        right_eef_action[:2] = xy_list[i][2:]
        noscale_action[:6] = robot._action_to_action_B(left_eef_action)
        noscale_action[14:20] = robot._action_to_action_B(right_eef_action)

    return noscale_action_list

def _get_target_xy(env: DualArmEnv, demo_data: dict, target_obj: str) -> tuple[np.ndarray, np.ndarray]:
    target_obj_full_name = env.body(target_obj) + "_joint"
    env_objects = env.objects
    demo_objects = demo_data.get("objects", [])
    
    # print("Demo Objects: ", demo_objects)
    # print("Target Object: ", target_obj_full_name)
    # print("env_objects: ", env_objects)

    def _get_object_info(obj_name: str, objects: list[tuple]) -> dict:
        for obj in objects:
            name = obj[0]
            # print("Checking object name: ", name)
            if isinstance(name, (bytes, bytearray)):
                name = name.decode('utf-8')
            if name == obj_name:
                return {
                    "joint_name": name,
                    "position": np.array(obj[1], dtype=np.float32),
                    "orientation": np.array(obj[2], dtype=np.float32)
                }
            
        print(f"Warning: Object '{obj_name}' not found in the provided objects list.")
        return {}
    
    target_obj_demo_info = _get_object_info(target_obj_full_name, demo_objects)
    target_obj_env_info = _get_object_info(target_obj_full_name, env_objects)

    # print("Target Object Demo Info: ", target_obj_demo_info)
    # print("Target Object Env Info: ", target_obj_env_info)

    demo_xy = target_obj_demo_info.get("position", np.zeros(3, dtype=np.float32))[:2]
    env_xy = target_obj_env_info.get("position", np.zeros(3, dtype=np.float32))[:2]
    return demo_xy, env_xy

def _find_closest_step(xy_list: np.ndarray, target_xy: np.ndarray) -> int:
    closest_step_id = 0
    closest_step_delta = float('inf')
    for i, xy in enumerate(xy_list):
        left_eef_xy = xy[:2]
        right_eef_xy = xy[2:]
        delta_left_xy = np.linalg.norm(left_eef_xy - target_xy)
        delta_right_xy = np.linalg.norm(right_eef_xy - target_xy)
        if delta_left_xy < 0.05 or delta_right_xy < 0.05:
            # print("Env: close to target object, left: ", delta_left_xy, " right: ", delta_right_xy, "in action: ", i)
            if delta_left_xy < closest_step_delta or delta_right_xy < closest_step_delta:
                closest_step_delta = min(delta_left_xy, delta_right_xy)
                closest_step_id = i

    return closest_step_id

def _transform_xy_list(source_xy_list: np.ndarray, source_xy: np.ndarray, target_xy: np.ndarray, skip_steps: int) -> np.ndarray:
    transform_matrix = calculate_transform_matrix(
        source_xy[0], source_xy[1], target_xy[0], target_xy[1]
    )
    print("Transform Matrix: ", transform_matrix)

    target_xy_list = np.zeros_like(source_xy_list)
    for i, xy in enumerate(source_xy_list):
        if i <= skip_steps:
            target_xy_list[i] = xy
            continue
        left_eef_xy = xy[:2]
        right_eef_xy = xy[2:]
        transformed_left_xy = apply_transform(transform_matrix, left_eef_xy[0], left_eef_xy[1])
        transformed_right_xy = apply_transform(transform_matrix, right_eef_xy[0], right_eef_xy[1])
        target_xy_list[i] = np.concatenate([transformed_left_xy, transformed_right_xy])

    return target_xy_list

def resample_actions(
        env: DualArmEnv, 
        demo_data: dict,
    ) -> np.ndarray:
    """
    两个关注点：1. 夹爪与物体最接近的点，是第一个目标点， 2. 样本中最后一个点是第二个目标点

    首先根据第一个目标点重采样后，与原始数据之间的变化，建立一个变换矩阵（平移+缩放）
    对所有轨迹应用变换矩阵，则夹爪可以移动到新的目标位置
    这时最终目标点也会有变化。因此计算第二个目标点的变换矩阵，
    然后对分界点后的轨迹，再次应用变换矩阵，获得第二段轨迹

    最终效果：夹爪到达新物体位置，然后再到达原轨迹中的最终位置
    """
    demo_action_list = demo_data['actions']
    target_obj, _ = get_target_object_and_goal(demo_data)

    if target_obj is None:
        print("Warning: No target object found in the demo data.")
        return demo_action_list
    
    noscale_action_list = _get_noscale_action_list(env, demo_action_list)
    demo_xy_list = _get_eef_xy_from_action(env, noscale_action_list)
    # print("Get xy_list from demo actions: ", demo_xy_list, "shape: ", demo_xy_list.shape)
    # sys.exit(0)

    demo_xy, env_xy = _get_target_xy(env, demo_data, target_obj)
    final_demo_xy = demo_xy_list[-1]
    print("Demo XY: ", demo_xy, "Env XY: ", env_xy, "Final Demo XY: ", final_demo_xy)

    closest_demo_step_id = _find_closest_step(demo_xy_list, demo_xy)
    print("Closest Demo Step ID: ", closest_demo_step_id)

    env_xy_list = _transform_xy_list(demo_xy_list, demo_xy, env_xy, 0)
    final_env_xy = env_xy_list[-1]
    print("Final Env XY: ", final_env_xy)

    closest_env_step_id = _find_closest_step(env_xy_list, env_xy)
    print("Closest Env Step ID: ", closest_env_step_id)

    env_xy_list = _transform_xy_list(env_xy_list, final_env_xy, final_demo_xy, closest_demo_step_id)
    new_final_env_xy = env_xy_list[-1]
    print("New Final Env XY: ", new_final_env_xy, "Demo XY: ", final_demo_xy, "distance: ", np.linalg.norm(new_final_env_xy - final_demo_xy))
    

    noscale_action_list = _set_eef_xy_to_action(env, noscale_action_list, env_xy_list)

    scaled_action_list = _get_scaled_action_list(env, noscale_action_list)

    return scaled_action_list
    
def do_playback(env : DualArmEnv, 
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
        env.objects = demo_data['objects']
        reset_playback_env(env, demo_data)
        playback_episode(env, action_list, done_list, action_step, realtime)
        time.sleep(1)

def augment_episode(env : DualArmEnv, 
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

    target_obj, target_goal = get_target_object_and_goal(demo_data)

    in_goal=False
    

    # 轨迹增广需要重新采样动作
    if sample_range > 0.0:
        action_list = resample_actions(env, demo_data)
    else:
        action_list = demo_data['actions']

    action_index_list = list(range(len(action_list)))
    holdon_action_index_list = action_index_list[-1] * np.ones(20, dtype=int)
    action_index_list = np.concatenate([action_index_list, holdon_action_index_list]).flatten()
    original_dones = np.array(demo_data['dones'], dtype=int)
    T = len(original_dones)
    
    for i in action_index_list:
        action = action_list[i]
        last_action = action_list[i - 1] if i > 0 else action
    
        action_chunk = np.array([
            last_action + (action - last_action) * (j / action_step)
            for j in range(action_step)
        ], dtype=np.float32)
    
        if noise_scale > 0.0:
            noise = np.random.normal(0, noise_scale, action_chunk.shape)
            action_chunk += noise * np.abs(action_chunk)
    
        action_chunk = np.clip(action_chunk, -1.0, 1.0)
    
        for j in range(action_step):
            if realtime:
                start_time = datetime.now()
    
            obs, reward, terminated, truncated, info = env.step(action_chunk[j])
            terminated_times = terminated_times + 1 if terminated else 0
            timestep_list.append(env.gym.data.time)
    
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
                pos, _, _ = env.get_body_xpos_xmat_xquat([body_name])
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
            # time.sleep(0.01)    # wait for camera to get new frame
            for camera in cameras:
                camera_frame, _ = camera.get_frame(format='rgb24', size=rgb_size)
                camera_frames[camera.name].append(camera_frame)

        for obs_key, obs_data in obs.items():
            obs_list[obs_key].append(obs_data)
            
        reward_list.append(reward)
        done_list.append(int(original_dones[i]) if i < T else 1)
        info_list.append(info)

        

        if terminated_times >= 5 or truncated:
            return obs_list, reward_list, done_list, info_list, [], timestep_list,in_goal
        
    return obs_list, reward_list, done_list, info_list, [], timestep_list,in_goal
 

def do_augmentation(
        env : DualArmEnv, 
        cameras : list[CameraWrapper], 
        rgb_size : tuple,                    
        original_dataset_path : str, 
        augmented_dataset_path : str, 
        augmented_noise : float, 
        sample_range : float,
        augmented_rounds : int,
        action_step : int = 1
    ):
    
    realtime = False
    
    # Copy the original dataset to the augmented dataset
    dataset_reader = DatasetReader(file_path=original_dataset_path)
    dataset_writer = DatasetWriter(base_dir=os.path.dirname(augmented_dataset_path),
                                    env_name=dataset_reader.get_env_name(),
                                    env_version=dataset_reader.get_env_version(),
                                    env_kwargs=dataset_reader.get_env_kwargs())
    dataset_writer.set_UUIDPATH()

    for camera in cameras:
        camera.start()
    
    for round in range(augmented_rounds):    
        need_demo_count = dataset_reader.get_demo_count()
        done_demo_count = 0
            
        demo_names = dataset_reader.get_demo_names()

        for original_demo_name in demo_names:
            done = False
            trial_count = 0
            max_trials = 2
            while not done and trial_count < max_trials:
                demo_data = dataset_reader.get_demo_data(original_demo_name)
                env.objects = demo_data['objects']
                print("Augmenting original demo: ", original_demo_name)
                language_instruction = demo_data['language_instruction']
                
                obs_list, reward_list, done_list, info_list\
                    , camera_frames, timestep_list,in_goal = augment_episode(env, cameras,rgb_size,
                                                                    demo_data, noise_scale=augmented_noise, 
                                                                    sample_range=sample_range, realtime=realtime, 
                                                                    action_step=action_step)
                if  in_goal:
                    add_demo_to_dataset(dataset_writer, obs_list, reward_list, done_list, info_list, 
                                        camera_frames, timestep_list, language_instruction)
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
                augmented_noise : float,
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
            demo_action_type = dataset_reader.get_env_kwargs()["action_type"]
            env_name = dataset_reader.get_env_name()
            env_name = env_name.split("-OrcaGym-")[0]
            env_index = 0
            if task_config is None:
               task_config_dict = {}
            else:
                with open(task_config, 'r') as f:
                    task_config_dict = yaml.safe_load(f)
            env_id, kwargs = register_env(orcagym_addr, env_name, env_index, agent_names, pico_ports, RunMode.POLICY_NORMALIZED, demo_action_type, ctrl_device, max_episode_steps, sample_range, action_step, camera_config, task_config_dict)
            print("Registered environment: ", env_id)

            env = gym.make(env_id)
            env = env.unwrapped
            print("Starting simulation...")
            do_playback(env, dataset_reader, playback_mode, action_step, realtime_playback)

        elif run_mode == "teleoperation":
            env_name = ENV_NAME
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
            env = env.unwrapped
            print("Starting simulation...")
            kwargs["run_mode"] = RunMode.POLICY_NORMALIZED  # 此处用于训练的时候读取

            if RGB_SIZE is None:
                cameras = []
            else:
                cameras = [CameraWrapper(name=camera_name, port=camera_port) for camera_name, camera_port in camera_config.items()]


#             dataset_writer = DatasetWriter(file_path=record_path,
#                                         env_name=env_id,
#                                         env_version=env.get_env_version(),
#                                         env_kwargs=kwargs)


            # dataset_writer = DatasetWriter(file_path=record_path,
            #                             env_name=env_id,
            #                             env_version=env.unwrapped.get_env_version(),
            #                             env_kwargs=kwargs)
            dataset_writer = DatasetWriter(
                                    base_dir=os.path.dirname(record_path),           # 用 record_path 的父目录
                                    env_name=env_id,
                                    env_version=env.unwrapped.get_env_version(),
                                    env_kwargs=kwargs)


            do_teleoperation(env, dataset_writer, teleoperation_rounds,
                                                 cameras=cameras, rgb_size=RGB_SIZE, action_step=action_step,)
            dataset_writer.shuffle_demos()
            dataset_writer.finalize()

        elif run_mode == "imitation":
            dataset_reader = DatasetReader(file_path=record_path)
            env_name = dataset_reader.get_env_name()
            camera_config = dataset_reader.get_env_kwargs()["camera_config"]
            action_step = dataset_reader.get_env_kwargs()["action_step"]
            demo_action_type = dataset_reader.get_env_kwargs()["action_type"]
            env_name = env_name.split("-OrcaGym-")[0]
            env_index = 0
            env_id, kwargs = register_env(orcagym_addr, env_name, env_index, agent_names, pico_ports, RunMode.POLICY_NORMALIZED, demo_action_type, ctrl_device, max_episode_steps, sample_range, action_step, camera_config)
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
            demo_action_type = env_kwargs["action_type"]

            env_id, kwargs = register_env(orcagym_addr, env_name, env_index, agent_names, pico_ports, RunMode.POLICY_NORMALIZED, demo_action_type, ctrl_device, max_episode_steps, sample_range, action_step, camera_config)
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
            demo_action_type = dataset_reader.get_env_kwargs()["action_type"]

            if demo_action_type in [ActionType.END_EFFECTOR_OSC, ActionType.JOINT_MOTOR] and action_type == ActionType.JOINT_POS:
                raise ValueError(f"Augmentation mode: Action type {action_type} is conflicting with demo action type {demo_action_type}.")
            elif demo_action_type in [ActionType.END_EFFECTOR_IK, ActionType.JOINT_POS] and action_type == ActionType.JOINT_MOTOR:
                raise ValueError(f"Augmentation mode: Action type {action_type} is conflicting with demo action type {demo_action_type}.")
            
            if sample_range > 0.0 and action_type not in [ActionType.END_EFFECTOR_OSC, ActionType.END_EFFECTOR_IK]:
                raise ValueError(f"Augmentation mode: Action type {action_type} does not support sample range. Please use 'end_effector_osc' or 'end_effector_ik'.")

            env_name = env_name.split("-OrcaGym-")[0]
            env_index = 0
            if task_config is None:
               task_config_dict = {}
            else:
                with open(task_config, 'r') as f:
                    task_config_dict = yaml.safe_load(f)
            env_id, kwargs = register_env(orcagym_addr, env_name, env_index, agent_names, pico_ports, RunMode.POLICY_NORMALIZED, action_type, ctrl_device, max_episode_steps, sample_range, action_step, camera_config,task_config_dict)
            print("Registered environment: ", env_id)

            env = gym.make(env_id)
            env = env.unwrapped
            print("Starting simulation...")

            now = datetime.now()
            formatted_now = now.strftime("%Y-%m-%d_%H-%M-%S")
            agumented_dataset_file_path = f"{current_file_path}/augmented_datasets_tmp/augmented_dataset_{formatted_now}.hdf5"

            if RGB_SIZE is None:
                cameras = []
            else:
                cameras = [CameraWrapper(name=camera_name, port=camera_port) for camera_name, camera_port in camera_config.items()]

            do_augmentation(env, cameras, RGB_SIZE, record_path, agumented_dataset_file_path, augmented_noise, sample_range, augmented_rounds, action_step)
            print("Augmentation done! The augmented dataset is saved to: ", agumented_dataset_file_path)
        else:
            print("Invalid run mode! Please input 'teleoperation' or 'playback'.")

    finally:
        print("Simulation stopped")
        # if run_mode == "teleoperation":
        if run_mode == "teleoperation" and 'dataset_writer' in locals():
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


def run_dual_arm_sim(args, project_root : str = None, current_file_path : str = None):
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
    augmented_noise = args.augmented_noise
    augmented_rounds = args.augmented_rounds
    teleoperation_rounds = args.teleoperation_rounds
    sample_range = args.sample_range
    realtime_playback = args.realtime_playback
    level = args.level

    assert record_time > 0, "The record time should be greater than 0."
    assert teleoperation_rounds > 0, "The teleoperation rounds should be greater than 0."
    assert sample_range >= 0.0, "The sample range should be greater than or equal to 0."
    assert augmented_noise >= 0.0, "The augmented noise should be greater than or equal to 0."
    assert augmented_rounds > 0, "The augmented times should be greater than 0."

    create_tmp_dir("records_tmp")
    create_tmp_dir("trained_models_tmp")
    create_tmp_dir("augmented_datasets_tmp")

    algo_config = _get_algo_config(algo) if run_mode == "imitation" else ["none_algorithm"]

    if run_mode == "teleoperation":
        if record_path is None:
            now = datetime.now()
            formatted_now = now.strftime("%Y-%m-%d_%H-%M-%S")
            level_dir = os.path.join(current_file_path, "records_tmp", level)
            os.makedirs(level_dir, exist_ok=True)
            record_path = os.path.join(level_dir, f"dual_arm_{formatted_now}.hdf5")
            print(f"Auto-generated record path: {record_path}")
        if action_type not in [ActionType.END_EFFECTOR_OSC, ActionType.END_EFFECTOR_IK]:
            raise ValueError(f"Action type {action_type} is not supported in teleoperation mode. Please input 'end_effector_osc', 'end_effector_ik'.")
    if run_mode == "imitation" or run_mode == "playback" or run_mode == "augmentation":
        if record_path is None:
            raise ValueError("Please input the record file path.")
    if run_mode == "rollout":
        if ckpt_path is None:
            raise ValueError("Please input the model file path.")
    if run_mode not in ["teleoperation", "playback", "imitation", "rollout", "augmentation"]:
        raise ValueError("Invalid run mode! Please input 'teleoperation', 'playback', 'imitation', 'rollout' or 'augmentation'.")
    if action_type not in [ActionType.END_EFFECTOR_OSC, ActionType.END_EFFECTOR_IK, ActionType.JOINT_POS, ActionType.JOINT_MOTOR]:
        raise ValueError(f"Invalid action type! Please input 'end_effector_osc', 'end_effector_ik', 'joint_pos' or 'joint_motor'.")

    if args.ctrl_device == 'vr':
        ctrl_device = ControlDevice.VR
    else:
        raise ValueError("Invalid control device! Please input 'xbox' or 'keyboard'.")

    max_episode_steps = int(record_time / REALTIME_STEP)
    print(f"Run episode in {max_episode_steps} steps as {record_time} seconds.")

    # 启动 Monitor 子进程
    ports = [
        # 7070, 7080, 7090,        # Agent1
        # 8070, 8080, 8090,        # Agent2
        7070, 7090,        # Agent1
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
                    augmented_noise,
                    augmented_rounds,
                    teleoperation_rounds,
                    sample_range,
                    realtime_playback,
                    current_file_path,
                    task_config=task_config)

    # 终止 Monitor 子进程
    for process in monitor_processes:
        terminate_monitor(process)