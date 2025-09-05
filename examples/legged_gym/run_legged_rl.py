import os
import sys
import argparse
import time
import numpy as np
from datetime import datetime
import yaml
import json

current_file_path = os.path.abspath('')
project_root = os.path.dirname(os.path.dirname(current_file_path))

# 将项目根目录添加到 PYTHONPATH
if project_root not in sys.path:
    sys.path.append(project_root)


from envs.legged_gym.legged_config import LeggedEnvConfig, LeggedRobotConfig
from orca_gym.utils.dir_utils import create_tmp_dir
from scripts.scene_util import generate_height_map_file, clear_scene, publish_terrain, publish_scene

TIME_STEP = LeggedEnvConfig["TIME_STEP"]
FRAME_SKIP = LeggedEnvConfig["FRAME_SKIP"]
ACTION_SKIP = LeggedEnvConfig["ACTION_SKIP"]
EPISODE_TIME = LeggedEnvConfig["EPISODE_TIME_LONG"]

def export_config(config: dict, model_dir: str):
    agent_name = config['agent_name']
    agent_config = LeggedRobotConfig[agent_name]

    config['agent_config'] = agent_config

    # 输出到 json 文件
    with open(os.path.join(model_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=4)

def process_scene(
    orcagym_addresses: list[str],
    agent_name: str,
    agent_spawnable_name: str,
    agent_num: int,
    terrain_spawnable_names: list[str],
):
    # 清空场景
    clear_scene(
        orcagym_addresses=orcagym_addresses,
    )

    # 发布地形
    publish_terrain(
        orcagym_addresses=orcagym_addresses,
        terrain_spawnable_names=terrain_spawnable_names,
    )

    # 空场景生成高度图
    height_map_file = generate_height_map_file(
        orcagym_addresses=orcagym_addresses,
    )

    # 放置机器人
    publish_scene(
        orcagym_addresses=orcagym_addresses,
        agent_name=agent_name,
        agent_spawnable_name=agent_spawnable_name,
        agent_num=agent_num,
        terrain_spawnable_names=terrain_spawnable_names,
    )

    return height_map_file

def process_model_dir(
    config: dict, 
    run_mode: str, 
    ckpt: str, 
    subenv_num: int, 
    agent_num: int, 
    agent_name: str, 
    task: str
):
    create_tmp_dir("trained_models_tmp")

    if ckpt is not None:
        model_file = ckpt
        model_dir = os.path.dirname(model_file)
    elif run_mode == "training":
        formatted_now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        model_dir = f"./trained_models_tmp/{agent_name}_{subenv_num * agent_num}_{formatted_now}"
        os.makedirs(model_dir, exist_ok=True)
        model_file = os.path.join(model_dir, f"{agent_name}_{task}.zip")
        export_config(config, model_dir)
    else:
        raise ValueError("Invalid model file! Please provide a model file for testing / play.")

    return model_dir, model_file

def run_sb3_ppo_rl(
    config: dict,
    run_mode: str,
    ckpt: str,
    remote: str,
    visualize: bool,
):
    if remote is not None:
        orcagym_addresses = [remote]
    else:
        orcagym_addresses = config['orcagym_addresses']

    agent_name = config['agent_name']
    agent_spawnable_name = config['agent_spawnable_name']
    training_episode = config['training_episode']
    task = config['task']

    run_mode_config = config[run_mode]
    subenv_num = run_mode_config['subenv_num']
    agent_num = run_mode_config['agent_num']

    if visualize:
        render_mode = "human"
    else:
        render_mode = run_mode_config['render_mode']

    terrain_spawnable_names = run_mode_config['terrain_spawnable_names'][task]
    entry_point = 'envs.legged_gym.legged_gym_env:LeggedGymEnv'

    if task == 'rough_terrain' or task == 'no_action' or task == 'flat_terrain':
        max_episode_steps = int(1 / (TIME_STEP * FRAME_SKIP * ACTION_SKIP) * EPISODE_TIME)
    else:
        raise ValueError("Invalid task")

    total_steps = training_episode * subenv_num * agent_num * max_episode_steps

    model_dir, model_file = process_model_dir(
        config=config, 
        run_mode=run_mode, 
        ckpt=ckpt, 
        subenv_num=subenv_num, 
        agent_num=agent_num, 
        agent_name=agent_name, 
        task=task
    )

    height_map_file = process_scene(
        orcagym_addresses=orcagym_addresses,
        agent_name=agent_name,
        agent_spawnable_name=agent_spawnable_name,
        agent_num=agent_num,
        terrain_spawnable_names=terrain_spawnable_names,
    )

    import examples.legged_gym.scripts.sb3_ppo_vecenv_rl as sb3_rl

    if run_mode == "training":
        print("Start Training! task: ", task, " subenv_num: ", subenv_num, " agent_num: ", agent_num, " agent_name: ", agent_name)
        print("Total Steps: ", total_steps, "Max Episode Steps: ", max_episode_steps, " Frame Skip: ", FRAME_SKIP, " Action Skip: ", ACTION_SKIP)
        sb3_rl.train_model(
            orcagym_addresses=orcagym_addresses, 
            subenv_num=subenv_num, 
            agent_num=agent_num, 
            agent_name=agent_name, 
            agent_config=LeggedRobotConfig[agent_name],
            task=task, 
            entry_point=entry_point, 
            time_step=TIME_STEP, 
            max_episode_steps=max_episode_steps, 
            render_mode=render_mode,
            frame_skip=FRAME_SKIP, 
            action_skip=ACTION_SKIP,
            total_timesteps=total_steps, 
            model_file=model_file, 
            height_map_file=height_map_file, 
            curriculum_list=run_mode_config['curriculum_list'][task],
        )
    elif run_mode in ["testing", "play"]:
        print("Start Testing! Run mode: ", run_mode, "task: ", task, " subenv_num: ", subenv_num, " agent_num: ", agent_num, " agent_name: ", agent_name)
        print(" Total Steps: ", total_steps, "Max Episode Steps: ", max_episode_steps, " Frame Skip: ", FRAME_SKIP, " Action Skip: ", ACTION_SKIP)
        sb3_rl.test_model(
            orcagym_addresses=orcagym_addresses, 
            agent_num=agent_num, 
            agent_name=agent_name, 
            task=task, 
            run_mode=run_mode, 
            entry_point=entry_point, 
            time_step=TIME_STEP, 
            max_episode_steps=max_episode_steps, 
            render_mode=render_mode,
            frame_skip=FRAME_SKIP, 
            action_skip=ACTION_SKIP,
            model_file=model_file, 
            height_map_file=height_map_file,
            curriculum_list=run_mode_config['curriculum_list'][task],
        )  
  
    else:
        raise ValueError("Invalid run mode")


def run_rllib_appo_rl(
    config: dict,
    run_mode: str,
    ckpt: str,
    remote: str,
    visualize: bool,
):
    import examples.legged_gym.scripts.rllib_appo_rl as rllib_appo_rl
    import ray
    import torch

    # 在脚本开头调用
    if rllib_appo_rl.setup_cuda_environment():
        print("CUDA 环境验证通过")
    else:
        print("CUDA 环境设置失败，GPU 加速可能不可用")
    
    # 验证 PyTorch CUDA
    rllib_appo_rl.verify_pytorch_cuda()

    # 初始化Ray集群
    if 'ray_cluster_address' in config and config['ray_cluster_address']:
        print(f"连接到Ray集群: {config['ray_cluster_address']}")
        ray.init(
            # address=config['ray_cluster_address'],
            # ignore_reinit_error=True,
            # runtime_env={"working_dir": "."}
        )
    else:
        print("使用本地Ray实例")
        ray.init(
            ignore_reinit_error=True,
            # 确保GPU资源被正确注册
            num_gpus=torch.cuda.device_count() if torch.cuda.is_available() else 0
        )

    # 打印集群信息
    print(f"Ray集群状态: {ray.is_initialized()}")
    print(f"可用节点数量: {len(ray.nodes())}")
    print(f"可用资源: {ray.available_resources()}")
    
    # 检查GPU资源
    if torch.cuda.is_available():
        print(f"PyTorch检测到GPU数量: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    
    # 检测Ray集群中的GPU资源
    num_gpus_available = rllib_appo_rl.detect_ray_gpu_resources()
    print(f"Ray集群检测到的GPU数量: {num_gpus_available}")

    if remote is not None:
        orcagym_addresses = [remote]
    else:
        orcagym_addresses = config['orcagym_addresses']

    agent_name = config['agent_name']
    agent_spawnable_name = config['agent_spawnable_name']
    task = config['task']

    run_mode_config = config[run_mode]
    num_env_runners = run_mode_config['num_env_runners']
    num_envs_per_env_runner = run_mode_config['num_envs_per_env_runner']

    if num_env_runners == 0:
        num_env_runners = int(ray.available_resources()['CPU'] - 2) // 8 * 8    # 1 for ray scheduler, 1 for leaner

    if visualize:
        render_mode = "human"
    else:
        render_mode = run_mode_config['render_mode']

    terrain_spawnable_names = run_mode_config['terrain_spawnable_names'][task]

    model_dir, model_file = process_model_dir(
        config=config, 
        run_mode=run_mode, 
        ckpt=ckpt, 
        subenv_num=num_env_runners, 
        agent_num=num_envs_per_env_runner, 
        agent_name=agent_name, 
        task=task
    )

    height_map_file = process_scene(
        orcagym_addresses=orcagym_addresses,
        agent_name=agent_name,
        agent_spawnable_name=agent_spawnable_name,
        agent_num=32,   # 一个Mujoco Instance支持 32 个agent是最合理的，这是默认配置
        terrain_spawnable_names=terrain_spawnable_names,
    )

    import examples.legged_gym.scripts.rllib_appo_rl as rllib_appo_rl

    max_episode_steps = run_mode_config['max_episode_steps']
    total_steps = run_mode_config['iter'] * num_env_runners * num_envs_per_env_runner * max_episode_steps
    agent_num = 32
    subenv_num = (num_env_runners * num_envs_per_env_runner) // agent_num

    if run_mode == 'training':
        print("Start Training! task: ", task, " subenv_num: ", subenv_num, " agent_num: ", agent_num, " agent_name: ", agent_name, " iter: ", run_mode_config['iter'])
        print("Total Steps: ", total_steps, "Max Episode Steps: ", max_episode_steps, " Frame Skip: ", FRAME_SKIP, " Action Skip: ", ACTION_SKIP)
        print(f"环境运行器数量: {num_env_runners}, 每个运行器的环境数量: {num_envs_per_env_runner}")
        
        rllib_appo_rl.run_training(
            orcagym_addr=orcagym_addresses[0],
            env_name=config['env_name'],
            agent_name=agent_name,
            agent_config=LeggedRobotConfig[agent_name],
            task=task,
            max_episode_steps=run_mode_config['max_episode_steps'],
            num_env_runners=num_env_runners,
            num_envs_per_env_runner=num_envs_per_env_runner,
            num_gpus_available=num_gpus_available,
            async_env_runner=run_mode_config['async_env_runner'],
            iter=run_mode_config['iter'],
            total_steps=total_steps,
            render_mode=render_mode,
            height_map_file=height_map_file,
            frame_skip=FRAME_SKIP,
            action_skip=ACTION_SKIP,
            time_step=TIME_STEP,
        )
    elif run_mode == 'testing':
        if not ckpt:
            raise ValueError("Checkpoint path must be provided for testing.")
        rllib_appo_rl.test_model(
            checkpoint_path=ckpt,
            orcagym_addr=orcagym_addresses[0],
            env_name=config['env_name'],
            agent_name=agent_name,
            max_episode_steps=run_mode_config['max_episode_steps'],
            use_onnx_for_inference=False,
            explore_during_inference=False,
            render_mode=render_mode,
            async_env_runner=run_mode_config['async_env_runner'],
            height_map_file=height_map_file,
            task=task,
            frame_skip=FRAME_SKIP,
            action_skip=ACTION_SKIP,
            time_step=TIME_STEP,
        )
    else:
        raise ValueError("Invalid run mode. Use 'training' or 'testing'.")
    
    # 训练完成后关闭Ray
    if ray.is_initialized():
        ray.shutdown()

def run_rl(config: dict, run_mode: str, ckpt: str, remote: str, visualize: bool):
    if config['framework'] == 'sb3':
        run_sb3_ppo_rl(config, run_mode, ckpt, remote, visualize)
    elif config['framework'] == 'rllib':
        run_rllib_appo_rl(config, run_mode, ckpt, remote, visualize)
    else:
        raise ValueError("Invalid framework")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run legged RL.')
    parser.add_argument('--config', type=str, help='The path of the config file')
    parser.add_argument('--train', action='store_true', help='Train the model')
    parser.add_argument('--test', action='store_true', help='Test the model')
    parser.add_argument('--play', action='store_true', help='Play the model')
    parser.add_argument('--ckpt', type=str, help='The path to the checkpoint file for testing / play')
    parser.add_argument('--remote', type=str, help='[Optional] The remote address of the ORCA Lab Simulator. Example: 192.198.1.123:50051')
    parser.add_argument('--visualize', action='store_true', help='Visualize the training process')
    args = parser.parse_args()

    if args.config is None:
        raise ValueError("Config file is required")
    
    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    assert args.train or args.test or args.play, "Please specify one of --train, --test, or --play"
    assert not (args.train and args.test), "Please specify only one of --train, --test, or --play"
    assert not (args.train and args.play), "Please specify only one of --train, --test, or --play"
    assert not (args.test and args.play), "Please specify only one of --train, --test, or --play"

    if args.train:
        run_rl(config, 'training', args.ckpt, args.remote, args.visualize)
    elif args.test:
        run_rl(config, 'testing', args.ckpt, args.remote, args.visualize)
    elif args.play:
        run_rl(config, 'play', args.ckpt, args.remote, args.visualize)
    else:
        raise ValueError("Invalid run mode")

