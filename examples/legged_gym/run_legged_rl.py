import os
import sys
import argparse
import time
import numpy as np
from datetime import datetime
import yaml
import shutil
import uuid
from orca_gym.scene.orca_gym_scene import OrcaGymScene, Actor
from orca_gym.utils.rotations import euler2quat

current_file_path = os.path.abspath('')
project_root = os.path.dirname(os.path.dirname(current_file_path))

# 将项目根目录添加到 PYTHONPATH
if project_root not in sys.path:
    sys.path.append(project_root)


from envs.legged_gym.legged_config import LeggedEnvConfig, LeggedRobotConfig
import orca_gym.scripts.sb3_ppo_vecenv_rl as sb3_ppo_vecenv_rl
from orca_gym.utils.dir_utils import create_tmp_dir

TIME_STEP = LeggedEnvConfig["TIME_STEP"]
FRAME_SKIP = LeggedEnvConfig["FRAME_SKIP"]
ACTION_SKIP = LeggedEnvConfig["ACTION_SKIP"]
EPISODE_TIME = LeggedEnvConfig["EPISODE_TIME_LONG"]

def generate_height_map_file(
    orcagym_addresses: list[str],
    model_dir: str,
):
    print("=============> Generate height map file ...")

    # 调用 ../../orca_gym/tools/generate_height_map.py
    os.system(f"python ../../orca_gym/tools/height_map_generater.py --orcagym_addresses {orcagym_addresses[0]}")

    # 用UUID生成一个唯一的文件名，并重命名 height_map.npy
    height_map_file = os.path.join(model_dir, f"height_map_{uuid.uuid4()}.npy")
    os.makedirs(model_dir, exist_ok=True)
    shutil.move("height_map.npy", height_map_file)

    print("=============> Generate height map file done. Height map file: ", height_map_file)

    return height_map_file

def clear_scene(
    orcagym_addresses: list[str],
):
    print("=============> Clear scene ...")

    scene = OrcaGymScene(orcagym_addresses[0])
    scene.publish_scene()
    time.sleep(1)
    scene.close()
    time.sleep(1)

    print("=============> Clear scene done.")

def publish_scene(
    orcagym_addresses: list[str],
    agent_name: str,
    agent_spawnable_name: str,
    agent_num: int,
    terrain_spawnable_names: list[str],
):
    print("=============> Publish scene ...")
    scene = OrcaGymScene(orcagym_addresses[0])
    # 排列成一个方阵，每个机器人间隔1米
    sqrt_width = int(np.ceil(np.sqrt(agent_num)))  # 向上取整
    base_offset_x = -(sqrt_width) / 2
    base_offset_y = -(sqrt_width) / 2
    for i in range(agent_num):
        x_pos = (i % sqrt_width) + base_offset_x
        y_pos = (i // sqrt_width) + base_offset_y
        actor = Actor(
            name=f"{agent_name}_{i:03d}",
            spawnable_name=agent_spawnable_name,
            position=[x_pos, y_pos, 0],
            rotation=euler2quat([0, 0, 0]),
            scale=1.0,
        )
        scene.add_actor(actor)
        print(f"    =============> Add actor {agent_name}_{i:03d} ...")
        time.sleep(0.01)

    for terrain_spawnable_name in terrain_spawnable_names:
        terrain = Actor(
            name=f"{terrain_spawnable_name}",
            spawnable_name=terrain_spawnable_name,
            position=[0, 0, 0],
            rotation=euler2quat([0, 0, 0]),
            scale=1.0,
        )
        scene.add_actor(terrain)
        print(f"    =============> Add terrain {terrain_spawnable_name} ...")
        time.sleep(0.01)

    scene.publish_scene()
    time.sleep(1)
    scene.close()
    time.sleep(1)

    print("=============> Publish scene done.")

def run_sb3_ppo_rl(
    config: dict,
    run_mode: str,
    ckpt: str,
):
    orcagym_addresses = config['orcagym_addresses']
    agent_name = config['agent_name']
    agent_spawnable_name = config['agent_spawnable_name']
    training_episode = config['training_episode']

    run_mode_config = config[run_mode]
    subenv_num = run_mode_config['subenv_num']
    agent_num = run_mode_config['agent_num']
    task = run_mode_config['task']
    render_mode = run_mode_config['render_mode']
    terrain_spawnable_names = run_mode_config['terrain_spawnable_names']
    entry_point = 'envs.legged_gym.legged_gym_env:LeggedGymEnv'


    if task == 'stand' or task == 'move_forward' or task == 'no_action' or task == 'follow_command':
        max_episode_steps = int(1 / (TIME_STEP * FRAME_SKIP * ACTION_SKIP) * EPISODE_TIME)
    else:
        raise ValueError("Invalid task")

    total_steps = training_episode * subenv_num * agent_num * max_episode_steps

    create_tmp_dir("trained_models_tmp")

    if ckpt is not None:
        model_file = ckpt
        model_dir = os.path.dirname(model_file)
    elif run_mode == "training":
        formatted_now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        model_dir = f"./trained_models_tmp/{agent_name}_{subenv_num * agent_num}_{formatted_now}"
        os.makedirs(model_dir, exist_ok=True)
        model_file = os.path.join(model_dir, f"{agent_name}_{task}.zip")
    else:
        raise ValueError("Invalid model file! Please provide a model file for testing / play.")

    # 清空场景
    clear_scene(
        orcagym_addresses=orcagym_addresses,
    )

    # 空场景生成高度图
    height_map_file = generate_height_map_file(
        orcagym_addresses=orcagym_addresses,
        model_dir=model_dir,
    )

    # 放置机器人
    publish_scene(
        orcagym_addresses=orcagym_addresses,
        agent_name=agent_name,
        agent_spawnable_name=agent_spawnable_name,
        agent_num=agent_num,
        terrain_spawnable_names=terrain_spawnable_names,
    )

    if run_mode == "training":
        print("Start Training! task: ", task, " subenv_num: ", subenv_num, " agent_num: ", agent_num, " agent_name: ", agent_name)
        print("Total Steps: ", total_steps, "Max Episode Steps: ", max_episode_steps, " Frame Skip: ", FRAME_SKIP, " Action Skip: ", ACTION_SKIP)
        sb3_ppo_vecenv_rl.train_model(
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
        )
    elif run_mode in ["testing", "play"]:
        print("Start Testing! Run mode: ", run_mode, "task: ", task, " subenv_num: ", subenv_num, " agent_num: ", agent_num, " agent_name: ", agent_name)
        print(" Total Steps: ", total_steps, "Max Episode Steps: ", max_episode_steps, " Frame Skip: ", FRAME_SKIP, " Action Skip: ", ACTION_SKIP)
        sb3_ppo_vecenv_rl.test_model(
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
            height_map_file=height_map_file
        )  
  
    else:
        raise ValueError("Invalid run mode")


def run_rllib_appo_rl(
    config: dict,
):
    from orca_gym.scripts.rllib_appo_rl import env_creator, test_model, run_training, setup_cuda_environment

    # 在脚本开头调用
    if setup_cuda_environment():
        print("CUDA 环境验证通过")
    else:
        print("CUDA 环境设置失败，GPU 加速可能不可用")

    if run_mode == 'training':
        run_training(
            orcagym_addr=config['orcagym_addr'],
            env_name=config['env_name'],
            agent_name=config['agent_name'],
            max_episode_steps=config['max_episode_steps'],
            num_env_runners=config['num_env_runners'],
            num_envs_per_env_runner=config['num_envs_per_env_runner'],
            async_env_runner=config['async_env_runner'],
            iter=config['iter'],
            render_mode=config['render_mode'],
            height_map_file=config['height_map_file']
        )
    elif run_mode == 'testing':
        if not checkpoint_path:
            raise ValueError("Checkpoint path must be provided for testing.")
        test_model(
            checkpoint_path=checkpoint_path,
            orcagym_addr=config['orcagym_addr'],
            env_name=config['env_name'],
            agent_name=config['agent_name'],
            max_episode_steps=config['max_episode_steps'],
            use_onnx_for_inference=False,
            explore_during_inference=False   
        )
    else:
        raise ValueError("Invalid run mode. Use 'training' or 'testing'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run multiple instances of the script with different gRPC addresses.')
    parser.add_argument('--config', type=str, help='The path of the config file')
    parser.add_argument('--train', action='store_true', help='Train the model')
    parser.add_argument('--test', action='store_true', help='Test the model')
    parser.add_argument('--play', action='store_true', help='Play the model')
    parser.add_argument('--ckpt', type=str, help='The path to the checkpoint file for testing / play')
    args = parser.parse_args()
    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    assert args.train or args.test or args.play, "Please specify one of --train, --test, or --play"
    assert not (args.train and args.test), "Please specify only one of --train, --test, or --play"
    assert not (args.train and args.play), "Please specify only one of --train, --test, or --play"
    assert not (args.test and args.play), "Please specify only one of --train, --test, or --play"

    if args.train:
        run_sb3_ppo_rl(config, 'training', None)
    elif args.test:
        run_sb3_ppo_rl(config, 'testing', args.ckpt)
    elif args.play:
        run_sb3_ppo_rl(config, 'play', args.ckpt)
    else:
        raise ValueError("Invalid config file")

