import os
import sys
import argparse
import time
import numpy as np
from datetime import datetime

current_file_path = os.path.abspath('')
project_root = os.path.dirname(os.path.dirname(current_file_path))

# 将项目根目录添加到 PYTHONPATH
if project_root not in sys.path:
    sys.path.append(project_root)


from envs.legged_gym.legged_config import LeggedEnvConfig, LeggedRobotConfig
import orca_gym.scripts.multi_agent_rl as rl
from orca_gym.utils.dir_utils import create_tmp_dir

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run multiple instances of the script with different gRPC addresses.')
    parser.add_argument('--orcagym_addresses', type=str, nargs='+', default=['localhost:50051'], help='The gRPC addresses to connect to')
    parser.add_argument('--subenv_num', type=int, default=1, help='The number of subenvs for each gRPC address')
    parser.add_argument('--agent_num', type=int, default=1, help='The number of agents for each subenv')
    parser.add_argument('--agent_name', type=str, default='go2', help='The name of the agent')
    parser.add_argument('--task', type=str, default='follow_command', help='The task to run')
    parser.add_argument('--model_type', type=str, default='ppo', help='The model to use (ppo only now)')
    parser.add_argument('--run_mode', type=str, default='training', help='The mode to run (training / testing / play / nav)')
    parser.add_argument('--render_mode', type=str, default='human', help='The render mode (human / none)')
    parser.add_argument('--model_file', type=str, help='The model file to save/load. If not provided, a new model file will be created while training')
    parser.add_argument('--height_map_file', type=str, default='../../orca_gym/tools/height_map.npy', help='The height field map file')
    parser.add_argument('--load_existing_model', type=bool, default=False, help='Load existing model')
    parser.add_argument('--training_episode', type=int, default=200, help='The number of training episodes for each agent')
    parser.add_argument('--start_her_episode', type=float, default=1.0, help='Before start HER training, run each agent for some episodes to get experience')
    parser.add_argument('--nav_ip', type=str, default="localhost", help='The IP address of the navigation server, default is localhost, should be local pc ip address')
    args = parser.parse_args()

    TIME_STEP = LeggedEnvConfig["TIME_STEP"]

    FRAME_SKIP_REALTIME = LeggedEnvConfig["FRAME_SKIP_REALTIME"]
    FRAME_SKIP_SHORT = LeggedEnvConfig["FRAME_SKIP_SHORT"]
    FRAME_SKIP_LONG = LeggedEnvConfig["FRAME_SKIP_LONG"]

    EPISODE_TIME_VERY_SHORT = LeggedEnvConfig["EPISODE_TIME_VERY_SHORT"]
    EPISODE_TIME_SHORT = LeggedEnvConfig["EPISODE_TIME_SHORT"]
    EPISODE_TIME_LONG = LeggedEnvConfig["EPISODE_TIME_LONG"]

    orcagym_addresses = args.orcagym_addresses
    subenv_num = args.subenv_num
    agent_num = args.agent_num
    agent_name = args.agent_name
    task = args.task
    model_type = args.model_type
    height_map_file = args.height_map_file
    run_mode = args.run_mode
    render_mode = args.render_mode
    load_existing_model = args.load_existing_model
    training_episode = args.training_episode
    start_her_episode = args.start_her_episode
    nav_ip = args.nav_ip

    entry_point = 'envs.legged_gym.legged_gym_env:LeggedGymEnv'

    if task == 'stand' or task == 'move_forward' or task == 'no_action' or task == 'follow_command':
        frame_skip = FRAME_SKIP_SHORT # FRAME_SKIP_REALTIME
        max_episode_steps = int(1 / (TIME_STEP * frame_skip) * EPISODE_TIME_LONG)
    else:
        raise ValueError("Invalid task")

    total_timesteps = training_episode * subenv_num * agent_num * max_episode_steps

    create_tmp_dir("trained_models_tmp")

    if args.model_file is not None:
        model_file = args.model_file
    elif run_mode == "training" and not load_existing_model:
        formatted_now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        model_dir = f"./trained_models_tmp/{agent_name}_{model_type}_{subenv_num * agent_num}-agents_{training_episode}-episodes_{formatted_now}"
        os.makedirs(model_dir, exist_ok=True)
        model_file = os.path.join(model_dir, f"{agent_name}_{task}.zip")
    else:
        raise ValueError("Invalid model file! Please provide a model file for testing, or set `load_existing_model` to False for training")

    if run_mode == "training":
        print("Start Training! task: ", task, " subenv_num: ", subenv_num, " agent_num: ", agent_num, " agent_name: ", agent_name)
        print("Model Type: ", model_type, " Total Timesteps: ", total_timesteps, " HER Start Episode: ", start_her_episode)
        print("Max Episode Steps: ", max_episode_steps, " Frame Skip: ", frame_skip)
        rl.train_model(
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
            frame_skip=frame_skip, 
            model_type=model_type, 
            total_timesteps=total_timesteps, 
            start_her_episode=start_her_episode, 
            model_file=model_file, 
            height_map_file=height_map_file, 
            load_existing_model=load_existing_model
        )
    elif run_mode in ["testing", "play", "nav"]:
        print("Start Testing! Run mode: ", run_mode, "task: ", task, " subenv_num: ", subenv_num, " agent_num: ", agent_num, " agent_name: ", agent_name)
        print("Model Type: ", model_type, " Total Timesteps: ", total_timesteps, " HER Start Episode: ", start_her_episode)
        print("Max Episode Steps: ", max_episode_steps, " Frame Skip: ", frame_skip)
        rl.test_model(
            orcagym_addresses=orcagym_addresses, 
            agent_num=agent_num, 
            agent_name=agent_name, 
            task=task, 
            run_mode=run_mode, 
            nav_ip=nav_ip, 
            entry_point=entry_point, 
            time_step=TIME_STEP, 
            max_episode_steps=max_episode_steps, 
            render_mode="human",
            frame_skip=frame_skip, 
            model_type=model_type, 
            model_file=model_file, 
            height_map_file=height_map_file
        )    
    else:
        raise ValueError("Invalid run mode")

