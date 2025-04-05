import os

current_file_path = os.path.abspath('')
project_root = os.path.dirname(os.path.dirname(current_file_path))

# if project_root not in sys.path:
#     sys.path.append(project_root)


import orca_gym.scripts.openloong_manipulation as openloong_manipulation

import argparse

import logging

import examples.openpi.openloong_openpi_env as _env
from openpi_client import action_chunk_broker
from openpi_client import websocket_client_policy as _websocket_client_policy
from openpi_client.runtime import runtime as _runtime
from openpi_client.runtime.agents import policy_agent as _policy_agent
import tyro
import gymnasium as gym
import orca_gym.scripts.camera_monitor as camera_monitor
from envs.manipulation.openloong_env import ControlDevice, RunMode, OpenLoongEnv

ENV_ENTRY_POINT = {
    "AzureLoong_Manipulation": "envs.manipulation.openloong_env:OpenLoongEnv"
}

TIME_STEP = openloong_manipulation.TIME_STEP
FRAME_SKIP = openloong_manipulation.FRAME_SKIP
REALTIME_STEP = TIME_STEP * FRAME_SKIP
CONTROL_FREQ = 1 / REALTIME_STEP

CAMERA_CONFIG = openloong_manipulation.CAMERA_CONFIG
RGB_SIZE = openloong_manipulation.RGB_SIZE


def main(args) -> None:
    orcagym_addr = args.orcagym_address
    agent_name = args.agent_name
    record_time = args.record_length
    action_type = args.action_type
    action_step = args.action_step
    task_instruction = args.task
    sample_range = args.sample_range
        
    env_name = "OpenLoong"
    env_index = 0
    camera_config = CAMERA_CONFIG
    ctrl_device = ControlDevice.VR
    max_episode_steps = int(record_time / REALTIME_STEP)
    print(f"Run episode in {max_episode_steps} steps as {record_time} seconds.")
    
    env_id, kwargs = openloong_manipulation.register_env(
        orcagym_addr, 
        env_name, 
        env_index, 
        agent_name, 
        RunMode.POLICY_NORMALIZED, 
        action_type, 
        task_instruction, 
        ctrl_device, 
        max_episode_steps, 
        sample_range, 
        action_step, 
        camera_config)
    
    print("Registered Simulation Environment: ", env_id, " with kwargs: ", kwargs)

    # 启动 Monitor 子进程
    ports = [7070]
    monitor_processes = []
    for port in ports:
        process = camera_monitor.start_monitor(port=port)
        monitor_processes.append(process)

    action_horizon: int = 10
    host: str = "0.0.0.0"
    port: int = 8000
    seed: int = 0
    obs_type: str = "pixels_agent_pos"
    
        
    runtime = _runtime.Runtime(
        environment=_env.OpenLoongOpenpiEnv(
            env_id=env_id,
            seed=seed,
            prompt=task_instruction,
            obs_type=obs_type,
        ),
        agent=_policy_agent.PolicyAgent(
            policy=action_chunk_broker.ActionChunkBroker(
                policy=_websocket_client_policy.WebsocketClientPolicy(
                    host=host,
                    port=port,
                ),
                action_horizon=action_horizon,
            )
        ),
        subscribers=[
        ],
        max_hz=50,
    )

    runtime.run()

    # 终止 Monitor 子进程
    for process in monitor_processes:
        camera_monitor.terminate_monitor(process)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run multiple instances of the script with different gRPC addresses.')
    parser.add_argument('--orcagym_address', type=str, default='localhost:50051', help='The gRPC addresses to connect to')
    parser.add_argument('--agent_names', type=str, default='OpenLoongHand', help='The agent names to control, separated by space')
    parser.add_argument('--run_mode', type=str, default='teleoperation', help='The run mode of the environment (teleoperation / playback / imitation / rollout / augmentation)')
    parser.add_argument('--action_type', type=str, default='joint_pos', help='The action type of the environment (end_effector / joint_pos)')
    parser.add_argument('--action_step', type=int, default=1, help='How may simulation steps to take for each action. 5 for end_effector, 1 for joint_pos')
    parser.add_argument('--prompt', type=str, default="Do something.", help='The task instruction to do teleoperation')
    parser.add_argument('--task_config', type=str, help='The task config file to load')
    parser.add_argument('--algo', type=str, default='bc', help='The algorithm to use for training the policy')
    parser.add_argument('--dataset', type=str, help='The file path to save the record')
    parser.add_argument('--model_file', type=str, help='The model file to load for rollout the policy')
    parser.add_argument('--record_length', type=int, default=1200, help='The time length in seconds to record the teleoperation in 1 episode')
    parser.add_argument('--ctrl_device', type=str, default='vr', help='The control device to use ')
    parser.add_argument('--playback_mode', type=str, default='random', help='The playback mode of the environment (loop or random)')
    parser.add_argument('--rollout_times', type=int, default=10, help='The times to rollout the policy')
    parser.add_argument('--augmented_sacle', type=float, default=0.01, help='The scale to augment the dataset')
    parser.add_argument('--augmented_rounds', type=int, default=3, help='The times to augment the dataset')
    parser.add_argument('--teleoperation_rounds', type=int, default=20, help='The rounds to do teleoperation')
    parser.add_argument('--sample_range', type=float, default=0.0, help='The area range to sample the object and goal position')
    parser.add_argument('--realtime_playback', type=bool, default=True, help='The flag to enable the real-time playback or rollout')
    
    args = parser.parse_args()
    
    if args.run_mode == 'rollout':
        logging.basicConfig(level=logging.INFO, force=True)
        main(args)
    else:
        openloong_manipulation.run_openloong_sim(args, project_root, current_file_path)