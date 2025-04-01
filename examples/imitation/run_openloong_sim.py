import os
import sys
import time
import subprocess
import signal

current_file_path = os.path.abspath('')
project_root = os.path.dirname(os.path.dirname(current_file_path))

# if project_root not in sys.path:
#     sys.path.append(project_root)


import orca_gym.scripts.openloong_manipulation as openloong_manipulation


import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run multiple instances of the script with different gRPC addresses.')
    parser.add_argument('--orcagym_address', type=str, default='localhost:50051', help='The gRPC addresses to connect to')
    parser.add_argument('--agent_name', type=str, default='AzureLoong', help='The agent name to control')
    parser.add_argument('--run_mode', type=str, default='teleoperation', help='The run mode of the environment (teleoperation / playback / imitation / rollout / augmentation)')
    parser.add_argument('--action_type', type=str, default='end_effector', help='The action type of the environment (end_effector / joint_pos)')
    parser.add_argument('--action_step', type=int, default=5, help='How may simulation steps to take for each action, 5 for end_effector, 1 for joint_pos')
    parser.add_argument('--task', type=str, help='The task instruction to do teleoperation')
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
    
    openloong_manipulation.run_openloong_sim(args, project_root, current_file_path)