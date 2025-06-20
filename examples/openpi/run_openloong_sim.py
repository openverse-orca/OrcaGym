import os

current_file_path = os.path.abspath('')
project_root = os.path.dirname(os.path.dirname(current_file_path))

# if project_root not in sys.path:
#     sys.path.append(project_root)


import orca_gym.scripts.openloong_manipulation as openloong_manipulation

import argparse



if __name__ == "__main__":

    # 如果要使用多Pico设备，需要设置adb反向端口转发
    # adb device 命令查看设备序列号

    # PICO_01:8001 -----> PC:8001
    # adb -s <device_serial_number_01> reverse tcp:8001 tcp:8001

    #PICO_02:8001 -----> PC:8002
    # adb -s <device_serial_number_02> reverse tcp:8001 tcp:8002  
    
    # 脚本参数a增加--pico_ports参数，指定多个端口
    # 例如：--pico_ports "8001 8002"

    parser = argparse.ArgumentParser(description='Run multiple instances of the script with different gRPC addresses.')
    parser.add_argument('--orcagym_address', type=str, default='localhost:50051', help='The gRPC addresses to connect to')
    parser.add_argument('--agent_names', type=str, default='OpenLoongGripperAgent', help='The agent names to control, separated by space')
    parser.add_argument('--pico_ports', type=str, default='8001', help='The pico server port')
    parser.add_argument('--run_mode', type=str, default='teleoperation', help='The run mode of the environment (teleoperation / playback / imitation / rollout / augmentation)')
    parser.add_argument('--action_type', type=str, default='joint_pos', help='The action type of the environment (end_effector / joint_pos)')
    parser.add_argument('--action_step', type=int, default=1, help='How may simulation steps to take for each action. 5 for end_effector, 1 for joint_pos')
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
    parser.add_argument('--teleoperation_rounds', type=int, default=100, help='The rounds to do teleoperation')
    parser.add_argument('--sample_range', type=float, default=0.0, help='The area range to sample the object and goal position')
    parser.add_argument('--realtime_playback', type=bool, default=True, help='The flag to enable the real-time playback or rollout')
    
    args = parser.parse_args()
    openloong_manipulation.run_openloong_sim(args, project_root, current_file_path)