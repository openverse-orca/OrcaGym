import os

current_file_path = os.path.abspath('')
project_root = os.path.dirname(os.path.dirname(current_file_path))

# if project_root not in sys.path:
#     sys.path.append(project_root)


import orca_gym.scripts.dual_arm_manipulation as dual_arm_manipulation

import argparse
import subprocess
import threading
import time
import sys

def run_simulation_instance(args, port_offset, project_root, current_file_path):
    """处理单个模拟实例的线程函数"""
    # 复制参数避免共享状态
    import copy
    instance_args = copy.deepcopy(args)
    instance_args.orcagym_address = f"localhost:{port + port_offset}"

    # 确保新线程有自己的事件循环
    def thread_target():
        # 关键修复：初始化新线程的事件循环
        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        dual_arm_manipulation.run_dual_arm_sim(instance_args, project_root, current_file_path)

    thread = threading.Thread(target=thread_target)
    thread.start()
    return thread


if __name__ == "__main__":

    # 如果要使用多Pico设备，需要设置adb反向端口转发
    # adb device 命令查看设备序列号

    # PICO_01:8001 -----> PC:8001
    # adb -s <device_serial_number_01> reverse tcp:8001 tcp:8001

    #PICO_02:8001 -----> PC:8002
    # adb -s <device_serial_number_02> reverse tcp:8001 tcp:8002

    # 脚本参数a增加--pico_ports参数，指定多个端口
    # 例如：--pico_ports "8001 8002"

    parser = argparse.ArgumentParser(description='Run parallelism augmenta of the script ')
    parser.add_argument('--parallelism_num', type=int, default='4', help='The parallelism number')
    parser.add_argument('--datalink_auth_config', type=str, required=True, help='The datalink auth config abs path')
    parser.add_argument('--orcasim_path', type=str, required=True, help='The orcasim processor path')
    parser.add_argument('--level', type=str, default='shopscene', required=True,  help='The storage level or scenario for file storage (e.g., default, experiment, debug)')
    parser.add_argument('--levelorca', type=str, default='shopscene', required=True,  help='The Orcagym data store directory (e.g., default, experiment, debug)')

    parser.add_argument('--orcagym_address', type=str, default='localhost:50051', help='The gRPC addresses to connect to')
    parser.add_argument('--agent_names', type=str, default='openloong_gripper_2f85_fix_base_usda', help='The agent names to control, separated by space')
    parser.add_argument('--pico_ports', type=str, default='8001', help='The pico server port')
    parser.add_argument('--run_mode', type=str, default='teleoperation', help='The run mode of the environment (teleoperation / playback / imitation / rollout / augmentation)')
    parser.add_argument('--action_type', type=str, default='joint_pos', help='The action type of the environment (end_effector / joint_pos)')
    parser.add_argument('--action_step', type=int, default=1, help='How may simulation steps to take for each action. 5 for end_effector, 1 for joint_pos')
    parser.add_argument('--task_config', type=str, default='ashop_task.yaml',help='The task config file to load')
    parser.add_argument('--algo', type=str, default='bc', help='The algorithm to use for training the policy')
    parser.add_argument('--dataset', type=str, help='The file path to save the record')
    parser.add_argument('--model_file', type=str, help='The model file to load for rollout the policy')
    parser.add_argument('--record_length', type=int, default=1200, help='The time length in seconds to record the teleoperation in 1 episode')
    parser.add_argument('--ctrl_device', type=str, default='vr', help='The control device to use ')
    parser.add_argument('--playback_mode', type=str, default='random', help='The playback mode of the environment (loop or random)')
    parser.add_argument('--rollout_times', type=int, default=10, help='The times to rollout the policy')
    parser.add_argument('--augmented_noise', type=float, default=0.01, help='The noise to augment the dataset')
    parser.add_argument('--augmented_rounds', type=int, default=3, help='The times to augment the dataset')
    parser.add_argument('--teleoperation_rounds', type=int, default=100, help='The rounds to do teleoperation')
    parser.add_argument('--sample_range', type=float, default=0.0, help='The area range to sample the object and goal position')
    parser.add_argument('--realtime_playback', type=str, default='True', help='The flag to enable the real-time playback or rollout')
    parser.add_argument('--withvideo', type=str, default='True', help='The flag to enable the real-time playback or rollout')

    args = parser.parse_args()

    parallelism_num = args.parallelism_num
    datalink_auth_config = args.datalink_auth_config
    orcasim_path = args.orcasim_path
    level = args.level

    orcagym_address = args.orcagym_address
    host, port = orcagym_address.split(':')
    port = int(port)
    orcagym_address_list = []
    for i in range(parallelism_num):
        orcagym_address_list.append(f"0.0.0.0:{port + i}")

    process  = []
    for i in range(parallelism_num):
        adapterIndex = i % 2
        p = subprocess.Popen([orcasim_path, "--datalink_auth_config", datalink_auth_config,
                        "--mj_grpc_server",  orcagym_address_list[i],
                        "--forceAdapter", " NVIDIA GeForce RTX 4090",
                        "--adapterIndex", str(adapterIndex),
                        "--r_width", "128", "--r_height", "128",
                        f"--regset=\"/O3DE/Autoexec/ConsoleCommands/LoadLevel={level}\""], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        process.append(p)

    time.sleep(15)


    #调用增广脚本，如果需要额外参数自行添加
    threads = []
    for i in range(parallelism_num):
        p = subprocess.Popen(["python", "run_dual_arm_sim.py", 
                              "--orcagym_address", f"localhost:{port + i}",
                              "--run_mode", "augmentation",
                              "--dataset", args.dataset,
                              "--action_type",args.action_type,
                              "--task_config",args.task_config,
                              "--sample_range",str(args.sample_range),
                              "--augmented_noise",str(args.augmented_noise),
                              "--augmented_rounds",str(args.augmented_rounds),
                              "--realtime_playback",args.realtime_playback,
                              "--withvideo",args.withvideo,
                              "--level",args.levelorca


                              ], stdout=sys.stdout, stderr=sys.stderr, text=True)
        
        process.append(p)

    # 等待所有线程完成
    # for t in threads:
    #     t.join()

    # # dual_arm_manipulation.run_dual_arm_sim(args, project_root, current_file_path)

    for p in process:
        p.wait()