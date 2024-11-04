import os
import subprocess
import argparse
import sys
from datetime import datetime
import time
import psutil
import multiprocessing

current_file_path = os.path.abspath('')
project_root = os.path.dirname(os.path.dirname(current_file_path))

if project_root not in sys.path:
    sys.path.append(project_root)

import subprocess
import time
import psutil
import multiprocessing

def run_openloong_wbc_multi_agents(ip_addr, agent_name, agent_num, individual_control, enable_cpu_affinity=False):
    server_command = ["python", "run_server.py"]
    server_process = subprocess.Popen(server_command, cwd="../teleoperation/keyboard_input")

    # 获取 CPU 核心数量
    num_cpus = multiprocessing.cpu_count()

    # 定义基础命令
    base_command = ["python", "./openloong_wbc_joystick.py"]
    client_processes = []
    try:
        for i in range(agent_num):
            port = 50051 + i
            grpc_address = f"{ip_addr}:{port}"
            # 构建每个代理的命令
            command = base_command + [
                "--grpc_address", grpc_address,
                "--agent_name", agent_name,
                "--individual_control", individual_control
            ]
            print(f"Running command: {' '.join(command)}")
            # 启动每个代理的子进程
            process = subprocess.Popen(command)
            client_processes.append(process)

            # 如果启用 CPU 亲和性，设置每个子进程的 CPU 绑定
            if enable_cpu_affinity:
                cpu_num = i % num_cpus  # 循环分配 CPU 核心
                p = psutil.Process(process.pid)
                p.cpu_affinity([cpu_num])

        # 等待服务器进程完成
        server_process.wait()
        time.sleep(0.1)

        # 等待所有客户端进程完成
        for process in client_processes:
            process.wait()
            time.sleep(0.1)

    finally:
        for process in client_processes:
            process.kill()
            time.sleep(0.1)

        # 先杀掉客户端，再杀掉服务器，避免残留
        server_process.kill()



if __name__ == '__main__':
    """
    The startup script for the openloong walking wbc control using joystick.
    """
    parser = argparse.ArgumentParser(description='Simulation Configuration')
    parser.add_argument('--grpc_address', type=str, default="localhost", help='The gRPC address for the simulation')
    parser.add_argument('--agent_name', type=str, default="AzureLoong", help='The agent name for the simulation')
    parser.add_argument('--agent_num', type=int, default=10, help='The number of agents for the simulation')
    parser.add_argument('--individual_control', type=str, default="False", help='Control the robots individually')
    args = parser.parse_args()

    grpc_address = f"{args.grpc_address}"
    agent_name = f"{args.agent_name}"
    agent_num = args.agent_num
    individual_control = f"{args.individual_control}"

    run_openloong_wbc_multi_agents(grpc_address, agent_name, agent_num, individual_control, enable_cpu_affinity=False)