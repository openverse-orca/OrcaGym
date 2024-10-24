import os
import subprocess
import argparse
import sys
from datetime import datetime
import time

current_file_path = os.path.abspath('')
project_root = os.path.dirname(os.path.dirname(current_file_path))

if project_root not in sys.path:
    sys.path.append(project_root)

def run_openloong_wbc_multi_agents(ip_addr, agent_name, agent_num, individual_control):

    server_command = ["python", "run_server.py"]
    server_process = subprocess.Popen(server_command, cwd="../teleoperation/keyboard_input")
    processes = []
    processes.append(server_process)

    # Define the base command
    base_command = ["python", "./openloong_wbc_joystick.py"]

    try:
        for i in range(agent_num):
            port = 50051 + i
            grpc_address = f"{ip_addr}:{port}"
            # Construct the command for each agent
            command = base_command + [
                "--grpc_address", grpc_address,
                "--agent_name", agent_name,                
                "--individual_control", individual_control
            ]
            print(f"Running command: {' '.join(command)}")
            # Start the subprocess for each agent
            process = subprocess.Popen(command)
            processes.append(process)

        # Wait for all processes to complete
        for process in processes:
            process.wait()

    finally:
        # 先杀client再杀server，避免残留
        for process in processes[1:]:
            process.kill()

        time.sleep(1)
        processes[0].kill()

if __name__ == '__main__':
    """
    The startup script for the openloong walking wbc control using joystick.
    """
    parser = argparse.ArgumentParser(description='Simulation Configuration')
    parser.add_argument('--grpc_address', type=str, default="localhost", help='The gRPC address for the simulation')
    parser.add_argument('--agent_name', type=str, default="AzureLoong", help='The agent name for the simulation')
    parser.add_argument('--agent_num', type=int, default=8, help='The number of agents for the simulation')
    parser.add_argument('--individual_control', type=str, default="False", help='Control the robots individually')
    args = parser.parse_args()

    grpc_address = f"{args.grpc_address}"
    agent_name = f"{args.agent_name}"
    agent_num = args.agent_num
    individual_control = f"{args.individual_control}"

    run_openloong_wbc_multi_agents(grpc_address, agent_name, agent_num, individual_control)