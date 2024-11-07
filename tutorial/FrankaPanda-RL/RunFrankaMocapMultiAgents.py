import subprocess
import argparse
import sys

def run_franka_mocap_multi_agents(ip_addr, env_num, agents_per_env, task, model_type, run_mode, total_timesteps):
    # 定义基本命令和端口列表
    base_command = ["python", "./FrankaMocapMultiAgents.py"]
    ports = []
    for i in range(env_num):
        ports.append(50051 + i)

    try:
        grpc_addresses = [f"{ip_addr}:{port}" for port in ports]
        command = base_command + ["--grpc_addresses"] + grpc_addresses + ["--agents_per_env"] + agents_per_env + ["--task", task] + ["--model_type", model_type] + ["--run_mode", run_mode] + ["--total_timesteps", str(total_timesteps)]
        print(f"Running command: {' '.join(command)}")
        process = subprocess.Popen(command)
        process.wait()

    except KeyboardInterrupt:
        process.kill()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run multiple instances of the script with different gRPC addresses.')
    parser.add_argument('--ip_addr', type=str, default='localhost', help='The IP address to connect to')
    parser.add_argument('--env_num', type=int, help='The number of envs to run, 1~x (for OrcaStudio level FrankaMocap_Multiagent_x): ', required=True)
    parser.add_argument('--agents_per_env', type=int, default=1, help='The number of agents per env')
    parser.add_argument('--task', type=str, default='reach', help='The task to run (reach or pick_and_place)')
    parser.add_argument('--model_type', type=str, choices=['ppo', 'tqc', 'sac', 'ddpg'], help='The model to use (ppo/tqc/sac/ddpg)', required=True)
    parser.add_argument('--run_mode', type=str, default='training', help='The mode to run (training or testing)')
    parser.add_argument('--total_timesteps', type=int, default=100000, help='The total timesteps to train the model')    
    args = parser.parse_args()

    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)

    ip_addr = args.ip_addr
    env_num = args.env_num
    agents_per_env = args.agents_per_env
    task = args.task
    model_type = args.model_type
    run_mode = args.run_mode
    total_timesteps = args.total_timesteps

    run_franka_mocap_multi_agents(ip_addr, env_num, agents_per_env, task, model_type, run_mode, total_timesteps)