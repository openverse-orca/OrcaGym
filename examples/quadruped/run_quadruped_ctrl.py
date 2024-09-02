import os
import subprocess
import argparse


if __name__ == '__main__':
    """
    自足机器狗控制器的启动脚本，采用MPC控制器，详见 envs/quadruped/README.md
    """

    # 设置环境变量
    os.environ['LD_LIBRARY_PATH'] = os.environ.get('LD_LIBRARY_PATH', '') + "/home/superfhwl/repo/acados/lib"
    os.environ['ACADOS_SOURCE_DIR'] = "/home/superfhwl/repo/acados"

    # Parse the arguments
    parser = argparse.ArgumentParser(description='Simulation Configuration')
    parser.add_argument('--grpc_address', type=str, required=True, help='The gRPC address for the simulation')

    args = parser.parse_args()

    grpc_address = args.grpc_address

    # 运行目标 Python 脚本
    subprocess.run(["python", "./quadruped_ctrl.py", "--grpc_address", grpc_address])


