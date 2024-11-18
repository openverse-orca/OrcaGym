import os
import subprocess
import argparse


if __name__ == '__main__':
    """
    The startup script for the quadruped robot dog controller uses an MPC controller. For more details, refer to `envs/quadruped/README.md`.
    """

    current_file_path = os.path.abspath('')
    acados_path = os.path.join(current_file_path, "../../3rd_party/acados/")
    os.environ['LD_LIBRARY_PATH'] = os.environ.get('LD_LIBRARY_PATH', '') + acados_path + "lib"
    os.environ['ACADOS_SOURCE_DIR'] = acados_path


    parser = argparse.ArgumentParser(description='Simulation Configuration')
    parser.add_argument('--orcagym_addr', type=str, required=True, help='The gRPC address for the simulation')
    args = parser.parse_args()
    orcagym_addr = args.orcagym_addr

    subprocess.run(["python", "./quadruped_ctrl.py", "--orcagym_addr", orcagym_addr])


