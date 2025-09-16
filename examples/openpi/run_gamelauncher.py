import argparse
import subprocess
import time

def main():
    parser = argparse.ArgumentParser(description="Launch only OrcaSim Gamelauncher")
    parser.add_argument('--datalink_auth_config', type=str, required=True, help='The datalink auth config abs path')
    parser.add_argument('--orcasim_path', type=str, required=True, help='The orcasim executable path')
    parser.add_argument('--level', type=str, default='shopscene', help='The storage level')
    parser.add_argument('--orcagym_address', type=str, default='localhost:50051', help='The gRPC address (host:port)')
    parser.add_argument('--index', type=int, default=0, help='Index (for multi-instance port offset)')
    parser.add_argument('--useNvenc', type=int, default=1, help='Use Nvenc (1=yes, 0=no)')
    args = parser.parse_args()

    host, port = args.orcagym_address.split(':')
    port = int(port)
    orcagym_address = f"0.0.0.0:{port + args.index}"

    print(f"Launching OrcaSim Gamelauncher at: {orcagym_address}")

    adapterIndex = args.index % 2

    if args.useNvenc == 1:
        cmd = [
            args.orcasim_path, "--datalink_auth_config", args.datalink_auth_config,
            "--mj_grpc_server", orcagym_address,
            "--forceAdapter", "NVIDIA GeForce RTX 4090",
            "--adapterIndex", str(adapterIndex),
            # "--r_width", "128", "--r_height", "128",
            "--useNvenc", "1",
            "--lockFps30",
            f"--regset=/O3DE/Autoexec/ConsoleCommands/LoadLevel={args.level}"
        ]
    else:
        cmd = [
            args.orcasim_path, "--datalink_auth_config", args.datalink_auth_config,
            "--mj_grpc_server", orcagym_address,
            "--forceAdapter", "NVIDIA GeForce RTX 4090",
            "--adapterIndex", str(adapterIndex),
            # "--r_width", "128", "--r_height", "128",
            "--lockFps30",
            f"--regset=/O3DE/Autoexec/ConsoleCommands/LoadLevel={args.level}"
        ]

    print("Running command:", " ".join(cmd))
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    # 持续输出日志
    try:
        while True:
            output = p.stdout.readline()
            if output == '' and p.poll() is not None:
                break
            if output:
                print(output.strip())
    except KeyboardInterrupt:
        print("Stopping Gamelauncher...")
        p.terminate()

if __name__ == "__main__":
    main()