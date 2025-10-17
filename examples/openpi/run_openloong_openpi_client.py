

import examples.openpi.openloong_openpi_env as _env
from openpi_client import action_chunk_broker
from openpi_client import websocket_client_policy as _websocket_client_policy
from openpi_client.runtime import runtime as _runtime
from openpi_client.runtime.agents import policy_agent as _policy_agent
import tyro

import dataclasses
import gymnasium as gym
import orca_gym.scripts.camera_monitor as camera_monitor
from envs.manipulation.dual_arm_env import ControlDevice, RunMode
import orca_gym.scripts.dual_arm_manipulation as dual_arm_manipulation
from orca_gym.scripts.dual_arm_manipulation import ActionType
import logging
import yaml

TIME_STEP = dual_arm_manipulation.TIME_STEP
FRAME_SKIP = dual_arm_manipulation.FRAME_SKIP
REALTIME_STEP = TIME_STEP * FRAME_SKIP
CONTROL_FREQ = 1 / REALTIME_STEP

CAMERA_CONFIG = dual_arm_manipulation.CAMERA_CONFIG
RGB_SIZE = dual_arm_manipulation.RGB_SIZE



@dataclasses.dataclass
class Args:
    orca_gym_address: str = 'localhost:50051'
    env_name: str = "DualArmEnv"
    seed: int = 0
    agent_names: str = "openloong_gripper_2f85_fix_base_usda"
    record_time: int = 20
    task: str = "Manipulation"
    obs_type: str = "pixels_agent_pos"
    prompt: str = "level: tmp  object: Box1 to goal: basket"

    action_horizon: int = 10

    host: str = "0.0.0.0"
    port: int = 8000
    
    pico_ports: str = "8001"
    action_step: int = 1
    ctrl_device: str = "keyboard"
    sample_range: float = 0.0
    action_type: ActionType = ActionType.JOINT_POS

    display: bool = False
    
    task_config: str = "jiazi_task.yaml"

def main(args: Args) -> None:
    max_episode_steps = int(args.record_time * CONTROL_FREQ)    
    env_index = 0
    camera_config = CAMERA_CONFIG
    task_config_dict = {}
    if args.task_config is not None:
        with open(args.task_config, "r") as f:
            task_config_dict = yaml.safe_load(f)

    env_id, kwargs = dual_arm_manipulation.register_env(
        orcagym_addr=args.orca_gym_address,
        env_name=args.env_name, 
        env_index=env_index, 
        agent_names=args.agent_names, 
        pico_ports=args.pico_ports, 
        run_mode=RunMode.POLICY_NORMALIZED, 
        action_step=args.action_step, 
        ctrl_device=args.ctrl_device, 
        max_episode_steps=max_episode_steps, 
        sample_range=args.sample_range, 
        action_type=args.action_type,
        camera_config=camera_config, 
        task_config_dict=task_config_dict,
    )
        
    print("Registered Simulation Environment: ", env_id, " with kwargs: ", kwargs)

    # 启动 Monitor 子进程
    ports = [7070]
    monitor_processes = []
    for port in ports:
        process = camera_monitor.start_monitor(port=port)
        monitor_processes.append(process)
    
    runtime = _runtime.Runtime(
        environment=_env.OpenLoongOpenpiEnv(
            env_id=env_id,
            seed=args.seed,
            obs_type=args.obs_type,
            prompt=args.prompt,
        ),
        agent=_policy_agent.PolicyAgent(
            policy=action_chunk_broker.ActionChunkBroker(
                policy=_websocket_client_policy.WebsocketClientPolicy(
                    host=args.host,
                    port=args.port,
                ),
                action_horizon=args.action_horizon,
            )
        ),
        # agent=_policy_agent.PolicyAgent(
        #     policy=test_policy.TestPolicy(
        #         demo_path="/home/user/Desktop/manipulation/OrcaGym/examples/openpi/records_tmp/tmp/a7419694_1755250755"
        #     )
        # ),
        subscribers=[
        ],
        max_hz=50,
    )

    runtime.run()

    # 终止 Monitor 子进程
    for process in monitor_processes:
        camera_monitor.terminate_monitor(process)

def draw_action_csv(path: str) -> None:
    import matplotlib.pyplot as plt
    import pandas as pd
    df = pd.read_csv(path)
    # Plot all columns
    plt.figure(figsize=(12, 6))
    # filter columns
    for col in df.columns[0:]:
        if "right" in col:
            continue
        print(col)
        print(df[col])
        plt.plot(df.index, df[col], marker='o', label=col)
    plt.legend()
    plt.show()
    plt.savefig("y_p_left_2.png")
    
def draw_from_original_data(path: str) -> None:
    # read from hdf5 file
    import h5py
    import pandas as pd
    import matplotlib.pyplot as plt
    with h5py.File(path, "r") as f:
        data = f["data"]["demo_00000"]["actions"]
        # turn to pandas dataframe
        df = pd.DataFrame(data)
    # Plot all columns
    plt.figure(figsize=(12, 6))
    # for col in df.columns[6:14]:
    #     plt.plot(df.index, df[col], marker='o', label=col)
    for col in df.columns[20:28]:
        plt.plot(df.index, df[col], marker='o', label=col)
    plt.legend()
    # plt.show()
    plt.savefig("y_right.png")
    

def draw_comparison(path: str, joint_num: int) -> None:
    import matplotlib.pyplot as plt
    # read from hdf5 file
    import h5py
    import pandas as pd
    import matplotlib.pyplot as plt
    with h5py.File(path, "r") as f:
        data = f["data"]["demo_00000"]["actions"]
        # turn to pandas dataframe
        df = pd.DataFrame(data)
    count = 0
    for col in df.columns[6:14]:
        if count!= joint_num:
            count += 1
            continue
        plt.plot(df.index, df[col], marker='o', label=col)
        count += 1
    for col in df.columns[20:28]:
        if count!= joint_num:
            count += 1
            continue
        plt.plot(df.index, df[col], marker='o', label=col)
        count += 1

    import matplotlib.pyplot as plt
    import pandas as pd
    df = pd.read_csv("action.csv")
    # filter columns
    for col in df.columns[joint_num: joint_num+1]:
        plt.plot(df.index, df[col], marker='o', label=col)
    

    plt.savefig(f"logs/y_comparison_{joint_num}.png")
    # clear plt
    plt.clf()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    tyro.cli(main)
    # draw_action_csv("state.csv")
    # for joint_num in range(0, 16):
    #     draw_comparison("/home/user/Desktop/manipulation/OrcaGym/examples/openpi/records_tmp/tmp/e9a532a3_1755250723/proprio_stats/proprio_stats.hdf5", joint_num)
    # draw_from_original_data("/home/user/Desktop/manipulation/OrcaGym/examples/openpi/records_tmp/tmp/e9a532a3_1755250723/proprio_stats/proprio_stats.hdf5")
