

import examples.openpi.openloong_openpi_env as _env
from openpi_client import action_chunk_broker
from openpi_client import websocket_client_policy as _websocket_client_policy
from openpi_client.runtime import runtime as _runtime
from openpi_client.runtime.agents import policy_agent as _policy_agent
import tyro

import dataclasses
import gymnasium as gym
import orca_gym.scripts.camera_monitor as camera_monitor
from envs.manipulation.openloong_env import ControlDevice, RunMode, OpenLoongEnv
import orca_gym.scripts.openloong_manipulation as openloong_manipulation
from orca_gym.scripts.openloong_manipulation import ActionType
import logging


TIME_STEP = openloong_manipulation.TIME_STEP
FRAME_SKIP = openloong_manipulation.FRAME_SKIP
REALTIME_STEP = TIME_STEP * FRAME_SKIP
CONTROL_FREQ = 1 / REALTIME_STEP

CAMERA_CONFIG = openloong_manipulation.CAMERA_CONFIG
RGB_SIZE = openloong_manipulation.RGB_SIZE



@dataclasses.dataclass
class Args:
    orca_gym_address: str = 'localhost:50051'
    env_name: str = "OpenLoong"
    seed: int = 0
    agent_names: str = "OpenLoongHand"
    record_time: int = 20
    task: str = "Manipulation"
    obs_type: str = "pixels_agent_pos"
    prompt: str = "Pick up the apple"

    action_horizon: int = 10

    host: str = "0.0.0.0"
    port: int = 8000
    
    pico_ports: str = "8001"
    action_step: int = 1
    ctrl_device: str = "keyboard"
    sample_range: float = 0.0
    action_type: ActionType = ActionType.JOINT_POS

    display: bool = False

def main(args: Args) -> None:
    max_episode_steps = int(args.record_time * CONTROL_FREQ)    
    env_index = 0
    camera_config = CAMERA_CONFIG
    task_config_dict = {}

    env_id, kwargs = openloong_manipulation.register_env(
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
        subscribers=[
        ],
        max_hz=50,
    )

    runtime.run()

    # 终止 Monitor 子进程
    for process in monitor_processes:
        camera_monitor.terminate_monitor(process)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    tyro.cli(main)
