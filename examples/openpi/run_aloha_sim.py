import dataclasses
import logging
import pathlib

import examples.openpi.aloha_openpi_env as _env
from openpi_client import action_chunk_broker
from openpi_client import websocket_client_policy as _websocket_client_policy
from openpi_client.runtime import runtime as _runtime
from openpi_client.runtime.agents import policy_agent as _policy_agent
import tyro
import gymnasium as gym
import orca_gym.scripts.camera_monitor as camera_monitor

ENV_ENTRY_POINT = {
    "AlohaTransferCube": "envs.aloha.aloha_dm_env:AlohaDMEnv"
}

TIME_STEP = 0.0025
FRAME_SKIP = 1
REALTIME_STEP = TIME_STEP * FRAME_SKIP
CONTROL_FREQ = 1 / REALTIME_STEP

CAMERA_CONFIG = {"top": 7070}

def register_env(orcagym_addr : str, 
                 env_name : str, 
                 env_index : int, 
                 agent_name : str, 
                 max_episode_steps : int,
                 task : str,
                 obs_type : str,
                 ) -> str:
    orcagym_addr_str = orcagym_addr.replace(":", "-")
    env_id = env_name + "-OrcaGym-" + orcagym_addr_str + f"-{env_index:03d}"
    agent_names = [f"{agent_name}"]
    kwargs = {'frame_skip': FRAME_SKIP,   
                'orcagym_addr': orcagym_addr, 
                'agent_names': agent_names, 
                'time_step': TIME_STEP,
                'camera_config': CAMERA_CONFIG,
                'obs_type': obs_type,
                'task': task,
                'render_mode': "human",}           
    gym.register(
        id=env_id,
        entry_point=ENV_ENTRY_POINT[env_name],
        kwargs=kwargs,
        max_episode_steps= max_episode_steps,
        reward_threshold=0.0,
        # Even after seeding, the rendered observations are slightly different,
        # so we set `nondeterministic=True` to pass `check_env` tests        
        nondeterministic=True,
    )
    return env_id, kwargs


@dataclasses.dataclass
class Args:
    orca_gym_address: str = 'localhost:50051'
    env_name: str = "AlohaTransferCube"
    seed: int = 0
    agent_name: str = "bimanual_viperx_transfer_cube_usda"
    record_time: int = 60
    task: str = "transfer_cube"
    obs_type: str = "pixels"

    action_horizon: int = 10

    host: str = "0.0.0.0"
    port: int = 8000

    display: bool = False

def main(args: Args) -> None:
    max_episode_steps = int(args.record_time * CONTROL_FREQ)
    env_id, kwargs = register_env(
        orcagym_addr=args.orca_gym_address,
        env_name=args.env_name, 
        env_index=0, 
        agent_name=args.agent_name, 
        max_episode_steps=max_episode_steps,
        task=args.task,
        obs_type=args.obs_type,
    )
    
    print("Registered Simulation Environment: ", env_id, " with kwargs: ", kwargs)

    # 启动 Monitor 子进程
    ports = [7070]
    monitor_processes = []
    for port in ports:
        process = camera_monitor.start_monitor(port=port)
        monitor_processes.append(process)
    
    runtime = _runtime.Runtime(
        environment=_env.AlohaOpenpiEnv(
            env_id=env_id,
            seed=args.seed,
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
