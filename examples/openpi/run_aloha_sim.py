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

ENV_ENTRY_POINT = {
    "AlohaTransferCube": "envs.aloha.aloha_env:AlohaEnv"
}

TIME_STEP = 0.005
FRAME_SKIP = 8
REALTIME_STEP = TIME_STEP * FRAME_SKIP
CONTROL_FREQ = 1 / REALTIME_STEP

def register_env(orcagym_addr : str, 
                 env_name : str, 
                 env_index : int, 
                 agent_name : str, 
                 max_episode_steps : int,
                 ) -> str:
    orcagym_addr_str = orcagym_addr.replace(":", "-")
    env_id = env_name + "-OrcaGym-" + orcagym_addr_str + f"-{env_index:03d}"
    agent_names = [f"{agent_name}"]
    kwargs = {'frame_skip': FRAME_SKIP,   
                'orcagym_addr': orcagym_addr, 
                'agent_names': agent_names, 
                'time_step': TIME_STEP}           
    gym.register(
        id=env_id,
        entry_point=ENV_ENTRY_POINT[env_name],
        kwargs=kwargs,
        max_episode_steps= max_episode_steps,
        reward_threshold=0.0,
    )
    return env_id, kwargs


@dataclasses.dataclass
class Args:
    orca_gym_address: str = 'localhost:50051'
    task: str = "AlohaTransferCube"
    seed: int = 0
    agent_name: str = "bimanual_viperx_transfer_cube_usda"
    record_time: int = 60

    action_horizon: int = 10

    host: str = "0.0.0.0"
    port: int = 8000

    display: bool = False

def main(args: Args) -> None:
    max_episode_steps = args.record_time * CONTROL_FREQ
    env_id, kwargs = register_env(
        orcagym_addr=args.orca_gym_address,
        env_name=args.task, 
        env_index=0, 
        agent_name=args.agent_name, 
        max_episode_steps=max_episode_steps
    )
    
    print("Registered Simulation Environment: ", env_id, " with kwargs: ", kwargs)
    
    runtime = _runtime.Runtime(
        environment=_env.AlohaSimEnvironment(
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


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    tyro.cli(main)
