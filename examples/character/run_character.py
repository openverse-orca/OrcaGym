from orca_gym.scene.orca_gym_scene_runtime import OrcaGymSceneRuntime
import time
import gymnasium as gym
import sys
from datetime import datetime
from typing import Optional
from orca_gym.scene.orca_gym_scene import OrcaGymScene
import argparse

ENV_ENTRY_POINT = {
    "Character": "envs.character.character_env:CharacterEnv",
}

TIME_STEP = 0.001
FRAME_SKIP = 20
REALTIME_STEP = TIME_STEP * FRAME_SKIP
CONTROL_FREQ = 1 / REALTIME_STEP

def register_env(orcagym_addr : str, 
                 env_name : str, 
                 env_index : int, 
                 agent_name : str, 
                 max_episode_steps : int) -> tuple[ str, dict ]:
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



def run_simulation(orcagym_addr : str, 
                agent_name : str,
                env_name : str,
                scene_runtime: Optional[OrcaGymSceneRuntime] = None) -> None:
    env = None  # Initialize env to None
    try:
        print("simulation running... , orcagym_addr: ", orcagym_addr)

        env_index = 0
        env_id, kwargs = register_env(orcagym_addr, 
                                      env_name, 
                                      env_index, 
                                      agent_name, 
                                      sys.maxsize)
        print("Registered environment: ", env_id)

        env = gym.make(env_id)        
        print("Starting simulation...")

        if scene_runtime is not None:
            if hasattr(env, "set_scene_runtime"):
                env.set_scene_runtime(scene_runtime)
                print("Scene runtime is set.")
            else:
                env_unwarpped = env.unwrapped
                if hasattr(env_unwarpped, "set_scene_runtime"):
                    env_unwarpped.set_scene_runtime(scene_runtime)
                    print("Scene runtime is set.")
                else:
                    print("Scene runtime is not set. env: ", env)
                    print("Scene runtime is not set. env_unwarpped: ", env_unwarpped)

        obs = env.reset()
        while True:
            start_time = datetime.now()

            action = env.action_space.sample()
    
            obs, reward, terminated, truncated, info = env.step(action)

            env.render()

            elapsed_time = datetime.now() - start_time
            if elapsed_time.total_seconds() < REALTIME_STEP:
                time.sleep(REALTIME_STEP - elapsed_time.total_seconds())


    except KeyboardInterrupt:
        print("Simulation stopped")        
        if env is not None:
            env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Run character simulation")
    parser.add_argument("--orcagym_addr", type=str, default="localhost:50051")
    parser.add_argument("--agent_name", type=str, default="Remy")
    parser.add_argument("--env_name", type=str, default="Character")
    args = parser.parse_args()
    orcagym_addr = args.orcagym_addr
    agent_name = args.agent_name
    env_name = args.env_name

    scene = OrcaGymScene(orcagym_addr)
    scene_runtime = OrcaGymSceneRuntime(scene)
    run_simulation(orcagym_addr, agent_name, env_name, scene_runtime)
