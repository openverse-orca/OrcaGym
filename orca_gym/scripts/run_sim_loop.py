from orca_gym.scene.orca_gym_scene import OrcaGymScene, Actor, LightInfo, CameraSensorInfo, MaterialInfo
from orca_gym.scene.orca_gym_scene_runtime import OrcaGymSceneRuntime
import numpy as np
import orca_gym.utils.rotations as rotations
import time
import random
import gymnasium as gym
import sys
from datetime import datetime
import os
from typing import Optional
from orca_gym.scripts.mujoco_monitor import open_mujoco_monitor

ENV_ENTRY_POINT = {
    "SimulationLoop": "orca_gym.scripts.sim_env:SimEnv",
}

TIME_STEP = 0.005
FRAME_SKIP = 1
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

        if isinstance(env, gym.wrappers.TimeLimit):
            local_env = env.unwrapped
        else:
            local_env = env

        open_mujoco_monitor(local_env)

        if scene_runtime is not None:
            if hasattr(env, "set_scene_runtime"):
                print("Setting scene runtime...")
                env.set_scene_runtime(scene_runtime)
            elif hasattr(env.unwrapped, "set_scene_runtime"):
                print("Setting scene runtime...")
                env.unwrapped.set_scene_runtime(scene_runtime)

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
    orcagym_addr = "localhost:50051"
    agent_name = "NoRobot"
    env_name = "SimulationLoop"
    run_simulation(orcagym_addr, agent_name, env_name)
