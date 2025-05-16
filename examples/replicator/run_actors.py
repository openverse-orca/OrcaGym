from orca_gym.scene.orca_gym_scene import OrcaGymScene, Actor, LightInfo, CameraSensorInfo, MaterialInfo
import numpy as np
import orca_gym.utils.rotations as rotations
import time
import random
import gymnasium as gym
import sys
from datetime import datetime
import os

def create_scene() -> OrcaGymScene:
    """
    Run the Replicator scene.
    """
    grpc_addr = "localhost:50051"
    scene = OrcaGymScene(grpc_addr)

    actor = Actor(
        name=f"original_red_cup",
        spawnable_name="cup_of_coffee_usda",
        position=np.array([np.random.uniform(0.0, 0.5), 
                           np.random.uniform(0.0, 0.5), 
                           np.random.uniform(1.0, 2.0)]),
        rotation=rotations.euler2quat(np.array([np.random.uniform(-np.pi, np.pi), 
                                                np.random.uniform(-np.pi, np.pi), 
                                                np.random.uniform(-np.pi, np.pi)])),
        scale=1.0,
    )
    scene.add_actor(actor)

    for i in range(10):
        actor = Actor(
            name=f"cup_with_random_color_and_scale_{i}",
            spawnable_name="cup_of_coffee_usda",
            position=np.array([np.random.uniform(-1.2, 1.2), 
                            np.random.uniform(-1.2, 1.2), 
                            np.random.uniform(1.0, 2.0)]),
            rotation=rotations.euler2quat(np.array([np.random.uniform(-np.pi, np.pi), 
                                                    np.random.uniform(-np.pi, np.pi), 
                                                    np.random.uniform(-np.pi, np.pi)])),
            scale=np.random.uniform(1.0, 2.0),
        )
        scene.add_actor(actor)


    actor = Actor(
        name="cart_basket",
        spawnable_name="cart_basket_usda",
        position=np.array([0, 0, 0.0]),
        rotation=rotations.euler2quat(np.array([0.0, 0.0, np.random.uniform(-np.pi, np.pi)])),
        scale=1.0,
    )
    scene.add_actor(actor)

    actor = Actor(
        name="office_desk",
        spawnable_name="office_desk_7_mb_usda",
        position=np.array([0, 0, 0.0]),
        rotation=rotations.euler2quat(np.array([0.0, 0.0, 0])),
        scale=1.0,
    )
    scene.add_actor(actor)

    actor = Actor(
        name="dafault_camera",
        spawnable_name="cameraviewport",
        position=np.array([-1, -1, 1.2]),
        rotation=rotations.euler2quat(np.array([0, 0, -np.pi / 4])),
        scale=1.0,
    )
    scene.add_actor(actor)

    scene.publish_scene()

    # scene.make_camera_viewport_active("default_camera", "CameraViewport")
    # for i in range(10):
    #     material_info = MaterialInfo(
    #         base_color=np.array([
    #             np.random.uniform(0.0, 1.0),
    #             np.random.uniform(0.0, 1.0),
    #             np.random.uniform(0.0, 1.0),
    #             1.0]),
    #     )
    #     scene.set_material_info(f"cup_with_random_color_and_scale_{i}", material_info)


    print("Replicator scene published successfully.")

    return scene


def destroy_scene(scene: OrcaGymScene):
    scene.close()
    print("Replicator scene closed successfully.")


ENV_ENTRY_POINT = {
    "Actors": "examples.replicator.actors_env:ActorsEnv",
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
                agent_name : str,):
    env = None  # Initialize env to None
    try:
        print("simulation running... , orcagym_addr: ", orcagym_addr)

        env_name = "Actors"
        env_index = 0
        env_id, kwargs = register_env(orcagym_addr, 
                                      env_name, 
                                      env_index, 
                                      agent_name, 
                                      sys.maxsize)
        print("Registered environment: ", env_id)

        env = gym.make(env_id)        
        print("Starting simulation...")

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
    scene = create_scene()

    orcagym_addr = "localhost:50051"
    agent_name = "NoRobot"
    # run_simulation(orcagym_addr, agent_name)

    time.sleep(10)

    destroy_scene(scene)

