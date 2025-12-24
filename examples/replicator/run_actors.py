from orca_gym.scene.orca_gym_scene import OrcaGymScene, Actor, LightInfo, CameraSensorInfo, MaterialInfo
import numpy as np
import orca_gym.utils.rotations as rotations
import time
import random
import gymnasium as gym
import sys
from datetime import datetime
import os
import run_simulation as sim

from orca_gym.log.orca_log import get_orca_logger
_logger = get_orca_logger()


def create_scene() -> OrcaGymScene:
    """
    Run the Replicator scene.
    """
    grpc_addr = "localhost:50051"
    scene = OrcaGymScene(grpc_addr)

    actor = Actor(
        name=f"original_red_cup",
        asset_path="assets/prefabs/cup_of_coffee_usda",
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
            asset_path="assets/prefabs/cup_of_coffee_usda",
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
        asset_path="assets/prefabs/cart_basket_usda",
        position=np.array([0, 0, 0.0]),
        rotation=rotations.euler2quat(np.array([0.0, 0.0, 0.0])),
        scale=1.0,
    )
    scene.add_actor(actor)

    actor = Actor(
        name="office_desk",
        asset_path="assets/prefabs/office_desk_7_mb_usda",
        position=np.array([0, 0, 0.0]),
        rotation=rotations.euler2quat(np.array([0.0, 0.0, 0])),
        scale=1.0,
    )
    scene.add_actor(actor)

    actor = Actor(
        name="default_camera",
        asset_path="assets/prefabs/cameraviewport",
        position=np.array([-1, -1, 1.2]),
        rotation=rotations.euler2quat(np.array([0, 0, -np.pi / 4])),
        scale=1.0,
    )
    scene.add_actor(actor)

    scene.publish_scene()

    # scene.make_camera_viewport_active("default_camera", "CameraViewport")
    for i in range(10):
        material_info = MaterialInfo(
            base_color=np.array([
                np.random.uniform(0.0, 1.0),
                np.random.uniform(0.0, 1.0),
                np.random.uniform(0.0, 1.0),
                1.0]),
        )
        scene.set_material_info(f"cup_with_random_color_and_scale_{i}", material_info)


    _logger.info("Replicator scene published successfully.")

    return scene


def destroy_scene(scene: OrcaGymScene):
    scene.publish_scene()
    scene.close()
    _logger.info("Replicator scene closed successfully.")



if __name__ == "__main__":
    scene = create_scene()

    orcagym_addr = "localhost:50051"
    agent_name = "NoRobot"
    env_name = "Actors"
    sim.run_simulation(orcagym_addr, agent_name, env_name)

    # time.sleep(3)

    destroy_scene(scene)

