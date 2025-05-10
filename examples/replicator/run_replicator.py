from orca_gym.scene.orca_gym_scene import OrcaGymScene, Actor
import numpy as np
import orca_gym.utils.rotations as rotations
import time
import random

def run_replicator():
    """
    Run the Replicator scene.
    """
    grpc_addr = "localhost:50051"
    scene = OrcaGymScene(grpc_addr)


    # actor = Actor(
    #     name=f"cup_of_coffee",
    #     spawnable_name="cup_of_coffee_usda",
    #     position=np.array([np.random.uniform(0.0, 0.5), np.random.uniform(0.0, 0.5), 2.0]),
    #     rotation=rotations.euler2quat(np.array([np.random.uniform(-np.pi, np.pi), 
    #                                             np.random.uniform(-np.pi, np.pi), 
    #                                             np.random.uniform(-np.pi, np.pi)])),
    #     scale=np.random.uniform(1.0, 2.0),
    # )
    # scene.add_actor(actor)


    # actor = Actor(
    #     name="cart_basket",
    #     spawnable_name="cart_basket_usda",
    #     position=np.array([np.random.uniform(0.0, 0.5), np.random.uniform(0.0, 0.5), 1.2]),
    #     rotation=rotations.euler2quat(np.array([0.0, 0.0, np.random.uniform(-np.pi, np.pi)])),
    #     scale=1.0,
    # )
    # scene.add_actor(actor)

    actor = Actor(
        name="office_desk_7_mb",
        spawnable_name="office_desk_7_mb_usda",
        position=np.array([np.random.uniform(0.0, 0.5), np.random.uniform(0.0, 0.5), 1.2]),
        rotation=rotations.euler2quat(np.array([0.0, 0.0, np.random.uniform(-np.pi, np.pi)])),
        scale=1.0,
    )
    scene.add_actor(actor)


    scene.publish_scene()
    print("Replicator scene published successfully.")


    scene.close()

if __name__ == "__main__":
    run_replicator()
