import shutil
import uuid
from orca_gym.scene.orca_gym_scene import OrcaGymScene, Actor
from orca_gym.utils.rotations import euler2quat
import os
import time
import numpy as np

def generate_height_map_file(
    orcagym_addresses: list[str],
):
    print("=============> Generate height map file ...")

    # 调用 ../../orca_gym/tools/generate_height_map.py
    os.system(f"python ../../orca_gym/tools/terrains/height_map_generater.py --orcagym_addresses {orcagym_addresses[0]}")

    height_map_dir = os.path.join(os.path.expanduser("~"), ".orcagym", "height_map")

    # 用UUID生成一个唯一的文件名，并重命名 height_map.npy
    height_map_file = os.path.join(height_map_dir, f"height_map_{uuid.uuid4()}.npy")
    os.makedirs(height_map_dir, exist_ok=True)
    shutil.move("height_map.npy", height_map_file)

    print("=============> Generate height map file done. Height map file: ", height_map_file)

    return height_map_file

def clear_scene(
    orcagym_addresses: list[str],
):
    print("=============> Clear scene ...")

    scene = OrcaGymScene(orcagym_addresses[0])
    scene.publish_scene()
    time.sleep(1)
    scene.close()
    time.sleep(1)

    print("=============> Clear scene done.")


def publish_terrain(
    orcagym_addresses: list[str],
    terrain_spawnable_names: list[str],
):
    print("=============> Publish terrain ...")
    scene = OrcaGymScene(orcagym_addresses[0])
    for terrain_spawnable_name in terrain_spawnable_names:
        terrain = Actor(
            name=f"{terrain_spawnable_name}",
            spawnable_name=terrain_spawnable_name,
            position=[0, 0, 0],
            rotation=euler2quat([0, 0, 0]),
            scale=1.0,
        )
        scene.add_actor(terrain)
        print(f"    =============> Add terrain {terrain_spawnable_name} ...")
        time.sleep(0.01)

    scene.publish_scene()
    time.sleep(3)
    scene.close()
    time.sleep(1)

    print("=============> Publish terrain done.")

def publish_scene(
    orcagym_addresses: list[str],
    agent_name: str,
    agent_spawnable_name: str,
    agent_num: int,
    terrain_spawnable_names: list[str],
):
    print("=============> Publish scene ...")
    scene = OrcaGymScene(orcagym_addresses[0])
    # 排列成一个方阵，每个机器人间隔0.5米
    sqrt_width = int(np.ceil(np.sqrt(agent_num)))  # 向上取整
    base_offset_x = -(sqrt_width) / 2
    base_offset_y = -(sqrt_width) / 2
    for i in range(agent_num):
        x_pos = (i % sqrt_width) * 0.5 + base_offset_x
        y_pos = (i // sqrt_width) * 0.5 + base_offset_y
        actor = Actor(
            name=f"{agent_name}_{i:03d}",
            spawnable_name=agent_spawnable_name,
            position=[x_pos, y_pos, 0],
            rotation=euler2quat([0, 0, 0]),
            scale=1.0,
        )
        scene.add_actor(actor)
        print(f"    =============> Add agent {agent_name}_{i:03d} ...")
        time.sleep(0.01)

    print("=============> Publish terrain ...")
    scene = OrcaGymScene(orcagym_addresses[0])
    for terrain_spawnable_name in terrain_spawnable_names:
        terrain = Actor(
            name=f"{terrain_spawnable_name}",
            spawnable_name=terrain_spawnable_name,
            position=[0, 0, 0],
            rotation=euler2quat([0, 0, 0]),
            scale=1.0,
        )
        scene.add_actor(terrain)
        print(f"    =============> Add terrain {terrain_spawnable_name} ...")
        time.sleep(0.01)


    scene.publish_scene()
    time.sleep(3)
    scene.close()
    time.sleep(1)

    print("=============> Publish scene done.")
