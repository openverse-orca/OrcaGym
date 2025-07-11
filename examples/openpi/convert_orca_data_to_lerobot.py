"""
This script demonstrates how to convert data from the Orca Gym environment to the LeRobot dataset format.
"""

import shutil

from lerobot.common.datasets.lerobot_dataset import LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import tyro

from orca_gym.adapters.robomimic.dataset_util import DatasetReader
import dataclasses
import argparse
import numpy as np

REPO_NAME = "orca_gym/AzureLoong"  # Name of the output dataset, also used for the Hugging Face Hub



def main(args) -> None:
    data_dir = args.data_dir
    
    # Clean up any existing dataset in the output directory
    output_path = LEROBOT_HOME / REPO_NAME
    if output_path.exists():
        shutil.rmtree(output_path)

    # Create LeRobot dataset, define features to store
    # OpenPi assumes that proprio is stored in `state` and actions in `action`
    # LeRobot assumes that dtype of image data is `image`
    dataset = LeRobotDataset.create(
        repo_id=REPO_NAME,
        robot_type="AzureLoong",
        fps=50,
        features={       
            "observation.images.head": {
                "dtype": "image",
                "shape": (224, 224, 3),
                "names": ["height", "width", "channel"],
            },
            "observation.state": {
                "dtype": "float32",
                "shape": (16,),
                "names": ["state"],
            },
            "action": {
                "dtype": "float32",
                "shape": (16,),
                "names": ["actions"],
            },
        },
        image_writer_threads=10,
        image_writer_processes=5,
    )

    # Loop over raw Libero datasets and write episodes to the LeRobot dataset
    # You can modify this for your own data format
    dataset_reader = DatasetReader(data_dir)
    demo_names = dataset_reader.get_demo_names()
    for demo_name in demo_names:
        demo_data = dataset_reader.get_demo_data(demo_name)
        demo_steps = len(demo_data["actions"])
        for step in range(demo_steps):
            action_data = demo_data["actions"][step]
            action_joint_pos = np.concatenate([action_data[6:14], action_data[20:28]]).flatten()       
            joint_qpos = np.concatenate([
                demo_data["obs"]["arm_joint_qpos_l"][step], 
                demo_data["obs"]["grasp_value_l"][step],
                demo_data["obs"]["arm_joint_qpos_r"][step],
                demo_data["obs"]["grasp_value_r"][step],
                ]).flatten()
            dataset.add_frame(
                {
                    "observation.images.head": demo_data["obs"]["camera_head"][step],
                    "action": action_joint_pos,                                        # action 是下发到ctrl的位控的pos
                    "observation.state": joint_qpos,                              # state 是记录的当前关节的qpos
                    # "observation.velocity": demo_data["obs"]["velocity"][step],
                    # "observation.effort": demo_data["obs"]["effort"][step],
                }
            )

        dataset.save_episode(task=demo_data["language_instruction"].decode("utf-8"))

    # Consolidate the dataset, skip computing stats since we will do that later
    dataset.consolidate(run_compute_stats=False)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert Orca Gym data to LeRobot dataset format')
    parser.add_argument('--data_dir', type=str, default=None, help='Directory to store the raw Orca Gym data')

    args = parser.parse_args()
    main(args)