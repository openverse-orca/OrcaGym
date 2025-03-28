"""
This script demonstrates how to convert data from the Orca Gym environment to the LeRobot dataset format.
"""

import shutil

from lerobot.common.datasets.lerobot_dataset import LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import tensorflow_datasets as tfds
import tyro

from orca_gym.robomimic.dataset_util import DatasetReader
import dataclasses
import argparse

REPO_NAME = "orca_gym/libero"  # Name of the output dataset, also used for the Hugging Face Hub



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
        fps=5,
        features={
            "image": {
                "dtype": "image",
                "shape": (240, 320, 3),
                "names": ["height", "width", "channel"],
            },
            # "wrist_image": {
            #     "dtype": "image",
            #     "shape": (256, 256, 3),
            #     "names": ["height", "width", "channel"],
            # },
            "state": {
                "dtype": "float32",
                "shape": (145,),
                "names": ["state"],
            },
            "actions": {
                "dtype": "float32",
                "shape": (14,),
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
            dataset.add_frame(
                {
                    "image": demo_data["obs"]["camera_head"][step],
                    # "wrist_image": step["observation"]["wrist_image"],
                    "state": demo_data["states"][step],
                    "actions": demo_data["actions"][step],
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