import argparse

import h5py
import numpy as np
import os
from pathlib import Path
from orca_gym.dataset_util import DatasetReader

class KPSDataSet:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None

    def create(self, data: dict):
        openpi_action = data['actions']
        openpi_obs = data['obs']
        openpi_timesteps = data['timesteps']
        len = openpi_timesteps.shape[0]
        index = range(len)

        kps_action_effector_position_l, kps_action_effector_position_r = openpi_obs['grasp_joint_pos_l'], openpi_obs['grasp_joint_pos_r']
        kps_action_effector_position = np.concatenate((kps_action_effector_position_l, kps_action_effector_position_r), axis=1)

        kps_action_end_orientation_l, kps_action_end_orientation_r = openpi_obs['ee_quat_l'], openpi_obs['ee_quat_r']
        kps_action_end_orientation = np.stack((kps_action_end_orientation_l, kps_action_end_orientation_r), axis=1)
        kps_action_end_position_l, kps_action_end_position_r = openpi_obs['ee_pos_l'], openpi_obs['ee_pos_r']
        kps_action_end_position = np.stack((kps_action_end_position_l, kps_action_end_position_r), axis=1)

        kps_action_joint_position_l, kps_action_joint_position_r = openpi_obs['arm_joint_qpos_l'], openpi_obs['arm_joint_qpos_r']
        kps_action_joint_position = np.concatenate((kps_action_joint_position_l, kps_action_joint_position_r), axis=1)

        kps_state_end_angular_l, kps_state_end_angular_r = openpi_obs['ee_vel_angular_l'], openpi_obs['ee_vel_angular_r']
        kps_state_end_angular = np.stack((kps_state_end_angular_l, kps_state_end_angular_r), axis=1)
        kps_state_end_velocity_l, kps_state_end_velocity_r = openpi_obs['ee_vel_linear_l'], openpi_obs['ee_vel_linear_r']
        kps_state_end_velocity = np.stack((kps_state_end_velocity_l, kps_state_end_velocity_r), axis=1)

        kps_state_joint_velocity_l, kps_state_joint_velocity_r = openpi_obs['arm_joint_vel_l'], openpi_obs['arm_joint_vel_r']
        kps_state_joint_velocity = np.concatenate((kps_state_joint_velocity_l, kps_state_joint_velocity_r), axis=1)

        with h5py.File(self.file_path, 'w') as f:
            action_group = f.create_group('action')
            state_group = f.create_group('state')
            timestamps = f.create_dataset('timestamps', data=openpi_timesteps, compression="gzip", compression_opts=4)

            action_effector = action_group.create_group('effector')
            action_effector_index = action_effector.create_dataset('index', data=index)
            action_effector_position = action_effector.create_dataset('position', data=kps_action_effector_position, compression="gzip", compression_opts=4)

            action_end = action_group.create_group('end')
            action_end_orientation = action_end.create_dataset('orientation', data=kps_action_end_orientation, compression="gzip", compression_opts=4)
            action_end_position = action_end.create_dataset('position', data=kps_action_end_position, compression="gzip", compression_opts=4)
            action_end_index = action_end.create_dataset('index', data=index, compression="gzip", compression_opts=4)

            action_head = action_group.create_group('head')
            action_head_position = action_head.create_dataset('position', data=np.zeros((len, 2)), compression="gzip", compression_opts=4)
            action_head_index = action_head.create_dataset('index', data=index, compression="gzip", compression_opts=4)

            action_joint = action_group.create_group('joints')
            action_joint_position = action_joint.create_dataset('position', data=kps_action_joint_position, compression="gzip", compression_opts=4)
            action_joint_index = action_joint.create_dataset('index', data=index, compression="gzip", compression_opts=4)

            action_robot = action_group.create_group('robot')
            action_robot_velocity = action_robot.create_dataset('velocity', data=np.zeros((len, 2)), compression="gzip", compression_opts=4)
            action_robot_index = action_robot.create_dataset('index', data=index, compression="gzip", compression_opts=4)

            action_waist = action_group.create_group('waist')
            action_waist_velocity = action_waist.create_dataset('velocity', data=np.zeros((len, 2)), compression="gzip", compression_opts=4)
            action_waist_index = action_waist.create_dataset('index', data=index, compression="gzip", compression_opts=4)

            state_effector = state_group.create_group('effector')
            state_effector_force = state_effector.create_dataset('force', data=np.zeros((len, 2)), compression="gzip", compression_opts=4)
            state_effector_position = state_effector.create_dataset('position', data=kps_action_effector_position, compression="gzip", compression_opts=4)

            state_end = state_group.create_group('end')
            state_end_angular = state_end.create_dataset('angular', data=kps_state_end_angular , compression="gzip", compression_opts=4)
            state_end_orientation = state_end.create_dataset('orientation', data=kps_action_end_orientation, compression="gzip", compression_opts=4)
            state_end_position = state_end.create_dataset('position', data=kps_action_end_position, compression="gzip", compression_opts=4)
            state_end_volocity = state_end.create_dataset('velocity', data=kps_state_end_velocity, compression="gzip", compression_opts=4)
            state_end_wrench = state_end.create_dataset('wrench', data=np.zeros((len, 2, 6)), compression="gzip", compression_opts=4)

            state_head = state_group.create_group('head')
            state_head_effort = state_head.create_dataset('effort', data=np.zeros((len, 2)), compression="gzip", compression_opts=4)
            state_head_position = state_head.create_dataset('position', data=np.zeros((len, 2)), compression="gzip", compression_opts=4)
            state_head_velocity = state_head.create_dataset('velocity', data=np.zeros((len, 2)), compression="gzip", compression_opts=4)

            state_joint = state_group.create_group('joint')
            state_joint_current_value = state_joint.create_dataset('current_value', data=np.zeros((len, 14)), compression="gzip", compression_opts=4)
            state_joint_effort = state_joint.create_dataset('effort', data=np.zeros((len, 14)), compression="gzip", compression_opts=4)
            state_joint_position = state_joint.create_dataset('position', data=kps_action_joint_position, compression="gzip", compression_opts=4)
            state_joint_velocity = state_joint.create_dataset('velocity', data=kps_state_joint_velocity, compression="gzip", compression_opts=4)

            state_robot = state_group.create_group('robot')
            state_robot_orientation = state_robot.create_dataset('orientation', data=np.zeros((len, 4)), compression="gzip", compression_opts=4)
            state_robot_orientation_dirft = state_robot.create_dataset('orientation_dirft', data=np.zeros((len, 4)), compression="gzip", compression_opts=4)
            state_robot_position = state_robot.create_dataset('position', data=np.zeros((len, 3)), compression="gzip", compression_opts=4)
            state_robot_position_dirft = state_robot.create_dataset('position_dirft', data=np.zeros((len, 3)), compression="gzip", compression_opts=4)

            state_waist = state_group.create_group('waist')
            state_waist_effort = state_waist.create_dataset('effort', data=np.zeros((len, 2)), compression="gzip", compression_opts=4)
            state_waist_position = state_waist.create_dataset('position', data=np.zeros((len, 2)), compression="gzip", compression_opts=4)
            state_waist_velocity = state_waist.create_dataset('velocity', data=np.zeros((len, 2)), compression="gzip", compression_opts=4)


    def create_dataset(self, group, data):
        pass



if __name__ == '__main__':
    parse = argparse.ArgumentParser(description="Convert OpenPI dataset to KPS dataset")
    parse.add_argument("--dataset_path", help="Path to OpenPI dataset")

    args = parse.parse_args()
    dataset_path = Path(args.dataset_path).resolve()

    basic_dataset_list = []
    with os.scandir(dataset_path) as entries:
        for entry in entries:
            if entry.is_dir():
                basic_dataset_list.append(entry)

    for basic_dataset in basic_dataset_list:
        openpi_dataset_path = os.path.join(dataset_path, basic_dataset, "proprio_stats", "proprio_stats.hdf5")
        openpi_dataset_reader = DatasetReader(openpi_dataset_path)
        demo_names = openpi_dataset_reader.get_demo_names()
        demo_data = openpi_dataset_reader.get_demo_data(demo_names[0])

        kps_dataset_path = os.path.join(dataset_path, basic_dataset, "proprio_stats", "proprio_stats_kps.hdf5")
        kps_dataset = KPSDataSet(kps_dataset_path)
        kps_dataset.create(demo_data)