import robomimic.envs.env_base as EB

import h5py
import numpy as np
import json
import os
from envs.openloong.camera_wrapper import *

class DatasetWriter:
    def __init__(self, file_path, env_name, env_version, env_kwargs=None):
        """
        初始化 DatasetWriter。

        参数：
        - file_path: 要创建的 HDF5 文件路径。
        - env_name: 环境名称。
        - env_type: 环境类型。
        - env_version: 环境版本。
        - env_kwargs: 环境参数字典（可选）。
        """
        self.file_path = file_path
        self.env_args = {
            "env_name": env_name,
            "type": EB.EnvType.ORCA_GYM_TYPE,
            "env_version": env_version,
            "env_kwargs": env_kwargs or {}
        }

        # 如果文件已存在，删除它以清空内容
        if os.path.exists(self.file_path):
            os.remove(self.file_path)

        # 创建新的 HDF5 文件，写入初始数据，然后关闭文件
        with h5py.File(self.file_path, 'w') as f:
            data_group = f.create_group('data')
            data_group.attrs['env_args'] = json.dumps(self.env_args)
            data_group.attrs['total'] = 0  # 初始化总样本数为 0
            data_group.attrs['demo_count'] = 0  # 初始化演示计数为 0

            f.create_group('mask')  # 创建掩码组用于过滤器键（可选）

    def add_demo(self, demo_data, save_camera, camera_name_list, ts_list, model_file=None):
        """
        添加一个新的演示（trajectory）。

        参数：
        - demo_data: 包含演示数据的字典，结构如下：
            {
                'states': np.ndarray (N, D),
                'actions': np.ndarray (N, A),
                'rewards': np.ndarray (N,),
                'dones': np.ndarray (N,),
                'obs': dict of np.ndarrays
                # 'next_obs' 可选，如果未提供，将自动生成
            }
        - model_file: MJCF MuJoCo 模型的 XML 字符串（可选，仅用于 robosuite 数据集）。
        """
        # 打开文件进行读写
        with h5py.File(self.file_path, 'r+') as f:
            data_group = f['data']

            # 获取当前的演示计数和总样本数
            demo_count = data_group.attrs['demo_count']
            total_samples = data_group.attrs['total']

            demo_name = f'demo_{demo_count}'
            demo_group = data_group.create_group(demo_name)
            num_samples = demo_data['actions'].shape[0]
            demo_group.attrs['num_samples'] = num_samples

            if model_file:
                demo_group.attrs['model_file'] = model_file

            # 存储数据集
            for key in ['states', 'actions', 'rewards', 'dones']:
                data = demo_data.get(key)
                if data is not None:
                    demo_group.create_dataset(key, data=data)

            # 处理 obs
            obs_group = demo_group.create_group('obs')
            for obs_key, obs_data in demo_data['obs'].items():
                obs_group.create_dataset(obs_key, data=obs_data)

            if save_camera:
                print("Saving camera data")
                for camera_name in camera_name_list:
                    camera_group = demo_group.create_group("camera")
                    for camera_name in camera_name_list:
                        frames = []
                        for ts in ts_list:
                            parser = CameraDataParser(camera_name)
                            index, frame = parser.get_closed_frame(ts)
                            frames.append(frame)
                        camera_group.create_dataset(camera_name, data=np.array(frames))
                print("Camera data saved")

            # 自动生成 next_obs
            if 'next_obs' in demo_data:
                next_obs_data = demo_data['next_obs']
            else:
                next_obs_data = self._generate_next_obs(demo_data['obs'])

            next_obs_group = demo_group.create_group('next_obs')
            for obs_key, obs_data in next_obs_data.items():
                next_obs_group.create_dataset(obs_key, data=obs_data)

            # 更新总样本数和演示计数
            total_samples += num_samples
            demo_count += 1
            data_group.attrs['total'] = total_samples
            data_group.attrs['demo_count'] = demo_count

    def _generate_next_obs(self, obs):
        """
        从 obs 自动生成 next_obs。

        参数：
        - obs: 观测数据字典

        返回：
        - next_obs: 下一个观测数据字典，结构与 obs 相同。
        """
        next_obs = {obs_key: [] for obs_key in obs.keys()}
        for obs_key, obs_data in obs.items():
            next_obs_data = obs_data[1:]
            next_obs_data.append(obs_data[-1])
            next_obs[obs_key].append(next_obs_data)
        return next_obs

    def add_filter_key(self, filter_key_name, demo_names):
        """
        添加一个过滤器键，用于数据集的子集划分。

        参数：
        - filter_key_name: 过滤器键的名称。
        - demo_names: 包含演示名称的列表，例如 ['demo_0', 'demo_2', 'demo_5']。
        """
        # 打开文件进行读写
        with h5py.File(self.file_path, 'r+') as f:
            mask_group = f['mask']
            # 将演示名称转换为字节串，适用于 HDF5
            demo_names_bytes = np.array(demo_names, dtype=h5py.special_dtype(vlen=str))
            mask_group.create_dataset(filter_key_name, data=demo_names_bytes)

    def finalize(self):
        """
        由于每次操作都已保存并关闭文件，因此此方法可为空或用于其他清理操作。
        """
        pass  # 在此示例中，无需执行任何操作


class DatasetReader:
    def __init__(self, file_path):
        """
        初始化 DatasetReader。

        参数：
        - file_path: 数据集的 HDF5 文件路径。
        """
        self.file_path = file_path
        self.env_args = None

        # 打开文件并读取环境信息
        with h5py.File(self.file_path, 'r') as f:
            if 'data' not in f:
                raise ValueError("HDF5 文件中不存在 'data' 组")
            data_group = f['data']
            if 'env_args' in data_group.attrs:
                self.env_args = json.loads(data_group.attrs['env_args'])

    def get_demo_names(self):
        """
        获取所有演示的名称。

        返回：
        - demo_names: 包含演示名称的列表。
        """
        with h5py.File(self.file_path, 'r') as f:
            return list(f['data'].keys())

    def load_demo(self, demo_name):
        """
        加载指定的演示数据。

        参数：
        - demo_name: 要加载的演示名称。

        返回：
        - demo_data: 包含演示数据的字典。
        """
        with h5py.File(self.file_path, 'r') as f:
            data_group = f['data']
            if demo_name not in data_group:
                raise ValueError(f"演示 '{demo_name}' 不存在")

            demo_group = data_group[demo_name]
            demo_data = {
                "states": demo_group['states'][:],
                "actions": demo_group['actions'][:],
                "rewards": demo_group['rewards'][:],
                "dones": demo_group['dones'][:],
                "obs": {},
                "next_obs": {}
            }

            # 加载 obs
            obs_group = demo_group['obs']
            for obs_key in obs_group.keys():
                demo_data["obs"][obs_key] = obs_group[obs_key][:]

            # 加载 next_obs
            if 'next_obs' in demo_group:
                next_obs_group = demo_group['next_obs']
                for obs_key in next_obs_group.keys():
                    demo_data["next_obs"][obs_key] = next_obs_group[obs_key][:]
            else:
                # 如果没有 next_obs，可以选择生成
                demo_data["next_obs"] = self._generate_next_obs(demo_data["obs"])

            # 加载摄像机数据（如果存在）
            if 'camera' in demo_group:
                camera_group = demo_group['camera']
                demo_data['camera'] = {
                    camera_name: camera_group[camera_name][:]
                    for camera_name in camera_group.keys()
                }

            return demo_data

    def load_filtered_demos(self, filter_key_name):
        """
        加载指定过滤器键的演示。

        参数：
        - filter_key_name: 过滤器键的名称。

        返回：
        - demos: 包含过滤后的演示数据字典。
        """
        with h5py.File(self.file_path, 'r') as f:
            mask_group = f['mask']
            if filter_key_name not in mask_group:
                raise ValueError(f"过滤器键 '{filter_key_name}' 不存在")

            demo_names = mask_group[filter_key_name][:]
            demo_names = [name.decode('utf-8') for name in demo_names]

        # 加载演示
        return {demo_name: self.load_demo(demo_name) for demo_name in demo_names}

    def _generate_next_obs(self, obs):
        """
        从 obs 自动生成 next_obs。

        参数：
        - obs: 观测数据字典。

        返回：
        - next_obs: 下一个观测数据字典，结构与 obs 相同。
        """
        next_obs = {obs_key: [] for obs_key in obs.keys()}
        for obs_key, obs_data in obs.items():
            next_obs[obs_key] = np.concatenate([obs_data[1:], obs_data[-1:]], axis=0)
        return next_obs

    def get_env_args(self):
        """
        获取环境参数。

        返回：
        - env_args: 环境参数字典。
        """
        return self.env_args

