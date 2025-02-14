import robomimic.envs.env_base as EB

import h5py
import numpy as np
import json
import os

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
        # 如果文件已存在，跳过创建和初始化
        if os.path.exists(self.file_path):
            reader = DatasetReader(self.file_path)
            self._env_args = reader._get_env_args()
            return

        self._env_args = {
            "env_name": env_name,
            "type": EB.EnvType.ORCA_GYM_TYPE,
            "env_version": env_version,
            "env_kwargs": env_kwargs or {}
        }

        # 创建新的 HDF5 文件，写入初始数据，然后关闭文件
        with h5py.File(self.file_path, 'w') as f:
            data_group = f.create_group('data')
            data_group.attrs['env_args'] = json.dumps(self._env_args)
            data_group.attrs['total'] = 0  # 初始化总样本数为 0
            data_group.attrs['demo_count'] = 0  # 初始化演示计数为 0

            f.create_group('mask')  # 创建掩码组用于过滤器键（可选）

    def set_env_kwargs(self, env_kwargs):
        """
        设置环境参数。

        参数：
        - env_kwargs: 环境参数字典。
        """
        self._env_args['env_kwargs'] = env_kwargs
        with h5py.File(self.file_path, 'r+') as f:
            data_group = f['data']
            data_group.attrs['env_args'] = json.dumps(self._env_args)

    def add_demo_data(self, demo_data, model_file=None):
        """
        添加一个新的演示（trajectory）。

        参数：
        - demo_data: 包含演示数据的字典，结构如下：
            {
                'states': np.ndarray (N, D),    # 机器人关节、夹爪、物体 的位姿、速度
                'actions': np.ndarray (N, A),   # 机械臂末端的位姿、夹爪的开合程度
                'goals': np.ndarray (N, G)      # 目标位姿（可选）
                'rewards': np.ndarray (N,),     # 奖励
                'dones': np.ndarray (N,),       # 完成标志
                'obs': dict of np.ndarrays      # 观测数据字典
                'timesteps': np.ndarray (N,)    # 仿真时间，单位为秒
                'language_instruction': str     # 语言指令（可选）
                'next_obs'                      # 如果未提供，将自动生成
                'camera_frames'                 # 可选，用于存储相机帧
            }
        - model_file: MJCF MuJoCo 模型的 XML 字符串（可选，仅用于 robosuite 数据集）。
        """
        # 打开文件进行读写
        with h5py.File(self.file_path, 'r+') as f:
            data_group = f['data']

            # 获取当前的演示计数和总样本数
            demo_count = data_group.attrs['demo_count']
            total_samples = data_group.attrs['total']

            demo_name = f'demo_{demo_count:05d}'   # 保存10万个演示
            demo_group = data_group.create_group(demo_name)
            num_samples = demo_data['actions'].shape[0]
            demo_group.attrs['num_samples'] = num_samples

            if model_file:
                demo_group.attrs['model_file'] = model_file

            # 存储数据集
            for key in ['states', 'actions', 'rewards', 'dones', 'goals', 'timesteps', 'language_instruction']:
                data = demo_data.get(key)
                if data is not None:
                    demo_group.create_dataset(key, data=data)

            # 处理 obs
            obs_group = demo_group.create_group('obs')
            for obs_key, obs_data in demo_data['obs'].items():
                obs_group.create_dataset(obs_key, data=obs_data, compression="gzip", compression_opts=4)

            if 'camera_frames' in demo_data:
                camera_frames = demo_data['camera_frames']
                for camera_name, frames in camera_frames.items():
                    camera_group = demo_group.create_group(camera_name)
                    for i, frame in enumerate(frames):
                        camera_group.create_dataset(f'frame_{i:05d}', data=frame, compression="gzip", compression_opts=4)

            # 自动生成 next_obs
            if 'next_obs' in demo_data:
                next_obs_data = demo_data['next_obs']
            else:
                next_obs_data = self._generate_next_obs(demo_data['obs'])

            next_obs_group = demo_group.create_group('next_obs')
            for obs_key, obs_data in next_obs_data.items():
                next_obs_group.create_dataset(obs_key, data=obs_data, compression="gzip", compression_opts=4)

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
            next_obs_data = np.concatenate([next_obs_data, np.expand_dims(obs_data[-1], axis=0)], axis=0)
            next_obs[obs_key] = next_obs_data
        return next_obs

    def add_filter_key(self, filter_key_name, demo_names):
        with h5py.File(self.file_path, 'r+') as f:
            mask_group = f['mask']
            # 将演示名称转换为字节串，适用于 HDF5
            demo_names_bytes = np.array([name.encode('utf-8') for name in demo_names], dtype='S')
            # 如果过滤键已存在，先删除再创建
            if filter_key_name in mask_group:
                del mask_group[filter_key_name]
            mask_group.create_dataset(filter_key_name, data=demo_names_bytes)
            print("Added filter key:", filter_key_name, "with", len(demo_names), "demos.")
            
    def remove_filter_key(self, filter_key_name):
        with h5py.File(self.file_path, 'r+') as f:
            mask_group = f['mask']
            if filter_key_name in mask_group:
                del mask_group[filter_key_name]
                print("Removed filter key:", filter_key_name)

    def get_demo_names(self):
        with h5py.File(self.file_path, 'r') as f:
            data_group = f['data']
            demo_names = [name for name in data_group.keys()]
        return demo_names
    
    def shuffle_demos(self, train_demo_ratio=0.8):
        """
        将 80% 的演示数据用于训练 (train)，剩余 20% 用于测试(valid)
        """
        self.remove_filter_key("train")
        self.remove_filter_key("valid")
        
        demo_names = self.get_demo_names()
        train_demos = []
        valid_demos = []
        for demo_name in demo_names:
            if np.random.rand() < train_demo_ratio:
                train_demos.append(demo_name)
                # print("Demo name: ", demo_name, " added to train.")
            else:
                valid_demos.append(demo_name)
                # print("Demo name: ", demo_name, " added to valid.")
                
        self.add_filter_key("train", train_demos)
        self.add_filter_key("valid", valid_demos)
            
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
        - file_path: 要读取的 HDF5 文件路径。
        """
        self.file_path = file_path

    def _get_env_args(self):
        """
        获取数据集的环境参数。

        返回：
        - env_args: 环境参数字典，结构如下：
            {
                'env_name': str,
                'type': str,
                'env_version': str,
                'env_kwargs': dict
            }
        """
        with h5py.File(self.file_path, 'r') as f:
            data_group = f['data']
            env_args = json.loads(data_group.attrs['env_args'])
        return env_args
    
    def get_env_name(self):
        """
        获取数据集的环境名称。

        返回：
        - env_name: 环境名称。
        """
        return self._get_env_args()['env_name']
    
    def get_env_version(self):
        """
        获取数据集的环境版本。

        返回：
        - env_version: 环境版本。
        """
        return self._get_env_args()['env_version']
    
    def get_env_kwargs(self):
        """
        获取数据集的环境参数。

        返回：
        - env_kwargs: 环境参数字典。
        """
        return self._get_env_args()['env_kwargs']
    
    def get_env_type(self):
        """
        获取数据集的环境类型。

        返回：
        - env_type: 环境类型。
        """
        return self._get_env_args()['type']

    def get_total_samples(self):
        """
        获取数据集中的总样本数。

        返回：
        - total_samples: 总样本数。
        """
        with h5py.File(self.file_path, 'r') as f:
            data_group = f['data']
            total_samples = data_group.attrs['total']
        return total_samples

    def get_demo_count(self):
        """
        获取数据集中的演示数量。

        返回：
        - demo_count: 演示数量。
        """
        with h5py.File(self.file_path, 'r') as f:
            data_group = f['data']
            demo_count = data_group.attrs['demo_count']
        return demo_count
    
    def get_demo_names(self, filter_key : str = None):
        """
        获取数据集中的所有演示名称。

        返回：
        - demo_names: 演示名称列表。
        """
        with h5py.File(self.file_path, 'r') as f:
            data_group = f['data']
            demo_names = [name for name in data_group.keys()]
            if filter_key is not None:
                mask_group = f['mask']
                filter_key_names = mask_group[filter_key]
                filter_key_names = [name.decode('utf-8') for name in filter_key_names]
                demo_names = [name for name in demo_names if name in filter_key_names]
        return demo_names
            
        #     data_group = f['data']
        #     demo_names = [name for name in data_group.keys()]
        # return demo_names

    def get_demo_data(self, demo_name):
        """
        获取指定名称的演示数据。

        参数：
        - demo_name: 演示名称。

        返回：
        - demo_data: 包含演示数据的字典，结构如下：
            {
                'states': np.ndarray (N, D),    # 机器人关节、夹爪、物体 的位姿、速度
                'actions': np.ndarray (N, A),   # 机械臂末端的位姿、夹爪的开合程度
                'goals': np.ndarray (N, G)      # 目标位姿（可选）
                'rewards': np.ndarray (N,),     # 奖励
                'dones': np.ndarray (N,),       # 完成标志
                'obs': dict of np.ndarrays      # 观测数据字典
                'timesteps': np.ndarray (N,)    # 仿真时间，单位为秒
                'language_instruction': str     # 语言指令（可选）
                'next_obs'                      # 如果未提供，将自动生成
                'camera_frames'                 # 用于存储相机帧 (可选)
            }
        """
        with h5py.File(self.file_path, 'r') as f:
            demo_group = f['data'][demo_name]
            demo_data = {
                'states': np.array(demo_group['states']),
                'actions': np.array(demo_group['actions']),
                'goals': np.array(demo_group['goals']) if 'goals' in demo_group else None,
                'rewards': np.array(demo_group['rewards']),
                'dones': np.array(demo_group['dones']),
                'obs': {key: np.array(demo_group['obs'][key]) for key in demo_group['obs'].keys()},
                'timesteps': np.array(demo_group['timesteps']),
                'language_instruction': demo_group['language_instruction'][()] if 'language_instruction' in demo_group else None,
                'next_obs': {key: np.array(demo_group['next_obs'][key]) for key in demo_group['next_obs'].keys()},
                'camera_frames': {camera_name: [np.array(frame_data) for _, frame_data in camera_group.items()]
                                  for camera_name, camera_group in demo_group.items() if camera_name.startswith('camera')},  
            }
        return demo_data