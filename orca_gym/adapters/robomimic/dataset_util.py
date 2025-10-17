import cv2
import robomimic.envs.env_base as EB
import h5py
import numpy as np
import json
import os
import uuid  # 新增导入uuid模块
import time  # 新增导入time模块
import shutil
import hashlib
class DatasetWriter:
    # def __init__(self, base_dir, env_name, env_version, env_kwargs=None):  # 修改1: 参数名改为base_dir
    #     """
    #     初始化 DatasetWriter。

    #     参数：
    #     - base_dir: 基础目录路径，用于创建UUID子目录
    #     - env_name: 环境名称。
    #     - env_version: 环境版本。
    #     - env_kwargs: 环境参数字典（可选）。
    #     """
    #     self.experiment_id = str(uuid.uuid4())[:8]  # 修改2: 使用8位UUID简化路径
    #     self.uuid_dir = os.path.join(base_dir, f"{self.experiment_id}_{int(time.time())}")  # 修改3: 结合时间戳
    #     os.makedirs(self.uuid_dir, exist_ok=True)  # 新增: 创建UUID目录
        
    #     self.mp4_save_path : str = None

    #     self._env_args = {
    #         "env_name": env_name,
    #         "type": EB.EnvType.ORCA_GYM_TYPE,
    #         "env_version": env_version,
    #         "env_kwargs": env_kwargs or {}
    #     }

    #     # 创建新的 HDF5 文件，写入初始数据，然后关闭文件
    #     self.hdf5_path = os.path.join(self.uuid_dir, "env_data.hdf5")  # 新增: 定义HDF5文件路径
    #     with h5py.File(self.hdf5_path, 'w') as f:  # 修改4: 使用hdf5_path
    #         data_group = f.create_group('data')
    #         data_group.attrs['env_args'] = json.dumps(self._env_args)
    #         data_group.attrs['total'] = 0  # 初始化总样本数为 0
    #         data_group.attrs['demo_count'] = 0  # 初始化演示计数为 0
    #         f.create_group('mask')  # 创建掩码组用于过滤器键（可选）
    def __init__(self, base_dir, env_name, env_version, env_kwargs=None, specific_file_path=None):  # 修改1: 参数名改为base_dir
        """
        初始化 DatasetWriter。

        参数：
        - base_dir: 基础目录路径，用于创建UUID子目录
        - env_name: 环境名称。
        - env_version: 环境版本。
        - env_kwargs: 环境参数字典（可选）。
        """

        self._env_args = {
            "env_name": env_name,
            "type": EB.EnvType.ORCA_GYM_TYPE,
            "env_version": env_version,
            "env_kwargs": env_kwargs or {}
        }
        self.pathremoved = False
       # print("base_dir..................:", base_dir)
        if specific_file_path is not None:  # 新增: 如果提供了特定文件路径
            self.hdf5_path = specific_file_path  # 使用特定文件路径
            self.create_hdf5_file()  # 新增: 创建HDF5文件
        else:
            self.basedir = base_dir
            # self.experiment_id = str(uuid.uuid4())[:8]  # 修改2: 使用8位UUID简化路径
            # self.uuid_dir = os.path.join(base_dir, f"{self.experiment_id}_{int(time.time())}")  # 修改3: 结合时间戳

            # uuid1 = str(uuid.uuid4())[:8]
            # uuid2 = str(uuid.uuid4())[:8]
            raw_uuid = str(uuid.uuid4())
            hash_digest = hashlib.sha256(raw_uuid.encode('utf-8')).hexdigest()
            short_id = hash_digest[:16]
            self.experiment_id = short_id
            # self.experiment_id = f"{uuid1}_{uuid2}"
            self.uuid_dir = os.path.join(base_dir, self.experiment_id)
            self.camera_dir = os.path.join(self.uuid_dir, "camera")
            self.parameters_dir = os.path.join(self.uuid_dir, "parameters")
            self.proprio_stats_dir = os.path.join(self.uuid_dir, "proprio_stats")
            self.depth_dir = os.path.join(self.camera_dir, "depth")
            self.video_dir = os.path.join(self.camera_dir, "video")
            print("self.uuid_dir..............:",self.uuid_dir)
            self.hdf5_path = os.path.join(self.proprio_stats_dir, "proprio_stats.hdf5") 
            self.mp4_save_path : str = None
            
        



    def create_hdf5_file(self):
        with h5py.File(self.hdf5_path, 'w') as f:  # 修改4: 使用hdf5_path
            data_group = f.create_group('data')
            data_group.attrs['env_args'] = json.dumps(self._env_args)
            data_group.attrs['total'] = 0  # 初始化总样本数为 0
            data_group.attrs['demo_count'] = 0  # 初始化演示计数为 0
            f.create_group('mask')  # 创建掩码组用于过滤器键（可选）

    def set_UUIDPATH(self):
        # self.experiment_id = str(uuid.uuid4())[:8]  # 修改2: 使用8位UUID简化路径
        # self.uuid_dir = os.path.join(self.basedir, f"{self.experiment_id}_{int(time.time())}")  # 修改3: 结合时间戳
        # uuid1 = str(uuid.uuid4())[:8]
        # uuid2 = str(uuid.uuid4())[:8]
        # self.experiment_id = f"{uuid1}_{uuid2}"
        raw_uuid = str(uuid.uuid4())
        hash_digest = hashlib.sha256(raw_uuid.encode('utf-8')).hexdigest()
        short_id = hash_digest[:16]
        self.experiment_id = short_id
        self.uuid_dir = os.path.join(self.basedir, self.experiment_id)
        self.camera_dir = os.path.join(self.uuid_dir, "camera")
        self.proprio_stats_dir = os.path.join(self.uuid_dir, "proprio_stats")
        self.depth_dir = os.path.join(self.camera_dir, "depth")
        self.video_dir = os.path.join(self.camera_dir, "video")
        self.parameters_dir = os.path.join(self.uuid_dir, "parameters")
       # print("self.uuid_dir:",self.uuid_dir)
        os.makedirs(self.uuid_dir, exist_ok=True)  # 新增: 创建UUID目录
        os.makedirs(self.camera_dir, exist_ok=True)
        os.makedirs(self.parameters_dir, exist_ok=True)
        os.makedirs(self.proprio_stats_dir, exist_ok=True)
        os.makedirs(self.depth_dir, exist_ok=True)
        os.makedirs(self.video_dir, exist_ok=True)
        # 创建新的 HDF5 文件，写入初始数据，然后关闭文件
        self.hdf5_path = os.path.join(self.proprio_stats_dir, "proprio_stats.hdf5")  # 新增: 定义HDF5文件路径
        # self.hdf5_path = os.path.join(self.uuid_dir)
        # print("hdf5_path11112222:",self.hdf5_path)
        self.create_hdf5_file()  # 新增: 创建HDF5文件
        self.pathremoved = False
    
    def get_UUIDPath(self):
        return self.uuid_dir

    def remove_path(self):
        rmpath = self.uuid_dir
        print(f"删除目录: {rmpath}")
        if os.path.exists(rmpath) and os.path.isdir(rmpath):
            shutil.rmtree(rmpath)
        self.pathremoved = True


    @staticmethod
    def remove_episode_from_json(json_path, episode_id):
    # 删除 JSON 文件中指定 episode_id 的记录
        if not os.path.exists(json_path):
            print(f"JSON file {json_path} not found!")
            return

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # 保留不是这个 episode_id 的条目
        new_data = [item for item in data if item["episode_id"] != episode_id]

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(new_data, f, ensure_ascii=False, indent=4)
        
        print(f"Removed episode {episode_id} from {json_path}")


        


    def set_env_kwargs(self, env_kwargs):
        """
        设置环境参数。

        参数：
        - env_kwargs: 环境参数字典。
        """
        self._env_args['env_kwargs'] = env_kwargs
        with h5py.File(self.hdf5_path, 'r+') as f:  # 修改5: 使用hdf5_path
            data_group = f['data']
            data_group.attrs['env_args'] = json.dumps(self._env_args)

    def get_cur_demo_name(self) -> str:
        with h5py.File(self.hdf5_path, 'r') as f:  # 修改6: 使用hdf5_path
            data_group = f['data']
            demo_count = data_group.attrs['demo_count']
        return f'demo_{demo_count:05d}'  # 返回当前演示的名称格式
    
    
    def get_mp4_save_path(self):  # 修改7: 增加demo_name和camera_name参数
        """
        生成视频保存路径。

        参数：
        - demo_name: 演示名称，如 'demo_00000'
        - camera_name: 相机名称，如 'camera_head_color'

        返回：
        - 视频文件的完整保存路径
        """
        # print("file_path:",self.file_path)
      #  print("uuid_dir 111111111111:",self.uuid_dir)

        # demo_dir = os.path.join(self.uuid_dir)  # 新增: 创建demo子目录
        demo_dir = os.path.join(self.camera_dir)
     #   print("Demo dir:",demo_dir)
        os.makedirs(demo_dir, exist_ok=True)  # 新增: 确保目录存在
        retpath = os.path.join(demo_dir) 
        #print("retpath:",retpath)
        return retpath  # 修改8: 返回demo子目录中的路径

    # def get_mp4_save_path(self) -> str:
    #     self.mp4_save_path = os.path.join(self.file_path.removesuffix('.hdf5'), self.get_cur_demo_name())
    #     return self.mp4_save_path
    
    def add_demo_data(self, demo_data, model_file=None):
        """
        添加一个新的演示（trajectory）。

        参数：
        - demo_data: 包含演示数据的字典，结构如下：
            {
                'states': np.ndarray (N, D),    # 机器人关节、夹爪、物体 的位姿、速度
                'actions': np.ndarray (N, A),   # 机械臂末端的位姿、夹爪的开合程度
                'objects': str,   # 物体的相关信息 json格式
                'goals': str     # 目标的相关信息 json格式
                'rewards': np.ndarray (N,),     # 奖励
                'dones': np.ndarray (N,),       # 完成标志
                'obs': dict of np.ndarrays      # 观测数据字典
                'timesteps': np.ndarray (N,)    # 仿真时间，单位为秒
                'language_instruction': str     # 语言指令（可选）
                'next_obs'                      # 如果未提供，将自动生成
                'camera_frames'                 # 可选，用于存储相机帧，格式为字典，键为相机名称，值为帧列表
                'camera_time_stamp': dict       # 相机时间戳，键为相机名称，值为时间戳列表
            }
        - model_file: MJCF MuJoCo 模型的 XML 字符串（可选，仅用于 robosuite 数据集）。
        """
        # 打开文件进行读写
        with h5py.File(self.hdf5_path, 'r+') as f:  # 修改9: 使用hdf5_path
            data_group = f['data']

            # 获取当前的演示计数和总样本数
            total_samples = data_group.attrs['total']
            # demo_name_h5 = self.get_cur_demo_name()   # 获取HDF5中的演示名称
            demo_name = self.get_cur_demo_name()
            demo_count = data_group.attrs['demo_count']  # 获取当前演示计数

            # demo_group = data_group.create_group(demo_name_h5)
            demo_group = data_group.create_group(demo_name)
            num_samples = demo_data['actions'].shape[0]
            demo_group.attrs['num_samples'] = num_samples

            if model_file:
                demo_group.attrs['model_file'] = model_file

            for key in ['states', 'actions', 'language_instruction', 'objects', 'goals', 'rewards', 'dones', 'timesteps', 'timestamps']:
                if key in demo_data:
                    demo_group.create_dataset(key, data=demo_data[key])
            
            # 处理task_info（任务信息）
            if 'task_info' in demo_data:
                task_info_group = demo_group.create_group('task_info')
                for task_key, task_value in demo_data['task_info'].items():
                    if isinstance(task_value, str):
                        # 字符串类型，转换为字节存储
                        task_info_group.create_dataset(task_key, data=task_value.encode('utf-8'))
                    else:
                        # 其他类型，直接存储
                        task_info_group.create_dataset(task_key, data=task_value)

            # 处理 obs
            obs_group = demo_group.create_group('obs')
            for obs_key, obs_data in demo_data['obs'].items():
                obs_group.create_dataset(obs_key, data=obs_data, compression="gzip", compression_opts=4)
 
            camera_group = demo_group.create_group('camera')
            for key in ['camera_frames', 'camera_time_stamp']:
                if key == 'camera_frames' and 'camera_frames' in demo_data:
                    # 如果存在 camera_frames，则创建相机帧数据集
                    demo_group.create_dataset('camera_frames', data=demo_data['camera_frames'], compression="gzip", compression_opts=4)
                if key == 'camera_time_stamp' and 'camera_time_stamp' in demo_data:
                    for camera_name, time_stamps in demo_data['camera_time_stamp'].items():
                        camera_group.create_dataset(camera_name, data=time_stamps, compression="gzip", compression_opts=4)

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

    def save_single_camera_video(self, frames, video_path, fps):  # 新增: 单个相机视频保存方法
        """保存单个相机视频"""
        os.makedirs(os.path.dirname(video_path), exist_ok=True)
        print(f"Saving video at {video_path}")
        frame_height, frame_width = frames[0].shape[:2]
        isColor = (frames[0].ndim == 3)
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(video_path, fourcc, fps, (frame_width, frame_height), isColor=isColor)
        for frame in frames:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            out.write(frame)
        out.release()

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
        with h5py.File(self.hdf5_path, 'r+') as f:  # 修改13: 使用hdf5_path
            mask_group = f['mask']
            # 将演示名称转换为字节串，适用于 HDF5
            demo_names_bytes = np.array([name.encode('utf-8') for name in demo_names], dtype='S')
            # 如果过滤键已存在，先删除再创建
            if filter_key_name in mask_group:
                del mask_group[filter_key_name]
            mask_group.create_dataset(filter_key_name, data=demo_names_bytes)
            print("Added filter key:", filter_key_name, "with", len(demo_names), "demos.")
            
    def remove_filter_key(self, filter_key_name):
        with h5py.File(self.hdf5_path, 'r+') as f:  # 修改14: 使用hdf5_path
            mask_group = f['mask']
            if filter_key_name in mask_group:
                del mask_group[filter_key_name]
                print("Removed filter key:", filter_key_name)

    def get_demo_names(self):
        with h5py.File(self.hdf5_path, 'r') as f:  # 修改15: 使用hdf5_path
            data_group = f['data']
            demo_names = [name for name in data_group.keys()]
        return demo_names
    
    def shuffle_demos(self, train_demo_ratio=0.8):
        """
        将 80% 的演示数据用于训练 (train)，剩余 20% 用于测试(valid)
        """

        if self.pathremoved:
            return
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
    
    # 删除原save_camera_video方法，使用新的save_single_camera_video方法

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
                'objects': np.ndarray (N, O),   # 物体的位姿
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
                'objects': np.array(demo_group['objects']) if 'objects' in demo_group else None,
                'goals': np.array(demo_group['goals']) if 'goals' in demo_group else None,
                'rewards': np.array(demo_group['rewards']),
                'dones': np.array(demo_group['dones']),
                'obs': {key: np.array(demo_group['obs'][key]) for key in demo_group['obs'].keys()},
                'timesteps': np.array(demo_group['timesteps']),
                'language_instruction': demo_group['language_instruction'][()] if 'language_instruction' in demo_group else None,
                'next_obs': {key: np.array(demo_group['next_obs'][key]) for key in demo_group['next_obs'].keys()},
                'camera_frames': np.array(demo_group['camera_frames'])
            }

            if 'timestamps' in demo_group:
                demo_data['timestamps'] = np.array(demo_group['timestamps'])
            if 'camera' in demo_group:
                demo_data['camera'] = {key: np.array(demo_group['camera'][key]) for key in demo_group['camera'].keys()}
        return demo_data
