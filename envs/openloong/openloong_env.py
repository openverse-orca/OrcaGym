import sys
import os

# 添加 libs 目录到 sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))  # 获取当前文件的目录
libs_dir = os.path.join(current_dir, 'libs')  # 构建 libs 目录路径
sys.path.append(libs_dir)  # 将 libs 目录添加到 sys.path

from openloong_dyn_ctrl import OpenLoongWBC, OrcaGym_Interface, ButtonState

import numpy as np
from gymnasium.core import ObsType
from envs.robot_env import MujocoRobotEnv
from orca_gym.utils import rotations
from typing import Optional, Any, SupportsFloat
from gymnasium import spaces
from orca_gym.devices.keyboard import KeyboardInput
import h5py

class RecordState:
    RECORD = "record"
    REPLAY = "replay"
    REPLAY_FINISHED = "replay_finished"
    NONE = "none"

class OpenLoongEnv(MujocoRobotEnv):
    """
    Control the walking of the OpenLoong robot.

    The OpenLoong project is an open-source project operated by Shanghai Humanoid Robot Co., Ltd., Shanghai Humanoid Robot Manufacturing Innovation Center, and OpenAtom Foundation.
    This environment adapts the motion control function of the OpenLoong robot, based on the "OpenLoong" robot model developed by Shanghai Humanoid Robot Innovation Center. 
    It provides three motion examples: walking, jumping, and blind stepping on obstacles. 
    Refer to the OpenLoong Dynamics Control project for more information (https://atomgit.com/openloong/openloong-dyn-control).
    """
    def __init__(
        self,
        frame_skip: int = 5,        
        grpc_address: str = 'localhost:50051',
        agent_names: list = ['Agent0'],
        time_step: float = 0.016,  # 0.016 for 60 fps        
        record_state: str = RecordState.NONE,        
        record_file: Optional[str] = None,
        urdf_path: str = "",
        json_path: str = "",
        log_path: str = "",
        **kwargs,
    ):

        self.record_state = record_state
        self.record_file = record_file
        self.record_pool = []
        self.RECORD_POOL_SIZE = 1000
        self.record_cursor = 0

        action_size = 3 # 实际并不使用

        super().__init__(
            frame_skip = frame_skip,
            grpc_address = grpc_address,
            agent_names = agent_names,
            time_step = time_step,            
            n_actions=action_size,
            observation_space = None,
            **kwargs,
        )

        # Interface 用于传递仿真状态，并接收控制指令
        self._orcagym_interface = OrcaGym_Interface(time_step)
        joint_name_list = self._orcagym_interface.getJointName()
        self._build_orcagym_interface_map(joint_name_list)
        self._sensor_name_list = []
        # self._sensor_name_list.append(self.sensor(self._orcagym_interface.getBaseName()))
        self._sensor_name_list.append(self.sensor(self._orcagym_interface.getOrientationSensorName()))
        self._sensor_name_list.append(self.sensor(self._orcagym_interface.getVelSensorName()))
        self._sensor_name_list.append(self.sensor(self._orcagym_interface.getGyroSensorName()))
        self._sensor_name_list.append(self.sensor(self._orcagym_interface.getAccSensorName()))

        # OpenLoongWBC调用青龙控制算法接口，解算控制数据
        self._openloong_wbc = OpenLoongWBC(urdf_path, time_step, json_path, log_path, self.model.nq, self.model.nv)

        self._openloong_wbc.InitLogger()

        self._keyboard_controller = KeyboardInput()
        self._button_state = ButtonState()

        self.ctrl = np.zeros(self.model.nu) # 初始化控制数组

    
    def _build_orcagym_interface_map(self, joint_name_list):
        jntId_qpos = np.array([self.model.joint_name2id(self.joint(jntName)) for jntName in joint_name_list])
        print("joint_name_list: ", joint_name_list)
        print("jntId_qpos: ", jntId_qpos)
        self._orcagym_interface.setJntIdQpos(jntId_qpos)
        jntId_qvel = np.array([self.model.joint_name2id(self.joint(jntName)) for jntName in joint_name_list])
        print("jntId_qvel: ", jntId_qvel)
        self._orcagym_interface.setJntIdQvel(jntId_qvel)
        return
        

    def step(self, action) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        self._set_action()
        self.do_simulation(self.ctrl, self.frame_skip)
        obs = self._get_obs().copy()

        info = {}
        terminated = False
        truncated = False
        reward = 0

        return obs, reward, terminated, truncated, info

    
    def _load_record(self) -> None:
        if self.record_file is None:
            raise ValueError("record_file is not set.")
        
        # 读取record_file中的数据，存储到record_pool中
        with h5py.File(self.record_file, 'r') as f:
            if "float_data" in f:
                dset = f["float_data"]
                if self.record_cursor >= dset.shape[0]:
                    return False

                self.record_pool = dset[self.record_cursor:self.record_cursor + self.RECORD_POOL_SIZE].tolist()
                self.record_cursor += self.RECORD_POOL_SIZE
                return True

        return False
    
    def save_record(self) -> None:
        if self.record_state != RecordState.RECORD:
            return
        
        if self.record_file is None:
            raise ValueError("record_file is not set.")

        with h5py.File(self.record_file, 'a') as f:
            # 如果数据集存在，获取其大小；否则，创建新的数据集
            if "float_data" in f:
                dset = f["float_data"]
                self.record_cursor = dset.shape[0]
            else:
                dset = f.create_dataset("float_data", (0, len(self.ctrl)), maxshape=(None, len(self.ctrl)), dtype='f', compression="gzip")
                self.record_cursor = 0

            # 将record_pool中的数据写入数据集
            dset.resize((self.record_cursor + len(self.record_pool), len(self.ctrl)))
            dset[self.record_cursor:] = np.array(self.record_pool)
            self.record_cursor += len(self.record_pool)
            self.record_pool.clear()

            print("Record saved.")


    def _replay(self) -> None:
        if self.record_state == RecordState.REPLAY_FINISHED:
            return
        
        if len(self.record_pool) == 0:
            if not self._load_record():
                self.record_state = RecordState.REPLAY_FINISHED
                print("Replay finished.")
                return

        self.ctrl = self.record_pool.pop(0)


    def _set_action(self) -> None:
        if self.record_state == RecordState.REPLAY or self.record_state == RecordState.REPLAY_FINISHED:
            self._replay()
            return
        
        # 调用青龙控制算法接口，获取控制数据
        xpos, _, _ = self.get_body_xpos_xmat_xquat([self.body("base_link")])
        sensor_dict = self.query_sensor_data(self._sensor_name_list)
        # print("sensor_dict: ", sensor_dict)
        self._orcagym_interface.updateSensorValues(self.data.qpos, self.data.qvel, 
                                                   sensor_dict[self.sensor('baselink-quat')]['values'], sensor_dict[self.sensor('baselink-velocity')]['values'], 
                                                   sensor_dict[self.sensor('baselink-gyro')]['values'], sensor_dict[self.sensor('baselink-baseAcc')]['values'], xpos)
        raise Exception("Test")
        # print("Run simulation, time: ", self.data.time)
        try:
            self._openloong_wbc.Runsimulation(self._button_state, self._orcagym_interface, self.data.time)
        except Exception as e:
            print("Error: ", e)

        self.ctrl = self._orcagym_interface.getMotorCtrl()


        # 将控制数据存储到record_pool中
        if self.record_state == RecordState.RECORD:
            self._save_record()


    def _save_record(self) -> None:
        self.record_pool.append(self.ctrl.copy())   
        if (len(self.record_pool) >= self.RECORD_POOL_SIZE):
            self.save_record()

    def _get_obs(self) -> dict:
        # robot
        achieved_goal = np.array([0,0,0])
        desired_goal = self.goal.copy()
        obs = np.concatenate(
                [
                    [0, 0, 0],
                    [0, 0, 0 ],
                    [0]
                ]).copy()            
        result = {
            "observation": obs,
            "achieved_goal": achieved_goal,
            "desired_goal": desired_goal,
        }
        return result

    def _render_callback(self) -> None:
        pass

    def reset_model(self):
        # Robot_env 统一处理，这里实现空函数就可以
        pass

    def _reset_sim(self) -> bool:
        # print("reset simulation")
        self.do_simulation(self.ctrl, self.frame_skip)
        return True

    def _sample_goal(self) -> np.ndarray:
        # 训练reach时，任务是移动抓夹，goal以抓夹为原点采样
        goal = np.array([0, 0, 0])
        return goal