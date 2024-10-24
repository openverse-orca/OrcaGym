import sys
import os
from datetime import datetime

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
from orca_gym.devices.keyboard import KeyboardClient

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
        joint_name_list = [self.joint(jntName) for jntName in joint_name_list]        
        qpos_offsets, qvel_offsets, _ = self.query_joint_offsets(joint_name_list)
        self._orcagym_interface.setJointOffsetQpos(qpos_offsets)
        self._orcagym_interface.setJointOffsetQvel(qvel_offsets)

        self._actuator_idmap = self._build_acutator_idmap()

        self._sensor_name_list = []
        # self._sensor_name_list.append(self.sensor(self._orcagym_interface.getBaseName()))
        self._sensor_name_list.append(self.sensor(self._orcagym_interface.getOrientationSensorName()))
        self._sensor_name_list.append(self.sensor(self._orcagym_interface.getVelSensorName()))
        self._sensor_name_list.append(self.sensor(self._orcagym_interface.getGyroSensorName()))
        self._sensor_name_list.append(self.sensor(self._orcagym_interface.getAccSensorName()))

        # OpenLoongWBC调用青龙控制算法接口，解算控制数据
        self._openloong_wbc = OpenLoongWBC(urdf_path, time_step, json_path, log_path, self.model.nq, self.model.nv)

        self._openloong_wbc.InitLogger()

        self._keyboard_controller = KeyboardClient()
        self._button_state = ButtonState()
        self._key_status = {"W": 0, "A": 0, "S": 0, "D": 0, "Space": 0, "Up": 0, "Down": 0}

        self.ctrl = np.zeros(self.model.nu) # 初始化控制数组

    
    def _build_acutator_idmap(self) -> list[int]:
        acutator_idmap = []

        # 来自于 external/openloong-dyn-control/models/AzureLoong.xml
        actuator_name_list = ['M_arm_l_01', 'M_arm_l_02', 'M_arm_l_03', 'M_arm_l_04', 'M_arm_l_05', 
                                'M_arm_l_06', 'M_arm_l_07', 'M_arm_r_01', 'M_arm_r_02', 'M_arm_r_03', 
                                'M_arm_r_04', 'M_arm_r_05', 'M_arm_r_06', 'M_arm_r_07', 'M_head_yaw', 
                                'M_head_pitch', 'M_waist_pitch', 'M_waist_roll', 'M_waist_yaw', 
                                'M_hip_l_roll', 'M_hip_l_yaw', 'M_hip_l_pitch', 'M_knee_l_pitch', 
                                'M_ankle_l_pitch', 'M_ankle_l_roll', 'M_hip_r_roll', 'M_hip_r_yaw', 
                                'M_hip_r_pitch', 'M_knee_r_pitch', 'M_ankle_r_pitch', 'M_ankle_r_roll']
        
        for i, actuator_name in enumerate(actuator_name_list):
            actuator_id = self.model.actuator_name2id(self.actuator(actuator_name))
            acutator_idmap.append(actuator_id)

        return acutator_idmap

    def step(self, action) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        # start_set_action_time = datetime.now()
        self._set_action()
        # elapsed_set_action_time = datetime.now() - start_set_action_time
        # print(f"elapsed_set_action_time (ms): {elapsed_set_action_time.total_seconds() * 1000}")

        self.do_simulation(self.ctrl, self.frame_skip)
        obs = self._get_obs().copy()

        info = {}
        terminated = False
        truncated = False
        reward = 0

        return obs, reward, terminated, truncated, info

    
    def _update_keyboard_control(self) -> None:
        self._keyboard_controller.update()
        key_status = self._keyboard_controller.get_state()

        self._button_state.key_w = False
        self._button_state.key_a = False
        self._button_state.key_s = False
        self._button_state.key_d = False
        self._button_state.key_space = False
        
        if self._key_status["W"] == 0 and key_status["W"] == 1:
            self._button_state.key_w = True
        if self._key_status["A"] == 0 and key_status["A"] == 1:
            self._button_state.key_a = True
        if self._key_status["S"] == 0 and key_status["S"] == 1:
            self._button_state.key_s = True
        if self._key_status["D"] == 0 and key_status["D"] == 1:
            self._button_state.key_d = True
        if self._key_status["Space"] == 0 and key_status["Space"] == 1:
            self._button_state.key_space = True
        if self._key_status["Up"] == 0 and key_status["Up"] == 1:
            self._openloong_wbc.SetBaseUp(0.025)
        if self._key_status["Down"] == 0 and key_status["Down"] == 1:
            self._openloong_wbc.SetBaseDown(0.025)

        self._key_status = key_status.copy()


        # print(f"key_w: {self._button_state.key_w}, key_a: {self._button_state.key_a}, key_s: {self._button_state.key_s}, key_d: {self._button_state.key_d}, key_space: {self._button_state.key_space}")

    def _set_action(self) -> None:
        # 调用青龙控制算法接口，获取控制数据
        xpos, _, _ = self.get_body_xpos_xmat_xquat([self.body("base_link")])
        sensor_dict = self.query_sensor_data(self._sensor_name_list)
        self._orcagym_interface.updateSensorValues(self.data.qpos, self.data.qvel, 
                                                   sensor_dict[self.sensor('baselink-quat')]['values'], sensor_dict[self.sensor('baselink-velocity')]['values'], 
                                                   sensor_dict[self.sensor('baselink-gyro')]['values'], sensor_dict[self.sensor('baselink-baseAcc')]['values'], xpos)
        
        self._update_keyboard_control()
        self._openloong_wbc.Runsimulation(self._button_state, self._orcagym_interface, self.data.time)

        ctrl = self._orcagym_interface.getMotorCtrl()
        for i, actuator_id in enumerate(self._actuator_idmap):
            self.ctrl[actuator_id] = ctrl[i]


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