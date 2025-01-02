import sys
import os
from datetime import datetime
import time

# 添加 libs 目录到 sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))  # 获取当前文件的目录
libs_dir = os.path.join(current_dir, 'libs')  # 构建 libs 目录路径
sys.path.append(libs_dir)  # 将 libs 目录添加到 sys.path

from openloong_dyn_ctrl import OpenLoongWBC, OrcaGym_Interface, ButtonState

import numpy as np
from gymnasium.core import ObsType
from orca_gym.utils import rotations
from orca_gym.environment import OrcaGymRemoteEnv, OrcaGymLocalEnv
from typing import Optional, Any, SupportsFloat
from gymnasium import spaces
from orca_gym.devices.keyboard import KeyboardClient, KeyboardInput

class OpenLoongEnv(OrcaGymLocalEnv):
    metadata = {'render_modes': ['human', 'none'], 'version': '0.0.1', 'render_fps': 30}
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
        orcagym_addr: str = 'localhost:50051',
        agent_names: list = ['Agent0'],
        time_step: float = 0.016,  # 0.016 for 60 fps        
        render_mode: str = "human",
        urdf_path: str = "",
        json_path: str = "",
        log_path: str = "",
        **kwargs,
    ):

        individual_control = kwargs['individual_control']
        self._render_mode = render_mode
        print("Render_mode is: ", self._render_mode)

        super().__init__(
            frame_skip = frame_skip,
            orcagym_addr = orcagym_addr,
            agent_names = agent_names,
            time_step = time_step,            
            **kwargs,
        )

        self._sim_init_time = datetime.now()
        self._agent_num = len(agent_names)
        self._agent0_nq = int(self.model.nq / self._agent_num)
        self._agent0_nv = int(self.model.nv / self._agent_num)
        self._agent0_nu = int(self.model.nu / self._agent_num)

        # Interface 用于传递仿真状态，并接收控制指令
        self._orcagym_interface = OrcaGym_Interface(time_step)
        joint_name_list = self._orcagym_interface.getJointName()
        self._joint_name_list = [self.joint(jntName) for jntName in joint_name_list]        
        qpos_offsets, qvel_offsets, _ = self.query_joint_offsets(self._joint_name_list)
        self._orcagym_interface.setJointOffsetQpos(qpos_offsets)
        self._orcagym_interface.setJointOffsetQvel(qvel_offsets)

        self._actuator_idmap = []
        for agent_id in range(self._agent_num):
            self._actuator_idmap.append(self._build_acutator_idmap(agent_id))

        self._sensor_name_list = []
        # self._sensor_name_list.append(self.sensor(self._orcagym_interface.getBaseName()))
        self._sensor_name_list.append(self.sensor(self._orcagym_interface.getOrientationSensorName()))
        self._sensor_name_list.append(self.sensor(self._orcagym_interface.getVelSensorName()))
        self._sensor_name_list.append(self.sensor(self._orcagym_interface.getGyroSensorName()))
        self._sensor_name_list.append(self.sensor(self._orcagym_interface.getAccSensorName()))

        # OpenLoongWBC调用青龙控制算法接口，解算控制数据
        self._openloong_wbc = OpenLoongWBC(urdf_path, time_step, json_path, log_path, self._agent0_nq, self._agent0_nv)

        # self._openloong_wbc.InitLogger()

        if individual_control:
            self._keyboard_controller = KeyboardInput()
        else:
            self._keyboard_controller = KeyboardClient()

        self._button_state = ButtonState()
        self._key_status = {"W": 0, "A": 0, "S": 0, "D": 0, "Space": 0, "Up": 0, "Down": 0}

        self.ctrl = np.zeros(self.model.nu) # 初始化控制数组

        # Run generate_observation_space after initialization to ensure that the observation object's name is defined.
        self._set_obs_space()
        self._set_action_space()

    def _set_obs_space(self):
        self.observation_space = self.generate_observation_space(self._get_obs().copy())

    def _set_action_space(self):
        self.action_space = self.generate_action_space(self.model.get_actuator_ctrlrange())
    
    def _build_acutator_idmap(self, agent_id) -> list[int]:
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
            actuator_id = self.model.actuator_name2id(self.actuator(actuator_name, agent_id=agent_id))
            acutator_idmap.append(actuator_id)

        return acutator_idmap

    def step(self, action) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        # start_set_action_time = datetime.now()
        self._set_action()
        # elapsed_set_action_time = datetime.now() - start_set_action_time
        # print(f"elapsed_set_action_time (ms): {elapsed_set_action_time.total_seconds() * 1000}")

        # start_time = time.perf_counter()


        self.do_simulation(self.ctrl, self.frame_skip)


        # end_time = time.perf_counter()
        # if (OpenLoongEnv.time_counter % 1000 == 0):
        #     print("step elapsed_time (ms): ", (end_time - start_time) * 1000)

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

    time_counter = 0
    # def _set_action(self) -> None:
    #     # 调用青龙控制算法接口，获取控制数据
    #     # start_time = time.perf_counter()

    #     xpos, _, _ = self.get_body_xpos_xmat_xquat([self.body("base_link")])
    #     sensor_dict = self.query_sensor_data(self._sensor_name_list)
    #     print("sensor_dict:",sensor_dict)
    #     self._orcagym_interface.updateSensorValues(self.data.qpos, 
    #                                                self.data.qvel,
    #                                                sensor_dict[self.sensor('baselink-quat')]['values'], 
    #                                                sensor_dict[self.sensor('baselink-velocity')]['values'], 
    #                                                sensor_dict[self.sensor('baselink-gyro')]['values'], 
    #                                                sensor_dict[self.sensor('baselink-baseAcc')]['values'], 
    #                                                xpos)
        
    #     self._update_keyboard_control()
    #     sim_time = (datetime.now() - self._sim_init_time).total_seconds()
    #     self._openloong_wbc.Runsimulation(self._button_state, self._orcagym_interface, sim_time)
    #     ctrl = self._orcagym_interface.getMotorCtrl()

    #     # end_time = time.perf_counter()
    #     # OpenLoongEnv.time_counter += 1
    #     # if (OpenLoongEnv.time_counter % 1000 == 0):
    #     #     print("_set_action elapsed_time (ms): ", (end_time - start_time) * 1000)

    #     for actuator_idmap in self._actuator_idmap:
    #         for i, actuator_id in enumerate(actuator_idmap):
    #             self.ctrl[actuator_id] = ctrl[i]

    def _set_action(self) -> None:
        # 调用青龙控制算法接口，获取控制数据
        xpos, _, _ = self.get_body_xpos_xmat_xquat([self.body("base_link")])
        sensor_dict = self.query_sensor_data(self._sensor_name_list)

        # 直接访问传感器数据（不使用 ['values']）
        self._orcagym_interface.updateSensorValues(
            self.data.qpos, 
            self.data.qvel,
            sensor_dict[self.sensor('baselink-quat')],  # 直接使用数组
            sensor_dict[self.sensor('baselink-velocity')],  # 直接使用数组
            sensor_dict[self.sensor('baselink-gyro')],  # 直接使用数组
            sensor_dict[self.sensor('baselink-baseAcc')],  # 直接使用数组
            xpos
        )

        self._update_keyboard_control()
        sim_time = (datetime.now() - self._sim_init_time).total_seconds()
        self._openloong_wbc.Runsimulation(self._button_state, self._orcagym_interface, sim_time)
        ctrl = self._orcagym_interface.getMotorCtrl()

        for actuator_idmap in self._actuator_idmap:
            for i, actuator_id in enumerate(actuator_idmap):
                self.ctrl[actuator_id] = ctrl[i]

    def _get_obs(self) -> dict:
        obs = np.concatenate(
                [
                    [0, 0, 0],
                    [0, 0, 0 ],
                    [0]
                ]).copy()            
        result = {
            "observation": obs,
        }
        return result

    def reset_model(self):
        obs = self._get_obs().copy()
        return obs, {}

    def get_observation(self, obs=None):
        if obs is not None:
            return obs
        else:
            return self._get_obs().copy()