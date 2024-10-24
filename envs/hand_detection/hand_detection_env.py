import numpy as np
from gymnasium.core import ObsType
from envs.robot_env import MujocoRobotEnv
from orca_gym.utils import rotations
from typing import Optional, Any, SupportsFloat
from gymnasium import spaces
from orca_gym.devices.xbox_joystick import XboxJoystick
from orca_gym.devices.pico_joytsick import PicoJoystick
from orca_gym.devices.hand_joytstick import HandJoystick
from scipy.spatial.transform import Rotation as R
import os
from envs.openloong.camera_wrapper import CameraWrapper
import time

class HandDetectionEnv(MujocoRobotEnv):
    def __init__(
        self,
        frame_skip: int = 5,        
        grpc_address: str = 'localhost:50051',
        agent_names: list = ['Agent0'],
        time_step: float = 0.016,  # 0.016 for 60 fps        
        **kwargs,
    ):

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

        self.neutral_joint_values = np.zeros(11)

        # Three auxiliary variables to understand the component of the xml document but will not be used
        # number of actuators/controls: 7 arm joints and 2 gripper joints
        self.nu = self.model.nu
        # 16 generalized coordinates: 9 (arm + gripper) + 7 (object free joint: 3 position and 4 quaternion coordinates)
        self.nq = self.model.nq
        # 9 arm joints and 6 free joints
        self.nv = self.model.nv

        self._set_init_state()

        self.joystick = HandJoystick()

    def _set_init_state(self) -> None:
        # print("Set initial state")
        self.set_joint_neutral()

        self.ctrl = np.array([0.00] * 11)

        self.mj_forward()

    def step(self, action) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        self._set_action()
        self.do_simulation(self.ctrl, self.frame_skip)
        obs = self._get_obs().copy()

        info = {}
        terminated = False
        truncated = False
        reward = 0

        return obs, reward, terminated, truncated, info

    def _set_action(self) -> None:
        self.mj_forward()

        hand_infos = self.joystick.get_hand_infos()
        radian_tolerance = 0.0
        if hand_infos is not None:
            for hand_info in hand_infos:
                if hand_info.hand_index == 0:
                    hand_point_names = ["J1", "J2", "J3", "J4", "J5", "J6", "J7", "J8", "J9", "J10", "J11"]
                    real_hand_qpos_dict = self.query_joint_qpos(hand_point_names)
                    hand_qpos_list = {}
                    for name, value in zip(hand_point_names, hand_info.qpos):
                        if name in real_hand_qpos_dict:
                            real_value = real_hand_qpos_dict[name]
                            if abs(value - real_value) > radian_tolerance:
                                hand_qpos_list[name] = np.array([value])
                            else:
                                hand_qpos_list[name] = np.array([real_value])
                        # hand_qpos_list[name] = np.array([value])
                    # self.set_joint_qpos(hand_qpos_list)
                    self.ctrl[:11] = np.array([hand_qpos_list[name] for name in hand_point_names]).flat.copy()

    def handle_hand_joystick(self):
        return


    def _render_callback(self) -> None:
        return

    def reset_model(self):
        # Robot_env 统一处理，这里实现空函数就可以
        pass

    def _reset_sim(self) -> bool:
        self._set_init_state()
        self.mj_forward()
        return True

    def set_joint_neutral(self) -> None:
        return
        # assign value to arm joints
        arm_joint_qpos_list = {}
        for name, value in zip(self.arm_joint_names, self.neutral_joint_values[0:16]):
            arm_joint_qpos_list[name] = np.array([value])
        self.set_joint_qpos(arm_joint_qpos_list)

    def _sample_goal(self) -> np.ndarray:
        # 训练reach时，任务是移动抓夹，goal以抓夹为原点采样
        goal = np.array([0, 0, 0])
        return goal
    
    def _get_obs(self) -> dict:          
        result = {
            "observation": np.ndarray([0, 0, 0]),
            "achieved_goal": np.ndarray([0, 0, 0]),
            "desired_goal": np.ndarray([0, 0, 0]),
        }
        return result
