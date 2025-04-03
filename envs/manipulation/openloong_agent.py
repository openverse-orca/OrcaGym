from datetime import datetime
import numpy as np
from gymnasium.core import ObsType
from orca_gym.robomimic.dataset_util import DatasetWriter
from orca_gym.robomimic.robomimic_env import RobomimicEnv
from orca_gym.utils import rotations
from typing import Optional, Any, SupportsFloat
from gymnasium import spaces
from orca_gym.devices.xbox_joystick import XboxJoystickManager
from orca_gym.devices.pico_joytsick import PicoJoystick
from orca_gym.robosuite.controllers.controller_factory import controller_factory
import orca_gym.robosuite.controllers.controller_config as controller_config
import orca_gym.robosuite.utils.transform_utils as transform_utils
from orca_gym.environment import OrcaGymLocalEnv
from scipy.spatial.transform import Rotation as R
from orca_gym.task.abstract_task import AbstractTask
import time
from orca_gym.utils.joint_controller import JointController
import random

from orca_gym.environment.orca_gym_env import RewardType
from orca_gym.utils.reward_printer import RewardPrinter

from envs.manipulation.openloong_env import OpenLoongEnv, RunMode, ControlDevice
import threading

class OpenLoongAgentBase:
    def __init__(self, id: int, name: str) -> None:
        self._id = id
        self._name = name
    
    @property
    def id(self) -> int:
        return self._id
    
    @property
    def name(self) -> str:
        return self._name

    def init_agent(self, env: OpenLoongEnv, id: int):
        raise NotImplementedError("This method should be overridden by subclasses.")
    
    def set_joint_neutral(self, env: OpenLoongEnv) -> None:
        raise NotImplementedError("This method should be overridden by subclasses.")
    
    def on_reset_model(self, env: OpenLoongEnv) -> None:
        raise NotImplementedError("This method should be overridden by subclasses.")
    
    def get_obs(self, env: OpenLoongEnv) -> dict:
        raise NotImplementedError("This method should be overridden by subclasses.")
    
    def on_close(self):
        pass

class AzureLoongAgent(OpenLoongAgentBase):
    def __init__(self, env: OpenLoongEnv, id: int, name: str) -> None:
        super().__init__(id, name)
        

    def init_agent(self, env: OpenLoongEnv, id: int):
        # print("base_body_xpos: ", self._base_body_xpos)
        # print("base_body_xquat: ", self._base_body_xquat)

        self._neck_joint_names = [env.joint("J_head_yaw", id), env.joint("J_head_pitch", id)]
        self._neck_actuator_names = [env.actuator("M_head_yaw", id), env.actuator("M_head_pitch", id)]
        self._neck_actuator_id = [env.model.actuator_name2id(actuator_name) for actuator_name in self._neck_actuator_names]
        self._neck_neutral_joint_values = np.array([0, -0.7854])
        self._neck_ctrl_values = {"yaw": 0.0, "pitch": -0.7854}

        # index used to distinguish arm and gripper joints
        self._r_arm_joint_names = [env.joint("J_arm_r_01"), env.joint("J_arm_r_02"), 
                                 env.joint("J_arm_r_03"), env.joint("J_arm_r_04"), 
                                 env.joint("J_arm_r_05"), env.joint("J_arm_r_06"), env.joint("J_arm_r_07")]
        self._r_arm_motor_names = [env.actuator("M_arm_r_01"), env.actuator("M_arm_r_02"),
                                env.actuator("M_arm_r_03"),env.actuator("M_arm_r_04"),
                                env.actuator("M_arm_r_05"),env.actuator("M_arm_r_06"),env.actuator("M_arm_r_07")]
        self._r_arm_position_names = [env.actuator("P_arm_r_01"), env.actuator("P_arm_r_02"),
                                      env.actuator("P_arm_r_03"),env.actuator("P_arm_r_04"),
                                      env.actuator("P_arm_r_05"),env.actuator("P_arm_r_06"),env.actuator("P_arm_r_07")]
        if env.action_use_motor():
            env.disable_actuators(self._r_arm_position_names)
            self._r_arm_actuator_id = [env.model.actuator_name2id(actuator_name) for actuator_name in self._r_arm_motor_names]
        else:
            env.disable_actuators(self._r_arm_motor_names)
            self._r_arm_actuator_id = [env.model.actuator_name2id(actuator_name) for actuator_name in self._r_arm_position_names]
        self._r_neutral_joint_values = np.array([0.905, -0.735, -2.733, 1.405, -1.191, 0.012, -0.517])
        

        # print("arm_actuator_id: ", self._r_arm_actuator_id)
        # print("hand_actuator_id: ", self._r_hand_actuator_id)

        # index used to distinguish arm and gripper joints
        self._l_arm_joint_names = [env.joint("J_arm_l_01"), env.joint("J_arm_l_02"), 
                                 env.joint("J_arm_l_03"), env.joint("J_arm_l_04"), 
                                 env.joint("J_arm_l_05"), env.joint("J_arm_l_06"), env.joint("J_arm_l_07")]
        self._l_arm_moto_names = [env.actuator("M_arm_l_01"), env.actuator("M_arm_l_02"),
                                env.actuator("M_arm_l_03"),env.actuator("M_arm_l_04"),
                                env.actuator("M_arm_l_05"),env.actuator("M_arm_l_06"),env.actuator("M_arm_l_07")]
        self._l_arm_position_names = [env.actuator("P_arm_l_01"), env.actuator("P_arm_l_02"),
                                      env.actuator("P_arm_l_03"),env.actuator("P_arm_l_04"),
                                      env.actuator("P_arm_l_05"),env.actuator("P_arm_l_06"),env.actuator("P_arm_l_07")]
        if env.action_use_motor():
            env.disable_actuators(self._l_arm_position_names)
            self._l_arm_actuator_id = [env.model.actuator_name2id(actuator_name) for actuator_name in self._l_arm_moto_names]
        else:
            env.disable_actuators(self._l_arm_moto_names)
            self._l_arm_actuator_id = [env.model.actuator_name2id(actuator_name) for actuator_name in self._l_arm_position_names]
        self._l_neutral_joint_values = np.array([-0.905, 0.735, 2.733, 1.405, 1.191, 0.012, 0.517])
        # self._l_neutral_joint_values = np.zeros(7)

        # control range
        self._all_ctrlrange = env.model.get_actuator_ctrlrange()
        neck_ctrl_range = [self._all_ctrlrange[actoator_id] for actoator_id in self._neck_actuator_id]
        # print("ctrl_range: ", neck_ctrl_range)

        r_ctrl_range = [self._all_ctrlrange[actoator_id] for actoator_id in self._r_arm_actuator_id]
        # print("ctrl_range: ", r_ctrl_range)

        l_ctrl_range = [self._all_ctrlrange[actoator_id] for actoator_id in self._l_arm_actuator_id]
        # print("ctrl_range: ", l_ctrl_range)


        arm_qpos_range_l = env.model.get_joint_qposrange(self._l_arm_joint_names)
        arm_qpos_range_r = env.model.get_joint_qposrange(self._r_arm_joint_names)
        self._setup_action_range(arm_qpos_range_l, arm_qpos_range_r)
        self._setup_obs_scale(arm_qpos_range_l, arm_qpos_range_r)

        NECK_NAME  = env.site("neck_center_site")
        site_dict = env.query_site_pos_and_quat([NECK_NAME])
        self._initial_neck_site_xpos = site_dict[NECK_NAME]['xpos']
        self._initial_neck_site_xquat = site_dict[NECK_NAME]['xquat']

        self.set_neck_mocap(self._initial_neck_site_xpos, self._initial_neck_site_xquat)
        self._mocap_neck_xpos, self._mocap_neck_xquat = self._initial_neck_site_xpos, self._initial_neck_site_xquat
        self._neck_angle_x, self._neck_angle_y = 0, 0

        self._ee_site_l  = env.site("ee_center_site")
        site_dict = self.query_site_pos_and_quat([self._ee_site_l])
        self._initial_grasp_site_xpos = site_dict[self._ee_site_l]['xpos']
        self._initial_grasp_site_xquat = site_dict[self._ee_site_l]['xquat']
        self._grasp_value_l = 0.0

        self.set_grasp_mocap(self._initial_grasp_site_xpos, self._initial_grasp_site_xquat)

        self._ee_site_r  = env.site("ee_center_site_r")
        site_dict = self.query_site_pos_and_quat([self._ee_site_r])
        self._initial_grasp_site_xpos_r = site_dict[self._ee_site_r]['xpos']
        self._initial_grasp_site_xquat_r = site_dict[self._ee_site_r]['xquat']
        self._grasp_value_r = 0.0

        self.set_grasp_mocap_r(self._initial_grasp_site_xpos_r, self._initial_grasp_site_xquat_r)
        
        if env.run_mode == RunMode.TELEOPERATION:
            if env.ctrl_device == ControlDevice.VR:
                self._pico_joystick = PicoJoystick()
            elif env.ctrl_device == ControlDevice.RANDOM_SAMPLE:
                self._pico_joystick = None
            else:
                raise ValueError("Invalid control device: ", self._ctrl_device)

        # -----------------------------
        # Neck controller
        self._neck_controller_config = controller_config.load_config("osc_pose")
        # print("controller_config: ", self.controller_config)

        # Add to the controller dict additional relevant params:
        #   the robot name, mujoco sim, eef_name, joint_indexes, timestep (model) freq,
        #   policy (control) freq, and ndim (# joints)
        self._neck_controller_config["robot_name"] = self.name
        self._neck_controller_config["sim"] = env.gym
        self._neck_controller_config["eef_name"] = NECK_NAME
        # self.controller_config["eef_rot_offset"] = self.eef_rot_offset
        qpos_offsets, qvel_offsets, _ = env.query_joint_offsets(self._neck_joint_names)
        self._neck_controller_config["joint_indexes"] = {
            "joints": self._neck_joint_names,
            "qpos": qpos_offsets,
            "qvel": qvel_offsets,
        }
        self._neck_controller_config["actuator_range"] = neck_ctrl_range
        self._neck_controller_config["policy_freq"] = env.control_freq
        self._neck_controller_config["ndim"] = len(self._neck_joint_names)
        self._neck_controller_config["control_delta"] = False


        self._neck_controller = controller_factory(self._neck_controller_config["type"], self._neck_controller_config)
        self._neck_controller.update_initial_joints(self._neck_neutral_joint_values)

        # -----------------------------
        # Right controller
        self._r_controller_config = controller_config.load_config("osc_pose")
        # print("controller_config: ", self.controller_config)

        # Add to the controller dict additional relevant params:
        #   the robot name, mujoco sim, eef_name, joint_indexes, timestep (model) freq,
        #   policy (control) freq, and ndim (# joints)
        self._r_controller_config["robot_name"] = self.name
        self._r_controller_config["sim"] = env.gym
        self._r_controller_config["eef_name"] = self._ee_site_r
        # self.controller_config["eef_rot_offset"] = self.eef_rot_offset
        qpos_offsets, qvel_offsets, _ = env.query_joint_offsets(self._r_arm_joint_names)
        self._r_controller_config["joint_indexes"] = {
            "joints": self._r_arm_joint_names,
            "qpos": qpos_offsets,
            "qvel": qvel_offsets,
        }
        self._r_controller_config["actuator_range"] = r_ctrl_range
        self._r_controller_config["policy_freq"] = env.control_freq
        self._r_controller_config["ndim"] = len(self._r_arm_joint_names)
        self._r_controller_config["control_delta"] = False


        self._r_controller = controller_factory(self._r_controller_config["type"], self._r_controller_config)
        self._r_controller.update_initial_joints(self._r_neutral_joint_values)

        self._r_gripper_offset_rate_clip = 0.0


        # -----------------------------
        # Left controller
        self._l_controller_config = controller_config.load_config("osc_pose")
        # print("controller_config: ", self.controller_config)

        # Add to the controller dict additional relevant params:
        #   the robot name, mujoco sim, eef_name, joint_indexes, timestep (model) freq,
        #   policy (control) freq, and ndim (# joints)
        self._l_controller_config["robot_name"] = self.name
        self._l_controller_config["sim"] = env.gym
        self._l_controller_config["eef_name"] = self._ee_site_l
        # self.controller_config["eef_rot_offset"] = self.eef_rot_offset
        qpos_offsets, qvel_offsets, _ = env.query_joint_offsets(self._l_arm_joint_names)
        self._l_controller_config["joint_indexes"] = {
            "joints": self._l_arm_joint_names,
            "qpos": qpos_offsets,
            "qvel": qvel_offsets,
        }
        self._l_controller_config["actuator_range"] = l_ctrl_range
        self._l_controller_config["policy_freq"] = env.control_freq
        self._l_controller_config["ndim"] = len(self._l_arm_joint_names)
        self._l_controller_config["control_delta"] = False


        self._l_controller = controller_factory(self._l_controller_config["type"], self._l_controller_config)
        self._l_controller.update_initial_joints(self._l_neutral_joint_values)

        self._l_gripper_offset_rate_clip = 0.0

    def on_close(self):
        if hasattr(self, "_pico_joystick") and self._pico_joystick is not None:
            self._pico_joystick.close()        

    def set_joint_neutral(self, env: OpenLoongEnv) -> None:
        # assign value to arm joints
        arm_joint_qpos = {}
        for name, value in zip(self._r_arm_joint_names, self._r_neutral_joint_values):
            arm_joint_qpos[name] = np.array([value])
        for name, value in zip(self._l_arm_joint_names, self._l_neutral_joint_values):
            arm_joint_qpos[name] = np.array([value])     
        for name, value in zip(self._neck_joint_names, self._neck_neutral_joint_values):
            arm_joint_qpos[name] = np.array([value])
        env.set_joint_qpos(arm_joint_qpos)        

    def on_reset_model(self, env: OpenLoongEnv) -> None:
        self.set_grasp_mocap(env, self._initial_grasp_site_xpos, self._initial_grasp_site_xquat)
        self.set_grasp_mocap_r(env, self._initial_grasp_site_xpos_r, self._initial_grasp_site_xquat_r)
        self._reset_gripper()
        self._reset_neck_mocap()

    def set_neck_mocap(self, env: OpenLoongEnv, position, orientation) -> None:
        mocap_pos_and_quat_dict = {env.mocap("neckMocap", self.id): {'pos': position, 'quat': orientation}}
        env.set_mocap_pos_and_quat(mocap_pos_and_quat_dict)

    def set_grasp_mocap(self, env: OpenLoongEnv, position, orientation) -> None:
        mocap_pos_and_quat_dict = {env.mocap("leftHandMocap", self.id): {'pos': position, 'quat': orientation}}
        env.set_mocap_pos_and_quat(mocap_pos_and_quat_dict)

    def set_grasp_mocap_r(self, env: OpenLoongEnv, position, orientation) -> None:
        mocap_pos_and_quat_dict = {env.mocap("rightHandMocap", self.id): {'pos': position, 'quat': orientation}}
        # print("Set grasp mocap: ", position, orientation)
        env.set_mocap_pos_and_quat(mocap_pos_and_quat_dict)


    def _reset_gripper(self) -> None:
        self._l_gripper_offset_rate_clip = 0.0
        self._r_gripper_offset_rate_clip = 0.0

    def _reset_neck_mocap(self, env: OpenLoongEnv) -> None:
        self._mocap_neck_xpos, self._mocap_neck_xquat = self._initial_neck_site_xpos, self._initial_neck_site_xquat
        self.set_neck_mocap(env, self._mocap_neck_xpos, self._mocap_neck_xquat)
        self._neck_angle_x, self._neck_angle_y = 0, 0


    def get_obs(self, env: OpenLoongEnv) -> dict:
        ee_sites = env.query_site_pos_and_quat([self._ee_site_l, self._ee_site_r])
        ee_xvalp, ee_xvalr = env.query_site_xvalp_xvalr([self._ee_site_l, self._ee_site_r])

        arm_joint_values_l = self._get_arm_joint_values(env, self._l_arm_joint_names)
        arm_joint_values_r = self._get_arm_joint_values(env, self._r_arm_joint_names)
        arm_joint_velocities_l = self._get_arm_joint_velocities(env, self._l_arm_joint_names)
        arm_joint_velocities_r = self._get_arm_joint_velocities(env, self._r_arm_joint_names)

        self._obs = {
            "ee_pos_l": ee_sites[self._ee_site_l]["xpos"].flatten().astype(np.float32),
            "ee_quat_l": ee_sites[self._ee_site_l]["xquat"].flatten().astype(np.float32),
            "ee_pos_r": ee_sites[self._ee_site_r]["xpos"].flatten().astype(np.float32),
            "ee_quat_r": ee_sites[self._ee_site_r]["xquat"].flatten().astype(np.float32),

            "ee_vel_linear_l": ee_xvalp[self._ee_site_l].flatten().astype(np.float32),
            "ee_vel_angular_l": ee_xvalr[self._ee_site_l].flatten().astype(np.float32),
            "ee_vel_linear_r": ee_xvalp[self._ee_site_r].flatten().astype(np.float32),
            "ee_vel_angular_r": ee_xvalr[self._ee_site_r].flatten().astype(np.float32),

            "arm_joint_qpos_l": arm_joint_values_l.flatten().astype(np.float32),
            "arm_joint_qpos_sin_l": np.sin(arm_joint_values_l).flatten().astype(np.float32),
            "arm_joint_qpos_cos_l": np.cos(arm_joint_values_l).flatten().astype(np.float32),
            "arm_joint_vel_l": arm_joint_velocities_l.flatten().astype(np.float32),

            "arm_joint_qpos_r": arm_joint_values_r.flatten().astype(np.float32),
            "arm_joint_qpos_sin_r": np.sin(arm_joint_values_r).flatten().astype(np.float32),
            "arm_joint_qpos_cos_r": np.cos(arm_joint_values_r).flatten().astype(np.float32),
            "arm_joint_vel_r": arm_joint_velocities_r.flatten().astype(np.float32),

            "grasp_value_l": np.array([self._grasp_value_l], dtype=np.float32),
            "grasp_value_r": np.array([self._grasp_value_r], dtype=np.float32),
        }
        scaled_obs = {key : self._obs[key] * self._obs_scale[key] for key in self._obs.keys()}
        return scaled_obs

    
    def _get_arm_joint_values(self, env: OpenLoongEnv, joint_names) -> np.ndarray:
        qpos_dict = env.query_joint_qpos(joint_names)
        return np.array([qpos_dict[joint_name] for joint_name in joint_names]).flatten()
    
    def _get_arm_joint_velocities(self, env: OpenLoongEnv, joint_names) -> np.ndarray:
        qvel_dict = env.query_joint_qvel(joint_names)
        return np.array([qvel_dict[joint_name] for joint_name in joint_names]).flatten()

    def _setup_obs_scale(self, arm_qpos_range_l, arm_qpos_range_r) -> None:
        # 观测空间范围
        ee_xpos_scale = np.array([max(abs(act_range[0]), abs(act_range[1])) for act_range in self._action_range[:3]], dtype=np.float32)   # 末端位置范围
        ee_xquat_scale = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)   # 裁剪到 -pi, pi 的单位四元数范围
        max_ee_linear_vel = 2.0  # 末端线速度范围 m/s
        max_ee_angular_vel = np.pi # 末端角速度范围 rad/s

        arm_qpos_scale_l = np.array([max(abs(qpos_range[0]), abs(qpos_range[1])) for qpos_range in arm_qpos_range_l], dtype=np.float32)  # 关节角度范围
        arm_qpos_scale_r = np.array([max(abs(qpos_range[0]), abs(qpos_range[1])) for qpos_range in arm_qpos_range_r], dtype=np.float32)  # 关节角度范围
        max_arm_joint_vel = np.pi  # 关节角速度范围 rad/s
                
        self._obs_scale = {
            "ee_pos_l": 1.0 / ee_xpos_scale,
            "ee_quat_l": 1.0 / ee_xquat_scale,
            "ee_pos_r": 1.0 / ee_xpos_scale,
            "ee_quat_r": 1.0 / ee_xquat_scale,

            "ee_vel_linear_l": np.ones(3, dtype=np.float32) / max_ee_linear_vel,
            "ee_vel_angular_l": np.ones(3, dtype=np.float32) / max_ee_angular_vel,
            "ee_vel_linear_r": np.ones(3, dtype=np.float32) / max_ee_linear_vel,
            "ee_vel_angular_r": np.ones(3, dtype=np.float32) / max_ee_angular_vel,

            "arm_joint_qpos_l": 1.0 / arm_qpos_scale_l,
            "arm_joint_qpos_sin_l": np.ones(len(arm_qpos_scale_l), dtype=np.float32),
            "arm_joint_qpos_cos_l": np.ones(len(arm_qpos_scale_l), dtype=np.float32),
            "arm_joint_vel_l": np.ones(len(arm_qpos_scale_l), dtype=np.float32) / max_arm_joint_vel,

            "arm_joint_qpos_r": 1.0 / arm_qpos_scale_r,
            "arm_joint_qpos_sin_r": np.ones(len(arm_qpos_scale_r), dtype=np.float32),
            "arm_joint_qpos_cos_r": np.ones(len(arm_qpos_scale_r), dtype=np.float32),
            "arm_joint_vel_r": np.ones(len(arm_qpos_scale_r), dtype=np.float32) / max_arm_joint_vel,

            "grasp_value_l": np.ones(1, dtype=np.float32),
            "grasp_value_r": np.ones(1, dtype=np.float32),       
        }

    def _setup_action_range(self, arm_qpos_range_l, arm_qpos_range_r) -> None:
        # 支持的动作范围空间，遥操作时不能超过这个范围
        # 模型接收的是 [-1, 1] 的动作空间，这里是真实的物理空间，需要进行归一化
        self._action_range =  np.concatenate(
            [
                [[-2.0, 2.0], [-2.0, 2.0], [-2.0, 2.0], [-np.pi, np.pi], [-np.pi, np.pi], [-np.pi, np.pi]], # left hand ee pos and angle euler
                arm_qpos_range_l,                                                                           # left arm joint pos
                [[-1.0, 0.0]],                                                                                # left hand grasp value
                [[-2.0, 2.0], [-2.0, 2.0], [-2.0, 2.0], [-np.pi, np.pi], [-np.pi, np.pi], [-np.pi, np.pi]], # right hand ee pos and angle euler
                arm_qpos_range_r,                                                                           # right arm joint pos
                [[-1.0, 0.0]],                                                                                # right hand grasp value
            ],
            dtype=np.float32,
            axis=0
        )

        self._action_range_min = self._action_range[:, 0]
        self._action_range_max = self._action_range[:, 1]
        

class AzureLoongHandAgent(AzureLoongAgent):
    def __init__(self, env: OpenLoongEnv, id: int, name: str) -> None:
        super().__init__(env, id, name)
        
        self.init_agent(env, id)
        super().init_agent(env, id)

        
    def init_agent(self, env: OpenLoongEnv, id: int):
        # print("arm_actuator_id: ", self._l_arm_actuator_id)
        self._l_hand_moto_names = [env.actuator("M_zbll_J1"), env.actuator("M_zbll_J2"), env.actuator("M_zbll_J3")
                                    ,env.actuator("M_zbll_J4"),env.actuator("M_zbll_J5"),env.actuator("M_zbll_J6"),
                                    env.actuator("M_zbll_J7"),env.actuator("M_zbll_J8"),env.actuator("M_zbll_J9"),
                                    env.actuator("M_zbll_J10"),env.actuator("M_zbll_J11")]
        self._l_hand_actuator_id = [env.model.actuator_name2id(actuator_name) for actuator_name in self._l_hand_moto_names]        
        self._l_hand_body_names = [env.body("zbll_Link1"), env.body("zbll_Link2"), env.body("zbll_Link3"),
                                   env.body("zbll_Link4"), env.body("zbll_Link5"), env.body("zbll_Link6"), 
                                   env.body("zbll_Link7"), env.body("zbll_Link8"), env.body("zbll_Link9"),
                                   env.body("zbll_Link10"), env.body("zbll_Link11")]
        self._l_hand_gemo_ids = []
        for geom_info in env.model.get_geom_dict().values():
            if geom_info["BodyName"] in self._l_hand_body_names:
                self._l_hand_gemo_ids.append(geom_info["GeomId"])


        self._r_hand_motor_names = [env.actuator("M_zbr_J1"), env.actuator("M_zbr_J2"), env.actuator("M_zbr_J3")
                                   ,env.actuator("M_zbr_J4"),env.actuator("M_zbr_J5"),env.actuator("M_zbr_J6"),
                                   env.actuator("M_zbr_J7"),env.actuator("M_zbr_J8"),env.actuator("M_zbr_J9"),
                                   env.actuator("M_zbr_J10"),env.actuator("M_zbr_J11")]
        self._r_hand_actuator_id = [env.model.actuator_name2id(actuator_name) for actuator_name in self._r_hand_motor_names]
        self._r_hand_body_names = [env.body("zbr_Link1"), env.body("zbr_Link2"), env.body("zbr_Link3"),
                                   env.body("zbr_Link4"), env.body("zbr_Link5"), env.body("zbr_Link6"), 
                                   env.body("zbr_Link7"), env.body("zbr_Link8"), env.body("zbr_Link9"),
                                   env.body("zbr_Link10"), env.body("zbr_Link11")]
        self._r_hand_gemo_ids = []
        for geom_info in env.model.get_geom_dict().values():
            if geom_info["BodyName"] in self._r_hand_body_names:
                self._r_hand_gemo_ids.append(geom_info["GeomId"])