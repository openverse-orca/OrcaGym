import numpy as np
from gymnasium.core import ObsType
from orca_gym.utils import rotations
from typing import Optional, Any, SupportsFloat
from gymnasium import spaces
from orca_gym.devices.xbox_joystick import XboxJoystickManager
from orca_gym.devices.pico_joytsick import PicoJoystick
from orca_gym.robosuite.controllers.controller_factory import controller_factory
import orca_gym.robosuite.controllers.controller_config as controller_config
import orca_gym.robosuite.utils.transform_utils as transform_utils
from envs import OrcaGymRemoteEnv
from scipy.spatial.transform import Rotation as R

class OpenloongArmEnv(OrcaGymRemoteEnv):
    """
    控制青龙机器人机械臂
    """
    def __init__(
        self,
        frame_skip: int = 5,        
        orcagym_addr: str = 'localhost:50051',
        agent_names: list = ['Agent0'],
        time_step: float = 0.00333333,
        control_freq: int = 20,
        **kwargs,
    ):

        self.control_freq = control_freq

        super().__init__(
            frame_skip = frame_skip,
            orcagym_addr = orcagym_addr,
            agent_names = agent_names,
            time_step = time_step,            
            **kwargs,
        )


        # Three auxiliary variables to understand the component of the xml document but will not be used
        # number of actuators/controls: 7 arm joints and 2 gripper joints
        self.nu = self.model.nu
        # 16 generalized coordinates: 9 (arm + gripper) + 7 (object free joint: 3 position and 4 quaternion coordinates)
        self.nq = self.model.nq
        # 9 arm joints and 6 free joints
        self.nv = self.model.nv

        self.gym.opt.iterations = 150
        self.gym.opt.noslip_tolerance = 50
        self.gym.opt.mpr_iterations = 100
        self.gym.opt.sdf_iterations = 50
        self.set_opt_config()
        print("opt_config: ", self.query_opt_config())

        self.goal = self._sample_goal()

        self._base_body_name = [self.body("base_link")]
        self._base_body_xpos, _, self._base_body_xquat = self.get_body_xpos_xmat_xquat(self._base_body_name)
        print("base_body_xpos: ", self._base_body_xpos)
        print("base_body_xquat: ", self._base_body_xquat)

        self._neck_joint_names = [self.joint("J_head_yaw"), self.joint("J_head_pitch")]
        self._neck_actuator_names = [self.actuator("M_head_yaw"), self.actuator("M_head_pitch")]
        self._neck_actuator_id = [self.model.actuator_name2id(actuator_name) for actuator_name in self._neck_actuator_names]
        self._neck_neutral_joint_values = np.array([0.0, 0.0])
        self._neck_ctrl_values = {"yaw": 0.0, "pitch": 0.0}

        # index used to distinguish arm and gripper joints
        self._r_arm_joint_names = [self.joint("J_arm_r_01"), self.joint("J_arm_r_02"), 
                                 self.joint("J_arm_r_03"), self.joint("J_arm_r_04"), 
                                 self.joint("J_arm_r_05"), self.joint("J_arm_r_06"), self.joint("J_arm_r_07")]
        self._r_arm_moto_names = [self.actuator("M_arm_r_01"), self.actuator("M_arm_r_02"),
                                self.actuator("M_arm_r_03"),self.actuator("M_arm_r_04"),
                                self.actuator("M_arm_r_05"),self.actuator("M_arm_r_06"),self.actuator("M_arm_r_07")]
        self._r_arm_actuator_id = [self.model.actuator_name2id(actuator_name) for actuator_name in self._r_arm_moto_names]
        self._r_neutral_joint_values = np.array([0.905, -0.735, -2.733, 1.405, -1.191, 0.012, -0.517])
        
        self._r_hand_moto_names = [self.actuator("M_zbr_J1"), self.actuator("M_zbr_J2"), self.actuator("M_zbr_J3")
                                   ,self.actuator("M_zbr_J4"),self.actuator("M_zbr_J5"),self.actuator("M_zbr_J6"),
                                   self.actuator("M_zbr_J7"),self.actuator("M_zbr_J8"),self.actuator("M_zbr_J9"),
                                   self.actuator("M_zbr_J10"),self.actuator("M_zbr_J11")]
        self._r_hand_actuator_id = [self.model.actuator_name2id(actuator_name) for actuator_name in self._r_hand_moto_names]
        self._r_hand_body_names = [self.body("zbr_Link1"), self.body("zbr_Link2"), self.body("zbr_Link3"),
                                   self.body("zbr_Link4"), self.body("zbr_Link5"), self.body("zbr_Link6"), 
                                   self.body("zbr_Link7"), self.body("zbr_Link8"), self.body("zbr_Link9"),
                                   self.body("zbr_Link10"), self.body("zbr_Link11")]
        self._r_hand_gemo_ids = []
        for geom_info in self.model.get_geom_dict().values():
            if geom_info["BodyName"] in self._r_hand_body_names:
                self._r_hand_gemo_ids.append(geom_info["GeomId"])

        print("arm_actuator_id: ", self._r_arm_actuator_id)
        print("hand_actuator_id: ", self._r_hand_actuator_id)

        # index used to distinguish arm and gripper joints
        self._l_arm_joint_names = [self.joint("J_arm_l_01"), self.joint("J_arm_l_02"), 
                                 self.joint("J_arm_l_03"), self.joint("J_arm_l_04"), 
                                 self.joint("J_arm_l_05"), self.joint("J_arm_l_06"), self.joint("J_arm_l_07")]
        self._l_arm_moto_names = [self.actuator("M_arm_l_01"), self.actuator("M_arm_l_02"),
                                self.actuator("M_arm_l_03"),self.actuator("M_arm_l_04"),
                                self.actuator("M_arm_l_05"),self.actuator("M_arm_l_06"),self.actuator("M_arm_l_07")]
        self._l_arm_actuator_id = [self.model.actuator_name2id(actuator_name) for actuator_name in self._l_arm_moto_names]
        self._l_neutral_joint_values = np.array([-0.905, 0.735, 2.733, 1.405, 1.191, 0.012, 0.517])
        # self._l_neutral_joint_values = np.zeros(7)

        print("arm_actuator_id: ", self._l_arm_actuator_id)
        self._l_hand_moto_names = [self.actuator("M_zbll_J1"), self.actuator("M_zbll_J2"), self.actuator("M_zbll_J3")
                                    ,self.actuator("M_zbll_J4"),self.actuator("M_zbll_J5"),self.actuator("M_zbll_J6"),
                                    self.actuator("M_zbll_J7"),self.actuator("M_zbll_J8"),self.actuator("M_zbll_J9"),
                                    self.actuator("M_zbll_J10"),self.actuator("M_zbll_J11")]
        self._l_hand_actuator_id = [self.model.actuator_name2id(actuator_name) for actuator_name in self._l_hand_moto_names]        
        self._l_hand_body_names = [self.body("zbll_Link1"), self.body("zbll_Link2"), self.body("zbll_Link3"),
                                   self.body("zbll_Link4"), self.body("zbll_Link5"), self.body("zbll_Link6"), 
                                   self.body("zbll_Link7"), self.body("zbll_Link8"), self.body("zbll_Link9"),
                                   self.body("zbll_Link10"), self.body("zbll_Link11")]
        self._l_hand_gemo_ids = []
        for geom_info in self.model.get_geom_dict().values():
            if geom_info["BodyName"] in self._l_hand_body_names:
                self._l_hand_gemo_ids.append(geom_info["GeomId"])



        # control range
        self._all_ctrlrange = self.model.get_actuator_ctrlrange()
        neck_ctrl_range = [self._all_ctrlrange[actoator_id] for actoator_id in self._neck_actuator_id]
        print("ctrl_range: ", neck_ctrl_range)

        r_ctrl_range = [self._all_ctrlrange[actoator_id] for actoator_id in self._r_arm_actuator_id]
        print("ctrl_range: ", r_ctrl_range)

        l_ctrl_range = [self._all_ctrlrange[actoator_id] for actoator_id in self._l_arm_actuator_id]
        print("ctrl_range: ", l_ctrl_range)

        self.ctrl = np.zeros(self.nu)
        self._set_init_state()

        NECK_NAME  = self.site("neck_center_site")
        _site_dict = self.query_site_pos_and_quat([NECK_NAME])
        self._initial_neck_site_xpos = _site_dict[NECK_NAME]['xpos']
        self._initial_neck_site_xquat = _site_dict[NECK_NAME]['xquat']

        self.set_neck_mocap(self._initial_neck_site_xpos, self._initial_neck_site_xquat)
        self._mocap_neck_xpos, self._mocap_neck_xquat = self._initial_neck_site_xpos, self._initial_neck_site_xquat
        self._neck_angle_x, self._neck_angle_y = 0, 0

        EE_NAME  = self.site("ee_center_site")
        _site_dict = self.query_site_pos_and_quat([EE_NAME])
        self._initial_grasp_site_xpos = _site_dict[EE_NAME]['xpos']
        self._initial_grasp_site_xquat = _site_dict[EE_NAME]['xquat']

        self.set_grasp_mocap(self._initial_grasp_site_xpos, self._initial_grasp_site_xquat)

        EE_NAME_R  = self.site("ee_center_site_r")
        _site_dict_r = self.query_site_pos_and_quat([EE_NAME_R])
        self._initial_grasp_site_xpos_r = _site_dict_r[EE_NAME_R]['xpos']
        self._initial_grasp_site_xquat_r = _site_dict_r[EE_NAME_R]['xquat']

        self.set_grasp_mocap_r(self._initial_grasp_site_xpos_r, self._initial_grasp_site_xquat_r)
        
        self._pico_joystick = PicoJoystick()

        # -----------------------------
        # Neck controller
        self._neck_controller_config = controller_config.load_config("osc_pose")
        # print("controller_config: ", self.controller_config)

        # Add to the controller dict additional relevant params:
        #   the robot name, mujoco sim, eef_name, joint_indexes, timestep (model) freq,
        #   policy (control) freq, and ndim (# joints)
        self._neck_controller_config["robot_name"] = agent_names[0]
        self._neck_controller_config["sim"] = self.gym
        self._neck_controller_config["eef_name"] = NECK_NAME
        # self.controller_config["eef_rot_offset"] = self.eef_rot_offset
        qpos_offsets, qvel_offsets, _ = self.query_joint_offsets(self._neck_joint_names)
        self._neck_controller_config["joint_indexes"] = {
            "joints": self._neck_joint_names,
            "qpos": qpos_offsets,
            "qvel": qvel_offsets,
        }
        self._neck_controller_config["actuator_range"] = neck_ctrl_range
        self._neck_controller_config["policy_freq"] = self.control_freq
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
        self._r_controller_config["robot_name"] = agent_names[0]
        self._r_controller_config["sim"] = self.gym
        self._r_controller_config["eef_name"] = EE_NAME_R
        # self.controller_config["eef_rot_offset"] = self.eef_rot_offset
        qpos_offsets, qvel_offsets, _ = self.query_joint_offsets(self._r_arm_joint_names)
        self._r_controller_config["joint_indexes"] = {
            "joints": self._r_arm_joint_names,
            "qpos": qpos_offsets,
            "qvel": qvel_offsets,
        }
        self._r_controller_config["actuator_range"] = r_ctrl_range
        self._r_controller_config["policy_freq"] = self.control_freq
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
        self._l_controller_config["robot_name"] = agent_names[0]
        self._l_controller_config["sim"] = self.gym
        self._l_controller_config["eef_name"] = EE_NAME
        # self.controller_config["eef_rot_offset"] = self.eef_rot_offset
        qpos_offsets, qvel_offsets, _ = self.query_joint_offsets(self._l_arm_joint_names)
        self._l_controller_config["joint_indexes"] = {
            "joints": self._l_arm_joint_names,
            "qpos": qpos_offsets,
            "qvel": qvel_offsets,
        }
        self._l_controller_config["actuator_range"] = l_ctrl_range
        self._l_controller_config["policy_freq"] = self.control_freq
        self._l_controller_config["ndim"] = len(self._l_arm_joint_names)
        self._l_controller_config["control_delta"] = False


        self._l_controller = controller_factory(self._l_controller_config["type"], self._l_controller_config)
        self._l_controller.update_initial_joints(self._l_neutral_joint_values)

        self._l_gripper_offset_rate_clip = 0.0

        # Run generate_observation_space after initialization to ensure that the observation object's name is defined.
        self._set_obs_space()
        self._set_action_space()

    def _set_obs_space(self):
        self.observation_space = self.generate_observation_space(self._get_obs().copy())

    def _set_action_space(self):
        self.action_space = self.generate_action_space(self.model.get_actuator_ctrlrange())

    def _set_init_state(self) -> None:
        # print("Set initial state")
        self.set_joint_neutral()

        self.ctrl = np.zeros(self.nu)       
        self.set_ctrl(self.ctrl)
        self.mj_forward()

    def _reset_gripper(self) -> None:
        self._l_gripper_offset_rate_clip = 0.0
        self._r_gripper_offset_rate_clip = 0.0

    def _reset_neck_mocap(self) -> None:
        self._mocap_neck_xpos, self._mocap_neck_xquat = self._initial_neck_site_xpos, self._initial_neck_site_xquat
        self.set_neck_mocap(self._mocap_neck_xpos, self._mocap_neck_xquat)
        self._neck_angle_x, self._neck_angle_y = 0, 0

    def step(self, action) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        self._set_action()
        self.do_simulation(self.ctrl, self.frame_skip)
        obs = self._get_obs().copy()
        r_hand_force = self._query_hand_force(self._r_hand_gemo_ids)
        l_hand_force = self._query_hand_force(self._l_hand_gemo_ids)
        self._pico_joystick.send_force_message(l_hand_force, r_hand_force)

        info = {}
        terminated = False
        truncated = False
        reward = 0

        return obs, reward, terminated, truncated, info
    
    def _query_hand_force(self, hand_geom_ids):
        contact_simple_list = self.query_contact_simple()
        contact_force_query_ids = []
        for contact_simple in contact_simple_list:
            if contact_simple["Geom1"] in hand_geom_ids:
                contact_force_query_ids.append(contact_simple["ID"])
            if contact_simple["Geom2"] in hand_geom_ids:
                contact_force_query_ids.append(contact_simple["ID"])

        contact_force_dict = self.query_contact_force(contact_force_query_ids)
        compose_force = 0
        for force in contact_force_dict.values():
            compose_force += np.linalg.norm(force[:3])
        return compose_force

    def _set_head_ctrl(self, joystick_state) -> None:
        x_axis = joystick_state["rightHand"]["joystickPosition"][0]
        if x_axis == 0:
            x_axis = joystick_state["leftHand"]["joystickPosition"][0]

        y_axis = joystick_state["rightHand"]["joystickPosition"][1]
        if y_axis == 0:
            y_axis = joystick_state["leftHand"]["joystickPosition"][1]
            
        mocap_neck_xpos, mocap_neck_xquat = self._mocap_neck_xpos, self._mocap_neck_xquat

        # 将 x_axis 和 y_axis 输入转换为旋转角度，按需要调节比例系数
        angle_x = x_axis * np.pi / 180  # 转换为弧度，模拟绕 X 轴的旋转
        angle_y = y_axis * np.pi / 180  # 转换为弧度，模拟绕 Y 轴的旋转

        # 设置旋转角度的限制
        self._neck_angle_x += angle_x
        if self._neck_angle_x > np.pi / 3 or self._neck_angle_x < -np.pi / 3:
            self._neck_angle_x = np.clip(self._neck_angle_x, -np.pi / 3, np.pi / 3)
            angle_x = 0
        
        self._neck_angle_y += angle_y
        if self._neck_angle_y > np.pi / 3 or self._neck_angle_y < -np.pi / 3:
            self._neck_angle_y = np.clip(self._neck_angle_y, -np.pi / 3, np.pi / 3)
            angle_y = 0

        # 创建绕 X 轴和 Y 轴的旋转四元数 (局部坐标系旋转)
        rotation_x = R.from_euler('x', angle_x).as_quat()  # 绕 X 轴的旋转
        rotation_y = R.from_euler('y', angle_y).as_quat()  # 绕 Y 轴的旋转

        # 将初始的局部旋转四元数应用到增量旋转中，形成局部旋转的总四元数
        initial_neck_quat = R.from_quat(self._initial_neck_site_xquat)
        local_rotation = R.from_quat(rotation_x) * R.from_quat(rotation_y)
        new_neck_quat_local = initial_neck_quat * local_rotation  # 在局部坐标系应用旋转

        # 将局部坐标系的旋转转换为全局坐标系，乘以当前全局旋转四元数
        new_neck_quat_global = (R.from_quat(mocap_neck_xquat) * new_neck_quat_local).as_quat()

        # 将新的全局旋转四元数转换为轴角表示
        mocap_neck_axisangle = transform_utils.quat2axisangle(np.array([new_neck_quat_global[1], 
                                                                        new_neck_quat_global[2],
                                                                        new_neck_quat_global[3],
                                                                        new_neck_quat_global[0]]))

        # 可选：将轴角重新转换回四元数进行夹紧或其他操作
        new_neck_quat_cliped = transform_utils.axisangle2quat(mocap_neck_axisangle)

        # 将动作信息打包并发送到控制器
        action_neck = np.concatenate([mocap_neck_xpos, mocap_neck_axisangle])

        # 更新 _mocap_neck_xquat 为新的全局旋转值
        self._mocap_neck_xquat = new_neck_quat_global.copy()

        self._neck_controller.set_goal(action_neck)
        ctrl = self._neck_controller.run_controller()
        for i in range(len(self._neck_actuator_id)):
            self.ctrl[self._neck_actuator_id[i]] = ctrl[i]

        # 更新头部位置
        self.set_neck_mocap(mocap_neck_xpos, self._mocap_neck_xquat)

    def _set_gripper_ctrl(self, joystick_state) -> None:
        # Press secondary button to set gripper minimal value
        offset_rate_clip_adjust_rate = 0.1  # 10% per second
        if joystick_state["leftHand"]["secondaryButtonPressed"]:
            self._l_gripper_offset_rate_clip -= offset_rate_clip_adjust_rate * self.dt    
            self._l_gripper_offset_rate_clip = np.clip(self._l_gripper_offset_rate_clip, -1, 0)
        elif joystick_state["leftHand"]["primaryButtonPressed"]:
            self._l_gripper_offset_rate_clip += offset_rate_clip_adjust_rate * self.dt
            self._l_gripper_offset_rate_clip = np.clip(self._l_gripper_offset_rate_clip, -1, 0)

        # Press trigger to close gripper
        # Adjust sensitivity using an exponential function
        trigger_value = joystick_state["leftHand"]["triggerValue"]  # Value in [0, 1]
        k = np.e  # Adjust 'k' to change the curvature of the exponential function
        adjusted_value = (np.exp(k * trigger_value) - 1) / (np.exp(k) - 1)  # Maps input from [0, 1] to [0, 1]
        offset_rate = -adjusted_value
        offset_rate = np.clip(offset_rate, -1, self._l_gripper_offset_rate_clip)

        for actuator_id in self._l_hand_actuator_id:
            actuator_name = self.model.actuator_id2name(actuator_id)
            if actuator_name == self.actuator("M_zbll_J3"):
                offset_dir = -1
            else:
                offset_dir = 1

            abs_ctrlrange = self._all_ctrlrange[actuator_id][1] - self._all_ctrlrange[actuator_id][0]
            self.ctrl[actuator_id] = offset_rate * offset_dir * abs_ctrlrange
            self.ctrl[actuator_id] = np.clip(
                self.ctrl[actuator_id],
                self._all_ctrlrange[actuator_id][0],
                self._all_ctrlrange[actuator_id][1])

    def _set_gripper_ctrl_r(self, joystick_state) -> None:
        # Press secondary button to set gripper minimal value
        offset_rate_clip_adjust_rate = 0.1
        if joystick_state["rightHand"]["secondaryButtonPressed"]:
            self._r_gripper_offset_rate_clip -= offset_rate_clip_adjust_rate * self.dt
            self._r_gripper_offset_rate_clip = np.clip(self._r_gripper_offset_rate_clip, -1, 0)
        elif joystick_state["rightHand"]["primaryButtonPressed"]:
            self._r_gripper_offset_rate_clip += offset_rate_clip_adjust_rate * self.dt
            self._r_gripper_offset_rate_clip = np.clip(self._r_gripper_offset_rate_clip, -1, 0)

        # Adjust sensitivity using an exponential function
        trigger_value = joystick_state["rightHand"]["triggerValue"]  # Value in [0, 1]
        k = np.e  # Adjust 'k' to change the curvature of the exponential function
        adjusted_value = (np.exp(k * trigger_value) - 1) / (np.exp(k) - 1)  # Maps input from [0, 1] to [0, 1]
        offset_rate = -adjusted_value
        offset_rate = np.clip(offset_rate, -1, self._r_gripper_offset_rate_clip)

        for actuator_id in self._r_hand_actuator_id:
            actuator_name = self.model.actuator_id2name(actuator_id)
            if actuator_name == self.actuator("M_zbr_J2") or actuator_name == self.actuator("M_zbr_J3"):
                offset_dir = -1
            else:
                offset_dir = 1

            abs_ctrlrange = self._all_ctrlrange[actuator_id][1] - self._all_ctrlrange[actuator_id][0]
            self.ctrl[actuator_id] = offset_rate * offset_dir * abs_ctrlrange
            self.ctrl[actuator_id] = np.clip(
                self.ctrl[actuator_id],
                self._all_ctrlrange[actuator_id][0],
                self._all_ctrlrange[actuator_id][1])

 


    def _set_action(self) -> None:
        mocap_l_xpos, mocap_l_xquat, mocap_r_xpos, mocap_r_xquat = None, None, None, None

        if self._pico_joystick is not None:
            mocap_l_xpos, mocap_l_xquat, mocap_r_xpos, mocap_r_xquat = self._processe_pico_joystick_move()
            self.set_grasp_mocap(mocap_l_xpos, mocap_l_xquat)
            self.set_grasp_mocap_r(mocap_r_xpos, mocap_r_xquat)
            self._process_pico_joystick_operation()
            # print("base_body_euler: ", self._base_body_euler / np.pi * 180)
        else:
            return


        # 两个工具的quat不一样，这里将 qw, qx, qy, qz 转为 qx, qy, qz, qw
        mocap_r_axisangle = transform_utils.quat2axisangle(np.array([mocap_r_xquat[1], 
                                                                   mocap_r_xquat[2], 
                                                                   mocap_r_xquat[3], 
                                                                   mocap_r_xquat[0]]))              
        # mocap_axisangle[1] = -mocap_axisangle[1]
        action_r = np.concatenate([mocap_r_xpos, mocap_r_axisangle])
        # print("action r:", action_r)
        self._r_controller.set_goal(action_r)
        ctrl = self._r_controller.run_controller()
        # print("ctrl r: ", ctrl)
        for i in range(len(self._r_arm_actuator_id)):
            self.ctrl[self._r_arm_actuator_id[i]] = ctrl[i]


        mocap_l_axisangle = transform_utils.quat2axisangle(np.array([mocap_l_xquat[1], 
                                                                   mocap_l_xquat[2], 
                                                                   mocap_l_xquat[3], 
                                                                   mocap_l_xquat[0]]))  
        action_l = np.concatenate([mocap_l_xpos, mocap_l_axisangle])
        # print("action l:", action_l)        
        # print(action)
        self._l_controller.set_goal(action_l)
        ctrl = self._l_controller.run_controller()
        # print("ctrl l: ", ctrl)
        for i in range(len(self._l_arm_actuator_id)):
            self.ctrl[self._l_arm_actuator_id[i]] = ctrl[i]
        

    def _processe_pico_joystick_move(self):
        if self._pico_joystick.is_reset_pos():
            self._pico_joystick.set_reset_pos(False)
            self._set_init_state()
            self._reset_gripper()
            self._reset_neck_mocap()

        transform_list = self._pico_joystick.get_transform_list()
        if transform_list is None:
            return self._initial_grasp_site_xpos, self._initial_grasp_site_xquat, self._initial_grasp_site_xpos_r, self._initial_grasp_site_xquat_r

        left_relative_position, left_relative_rotation = self._pico_joystick.get_left_relative_move(transform_list)
        right_relative_position, right_relative_rotation = self._pico_joystick.get_right_relative_move(transform_list)

        # left_relative_position_org, left_relative_rotation_org = self._pico_joystick.get_left_relative_move_org(transform_list)
        # right_relative_position_org, right_relative_rotation_org = self._pico_joystick.get_right_relative_move_org(transform_list)

        # print("left_relative_position: ", left_relative_position)
        # print("left_relative_rotation: ", rotations.quat2euler(left_relative_rotation) * 180 / np.pi)
        # print("right_relative_position: ", right_relative_position)
        # print("right_relative_rotation: ", R.from_quat(right_relative_rotation, scalar_first=True).as_euler('xzy', degrees=True))
        # print("right_relative_rotation_org: ", R.from_quat(right_relative_rotation_org, scalar_first=True).as_euler('xzy', degrees=True))

        # def decompose(quat):
        #     v = R.from_quat(quat, scalar_first=True).as_rotvec(degrees=True)
        #     l = np.linalg.norm(v)
        #     v = v / l
        #     return [f'{v[0]:>12.6f} {v[1]:>12.6f} {v[2]:>12.6f}', l]
        

            # v = R.from_quat(quat, scalar_first=True).as_euler('zxy', degrees=True)
            # return f'{v[0]:>12.6f} {v[1]:>12.6f} {v[2]:>12.6f}'

        # print("rotation_org: ", decompose(right_relative_rotation_org))
        # print("rotation_mujo:", decompose(right_relative_rotation))

        mocap_l_xpos = self._initial_grasp_site_xpos + rotations.quat_rot_vec(self._base_body_xquat, left_relative_position)
        mocap_r_xpos = self._initial_grasp_site_xpos_r + rotations.quat_rot_vec(self._base_body_xquat, right_relative_position)

        mocap_l_xquat = rotations.quat_mul(self._initial_grasp_site_xquat, left_relative_rotation)
        # mocap_r_xquat = rotations.quat_mul(self._initial_grasp_site_xquat_r, right_relative_rotation)
        mocap_r_xquat = (R.from_quat(self._initial_grasp_site_xquat_r, scalar_first=True) * 
                         R.from_quat(right_relative_rotation, scalar_first=True)).as_quat(scalar_first=True, canonical=True)
        
   

        return mocap_l_xpos, mocap_l_xquat, mocap_r_xpos, mocap_r_xquat


    def _process_pico_joystick_operation(self):
        joystick_state = self._pico_joystick.get_key_state()
        if joystick_state is None:
            return

        self._set_gripper_ctrl_r(joystick_state)
        self._set_gripper_ctrl(joystick_state)
        self._set_head_ctrl(joystick_state)

    def _get_obs(self) -> dict:
        # robot
        EE_NAME = self.site("ee_center_site")
        ee_position = self.query_site_pos_and_quat([EE_NAME])[EE_NAME]['xpos'].copy()
        ee_xvalp, _ = self.query_site_xvalp_xvalr([EE_NAME])
        ee_velocity = ee_xvalp[EE_NAME].copy() * self.dt


        achieved_goal = np.zeros(3)
        desired_goal = self.goal.copy()
        obs = np.concatenate(
                [
                    ee_position,
                    ee_velocity,
                    np.zeros(1),
                ]).copy()            
        result = {
            "observation": obs,
            "achieved_goal": achieved_goal,
            "desired_goal": desired_goal,
        }
        return result


    def reset_model(self):
        self._set_init_state()
        self.set_grasp_mocap(self._initial_grasp_site_xpos, self._initial_grasp_site_xquat)
        self.set_grasp_mocap_r(self._initial_grasp_site_xpos_r, self._initial_grasp_site_xquat_r)
        self._reset_gripper()
        self._reset_neck_mocap()
        self.mj_forward()      
        obs = self._get_obs().copy()
        return obs

    # custom methods
    # -----------------------------
    def set_neck_mocap(self, position, orientation) -> None:
        mocap_pos_and_quat_dict = {self.mocap("neckMocap"): {'pos': position, 'quat': orientation}}
        self.set_mocap_pos_and_quat(mocap_pos_and_quat_dict)

    def set_grasp_mocap(self, position, orientation) -> None:
        mocap_pos_and_quat_dict = {self.mocap("leftHandMocap"): {'pos': position, 'quat': orientation}}
        self.set_mocap_pos_and_quat(mocap_pos_and_quat_dict)

    def set_grasp_mocap_r(self, position, orientation) -> None:
        mocap_pos_and_quat_dict = {self.mocap("rightHandMocap"): {'pos': position, 'quat': orientation}}
        # print("Set grasp mocap: ", position, orientation)
        self.set_mocap_pos_and_quat(mocap_pos_and_quat_dict)

    def set_goal_mocap(self, position, orientation) -> None:
        mocap_pos_and_quat_dict = {"goal_goal": {'pos': position, 'quat': orientation}}
        self.set_mocap_pos_and_quat(mocap_pos_and_quat_dict)

    def set_joint_neutral(self) -> None:
        # assign value to arm joints
        arm_joint_qpos = {}
        for name, value in zip(self._r_arm_joint_names, self._r_neutral_joint_values):
            arm_joint_qpos[name] = np.array([value])
        for name, value in zip(self._l_arm_joint_names, self._l_neutral_joint_values):
            arm_joint_qpos[name] = np.array([value])     
        for name, value in zip(self._neck_joint_names, self._neck_neutral_joint_values):
            arm_joint_qpos[name] = np.array([value])
        self.set_joint_qpos(arm_joint_qpos)
        # print("set init joint state: " , arm_joint_qpos_list)
        # assign value to finger joints
        # gripper_joint_qpos_list = {}
        # for name, value in zip(self._gripper_joint_names, self._neutral_joint_values[7:9]):
        #     gripper_joint_qpos_list[name] = np.array([value])
        # self.set_joint_qpos(gripper_joint_qpos_list)

    def _sample_goal(self) -> np.ndarray:
        # 训练reach时，任务是移动抓夹，goal以抓夹为原点采样
        goal = np.array([0, 0, 0])
        return goal


    def get_ee_xform(self) -> np.ndarray:
        pos_dict = self.query_site_pos_and_mat([self.site("ee_center_site")])
        xpos = pos_dict[self.site("ee_center_site")]['xpos'].copy()
        xmat = pos_dict[self.site("ee_center_site")]['xmat'].copy().reshape(3, 3)
        return xpos, xmat

    def close(self):
        self._pico_joystick.close()

    def get_observation(self, obs=None):
        if obs is not None:
            return obs
        else:
            return self._get_obs().copy()