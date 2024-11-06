import numpy as np
from gymnasium.core import ObsType
from orca_gym.utils import rotations
from envs import OrcaGymRemoteEnv
from typing import Optional, Any, SupportsFloat
from gymnasium import spaces
from orca_gym.devices.xbox_joystick import XboxJoystickManager
from orca_gym.robosuite.controllers.controller_factory import controller_factory
import orca_gym.robosuite.controllers.controller_config as controller_config
import orca_gym.robosuite.utils.transform_utils as transform_utils
from scipy.spatial.transform import Rotation as R


class OpenloongArmEnv(OrcaGymRemoteEnv):
    """
    控制青龙机器人机械臂
    """
    def __init__(
        self,
        frame_skip: int = 5,        
        grpc_address: str = 'localhost:50051',
        agent_names: list = ['Agent0'],
        time_step: float = 0.00333333,
        control_freq: int = 20,
        **kwargs,
    ):

        self.control_freq = control_freq

        super().__init__(
            frame_skip = frame_skip,
            grpc_address = grpc_address,
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

        print("arm_actuator_id: ", self._r_arm_actuator_id)
        print("hand_actuator_id: ", self._r_hand_actuator_id)
        # self._gripper_joint_names = [self.joint("finger_joint1"), self.joint("finger_joint2")]

        # index used to distinguish arm and gripper joints
        self._l_arm_joint_names = [self.joint("J_arm_l_01"), self.joint("J_arm_l_02"), 
                                 self.joint("J_arm_l_03"), self.joint("J_arm_l_04"), 
                                 self.joint("J_arm_l_05"), self.joint("J_arm_l_06"), self.joint("J_arm_l_07")]
        self._l_arm_moto_names = [self.actuator("M_arm_l_01"), self.actuator("M_arm_l_02"),
                                self.actuator("M_arm_l_03"),self.actuator("M_arm_l_04"),
                                self.actuator("M_arm_l_05"),self.actuator("M_arm_l_06"),self.actuator("M_arm_l_07")]
        self._l_arm_actuator_id = [self.model.actuator_name2id(actuator_name) for actuator_name in self._l_arm_moto_names]
        self._l_neutral_joint_values = np.array([-0.905, 0.735, 2.733, 1.405, 1.191, 0.012, 0.517])

        print("arm_actuator_id: ", self._l_arm_actuator_id)
        # self._gripper_joint_names = [self.joint("finger_joint1"), self.joint("finger_joint2")]

        # control range
        self._all_ctrlrange = self.model.get_actuator_ctrlrange()
        r_ctrl_range = [self._all_ctrlrange[actoator_id] for actoator_id in self._r_arm_actuator_id]
        print("ctrl_range: ", r_ctrl_range)

        l_ctrl_range = [self._all_ctrlrange[actoator_id] for actoator_id in self._l_arm_actuator_id]
        print("ctrl_range: ", l_ctrl_range)

        self.ctrl = np.zeros(self.nu)
        self._set_init_state()

        EE_NAME  = self.site("ee_center_site")
        _site_dict = self.query_site_pos_and_quat([EE_NAME])
        self._initial_grasp_site_xpos = _site_dict[EE_NAME]['xpos']
        self._initial_grasp_site_xquat = _site_dict[EE_NAME]['xquat']
        self._saved_xpos = self._initial_grasp_site_xpos
        self._saved_xquat = self._initial_grasp_site_xquat

        self.set_grasp_mocap(self._initial_grasp_site_xpos, self._initial_grasp_site_xquat)

        EE_NAME_R  = self.site("ee_center_site_r")
        _site_dict_r = self.query_site_pos_and_quat([EE_NAME_R])
        self._initial_grasp_site_xpos_r = _site_dict_r[EE_NAME_R]['xpos']
        self._initial_grasp_site_xquat_r = _site_dict_r[EE_NAME_R]['xquat']
        self._saved_xpos_r = self._initial_grasp_site_xpos_r
        self._saved_xquat_r = self._initial_grasp_site_xquat_r

        self.set_grasp_mocap_r(self._initial_grasp_site_xpos_r, self._initial_grasp_site_xquat_r)

        self._joystick_manager = XboxJoystickManager()
        joystick_names = self._joystick_manager.get_joystick_names()
        if len(joystick_names) == 0:
            raise ValueError("No joystick detected.")

        self._joystick = self._joystick_manager.get_joystick(joystick_names[0])
        if self._joystick is None:
            raise ValueError("Joystick not found.")

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


    def step(self, action) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        self._set_action()
        self.do_simulation(self.ctrl, self.frame_skip)
        obs = self._get_obs().copy()

        info = {}
        terminated = False
        truncated = False
        reward = 0

        return obs, reward, terminated, truncated, info
    
    def _set_reset_pos(self, joystick_state) -> None:
        if (joystick_state["buttons"]["X"] and joystick_state["buttons"]["Y"]):
            self._saved_xpos = self._initial_grasp_site_xpos
            self._saved_xquat = self._initial_grasp_site_xquat
            self._saved_xpos_r = self._initial_grasp_site_xpos_r
            self._saved_xquat_r = self._initial_grasp_site_xquat_r
            self._set_init_state()
            return True

    def _set_gripper_ctrl(self, joystick_state) -> None:
        offset_rate = 0
        if (joystick_state["buttons"]["A"]):
            offset_rate = -0.1 * self.dt
        elif (joystick_state["buttons"]["B"]):
            offset_rate = 0.1 * self.dt

        for actuator_id in self._r_hand_actuator_id:
            if self.model.actuator_id2name(actuator_id) == self.actuator("M_zbr_J2") or self.model.actuator_id2name(actuator_id) == self.actuator("M_zbr_J3"):
                offset_rate *= -1

            abs_ctrlrange = self._all_ctrlrange[actuator_id][1] - self._all_ctrlrange[actuator_id][0]
            self.ctrl[actuator_id] += offset_rate * abs_ctrlrange
            self.ctrl[actuator_id] = np.clip(self.ctrl[actuator_id], self._all_ctrlrange[actuator_id][0], self._all_ctrlrange[actuator_id][1])


    def _set_action(self) -> None:
        mocap_l_xpos, mocap_l_xquat, mocap_r_xpos, mocap_r_xquat = None, None, None, None

        # 根据xbox手柄的输入，设置机械臂的动作
        if self._joystick is not None:
            # mocap_l_xpos, mocap_l_xquat = self._process_xbox_controller(self._saved_xpos, self._saved_xquat)
            # self.set_grasp_mocap(mocap_l_xpos, mocap_l_xquat)
            # self._saved_xpos, self._saved_xquat = mocap_l_xpos, mocap_l_xquat
            mocap_l_xpos, mocap_l_xquat = self._saved_xpos, self._saved_xquat

            mocap_r_xpos, mocap_r_xquat = self._process_xbox_controller_r(self._saved_xpos_r, self._saved_xquat_r)
            self.set_grasp_mocap_r(mocap_r_xpos, mocap_r_xquat)
            self._saved_xpos_r, self._saved_xquat_r = mocap_r_xpos, mocap_r_xquat  
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
        

    def _process_xbox_controller_r(self, mocap_xpos, mocap_xquat) -> tuple[np.ndarray, np.ndarray]:
        self._joystick_manager.update()

        if self._set_reset_pos(self._joystick.get_state()):
            self._set_init_state()
            return self._saved_xpos_r, self._saved_xquat_r

        pos_ctrl_dict = self._joystick.capture_joystick_pos_ctrl()
        pos_ctrl = np.array([-pos_ctrl_dict['x'], pos_ctrl_dict['y'], pos_ctrl_dict['z']])
        rot_ctrl_dict = self._joystick.capture_joystick_rot_ctrl()
        rot_ctrl = np.array([rot_ctrl_dict['yaw'], rot_ctrl_dict['roll'] * 2, rot_ctrl_dict['pitch']])
        
        self._set_gripper_ctrl(self._joystick.get_state())

        # 考虑到手柄误差，只有输入足够的控制量，才移动mocap点
        CTRL_MIN = 0.10000000
        if np.linalg.norm(pos_ctrl) < CTRL_MIN and np.linalg.norm(rot_ctrl) < CTRL_MIN:
            return mocap_xpos, mocap_xquat

        mocap_xmat = rotations.quat2mat(mocap_xquat)

        # 平移控制
        MOVE_SPEED = self.gym.opt.timestep * 0.2
        mocap_xpos = mocap_xpos + np.dot(mocap_xmat, pos_ctrl) * MOVE_SPEED
        mocap_xpos[2] = np.max((0, mocap_xpos[2]))  # 确保在地面以上

        # 旋转控制
        ROUTE_SPEED = self.gym.opt.timestep * 0.5
        rot_offset = rot_ctrl * ROUTE_SPEED
        new_xmat = self.calc_rotate_matrix(rot_offset[0], rot_offset[1], rot_offset[2])
        mocap_xquat = rotations.mat2quat(np.dot(mocap_xmat, new_xmat))

        return mocap_xpos, mocap_xquat

    def calc_rotate_matrix(self, yaw, pitch, roll) -> np.ndarray:
        # x = roll, y = pitch, z = yaw
        R_yaw = np.array([
            [np.cos(yaw), -np.sin(yaw), 0],
            [np.sin(yaw), np.cos(yaw), 0],
            [0, 0, 1]
        ])

        R_pitch = np.array([
            [np.cos(pitch), 0, np.sin(pitch)],
            [0, 1, 0],
            [-np.sin(pitch), 0, np.cos(pitch)]
        ])

        R_roll = np.array([
            [1, 0, 0],
            [0, np.cos(roll), -np.sin(roll)],
            [0, np.sin(roll), np.cos(roll)]
        ])

        new_xmat = np.dot(R_yaw, np.dot(R_pitch, R_roll))
        return new_xmat

    def _get_obs(self) -> dict:
        # robot
        EE_NAME = self.site("ee_center_site")
        ee_position = self.query_site_pos_and_quat([EE_NAME])[EE_NAME]['xpos'].copy()
        ee_xvalp, _ = self.query_site_xvalp_xvalr([EE_NAME])
        ee_velocity = ee_xvalp[EE_NAME].copy() * self.dt


        achieved_goal = np.array([0,0,0])
        desired_goal = self.goal.copy()
        obs = np.concatenate(
                [
                    ee_position,
                    ee_velocity,
                    [0]
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
        self.mj_forward()
        obs = self._get_obs().copy()
        return obs

    # custom methods
    # -----------------------------
    def set_grasp_mocap(self, position, orientation) -> None:
        mocap_pos_and_quat_dict = {self.mocap("leftHandMocap"): {'pos': position, 'quat': orientation}}
        self.set_mocap_pos_and_quat(mocap_pos_and_quat_dict)

    def set_grasp_mocap_r(self, position, orientation) -> None:
        mocap_pos_and_quat_dict = {self.mocap("rightHandMocap"): {'pos': position, 'quat': orientation}}
        # print("Set grasp mocap: ", position, orientation)
        self.set_mocap_pos_and_quat(mocap_pos_and_quat_dict)


    def set_joint_neutral(self) -> None:
        # assign value to arm joints
        arm_joint_qpos = {}
        for name, value in zip(self._r_arm_joint_names, self._r_neutral_joint_values):
            arm_joint_qpos[name] = np.array([value])
        for name, value in zip(self._l_arm_joint_names, self._l_neutral_joint_values):
            arm_joint_qpos[name] = np.array([value])     
        self.set_joint_qpos(arm_joint_qpos)
        print("set init joint state: " , arm_joint_qpos)
        # assign value to finger joints
        # gripper_joint_qpos_list = {}
        # for name, value in zip(self._gripper_joint_names, self._neutral_joint_values[7:9]):
        #     gripper_joint_qpos_list[name] = np.array([value])
        # self.set_joint_qpos(gripper_joint_qpos_list)


    def get_ee_xform(self) -> np.ndarray:
        pos_dict = self.query_site_pos_and_mat([self.site("ee_center_site")])
        xpos = pos_dict[self.site("ee_center_site")]['xpos'].copy()
        xmat = pos_dict[self.site("ee_center_site")]['xmat'].copy().reshape(3, 3)
        return xpos, xmat

    def get_observation(self, obs=None):
        if obs is not None:
            return obs
        else:
            return self._get_obs().copy()