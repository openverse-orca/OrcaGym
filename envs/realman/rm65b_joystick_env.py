import numpy as np
from gymnasium.core import ObsType
from orca_gym.environment import OrcaGymRemoteEnv
from orca_gym.utils import rotations
from typing import Optional, Any, SupportsFloat
from gymnasium import spaces
from orca_gym.devices.xbox_joystick import XboxJoystickManager
from scipy.spatial.transform import Rotation as R
from orca_gym.adapters.robosuite.controllers.controller_factory import controller_factory
import orca_gym.adapters.robosuite.controllers.controller_config as controller_config
import orca_gym.adapters.robosuite.utils.transform_utils as transform_utils

from orca_gym.log.orca_log import get_orca_logger
_logger = get_orca_logger()


class GripperState:
    OPENNING = "openning"
    CLOSING = "closing"
    STOPPED = "stopped"
    
class RM65BJoystickEnv(OrcaGymRemoteEnv):
    """
    通过xbox手柄控制机械臂
    """
    def __init__(
        self,
        frame_skip: int = 5,        
        orcagym_addr: str = 'localhost:50051',
        agent_names: list = ['Agent0'],
        time_step: float = 0.016,  # 0.016 for 60 fps        
        control_freq: int = 20,        
        **kwargs,
    ):

        super().__init__(
            frame_skip = frame_skip,
            orcagym_addr = orcagym_addr,
            agent_names = agent_names,
            time_step = time_step,            
            **kwargs,
        )

        self.control_freq = control_freq

        self._neutral_joint_values = np.array([0.00, 0.8, 1, 0, 1.3, 0, 0, 0, 0, 0])

        # Three auxiliary variables to understand the component of the xml document but will not be used
        self.nu = self.model.nu
        self.nq = self.model.nq
        self.nv = self.model.nv
        self.goal = self._sample_goal()


        # index used to distinguish arm and gripper joints
        self._arm_joint_names = [self.joint("joint1"), self.joint("joint2"), self.joint("joint3"), self.joint("joint4"), self.joint("joint5"), self.joint("joint6")]
        self.gripper_joint_names = [self.joint("Gripper_Link1"), self.joint("Gripper_Link11"), self.joint("Gripper_Link2"), self.joint("Gripper_Link22"),]
        self.gripper_body_names = [self.body("Gripper_Link11"), self.body("Gripper_Link22")]
        self.gripper_geom_ids = []
        for geom_info in self.model.get_geom_dict().values():
            if geom_info["BodyName"] in self.gripper_body_names:
                self.gripper_geom_ids.append(geom_info["GeomId"])

        self._arm_moto_names = [self.actuator("actuator1"), self.actuator("actuator2"),
                                self.actuator("actuator3"),self.actuator("actuator4"),
                                self.actuator("actuator5"),self.actuator("actuator6")]
        self._arm_actuator_id = [self.model.actuator_name2id(actuator_name) for actuator_name in self._arm_moto_names]
        self._gripper_actuator_names = [self.actuator("actuator_gripper1"), self.actuator("actuator_gripper2"),
                                        self.actuator("actuator_gripper11"), self.actuator("actuator_gripper22")]
        self._gripper_actuator_id = [self.model.actuator_name2id(actuator_name) for actuator_name in self._gripper_actuator_names]

        # control range
        all_actuator_ctrlrange = self.model.get_actuator_ctrlrange()
        self._arm_ctrl_range = [all_actuator_ctrlrange[actoator_id] for actoator_id in self._arm_actuator_id]
        self._gripper_ctrl_range = {actuator_name: all_actuator_ctrlrange[actuator_id] for actuator_name, actuator_id in zip(self._gripper_actuator_names, self._gripper_actuator_id)}
        _logger.info(f"gripper ctrl range:  {self._gripper_ctrl_range}")
        actuators_dict = self.model.get_actuator_dict()
        self.gripper_force_limit = 1 #actuators_dict[self.actuator("actuator_gripper1")]["ForceRange"][1]
        self.gripper_state = GripperState.STOPPED

        self._set_init_state()

        EE_NAME  = self.site("ee_center_site")
        site_dict = self.query_site_pos_and_quat([EE_NAME])
        self._initial_grasp_site_xpos = site_dict[EE_NAME]['xpos']
        self._initial_grasp_site_xquat = site_dict[EE_NAME]['xquat']
        self._saved_xpos = self._initial_grasp_site_xpos
        self._saved_xquat = self._initial_grasp_site_xquat

        self.set_grasp_mocap(self._initial_grasp_site_xpos, self._initial_grasp_site_xquat)

        self._joystick_manager = XboxJoystickManager()
        joystick_names = self._joystick_manager.get_joystick_names()
        if len(joystick_names) == 0:
            raise ValueError("No joystick detected.")

        self._joystick = self._joystick_manager.get_joystick(joystick_names[0])
        if self._joystick is None:
            raise ValueError("Joystick not found.")

        self._controller_config = controller_config.load_config("osc_pose")
        # print("controller_config: ", self.controller_config)

        # Add to the controller dict additional relevant params:
        #   the robot name, mujoco sim, eef_name, joint_indexes, timestep (model) freq,
        #   policy (control) freq, and ndim (# joints)
        self._controller_config["robot_name"] = agent_names[0]
        self._controller_config["sim"] = self.gym
        self._controller_config["eef_name"] = EE_NAME
        # self.controller_config["eef_rot_offset"] = self.eef_rot_offset
        qpos_offsets, qvel_offsets, _ = self.query_joint_offsets(self._arm_joint_names)
        self._controller_config["joint_indexes"] = {
            "joints": self._arm_joint_names,
            "qpos": qpos_offsets,
            "qvel": qvel_offsets,
        }
        self._controller_config["actuator_range"] = self._arm_ctrl_range
        self._controller_config["policy_freq"] = self.control_freq
        self._controller_config["ndim"] = len(self._arm_joint_names)
        self._controller_config["control_delta"] = False


        self._controller = controller_factory(self._controller_config["type"], self._controller_config)
        self._controller.update_initial_joints(self._neutral_joint_values[0:6])        

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

        self.ctrl = np.zeros(self.model.nu)
        self.set_ctrl(self.ctrl)

        self.mj_forward()

    def _sample_goal(self) -> np.ndarray:
        # 训练reach时，任务是移动抓夹，goal以抓夹为原点采样
        goal = np.array([0, 0, 0])
        return goal
    def step(self, action) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        self._set_action()
        self.do_simulation(self.ctrl, self.frame_skip)
        obs = self._get_obs().copy()

        info = {}
        terminated = False
        truncated = False
        reward = 0

        return obs, reward, terminated, truncated, info
    
    def _query_gripper_contact_force(self) -> dict:
        contact_simple_list = self.query_contact_simple()
        contact_force_query_ids = []
        for contact_simple in contact_simple_list:
            if contact_simple["Geom1"] in self.gripper_geom_ids:
                contact_force_query_ids.append(contact_simple["ID"])
            if contact_simple["Geom2"] in self.gripper_geom_ids:
                contact_force_query_ids.append(contact_simple["ID"])

        # print("Contact force query ids: ", contact_force_query_ids)
        contact_force_dict = self.query_contact_force(contact_force_query_ids)
        return contact_force_dict

    def _set_gripper_ctrl(self, joystick_state) -> None:
        MOVE_STEP = self.gym.opt.timestep * 0.5

        gripper_ctrl_1 = self.ctrl[6]
        gripper_ctrl_2 = self.ctrl[7]
        gripper_ctrl_11 = self.ctrl[8]
        gripper_ctrl_22 = self.ctrl[9]

        if (joystick_state["buttons"]["A"]):
            self.gripper_state = GripperState.CLOSING
            _logger.info(f"Gripper closing at:  {gripper_ctrl_1, gripper_ctrl_2}")
        elif (joystick_state["buttons"]["B"]):
            _logger.info(f"Gripper opening at:  {gripper_ctrl_1, gripper_ctrl_2}")
            self.gripper_state = GripperState.OPENNING

        if self.gripper_state == GripperState.CLOSING:
            contact_force_dict = self._query_gripper_contact_force()
            compose_force = 0
            for force in contact_force_dict.values():
                _logger.info(f"Gripper contact force:  {force}")
                compose_force += np.linalg.norm(force[:3])

            if compose_force >= self.gripper_force_limit:
                self.gripper_state = GripperState.STOPPED
                _logger.info(f"Gripper force limit reached. Stop gripper at:  {gripper_ctrl_1, gripper_ctrl_2}")

        if self.gripper_state == GripperState.CLOSING:
            _logger.info("grpper closing")
            gripper_ctrl_1 += MOVE_STEP
            gripper_ctrl_11 -= MOVE_STEP
            gripper_ctrl_2 -= MOVE_STEP
            gripper_ctrl_22 -= MOVE_STEP
            if gripper_ctrl_1 > self._gripper_ctrl_range[self.actuator("actuator_gripper1")][1]:
                gripper_ctrl_1 = self._gripper_ctrl_range[self.actuator("actuator_gripper1")][1]
                self.gripper_state = GripperState.STOPPED
                _logger.info(f"Gripper Stop at:  {gripper_ctrl_1, gripper_ctrl_2}")
            if gripper_ctrl_2 < self._gripper_ctrl_range[self.actuator("actuator_gripper2")][0]:
                gripper_ctrl_2 = self._gripper_ctrl_range[self.actuator("actuator_gripper2")][0]
                self.gripper_state = GripperState.STOPPED
                _logger.info(f"Gripper Stop at:  {gripper_ctrl_1, gripper_ctrl_2}")
        elif self.gripper_state == GripperState.OPENNING:
            _logger.info("gripper openning")
            gripper_ctrl_1 -= MOVE_STEP
            gripper_ctrl_11 += MOVE_STEP
            gripper_ctrl_2 += MOVE_STEP
            gripper_ctrl_22 += MOVE_STEP
            if gripper_ctrl_1 < self._gripper_ctrl_range[self.actuator("actuator_gripper1")][0]:
                gripper_ctrl_1 = self._gripper_ctrl_range[self.actuator("actuator_gripper1")][0]
                self.gripper_state = GripperState.STOPPED
                _logger.info(f"Gripper Stop at:  {gripper_ctrl_1, gripper_ctrl_2}")
            if gripper_ctrl_2 > self._gripper_ctrl_range[self.actuator("actuator_gripper2")][1]:
                gripper_ctrl_2 = self._gripper_ctrl_range[self.actuator("actuator_gripper2")][1]
                self.gripper_state = GripperState.STOPPED
                _logger.info(f"Gripper Stop at:  {gripper_ctrl_1, gripper_ctrl_2}")

        self.ctrl[6] = gripper_ctrl_1
        self.ctrl[7] = gripper_ctrl_2
        self.ctrl[8] = gripper_ctrl_11
        self.ctrl[9] = gripper_ctrl_22

    def _set_action(self) -> None:
        mocap_xpos = self._saved_xpos
        mocap_xquat = self._saved_xquat
        
        # 根据xbox手柄的输入，设置机械臂的动作
        if self._joystick is not None:
            mocap_xpos, mocap_xquat = self._process_xbox_controller(mocap_xpos, mocap_xquat)
            self.set_grasp_mocap(mocap_xpos, mocap_xquat)
            self._saved_xpos = mocap_xpos
            self._saved_xquat = mocap_xquat

        # 两个工具的quat不一样，这里将 qw, qx, qy, qz 转为 qx, qy, qz, qw
        mocap_axisangle = transform_utils.quat2axisangle(np.array([mocap_xquat[1], 
                                                                   mocap_xquat[2], 
                                                                   mocap_xquat[3], 
                                                                   mocap_xquat[0]]))
        # mocap_axisangle[1] = -mocap_axisangle[1]
        action = np.concatenate([mocap_xpos, mocap_axisangle])
        # print("action:", action)
        self._controller.set_goal(action)
        
        arm_moto_ctrl = self._controller.run_controller()
        for i in range(len(self._arm_actuator_id)):
            self.ctrl[self._arm_actuator_id[i]] = arm_moto_ctrl[i]

        # 特殊处理，osc算出来的link1 moto 值太小，因此放大link1和link6的gear倍数，这里要裁剪回去
        self.ctrl[self._arm_actuator_id[0]] = np.clip(self.ctrl[self._arm_actuator_id[0]], self._arm_ctrl_range[0][0] * 0.1, self._arm_ctrl_range[0][1] * 0.1)
        self.ctrl[self._arm_actuator_id[5]] = np.clip(self.ctrl[self._arm_actuator_id[5]], self._arm_ctrl_range[5][0] * 0.1, self._arm_ctrl_range[5][1] * 0.1)
        # print("ctrl: ", self.ctrl)

    def _process_xbox_controller(self, mocap_xpos, mocap_xquat) -> tuple[np.ndarray, np.ndarray]:
        self._joystick_manager.update()

        pos_ctrl_dict = self._joystick.capture_joystick_pos_ctrl()
        pos_ctrl = np.array([pos_ctrl_dict['z'], -pos_ctrl_dict['x'], pos_ctrl_dict['y']])
        rot_ctrl_dict = self._joystick.capture_joystick_rot_ctrl()
        rot_ctrl = np.array([rot_ctrl_dict['yaw'], rot_ctrl_dict['pitch'], rot_ctrl_dict['roll']])
        
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
        # x = yaw, y = pitch, z = roll
        R_yaw = np.array([
            [1, 0, 0],
            [0, np.cos(yaw), -np.sin(yaw)],
            [0, np.sin(yaw), np.cos(yaw)]
        ])

        R_pitch = np.array([
            [np.cos(pitch), 0, np.sin(pitch)],
            [0, 1, 0],
            [-np.sin(pitch), 0, np.cos(pitch)]
        ])

        R_roll = np.array([
            [np.cos(roll), -np.sin(roll), 0],
            [np.sin(roll), np.cos(roll), 0],
            [0, 0, 1]
        ])

        new_xmat = np.dot(R_yaw, np.dot(R_pitch, R_roll))
        return new_xmat

    def _get_obs(self) -> dict:
        # robot
        EE_NAME = self.site("ee_center_site")
        ee_position = self.query_site_pos_and_quat([EE_NAME])[EE_NAME]['xpos'].copy()
        ee_xvalp, _ = self.query_site_xvalp_xvalr([EE_NAME])
        ee_velocity = ee_xvalp[EE_NAME].copy() * self.dt
        fingers_width = [0]


        achieved_goal = np.array([0,0,0])
        desired_goal = self.goal.copy()
        obs = np.concatenate(
                [
                    ee_position,
                    ee_velocity,
                    fingers_width
                ], dtype=np.float32).copy()            
        result = {
            "observation": obs,
            "achieved_goal": achieved_goal,
            "desired_goal": desired_goal,
        }
        return result


    def reset_model(self) -> tuple[dict, dict]:
        self._set_init_state()
        self.set_grasp_mocap(self._initial_grasp_site_xpos, self._initial_grasp_site_xquat)
        self.mj_forward()
        obs = self._get_obs().copy()
        return obs, {}

    # custom methods
    # -----------------------------

    def set_grasp_mocap(self, position, orientation) -> None:
        mocap_pos_and_quat_dict = {self.mocap("rm65b_mocap"): {'pos': position, 'quat': orientation}}
        # print("Set grasp mocap: ", position, orientation)
        self.set_mocap_pos_and_quat(mocap_pos_and_quat_dict)

    def set_joint_neutral(self) -> None:
        # assign value to arm joints
        arm_joint_qpos = {}
        for name, value in zip(self._arm_joint_names, self._neutral_joint_values[0:6]):
            arm_joint_qpos[name] = np.array([value])
        self.set_joint_qpos(arm_joint_qpos)

        # assign value to finger joints
        gripper_joint_qpos = {}
        for name, value in zip(self.gripper_joint_names, self._neutral_joint_values[6:10]):
            gripper_joint_qpos[name] = np.array([value])
        self.set_joint_qpos(gripper_joint_qpos)

    def get_observation(self, obs=None):
        if obs is not None:
            return obs
        else:
            return self._get_obs().copy()