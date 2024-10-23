import numpy as np
from gymnasium.core import ObsType
from orca_gym.utils import rotations
from typing import Optional, Any, SupportsFloat
from gymnasium import spaces
from orca_gym.devices.xbox_joystick import XboxJoystickManager
from orca_gym.robosuite.controllers.controller_factory import controller_factory
import orca_gym.robosuite.controllers.controller_config as controller_config
import orca_gym.robosuite.utils.transform_utils as transform_utils
import h5py
from envs.robomimic.robomimic_env import RobomimicEnv
from envs.robomimic.robomimic_env import ControlType
from envs.orca_gym_env import ActionSpaceType, RewardType


class FrankaTeleoperationEnv(RobomimicEnv):
    """
    通过遥操作控制franka机械臂
    """
    ENV_VERSION = "2024.10.1"

    def __init__(
        self,
        frame_skip: int,        
        grpc_address: str,
        agent_names: list,
        time_step: float,
        control_type: ControlType,
        control_freq: int,
        **kwargs,
    ):

        self.control_type = control_type
        self.control_freq = control_freq
        self.reward_type = kwargs["reward_type"]

        super().__init__(
            frame_skip = frame_skip,
            grpc_address = grpc_address,
            agent_names = agent_names,
            time_step = time_step,            
            observation_space = None,
            **kwargs,
        )

        self._neutral_joint_values = np.array([0.00, 0.41, 0.00, -1.85, 0.00, 2.26, 0.79, 0.00, 0.00])

        # Three auxiliary variables to understand the component of the xml document but will not be used
        # number of actuators/controls: 7 arm joints and 2 gripper joints
        self.nu = self.model.nu
        # 16 generalized coordinates: 9 (arm + gripper) + 7 (object free joint: 3 position and 4 quaternion coordinates)
        self.nq = self.model.nq
        # 9 arm joints and 6 free joints
        self.nv = self.model.nv

        # control range
        self._ctrl_range = self.model.get_actuator_ctrlrange()

        # index used to distinguish arm and gripper joints
        self._arm_joint_names = [self.joint("joint1"), self.joint("joint2"), self.joint("joint3"), self.joint("joint4"), self.joint("joint5"), self.joint("joint6"), self.joint("joint7")]
        self._gripper_joint_names = [self.joint("finger_joint1"), self.joint("finger_joint2")]

        self._set_init_state()

        self.EE_NAME  = self.site("ee_center_site")
        site_dict = self.query_site_pos_and_quat([self.EE_NAME])
        self._initial_grasp_site_xpos = site_dict[self.EE_NAME]['xpos']
        self._initial_grasp_site_xquat = site_dict[self.EE_NAME]['xquat']
        self._reset_grasp_mocap()

        self.OBJ_NAME = "Toys_Object"
        site_dict = self.query_site_pos_and_quat([self.OBJ_NAME])
        self._initial_obj_site_xpos = site_dict[self.OBJ_NAME]['xpos']
        self._initial_obj_site_xquat = site_dict[self.OBJ_NAME]['xquat']
        self._sample_object()

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
        self._controller_config["eef_name"] = self.EE_NAME
        # self.controller_config["eef_rot_offset"] = self.eef_rot_offset
        qpos_offsets, qvel_offsets, _ = self.query_joint_offsets(self._arm_joint_names)
        self._controller_config["joint_indexes"] = {
            "joints": self._arm_joint_names,
            "qpos": qpos_offsets,
            "qvel": qvel_offsets,
        }
        self._controller_config["actuator_range"] = self._ctrl_range
        self._controller_config["policy_freq"] = self.control_freq
        self._controller_config["ndim"] = len(self._arm_joint_names)
        self._controller_config["control_delta"] = False


        self._controller = controller_factory(self._controller_config["type"], self._controller_config)
        self._controller.update_initial_joints(self._neutral_joint_values[0:7])


    def _reset_grasp_mocap(self) -> None:
        self._saved_xpos = self._initial_grasp_site_xpos
        self._saved_xquat = self._initial_grasp_site_xquat
        self.set_grasp_mocap(self._initial_grasp_site_xpos, self._initial_grasp_site_xquat)

    def get_env_version(self):
        return FrankaTeleoperationEnv.ENV_VERSION

    def _set_init_state(self) -> None:
        # print("Set initial state")
        self.set_joint_neutral()

        self.ctrl = np.array(self._neutral_joint_values[0:9])
        self.set_ctrl(self.ctrl)
        self.mj_forward()

    def _compute_reward(self, achieved_goal, desired_goal, info) -> float:
        if self.reward_type == RewardType.SPARSE:
            return 1 if self._is_success(achieved_goal, desired_goal) else 0
        elif self.reward_type == RewardType.DENSE:
            return -np.linalg.norm(achieved_goal - desired_goal)
        else:
            raise ValueError("Invalid reward type")
    
    def _is_success(self, achieved_goal, desired_goal) -> bool:
        success_threshold = 0.01
        return np.linalg.norm(achieved_goal - desired_goal) < success_threshold

    def step(self, action) -> tuple:
        if (self.control_type == ControlType.TELEOPERATION):
            action = self._teleoperation_action().copy()
            
        self.do_simulation(action, self.frame_skip)
        obs = self._get_obs().copy()
        achieved_goal = self._get_achieved_goal().copy()
        desired_goal = self._get_desired_goal().copy()

        info = {"state": self.get_state(), "action": action}
        terminated = self._is_success(achieved_goal, desired_goal)
        truncated = False
        reward = self._compute_reward(achieved_goal, desired_goal, info)

        return obs, reward, terminated, truncated, info
    
    def get_state(self) -> dict:
        state = {
            "time": self.data.time,
            "qpos": self.data.qpos.copy(),
            "qvel": self.data.qvel.copy(),
            "qacc": self.data.qacc.copy(),
            "ctrl": self.ctrl.copy(),
        }
        return state
    
    def _set_gripper_ctrl(self, joystick_state) -> None:
        if (joystick_state["buttons"]["A"]):
            self.ctrl[7] += 0.001
            self.ctrl[8] += 0.001
        elif (joystick_state["buttons"]["B"]):
            self.ctrl[7] -= 0.001
            self.ctrl[8] -= 0.001

        self.ctrl[7] = np.clip(self.ctrl[7], 0, 0.08)
        self.ctrl[8] = np.clip(self.ctrl[8], 0, 0.08)


    def _teleoperation_action(self) -> np.ndarray:
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
        
        self.ctrl[0:7] = self._controller.run_controller()
        
        return self.ctrl


    def _process_xbox_controller(self, mocap_xpos, mocap_xquat) -> tuple[np.ndarray, np.ndarray]:
        self._joystick_manager.update()

        pos_ctrl_dict = self._joystick.capture_joystick_pos_ctrl()
        pos_ctrl = np.array([pos_ctrl_dict['y'], pos_ctrl_dict['x'], pos_ctrl_dict['z']])
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
        new_xmat = self._joystick.calc_rotate_matrix(rot_offset[0], rot_offset[1], rot_offset[2])
        mocap_xquat = rotations.mat2quat(np.dot(mocap_xmat, new_xmat))

        return mocap_xpos, mocap_xquat

    def _get_obs(self) -> dict:
        ee_position = self.query_site_pos_and_quat([self.EE_NAME])[self.EE_NAME].copy()
        ee_xvalp, ee_xvalr = self.query_site_xvalp_xvalr([self.EE_NAME])
        fingers_width = self.get_fingers_width().copy()
        obj_xpos, obj_xquat = self._query_obj_pos_and_quat()
        joint_values = self.get_arm_joint_values().copy()
        joint_values_sin = np.sin(joint_values)
        joint_values_cos = np.cos(joint_values)
        joint_velocities = self.get_arm_joint_velocities().copy()

        obs = {
            "object": np.concatenate([obj_xpos, obj_xquat]),
            "ee_pos": ee_position["xpos"],
            "ee_quat": ee_position["xquat"],
            "ee_vel_linear": ee_xvalp[self.EE_NAME],
            "ee_vel_angular": ee_xvalr[self.EE_NAME],
            "joint_qpos": joint_values,
            "joint_qpos_sin": joint_values_sin,
            "joint_qpos_cos": joint_values_cos,
            "joint_vel": joint_velocities,
            "fingers_width": fingers_width,
        }
        return obs
    
    def _get_achieved_goal(self) -> dict:
        obj_xpos, obj_xquat = self._query_obj_pos_and_quat()
        achieved_goal = {"achieved_goal": obj_xpos.copy()}
        return achieved_goal
    
    def _get_desired_goal(self) -> dict:
        desired_goal = {"desired_goal": self.goal.copy()}
        return desired_goal

    def _render_callback(self) -> None:
        pass

    def reset_model(self) -> dict:
        """
        Reset the environment, return observation
        """
        self._set_init_state()
        self._reset_grasp_mocap()
        self._sample_object()
        self.goal = self._sample_goal()
        self.mj_forward()
        obs = self._get_obs().copy()
        return obs

    # custom methods
    # -----------------------------
    def set_grasp_mocap(self, position, orientation) -> None:
        mocap_pos_and_quat_dict = {self.mocap("panda_mocap"): {'pos': position, 'quat': orientation}}
        self.set_mocap_pos_and_quat(mocap_pos_and_quat_dict)

    def set_object_mocap(self, position, orientation) -> None:
        mocap_pos_and_quat_dict = {"Toys_Box1": {'pos': position, 'quat': orientation}}
        self.set_mocap_pos_and_quat(mocap_pos_and_quat_dict)

    def set_joint_neutral(self) -> None:
        # assign value to arm joints
        arm_joint_qpos_list = {}
        for name, value in zip(self._arm_joint_names, self._neutral_joint_values[0:7]):
            arm_joint_qpos_list[name] = np.array([value])
        self.set_joint_qpos(arm_joint_qpos_list)

        # assign value to finger joints
        gripper_joint_qpos_list = {}
        for name, value in zip(self._gripper_joint_names, self._neutral_joint_values[7:9]):
            gripper_joint_qpos_list[name] = np.array([value])
        self.set_joint_qpos(gripper_joint_qpos_list)

    def _query_obj_pos_and_quat(self) -> tuple:
        site_dict = self.query_site_pos_and_quat([self.OBJ_NAME])
        obj_xpos, obj_xquat = site_dict[self.OBJ_NAME]['xpos'], site_dict[self.OBJ_NAME]['xquat']
        return obj_xpos, obj_xquat

    def _sample_goal(self) -> np.ndarray:
        obj_xpos, obj_xquat = self._query_obj_pos_and_quat()
        goal_xpos = obj_xpos.copy()
        goal_xpos[2] += 0.1
        return goal_xpos

    def _sample_object(self) -> None:
        """
        随机采样一个物体的位置
        """
        obj_xpos = self._initial_obj_site_xpos
        obj_xquat = self._initial_obj_site_xquat
        obj_euler = rotations.quat2euler(obj_xquat)

        obj_xpos[0] = np.random.uniform(-0.3, 0.3) + obj_xpos[0]
        obj_xpos[1] = np.random.uniform(-0.3, 0.3) + obj_xpos[1]
        obj_euler[2] = np.random.uniform(-np.pi, np.pi)
        obj_xquat = rotations.euler2quat(obj_euler)

        self.set_object_mocap(obj_xpos, obj_xquat)


    def get_ee_xform(self) -> np.ndarray:
        pos_dict = self.query_site_pos_and_mat([self.site("ee_center_site")])
        xpos = pos_dict[self.site("ee_center_site")]['xpos'].copy()
        xmat = pos_dict[self.site("ee_center_site")]['xmat'].copy().reshape(3, 3)
        return xpos, xmat

    def get_fingers_width(self) -> np.ndarray:
        qpos_dict = self.query_joint_qpos([self.joint("finger_joint1"), self.joint("finger_joint2")])
        finger1 = qpos_dict[self.joint("finger_joint1")]
        finger2 = qpos_dict[self.joint("finger_joint2")]
        return finger1 + finger2
    
    def get_arm_joint_values(self) -> np.ndarray:
        qpos_dict = self.query_joint_qpos(self._arm_joint_names)
        return np.array([qpos_dict[joint_name] for joint_name in self._arm_joint_names])
    
    def get_arm_joint_velocities(self) -> np.ndarray:
        qvel_dict = self.query_joint_qvel(self._arm_joint_names)
        return np.array([qvel_dict[joint_name] for joint_name in self._arm_joint_names])

    def get_observation(self, obs=None):
        """
        Return the current environment observation as a dictionary, unless obs is not None.
        This function should process the raw environment observation to align with the input expected by the policy model.
        For example, it should cast an image observation to float with value range 0-1 and shape format [C, H, W].
        """
        if obs is not None:
            return obs
        
        return self._get_obs().copy()