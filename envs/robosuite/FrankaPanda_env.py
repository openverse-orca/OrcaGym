import numpy as np
from gymnasium.core import ObsType
from envs.robot_env import MujocoRobotEnv
from orca_gym.utils import rotations
from typing import Optional, Any, SupportsFloat
from gymnasium import spaces
from orca_gym.devices.xbox_joystick import XboxJoystickManager
from orca_gym.robosuite.controllers.controller_factory import controller_factory
import orca_gym.robosuite.controllers.controller_config as controller_config
import orca_gym.robosuite.utils.transform_utils as transform_utils
import h5py

class RecordState:
    RECORD = "record"
    REPLAY = "replay"
    REPLAY_FINISHED = "replay_finished"
    NONE = "none"

class FrankaPandaEnv(MujocoRobotEnv):
    """
    通过xbox手柄控制franka机械臂
    """
    def __init__(
        self,
        frame_skip: int = 5,        
        grpc_address: str = 'localhost:50051',
        agent_names: list = ['Agent0'],
        time_step: float = 0.00333333,
        record_state: str = RecordState.NONE,        
        record_file: Optional[str] = None,
        control_freq: int = 20,
        **kwargs,
    ):

        self.record_state = record_state
        self.record_file = record_file
        self.record_pool = []
        self.RECORD_POOL_SIZE = 1000
        self.record_cursor = 0

        action_size = 3 # 实际并不使用

        self.control_freq = control_freq

        super().__init__(
            frame_skip = frame_skip,
            grpc_address = grpc_address,
            agent_names = agent_names,
            time_step = time_step,            
            n_actions=action_size,
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
        self._controller_config["actuator_range"] = self._ctrl_range
        self._controller_config["policy_freq"] = self.control_freq
        self._controller_config["ndim"] = len(self._arm_joint_names)
        self._controller_config["control_delta"] = False


        self._controller = controller_factory(self._controller_config["type"], self._controller_config)
        self._controller.update_initial_joints(self._neutral_joint_values[0:7])


    def _set_init_state(self) -> None:
        # print("Set initial state")
        self.set_joint_neutral()

        self.ctrl = np.array(self._neutral_joint_values[0:9])
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
    
    def _set_gripper_ctrl(self, joystick_state) -> None:
        if (joystick_state["buttons"]["A"]):
            self.ctrl[7] += 0.001
            self.ctrl[8] += 0.001
        elif (joystick_state["buttons"]["B"]):
            self.ctrl[7] -= 0.001
            self.ctrl[8] -= 0.001

        self.ctrl[7] = np.clip(self.ctrl[7], 0, 0.08)
        self.ctrl[8] = np.clip(self.ctrl[8], 0, 0.08)

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
        print("action:", action)
        self._controller.set_goal(action)
        
        self.ctrl[0:7] = self._controller.run_controller()
        print("ctrl: ", self.ctrl)

        # 将控制数据存储到record_pool中
        if self.record_state == RecordState.RECORD:
            self._save_record()

    def _process_xbox_controller(self, mocap_xpos, mocap_xquat) -> tuple[np.ndarray, np.ndarray]:
        self._joystick_manager.update()

        pos_ctrl = self._joystick.capture_joystick_pos_ctrl()
        rot_ctrl = self._joystick.capture_joystick_rot_ctrl()
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


    def _save_record(self) -> None:
        self.record_pool.append(self.ctrl.copy())   
        if (len(self.record_pool) >= self.RECORD_POOL_SIZE):
            self.save_record()

    def _get_obs(self) -> dict:
        # robot
        EE_NAME = self.site("ee_center_site")
        ee_position = self.query_site_pos_and_quat([EE_NAME])[EE_NAME]['xpos'].copy()
        ee_xvalp, _ = self.query_site_xvalp_xvalr([EE_NAME])
        ee_velocity = ee_xvalp[EE_NAME].copy() * self.dt
        fingers_width = self.get_fingers_width().copy()


        achieved_goal = np.array([0,0,0])
        desired_goal = self.goal.copy()
        obs = np.concatenate(
                [
                    ee_position,
                    ee_velocity,
                    fingers_width
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
        self._set_init_state()
        self.set_grasp_mocap(self._initial_grasp_site_xpos, self._initial_grasp_site_xquat)
        self.mj_forward()
        return True

    # custom methods
    # -----------------------------
    def set_grasp_mocap(self, position, orientation) -> None:
        mocap_pos_and_quat_dict = {self.mocap("panda_mocap"): {'pos': position, 'quat': orientation}}
        self.set_mocap_pos_and_quat(mocap_pos_and_quat_dict)

    def set_goal_mocap(self, position, orientation) -> None:
        mocap_pos_and_quat_dict = {"goal_goal": {'pos': position, 'quat': orientation}}
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

    def _sample_goal(self) -> np.ndarray:
        # 训练reach时，任务是移动抓夹，goal以抓夹为原点采样
        goal = np.array([0, 0, 0])
        return goal


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
