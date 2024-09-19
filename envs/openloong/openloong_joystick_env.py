import numpy as np
from gymnasium.core import ObsType
from envs.robot_env import MujocoRobotEnv
from orca_gym.utils import rotations
from typing import Optional, Any, SupportsFloat
from gymnasium import spaces
from orca_gym.devices.xbox_joystick import XboxJoystickManager
import h5py
from scipy.spatial.transform import Rotation as R

class RecordState:
    RECORD = "record"
    REPLAY = "replay"
    REPLAY_FINISHED = "replay_finished"
    NONE = "none"

class OpenloongJoystickEnv(MujocoRobotEnv):
    """
    通过xbox手柄控制机械臂
    """
    def __init__(
        self,
        frame_skip: int = 5,        
        grpc_address: str = 'localhost:50051',
        agent_names: list = ['Agent0'],
        time_step: float = 0.016,  # 0.016 for 60 fps        
        record_state: str = RecordState.NONE,        
        record_file: Optional[str] = None,
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

        self.neutral_joint_values = np.array([0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00])

        # Three auxiliary variables to understand the component of the xml document but will not be used
        # number of actuators/controls: 7 arm joints and 2 gripper joints
        self.nu = self.model.nu
        # 16 generalized coordinates: 9 (arm + gripper) + 7 (object free joint: 3 position and 4 quaternion coordinates)
        self.nq = self.model.nq
        # 9 arm joints and 6 free joints
        self.nv = self.model.nv

        # control range
        self.ctrl_range = self.model.get_actuator_ctrlrange()

        # index used to distinguish arm and gripper joints
        self.arm_joint_names = [self.joint("J_arm_l_01"), self.joint("J_arm_l_02"), self.joint("J_arm_l_03"), self.joint("J_arm_l_04"), self.joint("J_arm_l_05"), self.joint("J_arm_l_06"), self.joint("J_arm_l_07"), self.joint("J_arm_r_01"), self.joint("J_arm_r_02"), self.joint("J_arm_r_03"), self.joint("J_arm_r_04"), self.joint("J_arm_r_05"), self.joint("J_arm_r_06"), self.joint("J_arm_r_07"), self.joint("J_head_yaw"), self.joint("J_head_pitch")]
        self.left_arm_joint_names = [self.joint("J_arm_l_01"), self.joint("J_arm_l_02"), self.joint("J_arm_l_03"), self.joint("J_arm_l_04"), self.joint("J_arm_l_05"), self.joint("J_arm_l_06"), self.joint("J_arm_l_07")]
        self.right_arm_joint_names = [self.joint("J_arm_r_01"), self.joint("J_arm_r_02"), self.joint("J_arm_r_03"), self.joint("J_arm_r_04"), self.joint("J_arm_r_05"), self.joint("J_arm_r_06"), self.joint("J_arm_r_07")]

        self._set_init_state()

        EE_NAME  = self.site("ee_center_site")
        site_dict = self.query_site_pos_and_quat([EE_NAME])
        self.initial_grasp_site_xpos = site_dict[EE_NAME]['xpos']
        self.initial_grasp_site_xquat = site_dict[EE_NAME]['xquat']
        self.save_xquat = self.initial_grasp_site_xquat.copy()

        self.set_grasp_mocap(self.initial_grasp_site_xpos, self.initial_grasp_site_xquat)

        EE_NAME_R  = self.site("ee_center_site_r")
        site_dict_r = self.query_site_pos_and_quat([EE_NAME_R])
        self.initial_grasp_site_xpos_r = site_dict_r[EE_NAME_R]['xpos']
        self.initial_grasp_site_xquat_r = site_dict_r[EE_NAME_R]['xquat']
        self.save_xquat_r = self.initial_grasp_site_xquat_r.copy()

        self.set_grasp_mocap_r(self.initial_grasp_site_xpos_r, self.initial_grasp_site_xquat_r)

        self.joystick_manager = XboxJoystickManager()
        joystick_names = self.joystick_manager.get_joystick_names()
        if len(joystick_names) == 0:
            raise ValueError("No joystick detected.")

        self.joysticks = []
        for name in joystick_names:
            self.joysticks.append(self.joystick_manager.get_joystick(name))
            if len(self.joysticks) >= 2:
                break
        

    def _set_init_state(self) -> None:
        # print("Set initial state")
        self.set_joint_neutral()

        self.ctrl = np.array([0.00] * 31)

        self.set_ctrl(self.ctrl)

        self.reset_mocap_welds()

        self.mj_forward()


    def step(self, action) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        self._set_action()
        # self.ctrl = np.array([0.00, -0.8, -1, 0, -1.3, 1, 0.00, 0.00])         # for test joint control
        self.do_simulation(self.ctrl, self.frame_skip)
        obs = self._get_obs().copy()

        info = {}
        terminated = False
        truncated = False
        reward = 0

        return obs, reward, terminated, truncated, info

    def _load_record(self):
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

        # 根据xbox手柄的输入，设置机械臂的动作
        self.joystick_manager.update()

        changed = False
        for index, joystick in enumerate(self.joysticks):
            # print(index, joystick_state)
            pos_ctrl_dict = self._joystick.capture_joystick_pos_ctrl()
            pos_ctrl = np.array([pos_ctrl_dict['x'], pos_ctrl_dict['y'], pos_ctrl_dict['z']])
            rot_ctrl_dict = self._joystick.capture_joystick_rot_ctrl()
            rot_ctrl = np.array([rot_ctrl_dict['yaw'], rot_ctrl_dict['pitch'], rot_ctrl_dict['roll']])
            # self._set_gripper_ctrl(joystick_state)

            # 如果控制量太小，不执行动作
            CTRL_THRESHOLD = 0.1
            if np.linalg.norm(pos_ctrl) < CTRL_THRESHOLD and np.linalg.norm(rot_ctrl) < CTRL_THRESHOLD:
                continue

            changed = True

            ee_xpos, ee_xmat = self.get_ee_xform() if index == 0 else self.get_ee_r_xform()
            ee_xquat = rotations.mat2quat(ee_xmat)

            # 平移控制
            move_ctrl_rate = 0.02
            mocap_xpos = ee_xpos + np.dot(ee_xmat, pos_ctrl) * move_ctrl_rate
            mocap_xpos[2] = np.max((0, mocap_xpos[2]))  # 确保在地面以上

            # 旋转控制，如果输入量小，需要记录当前姿态并在下一帧还原（保持姿态）
            rot_ctrl_rate = 0.04
            if np.linalg.norm(rot_ctrl) < CTRL_THRESHOLD:
                mocap_xquat = self.save_xquat if index == 0 else self.save_xquat_r
            else:
                rot_offset = rot_ctrl * rot_ctrl_rate
                new_xmat = joystick.calc_rotate_matrix(rot_offset[0], rot_offset[1], rot_offset[2])
                mocap_xquat = rotations.mat2quat(np.dot(ee_xmat, new_xmat))
                if index == 0:
                    self.save_xquat = mocap_xquat
                else:
                    self.save_xquat_r = mocap_xquat

            # 直接根据新的qpos位置设置控制量，类似于牵引示教
            # print(mocap_xpos, mocap_xquat)
            if index == 0:
                self.set_grasp_mocap(mocap_xpos, mocap_xquat)
            else:
                self.set_grasp_mocap_r(mocap_xpos, mocap_xquat)
            # print(index, mocap_xpos, mocap_xquat)
        
        if not changed:
            if self.record_state == RecordState.RECORD:
                self._save_record()
            return

        self.mj_forward()
        joint_qpos = self.query_joint_qpos(self.arm_joint_names)
        self.ctrl[:7] = np.array([joint_qpos[joint_name] for joint_name in self.left_arm_joint_names]).flat.copy()
        self.ctrl[7:14] = np.array([joint_qpos[joint_name] for joint_name in self.right_arm_joint_names]).flat.copy()

        # 补偿旋转控制
        # if np.linalg.norm(rot_ctrl) < CTRL_THRESHOLD:
        #     # 如果输入量小，纠正当前旋转姿态到目标姿态
        #     rot_ctrl = self._calculate_yaw_pitch_roll(ee_xquat, mocap_xquat)
        #     print("Rot_ctrl: ", rot_ctrl)

        # self._joint_rot_ctrl_compensation(rot_ctrl)

        # 将控制数据存储到record_pool中
        if self.record_state == RecordState.RECORD:
            self._save_record()

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
        # fingers_width = self.get_fingers_width().copy()


        achieved_goal = np.array([0,0,0])
        desired_goal = self.goal.copy()
        obs = np.concatenate(
                [
                    ee_position,
                    ee_velocity
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
        self.set_grasp_mocap(self.initial_grasp_site_xpos, self.initial_grasp_site_xquat)
        self.set_grasp_mocap_r(self.initial_grasp_site_xpos_r, self.initial_grasp_site_xquat_r)
        self.mj_forward()
        return True

    # custom methods
    # -----------------------------
    def reset_mocap_welds(self) -> None:
        if self.model.nmocap > 0 and self.model.neq > 0:
            eq_list = self.model.get_eq_list()
            for eq in eq_list:
                if eq['eq_type'] == self.model.mjEQ_WELD:
                    obj1_id = eq['obj1_id']
                    obj2_id = eq['obj2_id']
                    eq_data = eq['eq_data'].copy()
                    eq_data[3:10] = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
                    self.update_equality_constraints([{"obj1_id": obj1_id, "obj2_id": obj2_id, "eq_data": eq_data}])
        self.mj_forward()


    def set_grasp_mocap(self, position, orientation) -> None:
        mocap_pos_and_quat_dict = {self.mocap("rm65b_mocap"): {'pos': position, 'quat': orientation}}
        # print("Set grasp mocap: ", position, orientation)
        self.set_mocap_pos_and_quat(mocap_pos_and_quat_dict)

    def set_grasp_mocap_r(self, position, orientation) -> None:
        mocap_pos_and_quat_dict = {self.mocap("rm65b_mocap_r"): {'pos': position, 'quat': orientation}}
        # print("Set grasp mocap: ", position, orientation)
        self.set_mocap_pos_and_quat(mocap_pos_and_quat_dict)

    def set_goal_mocap(self, position, orientation) -> None:
        mocap_pos_and_quat_dict = {"goal_goal": {'pos': position, 'quat': orientation}}
        self.set_mocap_pos_and_quat(mocap_pos_and_quat_dict)

    def set_joint_neutral(self) -> None:
        # assign value to arm joints
        arm_joint_qpos_list = {}
        for name, value in zip(self.arm_joint_names, self.neutral_joint_values[0:16]):
            arm_joint_qpos_list[name] = np.array([value])
        self.set_joint_qpos(arm_joint_qpos_list)

    def _sample_goal(self) -> np.ndarray:
        # 训练reach时，任务是移动抓夹，goal以抓夹为原点采样
        goal = np.array([0, 0, 0])
        return goal


    def get_ee_xform(self) -> np.ndarray:
        pos_dict = self.query_site_pos_and_mat([self.site("ee_center_site")])
        xpos = pos_dict[self.site("ee_center_site")]['xpos'].copy()
        xmat = pos_dict[self.site("ee_center_site")]['xmat'].copy().reshape(3, 3)
        return xpos, xmat

    def get_ee_r_xform(self) -> np.ndarray:
        pos_dict = self.query_site_pos_and_mat([self.site("ee_center_site_r")])
        xpos = pos_dict[self.site("ee_center_site_r")]['xpos'].copy()
        xmat = pos_dict[self.site("ee_center_site_r")]['xmat'].copy().reshape(3, 3)
        return xpos, xmat

    def get_fingers_width(self) -> np.ndarray:
        qpos_dict = self.query_joint_qpos([self.joint("Gripper_Link11"), self.joint("Gripper_Link22")])
        finger1 = qpos_dict[self.joint("Gripper_Link11")]
        finger2 = qpos_dict[self.joint("Gripper_Link22")]
        return finger1 + finger2
    
