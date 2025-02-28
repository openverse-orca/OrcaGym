import numpy as np
from gymnasium.core import ObsType
from orca_gym.utils import rotations
from typing import Optional, Any, SupportsFloat
from gymnasium import spaces
from orca_gym.devices.xbox_joystick import XboxJoystickManager
from orca_gym.devices.keyboard import KeyboardInput
import orca_gym.robosuite.utils.transform_utils as transform_utils
from orca_gym.environment.orca_gym_env import RewardType
from orca_gym.environment.orca_gym_local_env import OrcaGymLocalEnv

class ControlDevice:
    """
    Enum class for control
    """
    KEYBOARD = "keyboard"
    XBOX = "xbox"

    
class GripperEnv(OrcaGymLocalEnv):
    """
    Gripper Control for Scene Debugging
    
    Control by Xbox Joystick or Keyboard
    """

    def __init__(
        self,
        frame_skip: int,        
        orcagym_addr: str,
        agent_names: list,
        time_step: float,
        ctrl_device: ControlDevice,
        **kwargs,
    ):

        self._ctrl_device = ctrl_device
        
        super().__init__(
            frame_skip = frame_skip,
            orcagym_addr = orcagym_addr,
            agent_names = agent_names,
            time_step = time_step,            
            **kwargs,
        )

        self._ee_name  = self.site("ee_center_site")

        # Three auxiliary variables to understand the component of the xml document but will not be used
        # number of actuators/controls: 7 arm joints and 2 gripper joints
        self.nu = self.model.nu
        # 16 generalized coordinates: 9 (arm + gripper) + 7 (object free joint: 3 position and 4 quaternion coordinates)
        self.nq = self.model.nq
        # 9 arm joints and 6 free joints
        self.nv = self.model.nv

        # control range
        self._ctrl_range = self.model.get_actuator_ctrlrange()
        self._ctrl_range_min = self._ctrl_range[:, 0]
        self._ctrl_range_max = self._ctrl_range[:, 1]
        print("ctrl range: ", self._ctrl_range)

        # action scale for normalization
        self._setup_action_range(self._ctrl_range[:2])

        # index used to distinguish arm and gripper joints
        self._arm_joint_names = [self.joint("joint1"), self.joint("joint2"), self.joint("joint3"), self.joint("joint4"), self.joint("joint5"), self.joint("joint6"), self.joint("joint7")]
        self._gripper_joint_names = [self.joint("finger_joint1"), self.joint("finger_joint2")]

        gripper_qpos_range = self.model.get_joint_qposrange(self._gripper_joint_names)

        site_dict = self.query_site_pos_and_quat([self._ee_name])
        self._initial_grasp_site_xpos = site_dict[self._ee_name]['xpos']
        self._initial_grasp_site_xquat = site_dict[self._ee_name]['xquat']



        if self._ctrl_device == ControlDevice.XBOX:
            self._joystick_manager = XboxJoystickManager()
            joystick_names = self._joystick_manager.get_joystick_names()
            if len(joystick_names) == 0:
                raise ValueError("No joystick detected.")

            self._joystick = self._joystick_manager.get_joystick(joystick_names[0])
            if self._joystick is None:
                raise ValueError("Joystick not found.")
        elif self._ctrl_device == ControlDevice.KEYBOARD:
            self._keyboard = KeyboardInput()

        self._set_obs_space()
        self._set_action_space()

    def _set_obs_space(self):
        self.observation_space = self.generate_observation_space(self._get_obs().copy())

    def _set_action_space(self):
        scaled_action_range = np.ones(self._action_range.shape, dtype=np.float32)
        self.action_space = self.generate_action_space(scaled_action_range)

    def _setup_action_range(self, finger_range) -> None:
        # 支持的动作范围空间，遥操作时不能超过这个范围
        # 模型接收的是 [-1, 1] 的动作空间，这里是真实的物理空间，需要进行归一化
        # action range: [x, y, z, yaw, pitch, roll, finger1, finger2]
        self._action_range = np.array([finger_range[0], finger_range[1]], dtype=np.float32)
        self._action_range_min = self._action_range[:, 0]
        self._action_range_max = self._action_range[:, 1]
    
    def render_callback(self, mode='human') -> None:
        if mode == "human":
            self.render()
        else:
            raise ValueError("Invalid render mode")

    def step(self, action) -> tuple:

        ctrl = self._teleoperation_action()

        # print("runmode: ", self._run_mode, "no_scaled_action: ", noscaled_action, "scaled_action: ", scaled_action, "ctrl: ", ctrl)
        
        # step the simulation with original action space
        self.do_simulation(ctrl, self.frame_skip)
        obs = self._get_obs().copy()

        info = {}
        terminated = False
        truncated = False
        reward = 0.0

        return obs, reward, terminated, truncated, info
    
    
    def _set_gripper_ctrl(self, state) -> None:
        GRIPPER_SPEED = self.realtime_step * 0.05  # 0.05 m/s
        if self._ctrl_device == ControlDevice.XBOX:
            if (state["buttons"]["B"]):
                self.ctrl[0] += GRIPPER_SPEED
                self.ctrl[1] += GRIPPER_SPEED
            elif (state["buttons"]["A"]):
                self.ctrl[0] -= GRIPPER_SPEED
                self.ctrl[1] -= GRIPPER_SPEED
        elif self._ctrl_device == ControlDevice.KEYBOARD:
            if (state["Z"]):
                self.ctrl[0] += GRIPPER_SPEED
                self.ctrl[1] += GRIPPER_SPEED
            elif (state["X"]):
                self.ctrl[0] -= GRIPPER_SPEED
                self.ctrl[1] -= GRIPPER_SPEED

        self.ctrl[0] = np.clip(self.ctrl[0], self._ctrl_range_min[0], self._ctrl_range_max[0])
        self.ctrl[1] = np.clip(self.ctrl[1], self._ctrl_range_min[1], self._ctrl_range_max[1])


    def _teleoperation_action(self) -> np.ndarray:
        mocap_xpos = self._saved_xpos
        mocap_xquat = self._saved_xquat

        mocap_xpos, mocap_xquat = self._process_controller(mocap_xpos, mocap_xquat)
        self._set_grasp_mocap(mocap_xpos, mocap_xquat)
        self._saved_xpos = mocap_xpos
        self._saved_xquat = mocap_xquat
        
        return self.ctrl.copy()



    def _process_controller(self, mocap_xpos, mocap_xquat) -> tuple[np.ndarray, np.ndarray]:
        if self._ctrl_device == ControlDevice.XBOX:
            self._joystick_manager.update()
            pos_ctrl_dict = self._joystick.capture_joystick_pos_ctrl()
            rot_ctrl_dict = self._joystick.capture_joystick_rot_ctrl()
            
            self._set_gripper_ctrl(self._joystick.get_state())
        elif self._ctrl_device == ControlDevice.KEYBOARD:
            self._keyboard.update()
            pos_ctrl_dict = self._keyboard.capture_keyboard_pos_ctrl()
            rot_ctrl_dict = self._keyboard.capture_keyboard_rot_ctrl()
            self._set_gripper_ctrl(self._keyboard.get_state())
        pos_ctrl = np.array([pos_ctrl_dict['y'], pos_ctrl_dict['x'], pos_ctrl_dict['z'], ])
        rot_ctrl = np.array([-rot_ctrl_dict['yaw'], -rot_ctrl_dict['pitch'], rot_ctrl_dict['roll']])

        # 考虑到手柄误差，只有输入足够的控制量，才移动mocap点
        CTRL_MIN = 0.20000000
        if np.linalg.norm(pos_ctrl) < CTRL_MIN and np.linalg.norm(rot_ctrl) < CTRL_MIN:
            return mocap_xpos, mocap_xquat

        mocap_xmat = rotations.quat2mat(mocap_xquat)

        # 平移控制
        MOVE_SPEED = self.realtime_step * 0.5   # 0.5 m/s
        mocap_xpos = mocap_xpos + np.dot(mocap_xmat, pos_ctrl) * MOVE_SPEED
        mocap_xpos[2] = np.max((0, mocap_xpos[2]))  # 确保在地面以上

        # 旋转控制
        ROUTE_SPEED = self.realtime_step * np.pi / 4  # 45 degree/s    
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

    def _get_gripper_qpos(self) -> np.ndarray:
        qpos_dict = self.query_joint_qpos([self.joint("finger_joint1"), self.joint("finger_joint2")])
        finger1 = qpos_dict[self.joint("finger_joint1")]
        finger2 = qpos_dict[self.joint("finger_joint2")]
        return np.concatenate([finger1, finger2], dtype=np.float32)
    
    def _get_gripper_qvel(self) -> np.ndarray:
        qvel_dict = self.query_joint_qvel([self.joint("finger_joint1"), self.joint("finger_joint2")])
        finger1 = qvel_dict[self.joint("finger_joint1")]
        finger2 = qvel_dict[self.joint("finger_joint2")]
        return np.concatenate([finger1, finger2], dtype=np.float32)

    def _get_obs(self) -> dict:
        ee_position = self.query_site_pos_and_quat([self._ee_name])[self._ee_name]
        ee_xvalp, ee_xvalr = self.query_site_xvalp_xvalr([self._ee_name])
        gripper_qpos = self._get_gripper_qpos()
        gripper_qvel = self._get_gripper_qvel()     
           
        obs = {
            "ee_pos": ee_position["xpos"].flatten().astype(np.float32),
            "ee_quat": ee_position["xquat"].flatten().astype(np.float32),
            "ee_vel_linear": ee_xvalp[self._ee_name].flatten().astype(np.float32),
            "ee_vel_angular": ee_xvalr[self._ee_name].flatten().astype(np.float32),
            "gripper_qpos": gripper_qpos.flatten().astype(np.float32),
            "gripper_qvel": gripper_qvel.flatten().astype(np.float32),
        }
        return obs


    def reset_model(self) -> dict:
        """
        Reset the environment, return observation
        """
        
        # print("Reset model")
        self._saved_xpos = self._initial_grasp_site_xpos.copy()
        self._saved_xquat = self._initial_grasp_site_xquat.copy()
        
        self.ctrl = np.zeros(self.nu, dtype=np.float32)

        obs = self._get_obs().copy()
        return obs, {}
    

    # custom methods
    # -----------------------------
    def _set_grasp_mocap(self, position, orientation) -> None:
        mocap_pos_and_quat_dict = {self.mocap("panda_mocap"): {'pos': position, 'quat': orientation}}
        self.set_mocap_pos_and_quat(mocap_pos_and_quat_dict)

        

    def get_observation(self, obs=None):
        if obs is not None:
            return obs
        else:
            return self._get_obs().copy()
