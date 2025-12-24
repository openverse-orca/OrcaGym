import numpy as np
from gymnasium.core import ObsType
from orca_gym.utils import rotations
from typing import Optional, Any, SupportsFloat
from gymnasium import spaces
from orca_gym.devices.xbox_joystick import XboxJoystickManager
from orca_gym.devices.keyboard import KeyboardInput, KeyboardInputSourceType
from orca_gym.adapters.robosuite.controllers.controller_factory import controller_factory
import orca_gym.adapters.robosuite.controllers.controller_config as controller_config
import orca_gym.adapters.robosuite.utils.transform_utils as transform_utils
from orca_gym.adapters.robomimic.robomimic_env import RobomimicEnv
from orca_gym.environment.orca_gym_env import RewardType
from orca_gym.utils.reward_printer import RewardPrinter

from orca_gym.log.orca_log import get_orca_logger
_logger = get_orca_logger()


class RunMode:
    """
    Enum class for control type
    Teleoperation: control the robot with a teleoperation device. Ignore the action passed to the step function.
    Policy normalized: control the robot with a policy. Use the normalized action passed to the step function.
    Policy raw: control the robot with a policy. Use the raw action passed to the step function.
    """
    TELEOPERATION = "teleoperation"
    POLICY_NORMALIZED = "policy_normalized"
    POLICY_RAW = "policy_raw"

class ControlDevice:
    """
    Enum class for control
    """
    KEYBOARD = "keyboard"
    XBOX = "xbox"
class Task:
    """
    Enum class for task
    """
    LIFT = "lift"
    PUSH = "push"
    PICK_AND_PLACE = "pick_and_place"
    

class SingleArmEnv(RobomimicEnv):
    """
    Franka single arm environment. The environment is used to train a policy to control the Franka robot arm.
    
    Task types: 
    - lift: lift the object to a target position
    - push: push the object to a target position
    - pick_and_place: pick the object and place it to a box
    
    Control types: 
    - Teleoperation: control the robot with a teleoperation device. Ignore the action passed to the step function.
    - Policy: control the robot with a policy. Use the normalized action passed to the step function.
    """
    ENV_VERSION = "1.0.0"

    def __init__(
        self,
        frame_skip: int,        
        reward_type: str,
        orcagym_addr: str,
        agent_names: list,
        time_step: float,
        run_mode: RunMode,
        task: Task,
        ctrl_device: ControlDevice,
        control_freq: int,
        sample_range: float,
        **kwargs,
    ):

        self._run_mode = run_mode
        self._task = task
        assert self._task in [Task.LIFT, Task.PUSH, Task.PICK_AND_PLACE], f"Invalid task: {self._task}"
        self._ctrl_device = ctrl_device
        self._control_freq = control_freq
        self._sample_range = sample_range
        self._reward_type = reward_type
        self._setup_reward_functions(reward_type)

        self._reward_printer = RewardPrinter()
        self._sync_render = True        # 数采需要严格同步渲染，保证生成的视频与仿真数据一致
        
        super().__init__(
            frame_skip = frame_skip,
            orcagym_addr = orcagym_addr,
            agent_names = agent_names,
            time_step = time_step,            
            **kwargs,
        )

        self._neutral_joint_values = np.array([0.00, 0.41, 0.00, -1.85, 0.00, 2.26, 0.79, 0.00, 0.00])
        self._ee_name  = self.site("ee_center_site")
        self._obj_name = self.body("object")
        self._obj_site_name = self.site("object_site")
        self._obj_joint_name = self.joint("object_joint")
        self._goal_name = self.site("goal")
        self._goal_site_name = self.site("goal_site")
        self._box_name = self.body("box")
        self._box_site_name = self.site("box_site")

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
        _logger.info(f"ctrl range:  {self._ctrl_range}")

        # action scale for normalization
        self._setup_action_range(self._ctrl_range[7:9])

        # index used to distinguish arm and gripper joints
        self._arm_joint_names = [self.joint("joint1"), self.joint("joint2"), self.joint("joint3"), self.joint("joint4"), self.joint("joint5"), self.joint("joint6"), self.joint("joint7")]
        self._gripper_joint_names = [self.joint("finger_joint1"), self.joint("finger_joint2")]

        arm_qpos_range = self.model.get_joint_qposrange(self._arm_joint_names)
        gripper_qpos_range = self.model.get_joint_qposrange(self._gripper_joint_names)
        self._setup_obs_scale(arm_qpos_range, gripper_qpos_range)
        
        self._set_init_state()
        site_dict = self.query_site_pos_and_quat([self._ee_name])
        self._initial_grasp_site_xpos = site_dict[self._ee_name]['xpos']
        self._initial_grasp_site_xquat = site_dict[self._ee_name]['xquat']
        self._reset_grasp_mocap()

        site_dict = self.query_site_pos_and_quat([self._obj_site_name])
        self._initial_obj_site_xpos = site_dict[self._obj_site_name]['xpos']
        self._initial_obj_site_xquat = site_dict[self._obj_site_name]['xquat']
        
        site_dict = self.query_site_pos_and_quat([self._goal_site_name])
        self._initial_goal_site_xpos = site_dict[self._goal_site_name]['xpos']
        self._initial_goal_site_xquat = site_dict[self._goal_site_name]['xquat']
        
        site_dict = self.query_site_pos_and_quat([self._box_site_name])
        self._initial_box_site_xpos = site_dict[self._box_site_name]['xpos']
        self._initial_box_site_xquat = site_dict[self._box_site_name]['xquat']


        if self._run_mode == RunMode.TELEOPERATION:
            if self._ctrl_device == ControlDevice.XBOX:
                self._joystick_manager = XboxJoystickManager()
                joystick_names = self._joystick_manager.get_joystick_names()
                if len(joystick_names) == 0:
                    raise ValueError("No joystick detected.")

                self._joystick = self._joystick_manager.get_joystick(joystick_names[0])
                if self._joystick is None:
                    raise ValueError("Joystick not found.")
            elif self._ctrl_device == ControlDevice.KEYBOARD:
                self._keyboard = KeyboardInput(KeyboardInputSourceType.ORCASTUDIO, orcagym_addr)


        self._controller_config = controller_config.load_config("osc_pose")
        # print("controller_config: ", self.controller_config)

        # Add to the controller dict additional relevant params:
        #   the robot name, mujoco sim, eef_name, joint_indexes, timestep (model) freq,
        #   policy (control) freq, and ndim (# joints)
        self._controller_config["robot_name"] = agent_names[0]
        self._controller_config["sim"] = self.gym
        self._controller_config["eef_name"] = self._ee_name
        # self.controller_config["eef_rot_offset"] = self.eef_rot_offset
        qpos_offsets, qvel_offsets, _ = self.query_joint_offsets(self._arm_joint_names)
        self._controller_config["joint_indexes"] = {
            "joints": self._arm_joint_names,
            "qpos": qpos_offsets,
            "qvel": qvel_offsets,
        }
        self._controller_config["actuator_range"] = self._ctrl_range
        self._controller_config["policy_freq"] = self._control_freq
        self._controller_config["ndim"] = len(self._arm_joint_names)
        self._controller_config["control_delta"] = False


        self._controller = controller_factory(self._controller_config["type"], self._controller_config)
        self._controller.update_initial_joints(self._neutral_joint_values[0:7])

        # Run generate_observation_space after initialization to ensure that the observation object's name is defined.
        self._set_obs_space()
        self._set_action_space()

    def _set_obs_space(self):
        self.observation_space = self.generate_observation_space(self._get_obs().copy())

    def _set_action_space(self):
        scaled_action_range = np.ones(self._action_range.shape, dtype=np.float32)
        self.action_space = self.generate_action_space(scaled_action_range)

    def _reset_grasp_mocap(self) -> None:
        self._saved_xpos = self._initial_grasp_site_xpos.copy()
        self._saved_xquat = self._initial_grasp_site_xquat.copy()
        
        # if self._run_mode == RunMode.TELEOPERATION:
        self._set_grasp_mocap(self._saved_xpos, self._saved_xquat)
        # else:
        #     xpos = self._initial_grasp_site_xpos + np.array([0.0, 0.0, -100])
        #     self._set_grasp_mocap(xpos, self._initial_grasp_site_xquat) # set the gripper to a position that is not in the camera view

    def get_env_version(self):
        return SingleArmEnv.ENV_VERSION

    def check_success(self):
        """
        Check if the task condition(s) is reached. Should return a dictionary
        { str: bool } with at least a "task" key for the overall task success,
        and additional optional keys corresponding to other task criteria.
        """        
        achieved_goal = self._get_achieved_goal()
        desired_goal = self._get_desired_goal()
        success = self._is_success(achieved_goal, desired_goal)
        return {"task": success}
    
    def render_callback(self, mode='human') -> None:
        if mode == "human":
            self.render()
        else:
            raise ValueError("Invalid render mode")

    def _set_init_state(self) -> None:
        # print("Set initial state")
        self._set_joint_neutral()

        self.ctrl = np.array(self._neutral_joint_values[0:9])
        self.set_ctrl(self.ctrl)
        self.mj_forward()

    def _is_success(self, achieved_goal, desired_goal) -> bool:
        success_threshold = 0.03
        return np.linalg.norm(achieved_goal - desired_goal) < success_threshold
    
    def check_success(self):
        achieved_goal = self._get_achieved_goal()
        desired_goal = self._get_desired_goal()
        success = self._is_success(achieved_goal, desired_goal)
        # print("achieved goal: ", achieved_goal, "desired goal: ", desired_goal, "success: ", success)
        return {"task": success}

    def step(self, action) -> tuple:
        if self._run_mode == RunMode.TELEOPERATION:
            ctrl, noscaled_action = self._teleoperation_action()
            scaled_action = self.normalize_action(noscaled_action, self._action_range_min, self._action_range_max)
        elif self._run_mode == RunMode.POLICY_NORMALIZED:
            scaled_action = action
            noscaled_action = self.denormalize_action(action, self._action_range_min, self._action_range_max)
            ctrl = self._playback_action(noscaled_action)
        elif self._run_mode == RunMode.POLICY_RAW:
            noscaled_action = np.clip(action, self._action_range_min, self._action_range_max)
            scaled_action = self.normalize_action(noscaled_action, self._action_range_min, self._action_range_max)
            ctrl = self._playback_action(noscaled_action)
        else:
            raise ValueError("Invalid run mode : ", self._run_mode)
        
        # print("runmode: ", self._run_mode, "no_scaled_action: ", noscaled_action, "scaled_action: ", scaled_action, "ctrl: ", ctrl)
        
        # step the simulation with original action space
        self.do_simulation(ctrl, self.frame_skip)
        obs = self._get_obs().copy()
        achieved_goal = self._get_achieved_goal()
        desired_goal = self._get_desired_goal()
        
        obj_xpos, obj_xquat = self._query_obj_pos_and_quat()
        object = np.concatenate([obj_xpos, obj_xquat]).flatten()
        goal_xpos, goal_xquat = self._query_goal_pos_and_quat()
        goal = np.concatenate([goal_xpos, goal_xquat]).flatten()

        info = {"state": self.get_state(), "action": scaled_action, "object" : object, "goal": goal}
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
    
    def _set_gripper_ctrl(self, state) -> None:
        GRIPPER_SPEED = self.realtime_step * 0.05  # 0.05 m/s
        if self._ctrl_device == ControlDevice.XBOX:
            if (state["buttons"]["B"]):
                self.ctrl[7] += GRIPPER_SPEED
                self.ctrl[8] += GRIPPER_SPEED
            elif (state["buttons"]["A"]):
                self.ctrl[7] -= GRIPPER_SPEED
                self.ctrl[8] -= GRIPPER_SPEED
        elif self._ctrl_device == ControlDevice.KEYBOARD:
            if (state["Z"]):
                self.ctrl[7] += GRIPPER_SPEED
                self.ctrl[8] += GRIPPER_SPEED
            elif (state["X"]):
                self.ctrl[7] -= GRIPPER_SPEED
                self.ctrl[8] -= GRIPPER_SPEED

        self.ctrl[7] = np.clip(self.ctrl[7], self._ctrl_range_min[7], self._ctrl_range_max[7])
        self.ctrl[8] = np.clip(self.ctrl[8], self._ctrl_range_min[8], self._ctrl_range_max[8])


    def _teleoperation_action(self) -> np.ndarray:
        mocap_xpos = self._saved_xpos
        mocap_xquat = self._saved_xquat

        mocap_xpos, mocap_xquat = self._process_controller(mocap_xpos, mocap_xquat)
        self._set_grasp_mocap(mocap_xpos, mocap_xquat)
        self._saved_xpos = mocap_xpos
        self._saved_xquat = mocap_xquat

        # 两个工具的quat不一样，这里将 qw, qx, qy, qz 转为 qx, qy, qz, qw
        mocap_axisangle = transform_utils.quat2axisangle(np.array([mocap_xquat[1], 
                                                                   mocap_xquat[2], 
                                                                   mocap_xquat[3], 
                                                                   mocap_xquat[0]]))
        # mocap_axisangle[1] = -mocap_axisangle[1]
        # 裁剪到支持的动作空间范围
        mocap_xpos = np.clip(mocap_xpos, self._action_range_min[:3], self._action_range_max[:3])
        mocap_axisangle = np.clip(mocap_axisangle, self._action_range_min[3:6], self._action_range_max[3:6])
        arm_action = np.concatenate([mocap_xpos, mocap_axisangle])
        # print("action:", action)
        self._controller.set_goal(arm_action)
        
        self.ctrl[0:7] = self._controller.run_controller()
        action = np.concatenate([arm_action, self.ctrl[7:9]]).flatten()
        
        return self.ctrl.copy(), action
    
    def _playback_action(self, action) -> np.ndarray:
        assert(len(action) == self.action_space.shape[0])
        
        self._controller.set_goal(action)
        self.ctrl[0:7] = self._controller.run_controller()      
        self.ctrl[7:9] = action[6:8]

        mocap_xpos = action[:3]
        axisangle = action[3:6]
        quat = transform_utils.axisangle2quat(axisangle)
        mocap_xquat = np.array([quat[3], quat[0], quat[1], quat[2]])
        self._set_grasp_mocap(mocap_xpos, mocap_xquat)
        
        return self.ctrl.copy()


    def _process_controller(self, mocap_xpos, mocap_xquat) -> tuple[np.ndarray, np.ndarray]:
        if self._ctrl_device == ControlDevice.XBOX:
            self._joystick_manager.update()
            pos_ctrl_dict = self._joystick.capture_joystick_pos_ctrl()
            rot_ctrl_dict = self._joystick.capture_joystick_rot_ctrl()
            self._set_gripper_ctrl(self._joystick.get_state())
        elif self._ctrl_device == ControlDevice.KEYBOARD:
            self._keyboard.update()
            pos_ctrl_dict = self.capture_keyboard_pos_ctrl()
            rot_ctrl_dict = self.capture_keyboard_rot_ctrl()
            self._set_gripper_ctrl(self._keyboard.get_state())
        pos_ctrl = np.array([pos_ctrl_dict['y'], pos_ctrl_dict['x'], pos_ctrl_dict['z']])
        rot_ctrl = np.array([rot_ctrl_dict['yaw'], rot_ctrl_dict['pitch'], rot_ctrl_dict['roll']])

        # 考虑到手柄误差，只有输入足够的控制量，才移动mocap点
        CTRL_MIN = 0.20000000
        if np.linalg.norm(pos_ctrl) < CTRL_MIN and np.linalg.norm(rot_ctrl) < CTRL_MIN:
            return mocap_xpos, mocap_xquat

        mocap_xmat = rotations.quat2mat(mocap_xquat)

        # 平移控制
        MOVE_SPEED = self.realtime_step * 0.2   # 0.2 m/s
        mocap_xpos = mocap_xpos + np.dot(mocap_xmat, pos_ctrl) * MOVE_SPEED
        mocap_xpos[2] = np.max((0, mocap_xpos[2]))  # 确保在地面以上

        # 旋转控制
        ROUTE_SPEED = self.realtime_step * np.pi / 2  # 90 degree/s    
        rot_offset = rot_ctrl * ROUTE_SPEED
        new_xmat = self.calc_rotate_matrix(rot_offset[0], rot_offset[1], rot_offset[2])
        mocap_xquat = rotations.mat2quat(np.dot(mocap_xmat, new_xmat))

        return mocap_xpos, mocap_xquat

    def capture_keyboard_rot_ctrl(self) -> dict:
        # Capture rotational control based on keyboard input
        state = self._keyboard.get_state()
        yaw = all([state["D"], state["LShift"]]) - all([state["A"], state["LShift"]])
        pitch = all([state["W"], state["LShift"]]) - all([state["S"], state["LShift"]])
        roll = state["E"] - state["Q"]
        rot_ctrl = {'yaw': yaw, 'pitch': pitch, 'roll': roll}
        return rot_ctrl

    def capture_keyboard_pos_ctrl(self) -> dict:
        # Capture positional control based on keyboard input
        state = self._keyboard.get_state()
        move_x = state["D"] - state["A"]
        move_y = state["W"] - state["S"]
        move_z = state["F"] - state["R"]
        pos_ctrl = {'x': move_x, 'y': move_y, 'z': move_z}
        return pos_ctrl

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
        ee_position = self.query_site_pos_and_quat([self._ee_name])[self._ee_name]
        ee_xvalp, ee_xvalr = self.query_site_xvalp_xvalr([self._ee_name])
        gripper_qpos = self._get_gripper_qpos()
        gripper_qvel = self._get_gripper_qvel()
        obj_xpos, obj_xquat = self._query_obj_pos_and_quat()
        goal_xpos, goal_xquat = self._query_goal_pos_and_quat()
        joint_values = self._get_arm_joint_values()
        joint_values_sin = np.sin(joint_values)
        joint_values_cos = np.cos(joint_values)
        joint_velocities = self._get_arm_joint_velocities()

        self._obs = {
            "object": np.concatenate([obj_xpos, obj_xquat], dtype=np.float32).flatten(),
            "ee_pos": ee_position["xpos"].flatten().astype(np.float32),
            "ee_quat": ee_position["xquat"].flatten().astype(np.float32),
            "ee_vel_linear": ee_xvalp[self._ee_name].flatten().astype(np.float32),
            "ee_vel_angular": ee_xvalr[self._ee_name].flatten().astype(np.float32),
            "joint_qpos": joint_values.flatten().astype(np.float32),
            "joint_qpos_sin": joint_values_sin.flatten().astype(np.float32),
            "joint_qpos_cos": joint_values_cos.flatten().astype(np.float32),
            "joint_vel": joint_velocities.flatten().astype(np.float32),
            "gripper_qpos": gripper_qpos.flatten().astype(np.float32),
            "gripper_qvel": gripper_qvel.flatten().astype(np.float32),
        }
        scaled_obs = {key : self._obs[key] * self._obs_scale[key] for key in self._obs.keys()}
        return scaled_obs
    
    def _get_achieved_goal(self) -> np.ndarray:
        obj_xpos, _ = self._query_obj_pos_and_quat()
        # print("achieved goal position: ", obj_xpos)
        return obj_xpos
    
    def _get_desired_goal(self) -> np.ndarray:
        goal_xpos, _ = self._query_goal_pos_and_quat()
        # print("desired goal position: ", goal_xpos)
        return goal_xpos

    def reset_model(self) -> tuple[dict, dict]:
        """
        Reset the environment, return observation
        """
        
        # print("Reset model")
        
        self._set_init_state()
        self._reset_grasp_mocap()

        obj_xpos, obj_xquat, goal_xpos, goal_xquat = self.sample_obj_goal(self._initial_obj_site_xpos,
                                                                          self._initial_obj_site_xquat,
                                                                          self._initial_goal_site_xpos,
                                                                          self._initial_goal_site_xquat,
                                                                          self._sample_range)
        
        self.replace_obj_goal(obj_xpos, obj_xquat, goal_xpos, goal_xquat)
        obs = self._get_obs().copy()
        return obs, {}
    
    def replace_obj_goal(self, obj_xpos, obj_xquat, goal_xpos, goal_xquat) -> None:
        self._set_obj_qpos(obj_xpos, obj_xquat)
        self._set_goal_mocap(goal_xpos, goal_xquat)
        self.mj_forward()

    # custom methods
    # -----------------------------
    def _set_grasp_mocap(self, position, orientation) -> None:
        mocap_pos_and_quat_dict = {self.mocap("panda_mocap"): {'pos': position, 'quat': orientation}}
        self.set_mocap_pos_and_quat(mocap_pos_and_quat_dict)

    def _set_obj_qpos(self, position, orientation) -> None:
        obj_qpos = {self._obj_joint_name: np.concatenate([position, orientation])}
        self.set_joint_qpos(obj_qpos)
        
    def _set_goal_mocap(self, position, orientation) -> None:
        mocap_pos_and_quat_dict = {self._goal_name: {'pos': position, 'quat': orientation}}
        self.set_mocap_pos_and_quat(mocap_pos_and_quat_dict)

    def _set_joint_neutral(self) -> None:
        # assign value to arm joints
        arm_joint_qpos = {}
        for name, value in zip(self._arm_joint_names, self._neutral_joint_values[0:7]):
            arm_joint_qpos[name] = np.array([value])
        self.set_joint_qpos(arm_joint_qpos)

        # assign value to finger joints
        gripper_joint_qpos = {}
        for name, value in zip(self._gripper_joint_names, self._neutral_joint_values[7:9]):
            gripper_joint_qpos[name] = np.array([value])
        self.set_joint_qpos(gripper_joint_qpos)

    def _query_obj_pos_and_quat(self) -> tuple:
        site_dict = self.query_site_pos_and_quat([self._obj_site_name])
        obj_xpos, obj_xquat = site_dict[self._obj_site_name]['xpos'], site_dict[self._obj_site_name]['xquat']
        return obj_xpos, obj_xquat
    
    def _query_goal_pos_and_quat(self) -> tuple:
        site_dict = self.query_site_pos_and_quat([self._goal_site_name])
        goal_xpos, goal_xquat = site_dict[self._goal_site_name]['xpos'], site_dict[self._goal_site_name]['xquat']
        return goal_xpos, goal_xquat


    def sample_obj_goal(self, init_obj_xpos, init_obj_xquat, init_goal_xpos, init_goal_xquat, sample_range) -> tuple:
        """
        随机采样一个物体位置和目标位置
        """
        # print("Sample obj and goal position, range: ", sample_range)        
        if sample_range > 0:
            contacted = True
            object_offset = sample_range
            goal_offset = sample_range
            rotate_offset = np.pi / sample_range
            while contacted:
                obj_xpos = init_obj_xpos.copy()
                obj_xquat = init_obj_xquat.copy()
                goal_xpos = init_goal_xpos.copy()
                goal_xquat = init_goal_xquat.copy()

                # 如果是push任务，则按照50%概率交换物体和目标位置，增加足够的多样性, 否则固定物体位置
                if np.random.uniform() < 0.5 and self._task == Task.PUSH:
                    obj_xpos = init_goal_xpos.copy()
                    obj_xquat = init_goal_xquat.copy()
                    goal_xpos = init_obj_xpos.copy()
                    goal_xquat = init_obj_xquat.copy()
                
                
                obj_euler = rotations.quat2euler(obj_xquat)
                obj_xpos[0] = np.random.uniform(-object_offset, object_offset) + obj_xpos[0]
                obj_xpos[1] = np.random.uniform(-object_offset, object_offset) + obj_xpos[1]
                obj_euler[2] = np.random.uniform(-rotate_offset, rotate_offset)
                obj_xquat = rotations.euler2quat(obj_euler)

                goal_euler = rotations.quat2euler(goal_xquat)
                goal_xpos[0] = np.random.uniform(-goal_offset, goal_offset) + goal_xpos[0]
                goal_xpos[1] = np.random.uniform(-goal_offset, goal_offset) + goal_xpos[1]
                goal_euler[2] = np.random.uniform(-rotate_offset, rotate_offset)
                goal_xquat = rotations.euler2quat(goal_euler)

                contacted = self._is_success(obj_xpos, goal_xpos)
        else:
            # 固定采样，物体位置固定不变
            obj_xpos = init_obj_xpos.copy()
            obj_xquat = init_obj_xquat.copy()
            goal_xpos = init_goal_xpos.copy()
            goal_xquat = init_goal_xquat.copy()
            
        # 任务不同，目标位置也不同
        if self._task == Task.LIFT:
            goal_xpos = obj_xpos.copy()
            goal_xpos[2] += 0.1
            goal_xquat = obj_xquat.copy()
        elif self._task == Task.PICK_AND_PLACE:
            goal_xpos = self._initial_box_site_xpos.copy()
            goal_xquat = self._initial_box_site_xquat.copy()

        return obj_xpos, obj_xquat, goal_xpos, goal_xquat

    def _get_ee_xform(self) -> np.ndarray:
        pos_dict = self.query_site_pos_and_mat([self.site("ee_center_site")])
        xpos = pos_dict[self.site("ee_center_site")]['xpos'].copy()
        xmat = pos_dict[self.site("ee_center_site")]['xmat'].copy().reshape(3, 3)
        return xpos, xmat

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
    
    def _get_arm_joint_values(self) -> np.ndarray:
        qpos_dict = self.query_joint_qpos(self._arm_joint_names)
        return np.array([qpos_dict[joint_name] for joint_name in self._arm_joint_names]).flatten()
    
    def _get_arm_joint_velocities(self) -> np.ndarray:
        qvel_dict = self.query_joint_qvel(self._arm_joint_names)
        return np.array([qvel_dict[joint_name] for joint_name in self._arm_joint_names]).flatten()

    def get_observation(self, obs=None):
        if obs is not None:
            return obs
        else:
            return self._get_obs().copy()

    def _print_reward(self, message : str, reward : Optional[float] = 0, coeff : Optional[float] = 1) -> None:
        if self._reward_printer is not None and self._reward_type == RewardType.DENSE:
            self._reward_printer.print_reward(message, reward, coeff)
        
    def _compute_reward_obj_goal_distance(self) -> float:
        tracking_sigma = 0.025  # 奖励的衰减因子
        
        obj_xpos = self._obs["object"][:3]
        goal_xpos = self._obs["goal"][:3]
        distance_squared = np.sum((obj_xpos - goal_xpos)**2)
        distance_squared = np.clip(distance_squared, 0.01, distance_squared)   # 当距离足够近的时候，避免从错误的方向靠近
        
        # 计算指数衰减奖励
        reward = np.exp(-distance_squared / tracking_sigma)
        return reward
    
    def _compute_reward_obj_grasp_distance(self) -> float:
        tracking_sigma = 0.025
        
        obj_xpos = self._obs["object"][:3]
        ee_xpos = self._obs["ee_pos"]
        distance_squared = np.sum((obj_xpos - ee_xpos)**2)
        
        # 计算指数衰减奖励
        reward = np.exp(-distance_squared / tracking_sigma)
        return reward
    
    def _compute_reward_success(self) -> float:
        achieved_goal = self._achieved_goal
        desired_goal = self._desired_goal
        return 1 if self._is_success(achieved_goal, desired_goal) else 0

    def _setup_reward_functions(self, reward_type : RewardType) -> None:
        if reward_type == RewardType.SPARSE:
            self._reward_functions = [
                {"function": self._compute_reward_success, "coeff": 1.0}
            ]
        elif reward_type == RewardType.DENSE:
            self._reward_functions = [
                {"function": self._compute_reward_obj_goal_distance, "coeff": 0.1},
                {"function": self._compute_reward_obj_grasp_distance, "coeff": 0.1},
                {"function": self._compute_reward_success, "coeff": 1.0},
            ]
        else:
            raise ValueError("Invalid reward type: ", reward_type)
        
    def _compute_reward(self, achieved_goal, desired_goal, info) -> float:
        total_reward = 0.0
        self._achieved_goal = achieved_goal
        self._desired_goal = desired_goal

        for reward_function in self._reward_functions:
            if reward_function["coeff"] == 0:
                continue
            else:    
                reward = reward_function["function"]() * reward_function["coeff"]
                total_reward += reward
                self._print_reward(reward_function["function"].__name__, reward, reward_function["coeff"])

        self._print_reward("Total reward: ", total_reward)
        return total_reward
        
    
    def _setup_action_range(self, finger_range) -> None:
        # 支持的动作范围空间，遥操作时不能超过这个范围
        # 模型接收的是 [-1, 1] 的动作空间，这里是真实的物理空间，需要进行归一化
        # action range: [x, y, z, yaw, pitch, roll, finger1, finger2]
        self._action_range = np.array([[-2.0, 2.0], [-2.0, 2.0], [-2.0, 2.0],
                                 [-np.pi, np.pi], [-np.pi, np.pi], [-np.pi, np.pi],
                                 finger_range[0], finger_range[1]], dtype=np.float32)
        self._action_range_min = self._action_range[:, 0]
        self._action_range_max = self._action_range[:, 1]

    def _setup_obs_scale(self, arm_qpos_range, gripper_qpos_range) -> None:
        # 观测空间范围
        ee_xpos_scale = np.array([max(abs(act_range[0]), abs(act_range[1])) for act_range in self._action_range[:3]], dtype=np.float32)   # 末端位置范围
        ee_xquat_scale = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)   # 裁剪到 -pi, pi 的单位四元数范围
        max_ee_linear_vel = 2.0  # 末端线速度范围 m/s
        max_ee_angular_vel = np.pi # 末端角速度范围 rad/s

        arm_qpos_scale = np.array([max(abs(qpos_range[0]), abs(qpos_range[1])) for qpos_range in arm_qpos_range], dtype=np.float32)  # 关节角度范围
        max_arm_joint_vel = np.pi  # 关节角速度范围 rad/s
        
        gripper_qpos_scale = np.array([max(abs(qpos_range[0]), abs(qpos_range[1])) for qpos_range in gripper_qpos_range], dtype=np.float32)  # 夹爪角度范围
        max_gripper_vel = 1.0  # 夹爪角速度范围 m/s
                
        self._obs_scale = {
            "object": 1.0 / np.concatenate([ee_xpos_scale, ee_xquat_scale], dtype=np.float32).flatten(),  # 物体位置和四元数，保持和末端位置一致
            "ee_pos": 1.0 / ee_xpos_scale,
            "ee_quat": 1.0 / ee_xquat_scale,
            "ee_vel_linear": np.ones(3, dtype=np.float32) / max_ee_linear_vel,
            "ee_vel_angular": np.ones(3, dtype=np.float32) / max_ee_angular_vel,
            "joint_qpos": 1.0 / arm_qpos_scale,
            "joint_qpos_sin": np.ones(len(arm_qpos_scale), dtype=np.float32),
            "joint_qpos_cos": np.ones(len(arm_qpos_scale), dtype=np.float32),
            "joint_vel": np.ones(len(arm_qpos_scale), dtype=np.float32) / max_arm_joint_vel,
            "gripper_qpos": 1.0 / gripper_qpos_scale,
            "gripper_qvel": np.ones(len(gripper_qpos_scale), dtype=np.float32) / max_gripper_vel,
        }
        
        # print("obs scale: ", self._obs_scale)