import numpy as np
from gymnasium.core import ObsType
from orca_gym.utils import rotations
from typing import Optional, Any, SupportsFloat
from gymnasium import spaces
from orca_gym.devices.xbox_joystick import XboxJoystickManager
from orca_gym.devices.keyboard import KeyboardInput
from orca_gym.robosuite.controllers.controller_factory import controller_factory
import orca_gym.robosuite.controllers.controller_config as controller_config
import orca_gym.robosuite.utils.transform_utils as transform_utils
from orca_gym.robomimic.robomimic_env import RobomimicEnv
from orca_gym.robomimic.robomimic_env import RunMode, ControlDevice
from orca_gym.environment.orca_gym_env import RewardType
from orca_gym.utils.reward_printer import RewardPrinter

class FrankaEnv(RobomimicEnv):
    """
    通过遥操作控制franka机械臂
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
        ctrl_device: ControlDevice,
        control_freq: int,
        **kwargs,
    ):

        self._run_mode = run_mode
        self._ctrl_device = ctrl_device
        self._control_freq = control_freq
        self._reward_type = reward_type
        self._setup_reward_functions()

        self._reward_printer = RewardPrinter()
        
        super().__init__(
            frame_skip = frame_skip,
            orcagym_addr = orcagym_addr,
            agent_names = agent_names,
            time_step = time_step,            
            **kwargs,
        )

        self._neutral_joint_values = np.array([0.00, 0.41, 0.00, -1.85, 0.00, 2.26, 0.79, 0.00, 0.00])
        self._ee_name  = self.site("ee_center_site")
        self._obj_name = self.body("item")
        self._obj_site_name = self.site("item_site")
        self._obj_joint_name = self.joint("item_joint")
        self._goal_name = self.site("goal")
        self._goal_site_name = self.site("goal_site")
        self._goal_joint_name = self.joint("goal_joint")

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

        # action range: [x, y, z, yaw, pitch, roll, finger1, finger2]
        self._action_range = np.array([[-3.0, 3.0], [-3.0, 3.0], [-3.0, 3.0],
                                 [-2 * np.pi, 2 * np.pi], [-2 * np.pi, 2 * np.pi], [-2 * np.pi, 2 * np.pi],
                                 self._ctrl_range[7], self._ctrl_range[8]], dtype=np.float32)
        self._action_range_min = self._action_range[:, 0]
        self._action_range_max = self._action_range[:, 1]

        # index used to distinguish arm and gripper joints
        self._arm_joint_names = [self.joint("joint1"), self.joint("joint2"), self.joint("joint3"), self.joint("joint4"), self.joint("joint5"), self.joint("joint6"), self.joint("joint7")]
        self._gripper_joint_names = [self.joint("finger_joint1"), self.joint("finger_joint2")]

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
                self._keyboard = KeyboardInput()


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
        self.action_space = self.generate_action_space(self._action_range)

    def _reset_grasp_mocap(self) -> None:
        self._saved_xpos = self._initial_grasp_site_xpos.copy()
        self._saved_xquat = self._initial_grasp_site_xquat.copy()
        
        if self._run_mode == RunMode.TELEOPERATION:
            self._set_grasp_mocap(self._saved_xpos, self._saved_xquat)
        else:
            xpos = self._initial_grasp_site_xpos + np.array([0.0, 0.0, -100])
            self._set_grasp_mocap(xpos, self._initial_grasp_site_xquat) # set the gripper to a position that is not in the camera view


    def get_env_version(self):
        return FrankaEnv.ENV_VERSION

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

    def step(self, action) -> tuple:
        if self._run_mode == RunMode.TELEOPERATION:
            ctrl, noscaled_action = self._teleoperation_action()
            scaled_action = self.normalize_action(noscaled_action, self._action_range_min, self._action_range_max)
        elif self._run_mode in [RunMode.PLAYBACK, RunMode.IMITATION, RunMode.ROLLOUT, RunMode.AUGMENTATION]:
            scaled_action = action
            noscaled_action = self.denormalize_action(action, self._action_range_min, self._action_range_max)
            ctrl = self._playback_action(noscaled_action)
        else:
            raise ValueError("Invalid run mode : ", self._run_mode)
        
        # print("runmode: ", self._run_mode, "no_scaled_action: ", noscaled_action, "scaled_action: ", scaled_action, "ctrl: ", ctrl)
        
        # step the simulation with original action space
        self.do_simulation(ctrl, self.frame_skip)
        obs = self._get_obs().copy()
        achieved_goal = self._get_achieved_goal()
        desired_goal = self._get_desired_goal()


        info = {"state": self.get_state(), "action": scaled_action}
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
            "object": np.concatenate([obj_xpos, obj_xquat], dtype=np.float32),
            "goal": np.concatenate([goal_xpos, goal_xquat], dtype=np.float32),
            "ee_pos": ee_position["xpos"],
            "ee_quat": ee_position["xquat"],
            "ee_vel_linear": ee_xvalp[self._ee_name],
            "ee_vel_angular": ee_xvalr[self._ee_name],
            "joint_qpos": joint_values,
            "joint_qpos_sin": joint_values_sin,
            "joint_qpos_cos": joint_values_cos,
            "joint_vel": joint_velocities,
            "gripper_qpos": gripper_qpos,
            "gripper_qvel": gripper_qvel,
        }
        return self._obs
    
    def _get_achieved_goal(self) -> np.ndarray:
        obj_xpos, _ = self._query_obj_pos_and_quat()
        # print("achieved goal position: ", obj_xpos)
        return obj_xpos
    
    def _get_desired_goal(self) -> np.ndarray:
        goal_xpos, _ = self._query_goal_pos_and_quat()
        # print("desired goal position: ", goal_xpos)
        return goal_xpos

    def reset_model(self) -> dict:
        """
        Reset the environment, return observation
        """
        
        # print("Reset model")
        
        self._set_init_state()
        self._reset_grasp_mocap()

        obj_xpos, obj_xquat, goal_xpos, goal_xquat = self._sample_obj_goal()
        
        self._set_obj_qpos(obj_xpos, obj_xquat)
        self._set_goal_qpos(goal_xpos, goal_xquat)

        self.mj_forward()
        obs = self._get_obs().copy()
        return obs, {}
    
    def replace_obj_goal(self, obj_xpos, obj_xquat, goal_xpos, goal_xquat) -> None:
        self._set_obj_qpos(obj_xpos, obj_xquat)
        self._set_goal_qpos(goal_xpos, goal_xquat)
        self.mj_forward()

    # custom methods
    # -----------------------------
    def _set_grasp_mocap(self, position, orientation) -> None:
        mocap_pos_and_quat_dict = {self.mocap("panda_mocap"): {'pos': position, 'quat': orientation}}
        self.set_mocap_pos_and_quat(mocap_pos_and_quat_dict)

    def _set_obj_qpos(self, position, orientation) -> None:
        obj_qpos = {self._obj_joint_name: np.concatenate([position, orientation])}
        self.set_joint_qpos(obj_qpos)
        
    def _set_goal_qpos(self, position, orientation) -> None:
        goal_qpos = {self._goal_joint_name: np.concatenate([position, orientation])}
        self.set_joint_qpos(goal_qpos)

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


    def _sample_obj_goal(self) -> None:
        """
        随机采样一个物体位置和目标位置
        """        
        contacted = True
        object_offset = 0.2
        goal_offset = 0.2
        while contacted:
            # 50%概率交换物体和目标位置
            if np.random.uniform() < 0.5:
                obj_xpos = self._initial_obj_site_xpos.copy()
                obj_xquat = self._initial_obj_site_xquat.copy()
                goal_xpos = self._initial_goal_site_xpos.copy()
                goal_xquat = self._initial_goal_site_xquat.copy()
            else:
                obj_xpos = self._initial_goal_site_xpos.copy()
                obj_xquat = self._initial_goal_site_xquat.copy()
                goal_xpos = self._initial_obj_site_xpos.copy()
                goal_xquat = self._initial_obj_site_xquat.copy()
            
            
            obj_euler = rotations.quat2euler(obj_xquat)
            obj_xpos[0] = np.random.uniform(-object_offset, object_offset) + obj_xpos[0]
            obj_xpos[1] = np.random.uniform(-object_offset, object_offset) + obj_xpos[1]
            obj_euler[2] = np.random.uniform(-np.pi, np.pi)
            obj_xquat = rotations.euler2quat(obj_euler)

            goal_euler = rotations.quat2euler(goal_xquat)
            goal_xpos[0] = np.random.uniform(-goal_offset, goal_offset) + goal_xpos[0]
            goal_xpos[1] = np.random.uniform(-goal_offset, goal_offset) + goal_xpos[1]
            goal_euler[2] = np.random.uniform(-np.pi, np.pi)
            goal_xquat = rotations.euler2quat(goal_euler)

            contacted = np.linalg.norm(obj_xpos - goal_xpos) < 0.2

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
        if self._reward_printer is not None and self._run_mode == RunMode.TELEOPERATION:
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

    def _setup_reward_functions(self) -> None:
        self._reward_functions = [
            {"function": self._compute_reward_obj_goal_distance, "coeff": 0.1},
            {"function": self._compute_reward_obj_grasp_distance, "coeff": 0.1},
            {"function": self._compute_reward_success, "coeff": 10},
        ]
        
    def _compute_dense_reward(self, achieved_goal, desired_goal,) -> float:
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

    def _compute_reward(self, achieved_goal, desired_goal, info) -> float:
        if self._reward_type == RewardType.SPARSE:
            return 1 if self._is_success(achieved_goal, desired_goal) else 0
        elif self._reward_type == RewardType.DENSE:
            return self._compute_dense_reward(achieved_goal, desired_goal)
        else:
            raise ValueError("Invalid reward type")
        
    
        