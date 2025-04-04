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
from orca_gym.task.abstract_task import AbstractTask, TaskType
import time
import random

from orca_gym.environment.orca_gym_env import RewardType
from orca_gym.utils.reward_printer import RewardPrinter

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
    VR = "vr"

class OpenLoongTask(AbstractTask):
    def __init__(self, task_type: str, config: dict):

        super().__init__(task_type, config)

    def get_task(self, env: RobomimicEnv):
        """ 从object,goal中选择数量最少的
            在生成一个随机整数x，在挑选x个object, goal
        """
        self.task_dict.clear()
        self.random_objs_and_goals(env, bounds=0.1)

        object_len = len(self.object_bodys)
        goal_len = len(self.goal_bodys)
        min_len = min(object_len, goal_len)
        if min_len == 0:
            return self.task_dict
        
        random_x = random.randint(1, min_len)

        ind_list = random.sample(range(object_len), random_x)
        for ind in ind_list:
            self.task_dict[self.object_bodys[ind]] = self.goal_bodys[ind]

        return self.task_dict

    def is_success(self, env: RobomimicEnv):
        pass

    def get_language_instruction(self) -> str:
        if len(self.task_dict) == 0:
            return "\033[91m No task \033[0m"

        object_str = "object: "
        for key in self.task_dict.keys():
            object_str += key + " "
        
        goal_str = "goal: "
        for value in self.task_dict.values():
            goal_str += value + " "
        
        language_instruction = f"level: {self.level_name}\ntask: {self.prompt}"
        language_instruction += f" \033[91m{object_str}\033[0m to \033[91m{goal_str}\033[0m"                               

        return language_instruction

class TaskStatus:
    """
    Enum class for task status
    """
    NOT_STARTED = "not_started"
    GET_READY = "get_ready"
    BEGIN = "begin"
    SUCCESS = "success"
    FAILURE = "failure"
    END = "end"

class OpenLoongEnv(RobomimicEnv):
    """
    OpenLoong Humandroid environment for manipulation tasks.
    
    Task types: 
    - pick_and_place: pick the object and place it to a box
    
    Control types: 
    - Teleoperation: control the robot with a teleoperation device. Ignore the action passed to the step function.
    - Policy Normalized: control the robot with a policy. Use the normalized action passed to the step function.
    - Policy Raw: control the robot with a policy. Use the raw action passed to the step function.
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
        task: str,
        ctrl_device: ControlDevice,
        control_freq: int,
        sample_range: float,
        **kwargs,
    ):

        self._run_mode = run_mode

        task_config_dict = kwargs["task_config_dict"]
        self._task = OpenLoongTask(task, task_config_dict)

        assert self._task.prompt in [TaskType.PICK_AND_PLACE], f"Invalid task: {self._task}"
        self._ctrl_device = ctrl_device
        self._control_freq = control_freq
        self._sample_range = sample_range
        self._reward_type = reward_type
        self._setup_reward_functions(reward_type)

        self._reward_printer = RewardPrinter()
        
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
        self.gym.opt.ccd_iterations = 100
        self.gym.opt.sdf_iterations = 50
        self.gym.set_opt_config()

        self._base_body_name = [self.body("base_link")]
        self._base_body_xpos, _, self._base_body_xquat = self.get_body_xpos_xmat_xquat(self._base_body_name)
        print("base_body_xpos: ", self._base_body_xpos)
        print("base_body_xquat: ", self._base_body_xquat)

        self._neck_joint_names = [self.joint("J_head_yaw"), self.joint("J_head_pitch")]
        self._neck_actuator_names = [self.actuator("M_head_yaw"), self.actuator("M_head_pitch")]
        self._neck_actuator_id = [self.model.actuator_name2id(actuator_name) for actuator_name in self._neck_actuator_names]
        self._neck_neutral_joint_values = np.array([0, -0.7854])
        self._neck_ctrl_values = {"yaw": 0.0, "pitch": -0.7854}

        # index used to distinguish arm and gripper joints
        self._r_arm_joint_names = [self.joint("J_arm_r_01"), self.joint("J_arm_r_02"), 
                                 self.joint("J_arm_r_03"), self.joint("J_arm_r_04"), 
                                 self.joint("J_arm_r_05"), self.joint("J_arm_r_06"), self.joint("J_arm_r_07")]
        self._r_arm_moto_names = [self.actuator("M_arm_r_01"), self.actuator("M_arm_r_02"),
                                self.actuator("M_arm_r_03"),self.actuator("M_arm_r_04"),
                                self.actuator("M_arm_r_05"),self.actuator("M_arm_r_06"),self.actuator("M_arm_r_07")]
        self._r_arm_actuator_id = [self.model.actuator_name2id(actuator_name) for actuator_name in self._r_arm_moto_names]
        self._r_neutral_joint_values = np.array([0.905, -0.735, -2.733, 1.405, -1.191, 0.012, -0.517])
        
        self._r_hand_actuator_names = [self.actuator("r_fingers_actuator")]

        self._r_hand_actuator_id = [self.model.actuator_name2id(actuator_name) for actuator_name in self._r_hand_actuator_names]
        self._r_hand_body_names = [self.body("r_left_pad"), self.body("r_right_pad")]
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
        self._l_hand_actuator_names = [self.actuator("l_fingers_actuator")]
        
        self._l_hand_actuator_id = [self.model.actuator_name2id(actuator_name) for actuator_name in self._l_hand_actuator_names]        
        self._l_hand_body_names = [self.body("l_left_pad"), self.body("l_right_pad")]
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

        self._setup_action_range()
        arm_qpos_range_l = self.model.get_joint_qposrange(self._l_arm_joint_names)
        arm_qpos_range_r = self.model.get_joint_qposrange(self._r_arm_joint_names)
        self._setup_obs_scale(arm_qpos_range_l, arm_qpos_range_r)

        NECK_NAME  = self.site("neck_center_site")
        site_dict = self.query_site_pos_and_quat([NECK_NAME])
        self._initial_neck_site_xpos = site_dict[NECK_NAME]['xpos']
        self._initial_neck_site_xquat = site_dict[NECK_NAME]['xquat']

        self.set_neck_mocap(self._initial_neck_site_xpos, self._initial_neck_site_xquat)
        self._mocap_neck_xpos, self._mocap_neck_xquat = self._initial_neck_site_xpos, self._initial_neck_site_xquat
        self._neck_angle_x, self._neck_angle_y = 0, 0

        self._ee_site_l  = self.site("ee_center_site")
        site_dict = self.query_site_pos_and_quat([self._ee_site_l])
        self._initial_grasp_site_xpos = site_dict[self._ee_site_l]['xpos']
        self._initial_grasp_site_xquat = site_dict[self._ee_site_l]['xquat']
        self._grasp_value_l = 0.0

        self.set_grasp_mocap(self._initial_grasp_site_xpos, self._initial_grasp_site_xquat)

        self._ee_site_r  = self.site("ee_center_site_r")
        site_dict = self.query_site_pos_and_quat([self._ee_site_r])
        self._initial_grasp_site_xpos_r = site_dict[self._ee_site_r]['xpos']
        self._initial_grasp_site_xquat_r = site_dict[self._ee_site_r]['xquat']
        self._grasp_value_r = 0.0

        self.set_grasp_mocap_r(self._initial_grasp_site_xpos_r, self._initial_grasp_site_xquat_r)
        
        if self._run_mode == RunMode.TELEOPERATION:
            if self._ctrl_device == ControlDevice.VR:
                self._pico_joystick = PicoJoystick()
            else:
                raise ValueError("Invalid control device: ", self._ctrl_device)

        # -----------------------------
        # Neck controller
        self._neck_controller_config = controller_config.load_config("osc_pose")
        # print("controller_config: ", self.controller_config)

        # Add to the controller dict additional relevant params:
        #   the robot name, mujoco sim, eef_name, joint_indexes, timestep (model) freq,
        #   policy (control) freq, and ndim (# joints)
        self._neck_controller_config["robot_name"] = agent_names[0] if len(agent_names) > 0 else "robot"
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
        self._neck_controller_config["policy_freq"] = self._control_freq
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
        self._r_controller_config["robot_name"] = agent_names[0] if len(agent_names) > 0 else "robot"
        self._r_controller_config["sim"] = self.gym
        self._r_controller_config["eef_name"] = self._ee_site_r
        # self.controller_config["eef_rot_offset"] = self.eef_rot_offset
        qpos_offsets, qvel_offsets, _ = self.query_joint_offsets(self._r_arm_joint_names)
        self._r_controller_config["joint_indexes"] = {
            "joints": self._r_arm_joint_names,
            "qpos": qpos_offsets,
            "qvel": qvel_offsets,
        }
        self._r_controller_config["actuator_range"] = r_ctrl_range
        self._r_controller_config["policy_freq"] = self._control_freq
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
        self._l_controller_config["robot_name"] = agent_names[0] if len(agent_names) > 0 else "robot"
        self._l_controller_config["sim"] = self.gym
        self._l_controller_config["eef_name"] = self._ee_site_l
        # self.controller_config["eef_rot_offset"] = self.eef_rot_offset
        qpos_offsets, qvel_offsets, _ = self.query_joint_offsets(self._l_arm_joint_names)
        self._l_controller_config["joint_indexes"] = {
            "joints": self._l_arm_joint_names,
            "qpos": qpos_offsets,
            "qvel": qvel_offsets,
        }
        self._l_controller_config["actuator_range"] = l_ctrl_range
        self._l_controller_config["policy_freq"] = self._control_freq
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
        scaled_action_range = np.ones(self._action_range.shape, dtype=np.float32)
        self.action_space = self.generate_action_space(scaled_action_range)

    def get_env_version(self):
        return OpenLoongEnv.ENV_VERSION

    def check_success(self):
        """
        Check if the task condition(s) is reached. Should return a dictionary
        { str: bool } with at least a "task" key for the overall task success,
        and additional optional keys corresponding to other task criteria.
        """        
        success = self._is_success()
        return {"task": success}
    
    def render_callback(self, mode='human') -> None:
        if mode == "human":
            self.render()
        else:
            raise ValueError("Invalid render mode")

    def _set_init_state(self) -> None:
        # print("Set initial state")
        self._task_status = TaskStatus.NOT_STARTED
        self._set_joint_neutral()

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

    def _is_success(self) -> bool:
        return self._task_status == TaskStatus.SUCCESS
    
    def _is_truncated(self) -> bool:
        return self._task_status == TaskStatus.FAILURE

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

        if self._run_mode == RunMode.TELEOPERATION:
            r_hand_force = self._query_hand_force(self._r_hand_gemo_ids)
            l_hand_force = self._query_hand_force(self._l_hand_gemo_ids)
            self._pico_joystick.send_force_message(l_hand_force, r_hand_force)

        obs = self._get_obs().copy()

        info = {"state": self.get_state(), "action": scaled_action, "object" : np.zeros(3), "goal": np.zeros(3),
                "task_status": self._task_status, "language_instruction": self._task.get_language_instruction()}
        terminated = self._is_success()
        truncated = self._is_truncated()
        reward = self._compute_reward(info)

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


    def _teleoperation_action(self) -> tuple:
        mocap_l_xpos, mocap_l_xquat, mocap_r_xpos, mocap_r_xquat = None, None, None, None

        if self._pico_joystick is not None:
            mocap_l_xpos, mocap_l_xquat, mocap_r_xpos, mocap_r_xquat = self._processe_pico_joystick_move()
            self.set_grasp_mocap(mocap_l_xpos, mocap_l_xquat)
            self.set_grasp_mocap_r(mocap_r_xpos, mocap_r_xquat)
            self._process_pico_joystick_operation()
            # print("base_body_euler: ", self._base_body_euler / np.pi * 180)
        else:
            return self.ctrl.copy(), np.zeros(14)


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
        self._set_arm_ctrl(self._r_arm_actuator_id, ctrl)

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
        self._set_arm_ctrl(self._l_arm_actuator_id, ctrl)
        
        return self.ctrl.copy(), np.concatenate([action_l, [self._grasp_value_l], action_r, [self._grasp_value_r]]).flatten()

    def _set_arm_ctrl(self, arm_actuator_id, ctrl) -> None:
        for i in range(len(arm_actuator_id)):
            self.ctrl[arm_actuator_id[i]] = ctrl[i]


    def _playback_action(self, action) -> np.ndarray:
        assert(len(action) == self.action_space.shape[0])
        
        action_l = action[:6]
        action_r = action[7:13]

        self._l_controller.set_goal(action_l)
        ctrl = self._l_controller.run_controller()
        self._set_arm_ctrl(self._l_arm_actuator_id, ctrl)

        self._grasp_value_l = action[6]
        self._set_l_hand_actuator_ctrl(self._grasp_value_l)

        self._r_controller.set_goal(action_r)
        ctrl = self._r_controller.run_controller()
        self._set_arm_ctrl(self._r_arm_actuator_id, ctrl)

        self._grasp_value_r = action[13]
        self._set_r_hand_actuator_ctrl(self._grasp_value_r)
        
        return self.ctrl.copy()
    

    def _get_obs(self) -> dict:
        ee_sites = self.query_site_pos_and_quat([self._ee_site_l, self._ee_site_r])
        ee_xvalp, ee_xvalr = self.query_site_xvalp_xvalr([self._ee_site_l, self._ee_site_r])

        arm_joint_values_l = self._get_arm_joint_values(self._l_arm_joint_names)
        arm_joint_values_r = self._get_arm_joint_values(self._r_arm_joint_names)
        arm_joint_velocities_l = self._get_arm_joint_velocities(self._l_arm_joint_names)
        arm_joint_velocities_r = self._get_arm_joint_velocities(self._r_arm_joint_names)

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

    def reset_model(self) -> tuple[dict, dict]:
        """
        Reset the environment, return observation
        """
        # self.reset_simulation()
        # print("Reset model")
        
        self._set_init_state()
        self.set_grasp_mocap(self._initial_grasp_site_xpos, self._initial_grasp_site_xquat)
        self.set_grasp_mocap_r(self._initial_grasp_site_xpos_r, self._initial_grasp_site_xquat_r)
        self._reset_gripper()

        self._reset_neck_mocap()

        self._task.get_task(self)

        # Tips for the operator
        print(self._task.get_language_instruction())
        print("Press left hand grip button to start recording task......")

        self.mj_forward()

        obs = self._get_obs().copy()
        return obs, {}

    # custom methods
    # -----------------------------

    def _set_joint_neutral(self) -> None:
        # assign value to arm joints
        arm_joint_qpos = {}
        for name, value in zip(self._r_arm_joint_names, self._r_neutral_joint_values):
            arm_joint_qpos[name] = np.array([value])
        for name, value in zip(self._l_arm_joint_names, self._l_neutral_joint_values):
            arm_joint_qpos[name] = np.array([value])     
        for name, value in zip(self._neck_joint_names, self._neck_neutral_joint_values):
            arm_joint_qpos[name] = np.array([value])
        self.set_joint_qpos(arm_joint_qpos)
    
    def _get_arm_joint_values(self, joint_names) -> np.ndarray:
        qpos_dict = self.query_joint_qpos(joint_names)
        return np.array([qpos_dict[joint_name] for joint_name in joint_names]).flatten()
    
    def _get_arm_joint_velocities(self, joint_names) -> np.ndarray:
        qvel_dict = self.query_joint_qvel(joint_names)
        return np.array([qvel_dict[joint_name] for joint_name in joint_names]).flatten()

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
        return 1 if self._is_success() else 0

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
        
    def _compute_reward(self, info) -> float:
        total_reward = 0.0

        for reward_function in self._reward_functions:
            if reward_function["coeff"] == 0:
                continue
            else:    
                reward = reward_function["function"]() * reward_function["coeff"]
                total_reward += reward
                self._print_reward(reward_function["function"].__name__, reward, reward_function["coeff"])

        self._print_reward("Total reward: ", total_reward)
        return total_reward
        
    
    def _setup_action_range(self) -> None:
        # 支持的动作范围空间，遥操作时不能超过这个范围
        # 模型接收的是 [-1, 1] 的动作空间，这里是真实的物理空间，需要进行归一化
        self._action_range = np.array(
            [
                [-2.0, 2.0], [-2.0, 2.0], [-2.0, 2.0],             # left hand ee pos
                [-np.pi, np.pi], [-np.pi, np.pi], [-np.pi, np.pi], # left hand ee angle euler
                [-1.0, 0.0],                                        # left hand grasp value
                [-2.0, 2.0], [-2.0, 2.0], [-2.0, 2.0],             # right hand ee pos
                [-np.pi, np.pi], [-np.pi, np.pi], [-np.pi, np.pi], # right hand ee angle euler
                [-1.0, 0.0],                                        # right hand grasp value
             ]
            , dtype=np.float32
            )
        self._action_range_min = self._action_range[:, 0]
        self._action_range_max = self._action_range[:, 1]

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
        
        # print("obs scale: ", self._obs_scale)

    def close(self):
        if hasattr(self, "_pico_joystick") and self._pico_joystick is not None:
            self._pico_joystick.close()

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

    def _set_task_status(self, joystick_state) -> None:
        if self._task_status == TaskStatus.NOT_STARTED and joystick_state["leftHand"]["gripButtonPressed"]:
            self._task_status = TaskStatus.GET_READY
            
        if self._task_status == TaskStatus.GET_READY and not joystick_state["leftHand"]["gripButtonPressed"]:
            print("Start to record task....... Press left hand grip if task failed, press right hand grip if task success.")            
            self._task_status = TaskStatus.BEGIN
            
        if self._task_status == TaskStatus.BEGIN and joystick_state["leftHand"]["gripButtonPressed"]:
            print("Task failed!")
            self._task_status = TaskStatus.FAILURE
        
        if self._task_status == TaskStatus.BEGIN and joystick_state["rightHand"]["gripButtonPressed"]:
            print("Task success!")
            self._task_status = TaskStatus.SUCCESS

    def _set_head_ctrl(self, joystick_state) -> None:
        x_axis = joystick_state["rightHand"]["joystickPosition"][0]
        if x_axis == 0:
            x_axis = joystick_state["leftHand"]["joystickPosition"][0]

        y_axis = joystick_state["rightHand"]["joystickPosition"][1]
        if y_axis == 0:
            y_axis = joystick_state["leftHand"]["joystickPosition"][1]
            
        mocap_neck_xpos, mocap_neck_xquat = self._mocap_neck_xpos, self._mocap_neck_xquat

        # 将 x_axis 和 y_axis 输入转换为旋转角度，按需要调节比例系数
        angle_x = -x_axis * np.pi / 180  # 转换为弧度，模拟绕 X 轴的旋转
        angle_y = -y_axis * np.pi / 180  # 转换为弧度，模拟绕 Y 轴的旋转

        # 设置旋转角度的限制
        self._neck_angle_x += angle_x
        if self._neck_angle_x > np.pi / 3 or self._neck_angle_x < -np.pi / 3:
            self._neck_angle_x = np.clip(self._neck_angle_x, -np.pi / 3, np.pi / 3)
            angle_x = 0
        
        self._neck_angle_y += angle_y
        if self._neck_angle_y > np.pi / 3 or self._neck_angle_y < -np.pi / 3:
            self._neck_angle_y = np.clip(self._neck_angle_y, -np.pi / 3, np.pi / 3)
            angle_y = 0

        new_neck_quat_local = rotations.euler2quat(np.array([0.0, angle_y, angle_x]))

        # 将局部坐标系的旋转转换为全局坐标系，乘以当前全局旋转四元数
        new_neck_quat_global = rotations.quat_mul(mocap_neck_xquat, new_neck_quat_local)

        # 将新的全局旋转四元数转换为轴角表示
        mocap_neck_axisangle = transform_utils.quat2axisangle(np.array([new_neck_quat_global[1], 
                                                                        new_neck_quat_global[2],
                                                                        new_neck_quat_global[3],
                                                                        new_neck_quat_global[0]]))

        # 可选：将轴角重新转换回四元数进行夹紧或其他操作
        new_neck_quat_cliped = transform_utils.axisangle2quat(mocap_neck_axisangle)

        # 将动作信息打包并发送到控制器
        action_neck = np.concatenate([mocap_neck_xpos, mocap_neck_axisangle])

        # # 更新 _mocap_neck_xquat 为新的全局旋转值
        self._mocap_neck_xquat = new_neck_quat_global

        self._neck_controller.set_goal(action_neck)
        ctrl = self._neck_controller.run_controller()
        for i in range(len(self._neck_actuator_id)):
            self.ctrl[self._neck_actuator_id[i]] = ctrl[i]

        # 更新头部位置
        self.set_neck_mocap(mocap_neck_xpos, self._mocap_neck_xquat)

    def _set_l_hand_actuator_ctrl(self, offset_rate) -> None:
        for actuator_id in self._l_hand_actuator_id:
            offset_dir = -1

            abs_ctrlrange = self._all_ctrlrange[actuator_id][1] - self._all_ctrlrange[actuator_id][0]
            self.ctrl[actuator_id] = offset_rate * offset_dir * abs_ctrlrange
            self.ctrl[actuator_id] = np.clip(
                self.ctrl[actuator_id],
                self._all_ctrlrange[actuator_id][0],
                self._all_ctrlrange[actuator_id][1])
            
    def _set_gripper_ctrl_l(self, joystick_state) -> None:
        # Press secondary button to set gripper minimal value
        offset_rate_clip_adjust_rate = 0.5  # 10% per second
        if joystick_state["leftHand"]["secondaryButtonPressed"]:
            self._l_gripper_offset_rate_clip -= offset_rate_clip_adjust_rate * self.dt    
            self._l_gripper_offset_rate_clip = np.clip(self._l_gripper_offset_rate_clip, -1, 0)
        elif joystick_state["leftHand"]["primaryButtonPressed"]:
            self._l_gripper_offset_rate_clip = 0

        # Press trigger to close gripper
        # Adjust sensitivity using an exponential function
        trigger_value = joystick_state["leftHand"]["triggerValue"]  # Value in [0, 1]
        k = np.e  # Adjust 'k' to change the curvature of the exponential function
        adjusted_value = (np.exp(k * trigger_value) - 1) / (np.exp(k) - 1)  # Maps input from [0, 1] to [0, 1]
        offset_rate = -adjusted_value
        offset_rate = np.clip(offset_rate, -1, self._l_gripper_offset_rate_clip)
        self._set_l_hand_actuator_ctrl(offset_rate)
        self._grasp_value_l = offset_rate
            
    def _set_r_hand_actuator_ctrl(self, offset_rate) -> None:
        for actuator_id in self._r_hand_actuator_id:
            offset_dir = -1

            abs_ctrlrange = self._all_ctrlrange[actuator_id][1] - self._all_ctrlrange[actuator_id][0]
            self.ctrl[actuator_id] = offset_rate * offset_dir * abs_ctrlrange
            self.ctrl[actuator_id] = np.clip(
                self.ctrl[actuator_id],
                self._all_ctrlrange[actuator_id][0],
                self._all_ctrlrange[actuator_id][1])

    def _set_gripper_ctrl_r(self, joystick_state) -> None:
        # Press secondary button to set gripper minimal value
        offset_rate_clip_adjust_rate = 0.5
        if joystick_state["rightHand"]["secondaryButtonPressed"]:
            self._r_gripper_offset_rate_clip -= offset_rate_clip_adjust_rate * self.dt
            self._r_gripper_offset_rate_clip = np.clip(self._r_gripper_offset_rate_clip, -1, 0)
        elif joystick_state["rightHand"]["primaryButtonPressed"]:
            self._r_gripper_offset_rate_clip = 0

        # Adjust sensitivity using an exponential function
        trigger_value = joystick_state["rightHand"]["triggerValue"]  # Value in [0, 1]
        k = np.e  # Adjust 'k' to change the curvature of the exponential function
        adjusted_value = (np.exp(k * trigger_value) - 1) / (np.exp(k) - 1)  # Maps input from [0, 1] to [0, 1]
        offset_rate = -adjusted_value
        offset_rate = np.clip(offset_rate, -1, self._r_gripper_offset_rate_clip)
        self._set_r_hand_actuator_ctrl(offset_rate)
        self._grasp_value_r = offset_rate

 
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
        
        # print("Josytick state: ", joystick_state)

        self._set_gripper_ctrl_r(joystick_state)
        self._set_gripper_ctrl_l(joystick_state)
        self._set_head_ctrl(joystick_state)
        self._set_task_status(joystick_state)


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


        # print("set init joint state: " , arm_joint_qpos_list)
        # assign value to finger joints
        # gripper_joint_qpos_list = {}
        # for name, value in zip(self._gripper_joint_names, self._neutral_joint_values[7:9]):
        #     gripper_joint_qpos_list[name] = np.array([value])
        # self.set_joint_qpos(gripper_joint_qpos_list)

    def _sample_goal(self) -> np.ndarray:
        # 训练reach时，任务是移动抓夹，goal以抓夹为原点采样
        goal = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        return goal


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