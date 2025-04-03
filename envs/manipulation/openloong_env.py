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
from orca_gym.task.abstract_task import AbstractTask
import time
from orca_gym.utils.joint_controller import JointController
import random

from orca_gym.environment.orca_gym_env import RewardType
from orca_gym.utils.reward_printer import RewardPrinter
from envs.manipulation.openloong_agent import OpenLoongAgentBase, AzureLoongHandAgent

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
    RANDOM_SAMPLE = "random_sample"

class ActionType:
    """
    Enum class for action type
    """
    END_EFFECTOR = "end_effector"
    JOINT_POS = "joint_pos"

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
            return f"\033[91m {self.prompt} \033[0m"

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
        action_type: ActionType,
        prompt: str,
        ctrl_device: ControlDevice,
        control_freq: int,
        sample_range: float,
        **kwargs,
    ):

        self._run_mode = run_mode
        self._action_type = action_type

        task_config_dict = kwargs["task_config_dict"]
        self._task = OpenLoongTask(prompt, task_config_dict)

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

        self.ctrl = np.zeros(self.nu)
        self._set_init_state()

        self._agents : dict[str, OpenLoongAgentBase] = {}
        for id, agent_name in enumerate(self._agent_names):
            if agent_name.startswith("AzureLoongHand"):
                self._agents[agent_name] = AzureLoongHandAgent(self, id)
        
        assert len(self._agents) > 0, "At least one agent should be created."

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

    @property
    def run_mode(self) -> RunMode:
        return self._run_mode
    
    @property
    def ctrl_device(self) -> ControlDevice:  
        return self._ctrl_device
    
    @property
    def control_freq(self) -> int:
        return self._control_freq
    
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
        [agent.set_joint_neutral(self) for agent in self._agents.values()]

        self.ctrl = np.zeros(self.nu)       
        self.set_ctrl(self.ctrl)
        self.mj_forward()



    def _is_success(self) -> bool:
        return self._task_status == TaskStatus.SUCCESS
    
    def _is_truncated(self) -> bool:
        return self._task_status == TaskStatus.FAILURE

    def step(self, action) -> tuple:
        if self._run_mode == RunMode.TELEOPERATION:
            ctrl, noscaled_action = self._teleoperation_action()
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
            noscaled_action = self._fill_arm_joint_pos(noscaled_action)
            scaled_action = self.normalize_action(noscaled_action, self._action_range_min, self._action_range_max)

            if self._pico_joystick is not None:
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
        elif self._ctrl_device == ControlDevice.RANDOM_SAMPLE:
            if self._task_status == TaskStatus.NOT_STARTED:
                self._task_status = TaskStatus.BEGIN
            if self._task_status == TaskStatus.BEGIN and np.random.uniform(0, 1) < 0.02:
                self._task_status = TaskStatus.SUCCESS
            return self.ctrl.copy(), np.random.uniform(-1.0, 1.0, 14)
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
        ctrl_r = self._r_controller.run_controller()
        # print("ctrl r: ", ctrl)
        self._set_arm_ctrl(self._r_arm_actuator_id, ctrl_r)

        mocap_l_axisangle = transform_utils.quat2axisangle(np.array([mocap_l_xquat[1], 
                                                                   mocap_l_xquat[2], 
                                                                   mocap_l_xquat[3], 
                                                                   mocap_l_xquat[0]]))  
        action_l = np.concatenate([mocap_l_xpos, mocap_l_axisangle])
        # print("action l:", action_l)        
        # print(action)
        self._l_controller.set_goal(action_l)
        ctrl_l = self._l_controller.run_controller()
        # print("ctrl l: ", ctrl)
        self._set_arm_ctrl(self._l_arm_actuator_id, ctrl_l)
        
        action = np.concatenate([action_l,                  # left eef pos and angle, 0-5
                                 np.zeros(7),               # left arm joint pos, 6-12 (will be fill after do simulation)
                                 [self._grasp_value_l],     # left hand grasp value, 13
                                 action_r,                  # right eef pos and angle, 14-19
                                 np.zeros(7),               # right arm joint pos, 20-26 (will be fill after do simulation)
                                 [self._grasp_value_r]]     # right hand grasp value, 27
                                ).flatten()

        return self.ctrl.copy(), action

    def _fill_arm_joint_pos(self, action) -> None:
        arm_joint_values_l = self._get_arm_joint_values(self._l_arm_joint_names)
        arm_joint_values_r = self._get_arm_joint_values(self._r_arm_joint_names)
        action[6:13] = arm_joint_values_l
        action[20:27] = arm_joint_values_r
        return action

    def _set_arm_ctrl(self, arm_actuator_id, ctrl) -> None:
        for i in range(len(arm_actuator_id)):
            self.ctrl[arm_actuator_id[i]] = ctrl[i]


    def _playback_action(self, action) -> np.ndarray:
        assert(len(action) == self.action_space.shape[0])
        
        self._grasp_value_l = action[13]
        self._set_l_hand_actuator_ctrl(self._grasp_value_l)
        self._grasp_value_r = action[27]
        self._set_r_hand_actuator_ctrl(self._grasp_value_r)

        if self._action_type == ActionType.END_EFFECTOR:
            action_l = action[:6]
            self._l_controller.set_goal(action_l)
            ctrl = self._l_controller.run_controller()
            self._set_arm_ctrl(self._l_arm_actuator_id, ctrl)

            action_r = action[14:20]
            self._r_controller.set_goal(action_r)
            ctrl = self._r_controller.run_controller()
            self._set_arm_ctrl(self._r_arm_actuator_id, ctrl)

        elif self._action_type == ActionType.JOINT_POS:
            l_arm_joint_action = action[6:13]
            # l_arm_joint_qpos = self._get_arm_joint_values(self._l_arm_joint_names)
            # l_arm_joint_qvel = self._get_arm_joint_velocities(self._l_arm_joint_names)
            # l_arm_torque = np.zeros(7)
            # for i in range(len(l_arm_joint_action)):
            #     l_arm_torque[i] = self._joint_controllers[self._l_arm_joint_names[i]].compute_torque(
            #         l_arm_joint_action[i], l_arm_joint_qpos[i], l_arm_joint_qvel[i], self.dt)
            # # print("l_arm_joint_action: ", l_arm_joint_action, "l_arm_torque: ", l_arm_torque)
            # self._set_arm_ctrl(self._l_arm_actuator_id, l_arm_torque)
            self._set_arm_ctrl(self._l_arm_actuator_id, l_arm_joint_action)

            r_arm_joint_action = action[20:27]
            # r_arm_joint_qpos = self._get_arm_joint_values(self._r_arm_joint_names)
            # r_arm_joint_qvel = self._get_arm_joint_velocities(self._r_arm_joint_names)
            # r_arm_torque = np.zeros(7)
            # for i in range(len(r_arm_joint_action)):
            #     r_arm_torque[i] = self._joint_controllers[self._r_arm_joint_names[i]].compute_torque(
            #         r_arm_joint_action[i], r_arm_joint_qpos[i], r_arm_joint_qvel[i], self.dt)
            # # print("r_arm_joint_action: ", r_arm_joint_action, "r_arm_torque: ", r_arm_torque)
            # self._set_arm_ctrl(self._r_arm_actuator_id, r_arm_torque)
            self._set_arm_ctrl(self._r_arm_actuator_id, r_arm_joint_action)
        else:
            raise ValueError("Invalid action type: ", self._action_type)
        
        return self.ctrl.copy()
    

    def _get_obs(self) -> dict:
        obs = {}
        for agent in self._agents.values():
            obs[agent.name] = agent.get_obs(self)
        return obs

    def reset_model(self) -> tuple[dict, dict]:
        """
        Reset the environment, return observation
        """
        # self.reset_simulation()
        # print("Reset model")
        
        self._set_init_state()
        
        [agent.on_reset_model(self) for agent in self._agents.values()]

        self._task.get_task(self)

        # Tips for the operator
        print(self._task.get_language_instruction())
        print("Press left hand grip button to start recording task......")

        self.mj_forward()

        obs = self._get_obs().copy()
        return obs, {}

    # custom methods
    # -----------------------------



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
        

    # def _setup_joint_controllers(self) -> dict[str, JointController]:
    #     joint_config = {
    #         "large" : {
    #             "kp": 400,
    #             "ki": 0.05,
    #             "kd": 1.0,
    #             "kv": 40,
    #             "max_speed": 80,
    #             "ctrlrange": (-80, 80),
    #         },
    #         "middle" : {
    #             "kp": 300,
    #             "ki": 0.1,
    #             "kd": 2.0,
    #             "kv": 30,
    #             "max_speed": 48,
    #             "ctrlrange": (-48, 48),
    #         },
    #         "small" : {
    #             "kp": 150,
    #             "ki": 0.1,
    #             "kd": 2.0,
    #             "kv": 15,
    #             "max_speed": 12.4,
    #             "ctrlrange": (-12.4, 12.4),
    #         },
    #     }

    #     joint_types = {
    #         "large": [self.joint("J_arm_l_01"), self.joint("J_arm_l_02"), self.joint("J_arm_r_01"), self.joint("J_arm_r_02")],
    #         "middle": [self.joint("J_arm_l_03"), self.joint("J_arm_l_04"), self.joint("J_arm_r_03"), self.joint("J_arm_r_04")],
    #         "small": [self.joint("J_arm_l_05"), self.joint("J_arm_l_06"), self.joint("J_arm_l_07"), self.joint("J_arm_r_05"), self.joint("J_arm_r_06"), self.joint("J_arm_r_07")],
    #     }

    #     controllers = {}
    #     for joint_name in self._l_arm_joint_names:
    #         config = joint_config["large"] if joint_name in joint_types["large"] else joint_config["middle"] if joint_name in joint_types["middle"] else joint_config["small"]
    #         controllers[joint_name] = JointController(config["kp"], config["ki"], config["kd"], config["kv"], config["max_speed"], config["ctrlrange"])

    #     for joint_name in self._r_arm_joint_names:
    #         config = joint_config["large"] if joint_name in joint_types["large"] else joint_config["middle"] if joint_name in joint_types["middle"] else joint_config["small"]
    #         controllers[joint_name] = JointController(config["kp"], config["ki"], config["kd"], config["kv"], config["max_speed"], config["ctrlrange"])
    #     return controllers


        # print("obs scale: ", self._obs_scale)

    def close(self):
        [agent.on_close() for agent in self._agents.values()]

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
            
    def _set_gripper_ctrl_l(self, joystick_state) -> None:
        # Press secondary button to set gripper minimal value
        offset_rate_clip_adjust_rate = 0.1  # 10% per second
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

    def _set_gripper_ctrl_r(self, joystick_state) -> None:
        # Press secondary button to set gripper minimal value
        offset_rate_clip_adjust_rate = 0.1
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

    def action_use_motor(self):
        if self._run_mode == RunMode.TELEOPERATION:
            # 遥操作采用OSC算法，因此采用电机控制
            return True
        else:
            if self._action_type == ActionType.END_EFFECTOR:
                # 回放采用OSC解算末端位姿，因此采用电机控制
                return True
            elif self._action_type == ActionType.JOINT_POS:
                # 回放直接采用关节角度，因此采用关节控制
                return False
            else:
                raise ValueError("Invalid action type: ", self._action_type)

    def disable_actuators(self, actuator_names):
        for actuator_name in actuator_names:
            actuator_id = self.model.actuator_name2id(actuator_name)
            self.disable_actuator(actuator_id)
            
            
            