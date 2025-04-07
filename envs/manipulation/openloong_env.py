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

class ActionType:
    """
    Enum class for action type
    """
    END_EFFECTOR = "end_effector"
    JOINT_POS = "joint_pos"

class OpenLoongTask(AbstractTask):
    def __init__(self, config: dict):

        super().__init__(config)

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
            return "Do something."

        object_str = "object: "
        for key in self.task_dict.keys():
            object_str += key + " "
        
        goal_str = "goal: "
        for value in self.task_dict.values():
            goal_str += value + " "
        
        language_instruction = f"level: {self.level_name} task: {self.prompt}"
        language_instruction += f"{object_str} to {goal_str}"                               

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
        pico_ports: list,
        time_step: float,
        run_mode: RunMode,
        action_type: ActionType,
        ctrl_device: ControlDevice,
        control_freq: int,
        sample_range: float,
        task_config_dict: dict,
        **kwargs,
    ):

        self._run_mode = run_mode
        self._action_type = action_type

        self._task = OpenLoongTask(task_config_dict)

        self._ctrl_device = ctrl_device
        self._control_freq = control_freq
        
        if self._ctrl_device == ControlDevice.VR:
            self._joystick = {}
            for i in range(len(agent_names)):
                self._joystick[agent_names[i]] = PicoJoystick(int(pico_ports[i]))
        
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
        self.mj_forward()   # Do this before initializing the controller, joints, sites, bodies, to avoid NaN error.

        self._agents : dict[str, OpenLoongAgent] = {}
        for id, agent_name in enumerate(self._agent_names):
            if agent_name.startswith("OpenLoongHand"):
                self._agents[agent_name] = OpenLoongHandAgent(self, id=id, name=agent_name)
        
        assert len(self._agents) > 0, "At least one agent should be created."
        self._set_init_state()
        
        # Run generate_observation_space after initialization to ensure that the observation object's name is defined.
        self._set_obs_space()
        self._set_action_space()


    def _set_obs_space(self):
        self.observation_space = self.generate_observation_space(self._get_obs().copy())

    def _set_action_space(self):
        env_action_range = np.concatenate([agent.action_range for agent in self._agents.values()], axis=0)
        self.env_action_range_min = env_action_range[:, 0]
        self.env_action_range_max = env_action_range[:, 1]
        # print("env action range: ", action_range)
        scaled_action_range = np.ones(env_action_range.shape, dtype=np.float32)
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
    
    @property
    def task_status(self) -> TaskStatus:
        return self._task_status
    
    @property
    def action_type(self) -> ActionType:
        return self._action_type
    
    @property
    def joystick(self) -> Optional[PicoJoystick]:
        if self._ctrl_device == ControlDevice.VR:
            return self._joystick
    
    def set_task_status(self, status: TaskStatus):
        if status == TaskStatus.SUCCESS:
            print("Task success!")
        elif status == TaskStatus.FAILURE:
            print("Task failure!")
        elif status == TaskStatus.BEGIN:
            print("Start to record task....... Press left hand grip if task failed, press right hand grip if task success.")            
        self._task_status = status
    
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
            noscaled_action = self.denormalize_action(action, self.env_action_range_min, self.env_action_range_max)
            ctrl = self._playback_action(noscaled_action)
        elif self._run_mode == RunMode.POLICY_RAW:
            noscaled_action = np.clip(action, self.env_action_range_min, self.env_action_range_max)
            scaled_action = self.normalize_action(noscaled_action, self.env_action_range_min, self.env_action_range_max)
            ctrl = self._playback_action(noscaled_action)
        else:
            raise ValueError("Invalid run mode : ", self._run_mode)
        
        # print("runmode: ", self._run_mode, "no_scaled_action: ", noscaled_action, "scaled_action: ", scaled_action, "ctrl: ", ctrl)
        
        # step the simulation with original action space
        self.do_simulation(ctrl, self.frame_skip)

        if self._run_mode == RunMode.TELEOPERATION:
            noscaled_action = self._fill_arm_joint_pos(noscaled_action)
            scaled_action = self.normalize_action(noscaled_action, self.env_action_range_min, self.env_action_range_max)

            [agent.update_force_feedback(self) for agent in self._agents.values()]


        obs = self._get_obs().copy()

        info = {"state": self.get_state(), "action": scaled_action, "object" : np.zeros(3), "goal": np.zeros(3),
                "task_status": self._task_status, "language_instruction": self._task.get_language_instruction()}
        terminated = self._is_success()
        truncated = self._is_truncated()
        reward = self._compute_reward(info)

        return obs, reward, terminated, truncated, info

    def _split_agent_action(self, action) -> dict:
        """
        Split the action into agent actions.
        """
        start = 0
        end = 0
        agent_action = {}
        for agent in self._agents.values():
            end += agent.action_range.shape[0]
            agent_action[agent.name] = action[start:end].copy()
            start = end
            # print(agent.name, "action: ", agent_action[agent.name])
        
        return agent_action

    def _fill_arm_joint_pos(self, action) -> np.ndarray:
        agent_action = self._split_agent_action(action)
        return np.concatenate([agent.fill_arm_joint_pos(self, agent_action[agent.name]) for agent in self._agents.values()], axis=0).flatten()
    
    

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
        agent_action = []
        for agent in self._agents.values():
            agent_action.append(agent.on_teleoperation_action(self))

        return self.ctrl.copy(), np.concatenate(agent_action).flatten()


    def _playback_action(self, action) -> np.ndarray:
        assert(len(action) == self.action_space.shape[0])
        
        # 将动作分配给每个agent
        agent_action = self._split_agent_action(action)
        for agent in self._agents.values():
            agent.on_playback_action(self, agent_action[agent.name])
        
        return self.ctrl.copy()
    

    def _get_obs(self) -> dict:
        if len(self._agents) == 1:
            # Use original observation if only one agent
            return self._agents[self._agent_names[0]].get_obs(self)

        # 将所有的agent obs 合并到一起，其中每个agent obs key 加上前缀 agent_name，确保不重复
        # 注意：这里需要兼容 gymnasium 的 obs dict 范式，因此不引入多级字典
        # 同理多agent的action采用拼接np.array方式，不采用字典分隔
        obs = {}
        for agent in self._agents.values():
            agent_obs = agent.get_obs(self)
            for key in agent_obs.keys():
                agent_key = f"{agent.name}_{key}"
                if agent_key in obs:
                    raise ValueError(f"Duplicate observation key: {agent_key}")
                obs[agent_key] = agent_obs[key]
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
        


    def close(self):
        [agent.on_close() for agent in self._agents.values()]


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


    # def get_ee_xform(self) -> np.ndarray:
    #     pos_dict = self.query_site_pos_and_mat([self.site("ee_center_site", id)])
    #     xpos = pos_dict[self.site("ee_center_site", id)]['xpos'].copy()
    #     xmat = pos_dict[self.site("ee_center_site", id)]['xmat'].copy().reshape(3, 3)
    #     return xpos, xmat


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

    def disable_actuators(self, actuator_names, trnid):
        for actuator_name in actuator_names:
            actuator_id = self.model.actuator_name2id(actuator_name)
            self.set_actuator_trnid(actuator_id, trnid)


## --------------------------------            
## Agent Class
## --------------------------------            
class OpenLoongAgent:
    def __init__(self, env: OpenLoongEnv, id: int, name: str) -> None:
        self._id = id
        self._name = name

    @property
    def id(self) -> int:
        return self._id
    
    @property
    def name(self) -> str:
        return self._name

    def init_agent(self, env: OpenLoongEnv, id: int):
        self._base_body_name = [env.body("base_link", id)]
        self._base_body_xpos, _, self._base_body_xquat = env.get_body_xpos_xmat_xquat(self._base_body_name)
        print("base_body_xpos: ", self._base_body_xpos)
        print("base_body_xquat: ", self._base_body_xquat)     
        
        dummy_joint_id = env.model.joint_name2id(env.joint("dummy_joint", id))

        self._neck_joint_names = [env.joint("J_head_yaw", id), env.joint("J_head_pitch", id)]
        self._neck_actuator_names = [env.actuator("M_head_yaw", id), env.actuator("M_head_pitch", id)]
        self._neck_actuator_id = [env.model.actuator_name2id(actuator_name) for actuator_name in self._neck_actuator_names]
        self._neck_neutral_joint_values = np.array([0, -0.7854])
        self._neck_ctrl_values = {"yaw": 0.0, "pitch": -0.7854}

        # index used to distinguish arm and gripper joints
        self._r_arm_joint_names = [env.joint("J_arm_r_01", id), env.joint("J_arm_r_02", id), 
                                 env.joint("J_arm_r_03", id), env.joint("J_arm_r_04", id), 
                                 env.joint("J_arm_r_05", id), env.joint("J_arm_r_06", id), env.joint("J_arm_r_07", id)]
        self._r_arm_motor_names = [env.actuator("M_arm_r_01", id), env.actuator("M_arm_r_02", id),
                                env.actuator("M_arm_r_03", id),env.actuator("M_arm_r_04", id),
                                env.actuator("M_arm_r_05", id),env.actuator("M_arm_r_06", id),env.actuator("M_arm_r_07", id)]
        self._r_arm_position_names = [env.actuator("P_arm_r_01", id), env.actuator("P_arm_r_02", id),
                                      env.actuator("P_arm_r_03", id),env.actuator("P_arm_r_04", id),
                                      env.actuator("P_arm_r_05", id),env.actuator("P_arm_r_06", id),env.actuator("P_arm_r_07", id)]
        if env.action_use_motor():
            env.disable_actuators(self._r_arm_position_names, dummy_joint_id)
            self._r_arm_actuator_id = [env.model.actuator_name2id(actuator_name) for actuator_name in self._r_arm_motor_names]
        else:
            env.disable_actuators(self._r_arm_motor_names, dummy_joint_id)
            self._r_arm_actuator_id = [env.model.actuator_name2id(actuator_name) for actuator_name in self._r_arm_position_names]
        self._r_neutral_joint_values = np.array([0.905, -0.735, -2.733, 1.405, -1.191, 0.012, -0.517])
        

        
        # print("hand_actuator_id: ", self._r_hand_actuator_id)

        # index used to distinguish arm and gripper joints
        self._l_arm_joint_names = [env.joint("J_arm_l_01", id), env.joint("J_arm_l_02", id), 
                                 env.joint("J_arm_l_03", id), env.joint("J_arm_l_04", id), 
                                 env.joint("J_arm_l_05", id), env.joint("J_arm_l_06", id), env.joint("J_arm_l_07", id)]
        self._l_arm_moto_names = [env.actuator("M_arm_l_01", id), env.actuator("M_arm_l_02", id),
                                env.actuator("M_arm_l_03", id),env.actuator("M_arm_l_04", id),
                                env.actuator("M_arm_l_05", id),env.actuator("M_arm_l_06", id),env.actuator("M_arm_l_07", id)]
        self._l_arm_position_names = [env.actuator("P_arm_l_01", id), env.actuator("P_arm_l_02", id),
                                      env.actuator("P_arm_l_03", id),env.actuator("P_arm_l_04", id),
                                      env.actuator("P_arm_l_05", id),env.actuator("P_arm_l_06", id),env.actuator("P_arm_l_07", id)]
        if env.action_use_motor():
            env.disable_actuators(self._l_arm_position_names, dummy_joint_id)
            self._l_arm_actuator_id = [env.model.actuator_name2id(actuator_name) for actuator_name in self._l_arm_moto_names]
        else:
            env.disable_actuators(self._l_arm_moto_names, dummy_joint_id)
            self._l_arm_actuator_id = [env.model.actuator_name2id(actuator_name) for actuator_name in self._l_arm_position_names]
        self._l_neutral_joint_values = np.array([-0.905, 0.735, 2.733, 1.405, 1.191, 0.012, 0.517])
        # self._l_neutral_joint_values = np.zeros(7)
        
        print("arm_actuator_id: ", self._r_arm_actuator_id, self._l_arm_actuator_id)

        # control range
        self._all_ctrlrange = env.model.get_actuator_ctrlrange()
        neck_ctrl_range = [self._all_ctrlrange[actoator_id] for actoator_id in self._neck_actuator_id]
        # print("ctrl_range: ", neck_ctrl_range)

        r_ctrl_range = [self._all_ctrlrange[actoator_id] for actoator_id in self._r_arm_actuator_id]
        # print("ctrl_range: ", r_ctrl_range)

        l_ctrl_range = [self._all_ctrlrange[actoator_id] for actoator_id in self._l_arm_actuator_id]
        # print("ctrl_range: ", l_ctrl_range)


        arm_qpos_range_l = env.model.get_joint_qposrange(self._l_arm_joint_names)
        arm_qpos_range_r = env.model.get_joint_qposrange(self._r_arm_joint_names)
        self._setup_action_range(arm_qpos_range_l, arm_qpos_range_r)
        self._setup_obs_scale(arm_qpos_range_l, arm_qpos_range_r)

        self.set_joint_neutral(env)
        env.mj_forward()

        NECK_NAME  = env.site("neck_center_site", id)
        site_dict = env.query_site_pos_and_quat([NECK_NAME])
        self._initial_neck_site_xpos = site_dict[NECK_NAME]['xpos']
        self._initial_neck_site_xquat = site_dict[NECK_NAME]['xquat']

        self.set_neck_mocap(env, self._initial_neck_site_xpos, self._initial_neck_site_xquat)
        self._mocap_neck_xpos, self._mocap_neck_xquat = self._initial_neck_site_xpos, self._initial_neck_site_xquat
        self._neck_angle_x, self._neck_angle_y = 0, 0

        self._ee_site_l  = env.site("ee_center_site", id)
        site_dict = env.query_site_pos_and_quat([self._ee_site_l])
        self._initial_grasp_site_xpos = site_dict[self._ee_site_l]['xpos']
        self._initial_grasp_site_xquat = site_dict[self._ee_site_l]['xquat']
        self._grasp_value_l = 0.0

        self.set_grasp_mocap(env, self._initial_grasp_site_xpos, self._initial_grasp_site_xquat)

        self._ee_site_r  = env.site("ee_center_site_r", id)
        site_dict = env.query_site_pos_and_quat([self._ee_site_r])
        self._initial_grasp_site_xpos_r = site_dict[self._ee_site_r]['xpos']
        self._initial_grasp_site_xquat_r = site_dict[self._ee_site_r]['xquat']
        self._grasp_value_r = 0.0

        self.set_grasp_mocap_r(env, self._initial_grasp_site_xpos_r, self._initial_grasp_site_xquat_r)
        
        if env.run_mode == RunMode.TELEOPERATION:
            if env.ctrl_device == ControlDevice.VR:
                self._pico_joystick = env.joystick[self.name]
            else:
                raise ValueError("Invalid control device: ", env.ctrl_device)

        # -----------------------------
        # Neck controller
        self._neck_controller_config = controller_config.load_config("osc_pose")
        # print("controller_config: ", self.controller_config)

        # Add to the controller dict additional relevant params:
        #   the robot name, mujoco sim, eef_name, joint_indexes, timestep (model) freq,
        #   policy (control) freq, and ndim (# joints)
        self._neck_controller_config["robot_name"] = self.name
        self._neck_controller_config["sim"] = env.gym
        self._neck_controller_config["eef_name"] = NECK_NAME
        # self.controller_config["eef_rot_offset"] = self.eef_rot_offset
        qpos_offsets, qvel_offsets, _ = env.query_joint_offsets(self._neck_joint_names)
        self._neck_controller_config["joint_indexes"] = {
            "joints": self._neck_joint_names,
            "qpos": qpos_offsets,
            "qvel": qvel_offsets,
        }
        self._neck_controller_config["actuator_range"] = neck_ctrl_range
        self._neck_controller_config["policy_freq"] = env.control_freq
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
        self._r_controller_config["robot_name"] = self.name
        self._r_controller_config["sim"] = env.gym
        self._r_controller_config["eef_name"] = self._ee_site_r
        # self.controller_config["eef_rot_offset"] = self.eef_rot_offset
        qpos_offsets, qvel_offsets, _ = env.query_joint_offsets(self._r_arm_joint_names)
        self._r_controller_config["joint_indexes"] = {
            "joints": self._r_arm_joint_names,
            "qpos": qpos_offsets,
            "qvel": qvel_offsets,
        }
        self._r_controller_config["actuator_range"] = r_ctrl_range
        self._r_controller_config["policy_freq"] = env.control_freq
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
        self._l_controller_config["robot_name"] = self.name
        self._l_controller_config["sim"] = env.gym
        self._l_controller_config["eef_name"] = self._ee_site_l
        # self.controller_config["eef_rot_offset"] = self.eef_rot_offset
        qpos_offsets, qvel_offsets, _ = env.query_joint_offsets(self._l_arm_joint_names)
        self._l_controller_config["joint_indexes"] = {
            "joints": self._l_arm_joint_names,
            "qpos": qpos_offsets,
            "qvel": qvel_offsets,
        }
        self._l_controller_config["actuator_range"] = l_ctrl_range
        self._l_controller_config["policy_freq"] = env.control_freq
        self._l_controller_config["ndim"] = len(self._l_arm_joint_names)
        self._l_controller_config["control_delta"] = False


        self._l_controller = controller_factory(self._l_controller_config["type"], self._l_controller_config)
        self._l_controller.update_initial_joints(self._l_neutral_joint_values)

        self._l_gripper_offset_rate_clip = 0.0

    def on_close(self):
        if hasattr(self, "_pico_joystick") and self._pico_joystick is not None:
            self._pico_joystick.close()        

    def set_joint_neutral(self, env: OpenLoongEnv) -> None:
        # assign value to arm joints
        arm_joint_qpos = {}
        for name, value in zip(self._r_arm_joint_names, self._r_neutral_joint_values):
            arm_joint_qpos[name] = np.array([value])
        for name, value in zip(self._l_arm_joint_names, self._l_neutral_joint_values):
            arm_joint_qpos[name] = np.array([value])     
        for name, value in zip(self._neck_joint_names, self._neck_neutral_joint_values):
            arm_joint_qpos[name] = np.array([value])
        env.set_joint_qpos(arm_joint_qpos)        

    def on_reset_model(self, env: OpenLoongEnv) -> None:
        self._reset_grasp_mocap(env)
        self._reset_gripper()
        self._reset_neck_mocap(env)

    def set_neck_mocap(self, env: OpenLoongEnv, position, orientation) -> None:
        mocap_pos_and_quat_dict = {env.mocap("neckMocap", self.id): {'pos': position, 'quat': orientation}}
        env.set_mocap_pos_and_quat(mocap_pos_and_quat_dict)

    def set_grasp_mocap(self, env: OpenLoongEnv, position, orientation) -> None:
        mocap_pos_and_quat_dict = {env.mocap("leftHandMocap", self.id): {'pos': position, 'quat': orientation}}
        env.set_mocap_pos_and_quat(mocap_pos_and_quat_dict)

    def set_grasp_mocap_r(self, env: OpenLoongEnv, position, orientation) -> None:
        mocap_pos_and_quat_dict = {env.mocap("rightHandMocap", self.id): {'pos': position, 'quat': orientation}}
        # print("Set grasp mocap: ", position, orientation)
        env.set_mocap_pos_and_quat(mocap_pos_and_quat_dict)

    def _reset_gripper(self) -> None:
        self._l_gripper_offset_rate_clip = 0.0
        self._r_gripper_offset_rate_clip = 0.0

    def _reset_neck_mocap(self, env: OpenLoongEnv) -> None:
        self._mocap_neck_xpos, self._mocap_neck_xquat = self._initial_neck_site_xpos, self._initial_neck_site_xquat
        self.set_neck_mocap(env, self._mocap_neck_xpos, self._mocap_neck_xquat)
        self._neck_angle_x, self._neck_angle_y = 0, 0
        
    def _reset_grasp_mocap(self, env: OpenLoongEnv) -> None:
        self.set_grasp_mocap(env, self._initial_grasp_site_xpos, self._initial_grasp_site_xquat)
        self.set_grasp_mocap_r(env, self._initial_grasp_site_xpos_r, self._initial_grasp_site_xquat_r)

    def get_obs(self, env: OpenLoongEnv) -> dict:
        ee_sites = env.query_site_pos_and_quat([self._ee_site_l, self._ee_site_r])
        ee_xvalp, ee_xvalr = env.query_site_xvalp_xvalr([self._ee_site_l, self._ee_site_r])

        arm_joint_values_l = self._get_arm_joint_values(env, self._l_arm_joint_names)
        arm_joint_values_r = self._get_arm_joint_values(env, self._r_arm_joint_names)
        arm_joint_velocities_l = self._get_arm_joint_velocities(env, self._l_arm_joint_names)
        arm_joint_velocities_r = self._get_arm_joint_velocities(env, self._r_arm_joint_names)

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

    
    def _get_arm_joint_values(self, env: OpenLoongEnv, joint_names) -> np.ndarray:
        qpos_dict = env.query_joint_qpos(joint_names)
        return np.array([qpos_dict[joint_name] for joint_name in joint_names]).flatten()
    
    def _get_arm_joint_velocities(self, env: OpenLoongEnv, joint_names) -> np.ndarray:
        qvel_dict = env.query_joint_qvel(joint_names)
        return np.array([qvel_dict[joint_name] for joint_name in joint_names]).flatten()

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

    def _setup_action_range(self, arm_qpos_range_l, arm_qpos_range_r) -> None:
        # 支持的动作范围空间，遥操作时不能超过这个范围
        # 模型接收的是 [-1, 1] 的动作空间，这里是真实的物理空间，需要进行归一化
        self._action_range =  np.concatenate(
            [
                [[-2.0, 2.0], [-2.0, 2.0], [-2.0, 2.0], [-np.pi, np.pi], [-np.pi, np.pi], [-np.pi, np.pi]], # left hand ee pos and angle euler
                arm_qpos_range_l,                                                                           # left arm joint pos
                [[-1.0, 0.0]],                                                                                # left hand grasp value
                [[-2.0, 2.0], [-2.0, 2.0], [-2.0, 2.0], [-np.pi, np.pi], [-np.pi, np.pi], [-np.pi, np.pi]], # right hand ee pos and angle euler
                arm_qpos_range_r,                                                                           # right arm joint pos
                [[-1.0, 0.0]],                                                                                # right hand grasp value
            ],
            dtype=np.float32,
            axis=0
        )

        self._action_range_min = self._action_range[:, 0]
        self._action_range_max = self._action_range[:, 1]
        
    @property
    def action_range(self) -> np.ndarray:
        return self._action_range

    def on_teleoperation_action(self, env: OpenLoongEnv) -> np.ndarray:
        mocap_l_xpos, mocap_l_xquat, mocap_r_xpos, mocap_r_xquat = None, None, None, None

        if self._pico_joystick is not None:
            mocap_l_xpos, mocap_l_xquat, mocap_r_xpos, mocap_r_xquat = self._processe_pico_joystick_move(env)
            self.set_grasp_mocap(env, mocap_l_xpos, mocap_l_xquat)
            self.set_grasp_mocap_r(env, mocap_r_xpos, mocap_r_xquat)
            self._process_pico_joystick_operation(env)
            # print("base_body_euler: ", self._base_body_euler / np.pi * 180)
        else:
            return np.zeros(14)


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
        self._set_arm_ctrl(env, self._r_arm_actuator_id, ctrl_r)

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
        self._set_arm_ctrl(env, self._l_arm_actuator_id, ctrl_l)
        
        action = np.concatenate([action_l,                  # left eef pos and angle, 0-5
                                 np.zeros(7),               # left arm joint pos, 6-12 (will be fill after do simulation)
                                 [self._grasp_value_l],     # left hand grasp value, 13
                                 action_r,                  # right eef pos and angle, 14-19
                                 np.zeros(7),               # right arm joint pos, 20-26 (will be fill after do simulation)
                                 [self._grasp_value_r]]     # right hand grasp value, 27
                                ).flatten()

        return action

    def fill_arm_joint_pos(self, env : OpenLoongEnv, action) -> None:
        arm_joint_values_l = self._get_arm_joint_values(env, self._l_arm_joint_names)
        arm_joint_values_r = self._get_arm_joint_values(env, self._r_arm_joint_names)
        action[6:13] = arm_joint_values_l
        action[20:27] = arm_joint_values_r
        return action

    def _set_arm_ctrl(self, env : OpenLoongEnv, arm_actuator_id, ctrl) -> None:
        for i in range(len(arm_actuator_id)):
            env.ctrl[arm_actuator_id[i]] = ctrl[i]


    def _process_pico_joystick_operation(self, env: OpenLoongEnv) -> None:
        joystick_state = self._pico_joystick.get_key_state()
        if joystick_state is None:
            return
        
        # print("Josytick state: ", joystick_state)

        self._set_gripper_ctrl_r(env, joystick_state)
        self._set_gripper_ctrl_l(env, joystick_state)
        self._set_head_ctrl(env, joystick_state)
        self._set_task_status(env, joystick_state)



    def _set_task_status(self, env : OpenLoongEnv, joystick_state) -> None:
        if self.id != 0:
            # Just for the first agent's controller
            return
            
        if env.task_status == TaskStatus.NOT_STARTED and joystick_state["leftHand"]["gripButtonPressed"]:
            env.set_task_status(TaskStatus.GET_READY)
        elif env.task_status == TaskStatus.GET_READY and not joystick_state["leftHand"]["gripButtonPressed"]:
            env.set_task_status(TaskStatus.BEGIN)
        elif env.task_status == TaskStatus.BEGIN and joystick_state["leftHand"]["gripButtonPressed"]:
            env.set_task_status(TaskStatus.FAILURE)
        elif env.task_status == TaskStatus.BEGIN and joystick_state["rightHand"]["gripButtonPressed"]:
            env.set_task_status(TaskStatus.SUCCESS)

    def _set_head_ctrl(self, env : OpenLoongEnv, joystick_state) -> None:
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
            env.ctrl[self._neck_actuator_id[i]] = ctrl[i]

        # 更新头部位置
        self.set_neck_mocap(env, mocap_neck_xpos, self._mocap_neck_xquat)
 
    def _processe_pico_joystick_move(self, env : OpenLoongEnv) -> tuple:
        if self._pico_joystick.is_reset_pos():
            self._pico_joystick.set_reset_pos(False)
            # self._set_init_state()
            self._reset_gripper()
            self._reset_neck_mocap(env)

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

    def on_playback_action(self, env : OpenLoongEnv, action) -> None:
        assert(len(action) == self.action_range.shape[0])
        
        self._grasp_value_l = action[13]
        self._set_l_hand_actuator_ctrl(env, self._grasp_value_l)
        self._grasp_value_r = action[27]
        self._set_r_hand_actuator_ctrl(env, self._grasp_value_r)

        if env.action_type == ActionType.END_EFFECTOR:
            action_l = action[:6]
            self._l_controller.set_goal(action_l)
            ctrl = self._l_controller.run_controller()
            self._set_arm_ctrl(env, self._l_arm_actuator_id, ctrl)

            action_r = action[14:20]
            self._r_controller.set_goal(action_r)
            ctrl = self._r_controller.run_controller()
            self._set_arm_ctrl(env, self._r_arm_actuator_id, ctrl)

        elif env.action_type == ActionType.JOINT_POS:
            l_arm_joint_action = action[6:13]
            self._set_arm_ctrl(env, self._l_arm_actuator_id, l_arm_joint_action)

            r_arm_joint_action = action[20:27]
            self._set_arm_ctrl(env, self._r_arm_actuator_id, r_arm_joint_action)
        else:
            raise ValueError("Invalid action type: ", self._action_type)
        
        return

    def _set_gripper_ctrl_r(self, env: OpenLoongEnv, joystick_state) -> None:
        raise NotImplementedError("This method should be implemented in the derived class.")
    
    def _set_gripper_ctrl_l(self, env: OpenLoongEnv, joystick_state) -> None:
        raise NotImplementedError("This method should be implemented in the derived class.")

    def update_force_feedback(self, env: OpenLoongEnv) -> None:
        raise NotImplementedError("This method should be implemented in the derived class.")

    def _set_l_hand_actuator_ctrl(self, env : OpenLoongEnv, offset_rate) -> None:
        raise NotImplementedError("This method should be implemented in the derived class.")
    
    def _set_r_hand_actuator_ctrl(self, env : OpenLoongEnv, offset_rate) -> None:
        raise NotImplementedError("This method should be implemented in the derived class.")    

class OpenLoongHandAgent(OpenLoongAgent):
    def __init__(self, env: OpenLoongEnv, id: int, name: str) -> None:
        super().__init__(env, id, name)
        
        self.init_agent(env, id)
        super().init_agent(env, id)

        
    def init_agent(self, env: OpenLoongEnv, id: int):
        # print("arm_actuator_id: ", self._l_arm_actuator_id)
        self._l_hand_moto_names = [env.actuator("M_zbll_J1", id), env.actuator("M_zbll_J2", id), env.actuator("M_zbll_J3", id)
                                    ,env.actuator("M_zbll_J4", id),env.actuator("M_zbll_J5", id),env.actuator("M_zbll_J6", id),
                                    env.actuator("M_zbll_J7", id),env.actuator("M_zbll_J8", id),env.actuator("M_zbll_J9", id),
                                    env.actuator("M_zbll_J10", id),env.actuator("M_zbll_J11", id)]
        self._l_hand_actuator_id = [env.model.actuator_name2id(actuator_name) for actuator_name in self._l_hand_moto_names]        
        self._l_hand_body_names = [env.body("zbll_Link1"), env.body("zbll_Link2"), env.body("zbll_Link3"),
                                   env.body("zbll_Link4"), env.body("zbll_Link5"), env.body("zbll_Link6"), 
                                   env.body("zbll_Link7"), env.body("zbll_Link8"), env.body("zbll_Link9"),
                                   env.body("zbll_Link10"), env.body("zbll_Link11")]
        self._l_hand_gemo_ids = []
        for geom_info in env.model.get_geom_dict().values():
            if geom_info["BodyName"] in self._l_hand_body_names:
                self._l_hand_gemo_ids.append(geom_info["GeomId"])


        self._r_hand_motor_names = [env.actuator("M_zbr_J1", id), env.actuator("M_zbr_J2", id), env.actuator("M_zbr_J3", id)
                                   ,env.actuator("M_zbr_J4", id),env.actuator("M_zbr_J5", id),env.actuator("M_zbr_J6", id),
                                   env.actuator("M_zbr_J7", id),env.actuator("M_zbr_J8", id),env.actuator("M_zbr_J9", id),
                                   env.actuator("M_zbr_J10", id),env.actuator("M_zbr_J11", id)]
        self._r_hand_actuator_id = [env.model.actuator_name2id(actuator_name) for actuator_name in self._r_hand_motor_names]
        self._r_hand_body_names = [env.body("zbr_Link1"), env.body("zbr_Link2"), env.body("zbr_Link3"),
                                   env.body("zbr_Link4"), env.body("zbr_Link5"), env.body("zbr_Link6"), 
                                   env.body("zbr_Link7"), env.body("zbr_Link8"), env.body("zbr_Link9"),
                                   env.body("zbr_Link10"), env.body("zbr_Link11")]
        self._r_hand_gemo_ids = []
        for geom_info in env.model.get_geom_dict().values():
            if geom_info["BodyName"] in self._r_hand_body_names:
                self._r_hand_gemo_ids.append(geom_info["GeomId"])            


    def _set_gripper_ctrl_l(self, env : OpenLoongEnv, joystick_state) -> None:
        # Press secondary button to set gripper minimal value
        offset_rate_clip_adjust_rate = 0.1  # 10% per second
        if joystick_state["leftHand"]["secondaryButtonPressed"]:
            self._l_gripper_offset_rate_clip -= offset_rate_clip_adjust_rate * env.dt    
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
        self._set_l_hand_actuator_ctrl(env, offset_rate)
        self._grasp_value_l = offset_rate
            

    def _set_gripper_ctrl_r(self, env : OpenLoongEnv, joystick_state) -> None:
        # Press secondary button to set gripper minimal value
        offset_rate_clip_adjust_rate = 0.1
        if joystick_state["rightHand"]["secondaryButtonPressed"]:
            self._r_gripper_offset_rate_clip -= offset_rate_clip_adjust_rate * env.dt
            self._r_gripper_offset_rate_clip = np.clip(self._r_gripper_offset_rate_clip, -1, 0)
        elif joystick_state["rightHand"]["primaryButtonPressed"]:
            self._r_gripper_offset_rate_clip = 0

        # Adjust sensitivity using an exponential function
        trigger_value = joystick_state["rightHand"]["triggerValue"]  # Value in [0, 1]
        k = np.e  # Adjust 'k' to change the curvature of the exponential function
        adjusted_value = (np.exp(k * trigger_value) - 1) / (np.exp(k) - 1)  # Maps input from [0, 1] to [0, 1]
        offset_rate = -adjusted_value
        offset_rate = np.clip(offset_rate, -1, self._r_gripper_offset_rate_clip)
        self._set_r_hand_actuator_ctrl(env, offset_rate)
        self._grasp_value_r = offset_rate
                
    def _set_r_hand_actuator_ctrl(self, env : OpenLoongEnv, offset_rate) -> None:
        for actuator_id in self._r_hand_actuator_id:
            actuator_name = env.model.actuator_id2name(actuator_id)
            if actuator_name == env.actuator("M_zbr_J2", self.id) or actuator_name == env.actuator("M_zbr_J3", self.id):
                offset_dir = -1
            else:
                offset_dir = 1

            abs_ctrlrange = self._all_ctrlrange[actuator_id][1] - self._all_ctrlrange[actuator_id][0]
            env.ctrl[actuator_id] = offset_rate * offset_dir * abs_ctrlrange
            env.ctrl[actuator_id] = np.clip(
                env.ctrl[actuator_id],
                self._all_ctrlrange[actuator_id][0],
                self._all_ctrlrange[actuator_id][1])
            
    def _set_l_hand_actuator_ctrl(self, env : OpenLoongEnv, offset_rate) -> None:
        for actuator_id in self._l_hand_actuator_id:
            actuator_name = env.model.actuator_id2name(actuator_id)
            if actuator_name == env.actuator("M_zbll_J3", self.id):
                offset_dir = -1
            else:
                offset_dir = 1

            abs_ctrlrange = self._all_ctrlrange[actuator_id][1] - self._all_ctrlrange[actuator_id][0]
            env.ctrl[actuator_id] = offset_rate * offset_dir * abs_ctrlrange
            env.ctrl[actuator_id] = np.clip(
                env.ctrl[actuator_id],
                self._all_ctrlrange[actuator_id][0],
                self._all_ctrlrange[actuator_id][1])            
            
    def update_force_feedback(self, env: OpenLoongEnv) -> None:
        if self._pico_joystick is not None:
            r_hand_force = self._query_hand_force(env, self._r_hand_gemo_ids)
            l_hand_force = self._query_hand_force(env, self._l_hand_gemo_ids)
            self._pico_joystick.send_force_message(l_hand_force, r_hand_force)            
            

    def _query_hand_force(self, env: OpenLoongEnv, hand_geom_ids):
        contact_simple_list = env.query_contact_simple()
        contact_force_query_ids = []
        for contact_simple in contact_simple_list:
            if contact_simple["Geom1"] in hand_geom_ids:
                contact_force_query_ids.append(contact_simple["ID"])
            if contact_simple["Geom2"] in hand_geom_ids:
                contact_force_query_ids.append(contact_simple["ID"])

        contact_force_dict = env.query_contact_force(contact_force_query_ids)
        compose_force = 0
        for force in contact_force_dict.values():
            compose_force += np.linalg.norm(force[:3])
        return compose_force            