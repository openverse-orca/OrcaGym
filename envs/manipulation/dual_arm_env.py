import random
import re
import time
from colorama import Fore, Style
import numpy as np
from orca_gym.adapters.robomimic.robomimic_env import RobomimicEnv
from typing import Optional
from orca_gym.devices.pico_joytsick import PicoJoystick
from orca_gym.environment.orca_gym_env import RewardType
from orca_gym.task.scan_QR_task import ScanQRTask
from orca_gym.task.close_open_task import CloseOpenTask
from orca_gym.utils.reward_printer import RewardPrinter
from orca_gym.task.abstract_task import TaskStatus
from orca_gym.task.pick_place_task import PickPlaceTask
import importlib

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
    END_EFFECTOR_OSC = "end_effector_osc"
    END_EFFECTOR_IK = "end_effector_ik"
    JOINT_MOTOR = "joint_motor"
    JOINT_POS = "joint_pos"



robot_entries = {
    "openloong_hand_fix_base": "envs.manipulation.robots.openloong_hand_fix_base:OpenLoongHandFixBase",
    "openloong_gripper_2f85_fix_base": "envs.manipulation.robots.openloong_gripper_fix_base:OpenLoongGripperFixBase",
    "openloong_gripper_2f85_mobile_base": "envs.manipulation.robots.openloong_gripper_mobile_base:OpenLoongGripperMobileBase",
}

ANIM_SPEED = 0.1
def get_robot_entry(name: str):
    for robot_name, entry in robot_entries.items():
        if name.startswith(robot_name):
            return entry
        
    raise ValueError(f"Robot entry for {name} not found in robot_entries.")

class DualArmEnv(RobomimicEnv):
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


        self._ctrl_device = ctrl_device
        self._control_freq = control_freq

        if self._ctrl_device == ControlDevice.VR and run_mode == RunMode.TELEOPERATION:
            self._joystick = {}
            pico_joystick = []
            for i in range(len(pico_ports)):
                pico_joystick.append(PicoJoystick(int(pico_ports[i])))
            for i in range(len(agent_names)):
                # 当agent数量大于pico数量时，使用最后一个pico
                self._joystick[agent_names[i]] = pico_joystick[min(i, len(pico_joystick) - 1)]
                print("Pico Joystick",agent_names[i])
        
        self._sample_range = sample_range
        self._reward_type = reward_type
        self._setup_reward_functions(reward_type)

        self._reward_printer = RewardPrinter()

        # todo: 这里应该改为工厂函数，
        self._config = task_config_dict
        self._config['grpc_addr'] = orcagym_addr
        if self._config["type"] == "pick_and_place":
            self._task = PickPlaceTask(self._config)
        elif self._config["type"] == "scan":
            self._task = ScanQRTask(self._config)
        elif self._config["type"] == "close_open":
            self._task = CloseOpenTask(self._config)

        self._task.register_init_env_callback(self.init_env)
        kwargs["task"] = self._task
        self._teleop_counter = 0
        self._got_task = False
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

        self._agents : dict[str, AgentBase] = {}
        for id, agent_name in enumerate(self._agent_names):
            self._agents[agent_name] = self.create_agent(id, agent_name)
        
        assert len(self._agents) > 0, "At least one agent should be created."
        self._set_init_state()
        
        # Run generate_observation_space after initialization to ensure that the observation object's name is defined.
        self._set_obs_space()
        self._set_action_space()

        #add animation joint for vr teleoperation
        self.animjoint_names = []
        self.animjoint = []
        self.animjointorigpos = []
        self.lastanimtime = 0
        self.animspeed = 0
        self.havestate = False
        if self._ctrl_device == ControlDevice.VR and run_mode == RunMode.TELEOPERATION:
            #add scene animation joint for vr teleoperation
          
            jointtv = self.model.get_joint_byname("openloong_gripper_2f85_fix_base_usda_J_TV_joint")
            if jointtv:
                print("Found animation joint for teleoperation: J_TV_joint")
                
                self.animjoint_names.append("openloong_gripper_2f85_fix_base_usda_J_TV_joint")    
                self.animjointorigpos = self.gym.query_joint_qpos(self.animjoint_names)         
                self.animjoint.append(jointtv)
                print("Animation jointpos: ",  self.animjointorigpos )

           


    def init_env(self):
        self.model, self.data = self.initialize_simulation()
        self._init_ctrl()
        self.init_agents()

    def create_agent(self, id, name):
        entry = get_robot_entry(name)
        module_name, class_name = entry.rsplit(":", 1)
        module = importlib.import_module(module_name)
        class_type = getattr(module, class_name)
        agent = class_type(self, id, name)
        return agent

    def _init_ctrl(self):
        self.nu = self.model.nu
        self.nq = self.model.nq
        self.nv = self.model.nv

        self.gym.opt.iterations = 150
        self.gym.opt.noslip_tolerance = 50
        self.gym.opt.ccd_iterations = 100
        self.gym.opt.sdf_iterations = 50
        self.gym.set_opt_config()

        self.ctrl = np.zeros(self.nu)
        self.mj_forward() 

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
        return DualArmEnv.ENV_VERSION

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
    def joystick(self):
        if self._ctrl_device == ControlDevice.VR:
            return self._joystick
    
    def set_task_status(self, status):
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
        [agent.set_joint_neutral() for agent in self._agents.values()]

        self.ctrl = np.zeros(self.nu)
        for agent in self._agents.values():
            agent.set_init_ctrl()

        self.set_ctrl(self.ctrl)

        self.mj_forward()

    def _is_success(self) -> bool:
        return self._task_status == TaskStatus.SUCCESS
    
    def _is_truncated(self) -> bool:
        return self._task_status == TaskStatus.FAILURE
    
    def  step_modelanim(self, animation_time: float) -> None:
        """
        Step the simulation for a certain amount of time, without changing the control inputs.
        This is used to animate the model in the GUI.
        """
        tt = np.array([animation_time])
        passtime = animation_time -  self.lastanimtime
        self.lastanimtime =  animation_time
        animjointpos = self.gym.query_joint_qpos(self.animjoint_names)['openloong_gripper_2f85_fix_base_usda_J_TV_joint']
        #add pos change wtih passtime and speed
        if len(self._joystick) > 0:
            picotemp = self._joystick['openloong_gripper_2f85_fix_base_usda']
            keystate = picotemp.get_key_state()
            if keystate:
                if self.havestate == False:
                    self.animspeed = ANIM_SPEED
                    self.havestate = True


                if keystate["leftHand"]["primaryButtonPressed"] and keystate["leftHand"]["secondaryButtonPressed"]:
                    self.animspeed = 0
                if keystate["rightHand"]["primaryButtonPressed"] and keystate["rightHand"]["secondaryButtonPressed"]:
                    self.animspeed = ANIM_SPEED
            else:
                if self.havestate == True:
                    self.havestate = False
                    self.animspeed = 0


               

            """
            if picotemp.running == False or picotemp.get_left_gripButton():
                print("get_left_gripButton true......................")
                animspeed = 0
            if picotemp.get_right_gripButton():
                print("get_right_gripButton true......................")
                animspeed = 0.1
            """ 
        animjointpos[0] += passtime * self.animspeed

       # print("step_modelanim time: ", animjointpos  )
        if self.animjoint:
            for animjoint in self.animjoint_names:
                
                self.set_joint_qpos({animjoint: animjointpos})
        self.mj_forward()



    def step(self, action) -> tuple:
        if self._run_mode == RunMode.TELEOPERATION:
            ctrl, noscaled_action = self._teleoperation_action()
        elif self._run_mode == RunMode.POLICY_NORMALIZED:
            noscaled_action = self.denormalize_action(action, self.env_action_range_min, self.env_action_range_max)
            ctrl, noscaled_action = self._playback_action(noscaled_action)
        else:
            raise ValueError("Invalid run mode : ", self._run_mode)
        
        # print("runmode: ", self._run_mode, "no_scaled_action: ", noscaled_action, "scaled_action: ", scaled_action, "ctrl: ", ctrl)
        
        self.step_modelanim(self.data.time)
        
        if self._run_mode == RunMode.TELEOPERATION:
            [agent.update_force_feedback() for agent in self._agents.values()]

        scaled_action = self.normalize_action(noscaled_action, self.env_action_range_min, self.env_action_range_max)

        # step the simulation with original action space
        self.do_simulation(ctrl, self.frame_skip)
        time_stamp = time.time_ns()

        obs = self._get_obs().copy()

        info = {"state": self.get_state(),
                "action": scaled_action,
                "object": self._task.get_objects_info(self),  # 提取第一个对象的位置
                "goal": self._task.get_goals_info(self),
                "task_status": self._task_status,
                "language_instruction": self._task.get_language_instruction(),
                "time_step": self.data.time,
                "time_stamp": time_stamp,}
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

    # def _fill_arm_joint_pos(self, action) -> np.ndarray:
    #     agent_action = self._split_agent_action(action)
    #     return np.concatenate([agent.fill_arm_joint_pos(agent_action[agent.name]) for agent in self._agents.values()], axis=0, dtype=np.float32).flatten()

    # def _fill_arm_ctrl(self, action) -> np.ndarray:
    #     agent_action = self._split_agent_action(action)
    #     return np.concatenate([agent.fill_arm_ctrl(agent_action[agent.name]) for agent in self._agents.values()], axis=0, dtype=np.float32).flatten()


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
            agent_action.append(agent.on_teleoperation_action())

        return self.ctrl.copy(), np.concatenate(agent_action).flatten()


    def _playback_action(self, action) -> tuple:
        assert(len(action) == self.action_space.shape[0])
        
        # 将动作分配给每个agent
        agent_action = self._split_agent_action(action)
        new_agent_action = []
        for agent in self._agents.values():
            new_agent_action.append(agent.on_playback_action(agent_action[agent.name]))

        return self.ctrl.copy(), np.concatenate(new_agent_action).flatten()


    def _get_obs(self) -> dict:
        if len(self._agents) == 1:
            # Use original observation if only one agent
            return self._agents[self._agent_names[0]].get_obs()

        # 将所有的agent obs 合并到一起，其中每个agent obs key 加上前缀 agent_name，确保不重复
        # 注意：这里需要兼容 gymnasium 的 obs dict 范式，因此不引入多级字典
        # 同理多agent的action采用拼接np.array方式，不采用字典分隔
        obs = {}
        for agent in self._agents.values():
            agent_obs = agent.get_obs()
            for key in agent_obs.keys():
                agent_key = f"{agent.name}_{key}"
                if agent_key in obs:
                    raise ValueError(f"Duplicate observation key: {agent_key}")
                obs[agent_key] = agent_obs[key]
        return obs

    def init_agents(self):
        for id, agent_name in enumerate(self._agent_names):
            self._agents[agent_name].init_agent(id)

    def _debug_list_loaded_objects(self):
        all_keys = list(self.model._joint_dict.keys())

        print("=== ALL JOINT KEYS IN MODEL ===")
        for k in all_keys:
            print("   ", k)
        print("================================")

        print("[Debug] before correction:", self._task.object_joints)
        corrected = []
        for short_jn in self._task.object_joints:
            sn_tokens = short_jn.lower().split("_")
            matches = []
            for full in all_keys:
                fn_tokens = full.lower().split("_")
                # 如果 full 的末尾几个 token 刚好和 short_jn token 一致，就算匹配
                if fn_tokens[-len(sn_tokens):] == sn_tokens:
                    matches.append(full)
            if not matches:
                print(f"[Debug] joint '{short_jn}' not found → SKIP")
                continue
            full_jn = matches[0]
            
            #print(f"[Loaded OBJ JOINT] {short_jn:25s} → {full_jn}")
        self._task.object_joints = full_jn
        #print("[Debug] after correction:", self._task.object_joints)
    
    def safe_get_task(self, env):
        while True:
            # 1) 生成一次 task（包括随机摆放）
            self._task.get_task(env)
    
            objs = self._task.randomized_object_positions
            goal_body = self._task.goal_bodys[0]
    
            # 2) 拿它的轴对齐包围盒
            bbox = self.get_goal_bounding_box(goal_body)
            min_xy = bbox['min'][:2]
            max_xy = bbox['max'][:2]
    
            bad = False
            for joint_name, qpos in objs.items():
                obj_xy = qpos[:2]
                if (min_xy <= obj_xy).all() and (obj_xy <= max_xy).all():
                    bad = True
                    break
                
            if not bad:
                return  # 成功退出
    
            # 否则继续尝试
    
    def reset_model(self) -> tuple[dict, dict]:
        self._set_init_state()
        for ag in self._agents.values():
            ag.on_reset_model()

        self._task.get_task(self)
        

        self.mj_forward()
        obs = self._get_obs().copy()
        return obs, {"objects": self._task.get_objects_info(self), "goals": self._task.get_goals_info(self)}

    def replace_objects(self, objects_data):
        """
        将 demo 里记录的 objects_data 写回到仿真。
        objects_data 可能是：
          1) 结构化 numpy array：dtype=[('joint_name',U100),('position',f4,3),('orientation',f4,4)]
          2) 扁平浮点 ndarray：长度 = num_objects * 7，或 shape=(num_objects,7)
        """
        qpos_dict = {}

        arr = objects_data
        # —— 情况 A：结构化数组（走原逻辑） ——
        if isinstance(arr, np.ndarray) and arr.dtype.fields is not None:
            # arr.shape = (num_objects,)
            for entry in arr:
                name = entry['joint_name']
                pos  = entry['position']
                quat = entry['orientation']
                qpos_dict[name] = np.concatenate([pos, quat], axis=0)

        else:
            # —— 情况 B：纯数值数组 ——
            flat = np.asarray(arr, dtype=np.float32)
            # 如果是一维 (num_objects*7,)
            if flat.ndim == 1:
                flat = flat.reshape(-1, 7)
            # 如果是二维 (timesteps, num_objects*7)，取首帧
            if flat.ndim == 2 and flat.shape[0] > 1:
                flat = flat[0].reshape(-1, 7)

            # 名字列表：从上一次 reset_model 里保存的 self.objects 拿 joint_name
            # （确保在遥操作模式里执行过一次 reset_model，使 self.objects 存在）
            names = [ent['joint_name'] for ent in self.objects]

            for idx, row in enumerate(flat):
                name = names[idx]
                pos  = row[0:3]
                quat = row[3:7]
                qpos_dict[name] = np.concatenate([pos, quat], axis=0)

        self.set_joint_qpos(qpos_dict)
        # 推进仿真
        self.mj_forward()

    def get_observation(self, obs=None):
        if obs is not None:
            return obs
        else:
            return self._get_obs().copy()

    def replace_goals(self, goals_data):
        """
        将 demo 里记录的 goals 写回环境，仅做数据格式转换。

        goals_data 可能是：
          1) 结构化 numpy array（dtype 包含 joint_name, position, orientation, min, max, size）
          2) 纯浮点 ndarray：长度 = num_goals * 16，或 shape=(num_goals,16)
             对应字段顺序 [pos(3), orient(4), min(3), max(3), size(3)]
        """
        arr = goals_data

        # 1) 如果已经是结构化数组，直接 copy 给 self.goals
        if isinstance(arr, np.ndarray) and arr.dtype.fields is not None:
            self.goals = arr.copy()
            return

        # 2) 否则把它变为 (num_goals, 16) 的纯数值数组
        flat = np.asarray(arr, dtype=np.float32)
        if flat.ndim == 1:
            flat = flat.reshape(-1, 16)
        elif flat.ndim == 2 and flat.shape[0] > 1:
            # 如果是时序数据，取第一帧
            flat = flat[0].reshape(-1, 16)

        # joint_name 列表从旧的 self.goals 拿，如果第一次用请先跑一次 reset_model() 初始化它
        names = [entry['joint_name'] for entry in self.goals]

        # 3) 重建结构化数组
        goal_dtype = np.dtype([
            ('joint_name',  'U100'),
            ('position',    'f4', (3,)),
            ('orientation', 'f4', (4,)),
            ('min',         'f4', (3,)),
            ('max',         'f4', (3,)),
            ('size',        'f4', (3,))
        ])
        entries = []
        for idx, row in enumerate(flat):
            name = names[idx]
            pos  = row[ 0:3].tolist()
            quat = row[ 3:7].tolist()
            mn   = row[ 7:10].tolist()
            mx   = row[10:13].tolist()
            sz   = row[13:16].tolist()
            entries.append((name, pos, quat, mn, mx, sz))

        self.goals = np.array(entries, dtype=goal_dtype)



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


    # def set_goal_mocap(self, position, orientation) -> None:
    #     mocap_pos_and_quat_dict = {"goal_goal": {'pos': position, 'quat': orientation}}
    #     self.set_mocap_pos_and_quat(mocap_pos_and_quat_dict)


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
        if self._action_type in [ActionType.END_EFFECTOR_OSC, ActionType.JOINT_MOTOR]:
            # OSC控制器和关节电机控制需要使用电机
            return True
        elif self._action_type in [ActionType.JOINT_POS, ActionType.END_EFFECTOR_IK]:
            # 关节位置控制和逆运动学控制不需要使用电机
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
class AgentBase:
    def __init__(self, env: DualArmEnv, id: int, name: str) -> None:
        self._id = id
        self._name = name
        self._env = env
        self._action_range = np.zeros(0, dtype=np.float32)

    @property
    def id(self) -> int:
        return self._id
    
    @property
    def name(self) -> str:
        return self._name

    @property
    def action_range(self) -> np.ndarray:
        return self._action_range

    def init_agent(self, id: int) -> None:
        raise NotImplementedError("This method should be overridden by subclasses")

    def set_joint_neutral(self) -> None:
        raise NotImplementedError("This method should be overridden by subclasses")

    def update_force_feedback(self) -> None:
        raise NotImplementedError("This method should be overridden by subclasses")

    def fill_arm_joint_pos(self, action: np.ndarray) -> np.ndarray:
        raise NotImplementedError("This method should be overridden by subclasses")

    def fill_arm_ctrl(self, action: np.ndarray) -> np.ndarray:
        raise NotImplementedError("This method should be overridden by subclasses")

    def on_teleoperation_action(self) -> np.ndarray:
        raise NotImplementedError("This method should be overridden by subclasses")

    def on_playback_action(self, action: np.ndarray) -> np.ndarray:
        raise NotImplementedError("This method should be overridden by subclasses")

    def get_obs(self) -> dict:
        raise NotImplementedError("This method should be overridden by subclasses")

    def on_reset_model(self) -> None:
        raise NotImplementedError("This method should be overridden by subclasses")

    def set_init_ctrl(self) -> None:
        raise NotImplementedError("This method should be overridden by subclasses")

    def on_close(self) -> None:
        pass
    
