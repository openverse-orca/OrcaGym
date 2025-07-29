from datetime import datetime
import sys
import numpy as np
from gymnasium.core import ObsType
from typing import Optional, Any, SupportsFloat
from gymnasium import spaces
from orca_gym.devices.xbox_joystick import XboxJoystickManager
from orca_gym.devices.pico_joytsick import PicoJoystick
from orca_gym.environment import OrcaGymLocalEnv
from scipy.spatial.transform import Rotation as R
from collections import defaultdict
import gymnasium as gym
from envs.legged_gym.legged_robot import LeggedRobot, get_legged_robot_name
from envs.legged_gym.legged_config import LeggedEnvConfig, LeggedRobotConfig
from orca_gym.devices.keyboard import KeyboardInput, KeyboardInputSourceType



class LeggedGymRLLibEnv(OrcaGymLocalEnv):
    metadata = {'render_modes': ['human', 'none'], 'version': '0.0.1', 'render_fps': 30}
    
    """
    Legged robot simulation environment.
    Robot's locomotion is controlled by ppo policy.
    User give command to the robot by keyboard or joystick.
    """
    ENV_VERSION = "1.0.0"

    def __init__(
        self,
        frame_skip: int,        
        orcagym_addr: str,
        env_id: str,
        agent_names: list,
        time_step: float,
        max_episode_steps: int,
        height_map_file: str,
        render_mode: str,
        **kwargs,
    ):

        self._render_mode = render_mode
        self.env_id = env_id
        self.max_episode_steps = max_episode_steps
        self._keyboard_addr = orcagym_addr

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

        self._init_height_map(height_map_file)

        self.ctrl = np.zeros(self.nu)
        self.mj_forward()   # Do this before initializing the controller, joints, sites, bodies, to avoid NaN error.

        agent_name = self._agent_names[0]
        if agent_name.startswith("go2"):
            self._agent = Go2Agent(self, id=0, name=agent_name)
        elif agent_name.startswith("Lite3"):
            self._agent = Lite3Agent(self, id=0, name=agent_name)
        elif agent_name.startswith("g1"):
            self._agent = G1Agent(self, id=0, name=agent_name)
        
        assert self._agent is not None, "At least one agent should be created."
        self._set_init_state()
        
        self.update_obs()
        # Run generate_observation_space after initialization to ensure that the observation object's name is defined.
        self._set_obs_space()
        self._set_action_space()

    def _init_height_map(self, height_map_file: str) -> None:
        if height_map_file is not None:
            try:
                self.height_map = np.load(height_map_file)
            except:
                gym.logger.warn("Height map file loading failed!, use default height map 200m x 200m")
                self.height_map = np.zeros((2000, 2000))  # default height map, 200m x 200m
        else:
            self.height_map = np.zeros((2000, 2000))  # default height map, 200m x 200m

    def _set_obs_space(self):
        self.observation_space = self.generate_observation_space(self._get_obs().copy())

    def _set_action_space(self):
        env_action_range = self._agent.action_range
        self.env_action_range_min = env_action_range[:, 0]
        self.env_action_range_max = env_action_range[:, 1]
        # print("env action range: ", action_range)
        # 归一化到 [-1, 1]区间
        scaled_action_range = np.concatenate([[[-1.0, 1.0]] * len(env_action_range)], dtype=np.float32)
        # print("Scaled action range: ", scaled_action_range)
        self.action_space = self.generate_action_space(scaled_action_range)


    def get_env_version(self):
        return LeggedGymRLLibEnv.ENV_VERSION


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
        self._agent.set_joint_neutral(self)

        self.ctrl = np.zeros(self.nu)       
        self.set_ctrl(self.ctrl)
        self.mj_forward()

    def _is_success(self) -> bool:
        return self._agent.is_success(self._achieved_goal, self._desired_goal)
    
    def _is_truncated(self) -> bool:
        return self._agent.truncated

    def _is_terminated(self) -> bool:
        return self._agent.is_terminated(self._achieved_goal, self._desired_goal)


    def update_obs(self):
        agent_obs_dict = self._agent.get_obs(self)
        self._obs = agent_obs_dict["observation"]
        self._achieved_goal = agent_obs_dict["achieved_goal"]
        self._desired_goal = agent_obs_dict["desired_goal"]

    def step(self, action) -> tuple:
        # print("runmode: ", self._run_mode, "no_scaled_action: ", noscaled_action, "scaled_action: ", scaled_action, "ctrl: ", ctrl)
        self._agent.on_step(self, action)
        
        # step the simulation with original action space
        self.do_simulation(self.ctrl, self.frame_skip)

        # render the environment
        self.render()

        self.update_obs()

        obs = self._get_obs().copy()

        info = {}
        terminated = self._is_terminated()
        truncated = self._is_truncated()
        reward = self.compute_reward(self._achieved_goal, self._desired_goal)

        # print(
        #     "action: ", action,
        #     "ctrl: ", self.ctrl, 
        #     # "obs: ", obs, 
        #     "reward: ", reward, 
        #     "terminated: ", terminated, 
        #     "truncated: ", truncated,
        #     "achieved_goal: ", self._achieved_goal,
        #     "desired_goal: ", self._desired_goal
        # )

        return obs, reward, terminated, truncated, info

    def compute_reward(self, achieved_goal, desired_goal) -> SupportsFloat:
        return self._agent.compute_reward(achieved_goal, desired_goal)

    def get_state(self) -> dict:
        state = {
            "time": self.data.time,
            "qpos": self.data.qpos.copy(),
            "qvel": self.data.qvel.copy(),
            "qacc": self.data.qacc.copy(),
            "ctrl": self.ctrl.copy(),
        }
        return state
    

    def _get_obs(self) -> np.ndarray:
        return self._obs

    def reset_model(self) -> tuple[dict, dict]:
        """
        Reset the environment, return observation
        """
        # self.reset_simulation()
        # print("Reset model")
        
        self._set_init_state()
        
        self._agent.on_reset_model(self)

        self.mj_forward()

        obs = self._get_obs().copy()
        info = {}
        return obs, info

    def get_observation(self, obs=None):
        if obs is not None:
            return obs
        else:
            return self._get_obs().copy()


    def close(self):
        self._agent.on_close()


    def generate_contact_dict(self) -> dict[str, set[str]]:
        contacts = self.query_contact_simple()

        contact_dict: dict[str, set[str]] = defaultdict(set)
        for contact in contacts:
            body_name1 = self.model.get_geom_body_name(contact["Geom1"])
            body_name2 = self.model.get_geom_body_name(contact["Geom2"])
            contact_dict[body_name1].add(body_name2)
            contact_dict[body_name2].add(body_name1)

        # print("Contact dict: ", contact_dict)

        return contact_dict



## --------------------------------            
## Agent Class
## --------------------------------            
class AgentBase:
    def __init__(self, env: LeggedGymRLLibEnv, id: int, name: str) -> None:
        self._id = id
        self._name = name
        self._config_name = get_legged_robot_name(name)

    @property
    def id(self) -> int:
        return self._id
    
    @property
    def name(self) -> str:
        return self._name
    
    @property
    def config_name(self) -> str:
        return self._config_name

    @property
    def action_range(self) -> np.ndarray:
        return self._action_range

    def init_agent(self, env: LeggedGymRLLibEnv, id: int):
        self._legged_agent = LeggedRobot(
            env_id = env.env_id,
            agent_name=self.name,
            task="follow_command",
            max_episode_steps=env.max_episode_steps,
            dt=env.dt,
        )
        
        self.dt = env.dt
        all_actuator = env.model.get_actuator_dict()
        self.agent.init_ctrl_info(all_actuator)
        
        init_joint_qpos = env.query_joint_qpos(self.agent.joint_names)
        init_site_pos_quat = env.query_site_pos_and_quat(self.agent.site_names)
        self.agent.set_init_state(
            joint_qpos=init_joint_qpos,
            init_site_pos_quat=init_site_pos_quat,
        )
        
        action_size = self.agent.get_action_size()
        self._action_range = np.array([[-3.0, 3.0]] * action_size, dtype=np.float32)
        action_space = spaces.Box(
            low=self._action_range[:, 0],
            high=self._action_range[:, 1],
            dtype=np.float32,
            shape=(action_size, ),
        )
        print("Action space: ", action_space)
        self.agent.set_action_space(action_space) 
        self.generate_action_scale_array(self._query_ctrl_info())

    @property
    def agent(self) -> LeggedRobot:
        return self._legged_agent

    def on_close(self):
        pass

    def set_joint_neutral(self, env: LeggedGymRLLibEnv) -> None:
        joint_names = self.agent.joint_names
        qpos_offset, qvel_offset, qacc_offset = env.query_joint_offsets(joint_names)
        qpos_length, qvel_length, qacc_length = env.query_joint_lengths(joint_names)
        self.agent.init_joint_index(qpos_offset, qvel_offset, qacc_offset, qpos_length, qvel_length, qacc_length)        
        
        agent_joint_qpos, agent_joint_qvel = self.agent.reset(env.np_random, height_map=env.height_map)
        # print("Reset joint qpos: ", joint_qpos)
        env.set_joint_qpos(agent_joint_qpos)
        env.set_joint_qvel(agent_joint_qvel)

        agent_cmd_mocap = self.agent.reset_command_indicator(env.data.qpos)    
        env.set_mocap_pos_and_quat(agent_cmd_mocap)

        env.mj_forward()
        env.update_data()      
    
    def on_step(self, env: LeggedGymRLLibEnv, action: np.ndarray) -> None:
        # print("Step agents: ", action)

        actuator_ctrl = self._action2ctrl(action)
        self.set_acatuator_ctrl(env, actuator_ctrl)
        

        self.agent.update_command(env.data.qpos)
        agent_ctrl, agent_mocap = self.agent.step(action, update_mocap=(env.render_mode == "human"))
        joint_qvel_dict = self.agent.push_robot(env.data.qvel)
        if len(joint_qvel_dict) > 0:
            env.set_joint_qvel(joint_qvel_dict)
        # self.ctrl[agent.ctrl_start : agent.ctrl_start + len(act)] = agent_ctrl

        if env.render_mode == "human":
            env.set_mocap_pos_and_quat(agent_mocap)

    def set_acatuator_ctrl(self, env : LeggedGymRLLibEnv, actuator_ctrl: np.ndarray) -> None:
        env.ctrl = actuator_ctrl.copy()
        # for i, actuator_name in enumerate(self.agent.actuator_names):
        #     actuator_id = env.model.actuator_name2id(actuator_name)
        #     env.ctrl[actuator_id] = actuator_ctrl[i]
    
    def get_obs(self, env: LeggedGymRLLibEnv) -> dict:
        # get_obs_start = datetime.datetime.now()
        # print("query joint qpos: ", self._agent_joint_names)
        sensor_data = self._query_sensor_data(env)
        # get_obs_sensor = (datetime.datetime.now() - get_obs_start).total_seconds() * 1000
        contact_dict = env.generate_contact_dict()
        # get_obs_contact = (datetime.datetime.now() - get_obs_start).total_seconds() * 1000
        site_pos_quat = env.query_site_pos_and_quat(self.agent.site_names)
        # get_obs_site = (datetime.datetime.now() - get_obs_start).total_seconds() * 1000

        scaled_obs = self.agent.get_obs(sensor_data, env.data.qpos, env.data.qvel, env.data.qacc, contact_dict, site_pos_quat, env.height_map)
        return scaled_obs

    def _query_sensor_data(self, env: LeggedGymRLLibEnv) -> dict[str, np.ndarray]:
        if self.agent._use_imu_sensor:
            return env.query_sensor_data(self.agent.sensor_names)
        else:
            return env.query_sensor_data(self.agent._foot_touch_sensor_names)

    def _action2ctrl(self, action: np.ndarray) -> np.ndarray:
        # 缩放后的 action
        if self._action_scale_mask is None:
            scaled_action = action * self._action_scale
        else:
            scaled_action = action * self._action_scale * self._action_scale_mask

        # 限制 scaled_action 在有效范围内
        clipped_action = np.clip(scaled_action, self._action_space_range[0], self._action_space_range[1])

        # 批量计算插值
        ctrl_delta = (
            self._ctrl_delta_range[:, 0] +  # fp1
            (self._ctrl_delta_range[:, 1] - self._ctrl_delta_range[:, 0]) *  # (fp2 - fp1)
            (clipped_action - self._action_space_range[0]) /  # (x - xp1)
            (self._action_space_range[1] - self._action_space_range[0])  # (xp2 - xp1)
        )

        actuator_ctrl = self._neutral_joint_values + ctrl_delta
        
        return actuator_ctrl    

    def on_reset_model(self, env: LeggedGymRLLibEnv) -> None:
        pass
    
    def on_close(self) -> None:
        pass

    
    def generate_action_scale_array(self, ctrl_info: dict) -> np.ndarray:
        self._action_scale = next(iter(ctrl_info.values()))["action_scale"]             # shape = (1)
        self._action_space_range = next(iter(ctrl_info.values()))["action_space_range"] # shape = (2)

        if next(iter(ctrl_info.values()))["action_scale_mask"] is None:
            self._action_scale_mask = None
        else:
            self._action_scale_mask = np.array([ctrl["action_scale_mask"] for key, ctrl in ctrl_info.items()]).flatten() # shape = (agent_num x actor_num)
            
        self._ctrl_delta_range = np.array([ctrl["ctrl_delta_range"] for key, ctrl in ctrl_info.items()]).reshape(-1, 2)  # shape = (agent_num x actor_num, 2)
        self._neutral_joint_values = np.array([ctrl["neutral_joint_values"] for key, ctrl in ctrl_info.items()]).reshape(-1) # shape = (agent_num x actor_num)
    
    def _query_ctrl_info(self):
        ctrl_info = {}
        ctrl_info[self.agent.name] = self.agent.get_ctrl_info()
        # # store ctrl_info to pkl file, whichn is used for 2real; commented otherwise
        # import pickle
        # with open(f"ctrl_info.pkl", "wb") as f:
        #     pickle.dump(ctrl_info, f)
        # exit()
        return ctrl_info    

    @property
    def truncated(self):
        return self.agent.truncated

    def compute_reward(self, achieved_goal, desired_goal) -> SupportsFloat:
        return self.agent.compute_reward(achieved_goal, desired_goal)

    def is_success(self, achieved_goal, desired_goal) -> np.float32:
        return self.agent.is_success(achieved_goal, desired_goal)

    def is_terminated(self, achieved_goal, desired_goal) -> bool:
        return self.agent.is_terminated(achieved_goal, desired_goal)

        
class Lite3Agent(AgentBase):
    def __init__(self, env: LeggedGymRLLibEnv, id: int, name: str) -> None:
        super().__init__(env, id, name)
        
        self.init_agent(env, id)

class Go2Agent(AgentBase):
    def __init__(self, env: LeggedGymRLLibEnv, id: int, name: str) -> None:
        super().__init__(env, id, name)
        
        self.init_agent(env, id)


class G1Agent(AgentBase):
    def __init__(self, env: LeggedGymRLLibEnv, id: int, name: str) -> None:
        super().__init__(env, id, name)
        
        self.init_agent(env, id)