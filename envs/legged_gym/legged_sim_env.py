import numpy as np
from typing import Optional
from orca_gym.devices.pico_joytsick import PicoJoystick
from orca_gym.environment import OrcaGymLocalEnv
from collections import defaultdict

from envs.legged_gym.legged_robot import LeggedRobot

from orca_gym.log.orca_log import get_orca_logger
_logger = get_orca_logger()


class ControlDevice:
    """
    Enum class for control
    """
    XBOX = "xbox"
    KEYBOARD = "keyboard"
    


class LeggedSimEnv(OrcaGymLocalEnv):
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
        action_skip: int,
        max_episode_steps: int,
        ctrl_device: ControlDevice,
        control_freq: int,
        robot_config: dict,
        legged_obs_config: dict,
        curriculum_config: dict,
        legged_env_config: dict,
        **kwargs,
    ):

        self._render_mode = "human"
        self.env_id = env_id
        self.max_episode_steps = max_episode_steps
        self._action_skip = action_skip
        self._ctrl_device = ctrl_device
        self._control_freq = control_freq     
        self._keyboard_addr = orcagym_addr

        self._robot_config = robot_config
        self._legged_obs_config = legged_obs_config
        self._curriculum_config = curriculum_config
        self._legged_env_config = legged_env_config

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

        self.height_map = np.zeros((2000, 2000))  # default height map, 200m x 200m

        self.ctrl = np.zeros(self.nu)
        self.mj_forward()   # Do this before initializing the controller, joints, sites, bodies, to avoid NaN error.

        self._agents : dict[str, AgentBase] = {}
        for id, agent_name in enumerate(self._agent_names):
            if agent_name.startswith("go2"):
                self._agents[agent_name] = Go2Agent(self, id=id, name=agent_name)
            elif agent_name.startswith("Lite3"):
                self._agents[agent_name] = Lite3Agent(self, id=id, name=agent_name)
            elif agent_name.startswith("g1"):
                self._agents[agent_name] = G1Agent(self, id=id, name=agent_name)
        
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
        _logger.info(f"env action range:  {env_action_range}")
        # 归一化到 [-1, 1]区间
        # scaled_action_range = np.concatenate([[[-1.0, 1.0]] * len(env_action_range)], dtype=np.float32)
        # print("Scaled action range: ", scaled_action_range)
        self.action_space = self.generate_action_space(env_action_range)


    def get_env_version(self):
        return LeggedSimEnv.ENV_VERSION

    @property
    def ctrl_device(self) -> ControlDevice:  
        return self._ctrl_device
    
    @property
    def control_freq(self) -> int:
        return self._control_freq
    
    @property
    def joystick(self) -> Optional[PicoJoystick]:
        if self._ctrl_device == ControlDevice.XBOX:
            return self._joystick
    

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
        [agent.set_joint_neutral(self) for agent in self._agents.values()]

        self.ctrl = np.zeros(self.nu)       
        self.set_ctrl(self.ctrl)
        self.mj_forward()

    def _is_success(self) -> bool:
        return False
    
    def _is_truncated(self) -> bool:
        return False

    def step(self, action) -> tuple:
        
        # print("runmode: ", self._run_mode, "no_scaled_action: ", noscaled_action, "scaled_action: ", scaled_action, "ctrl: ", ctrl)
        agent_action = self._split_agent_action(action)
        [agent.on_step(self, agent_action[agent.name]) for agent in self._agents.values()]
        
        for _ in range(self._action_skip):
            torque_ctrl_list = [
                agent.agent.compute_torques(self.data.qpos, self.data.qvel) for agent in self._agents.values()
            ]
            # print("--------------------------------")
            # print("action: ", action)
            # print("agent_action: ", agent_action)
            # print("torque_ctrl_list: ", torque_ctrl_list)
            [agent.set_acatuator_ctrl(self, torque_ctrl_list[i]) for i, agent in enumerate(self._agents.values())]

            # step the simulation with original action space
            self.do_simulation(self.ctrl, self.frame_skip)

        obs = self._get_obs().copy()

        info = {}
        terminated = self._is_success()
        truncated = self._is_truncated()
        reward = 0

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


    def get_state(self) -> dict:
        state = {
            "time": self.data.time,
            "qpos": self.data.qpos.copy(),
            "qvel": self.data.qvel.copy(),
            "qacc": self.data.qacc.copy(),
            "ctrl": self.ctrl.copy(),
        }
        return state
    

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
        [agent.on_close() for agent in self._agents.values()]

    def get_observation(self, obs=None):
        if obs is not None:
            return obs
        else:
            return self._get_obs().copy()


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

    def setup_command(self, command_dict : dict) -> None:
        [agent.agent.setup_command(command_dict) for agent in self._agents.values()]

    def setup_base_friction(self, base_friction: float) -> None:
        geom_dict = self.model.get_geom_dict()
        geom_friction_dict = self._agents[self._agent_names[0]].agent.scale_foot_friction(geom_dict, base_friction)
        self.set_geom_friction(geom_friction_dict)

        _logger.info(f"Setup base friction:  {geom_friction_dict}")



## --------------------------------            
## Agent Class
## --------------------------------            
class AgentBase:
    def __init__(self, env: LeggedSimEnv, id: int, name: str) -> None:
        self._id = id
        self._name = name

    @property
    def id(self) -> int:
        return self._id
    
    @property
    def name(self) -> str:
        return self._name


    @property
    def action_range(self) -> np.ndarray:
        return self.agent._action_range

    def init_agent(self, env: LeggedSimEnv, id: int):
        self._legged_agent = LeggedRobot(
            env_id = env.env_id,
            agent_name=self.name,
            task="flat_terrain",
            max_episode_steps=env.max_episode_steps,
            dt=env.dt,
            robot_config=env._robot_config,
            legged_obs_config=env._legged_obs_config,
            curriculum_config=env._curriculum_config,
            is_subenv=False,
        )
        
        self.dt = env.dt
        all_actuator = env.model.get_actuator_dict()
        all_joint = env.model.get_joint_dict()
        self.agent.init_ctrl_info(all_actuator, all_joint)
        
        init_joint_qpos = env.query_joint_qpos(self.agent.joint_names)
        init_site_pos_quat = env.query_site_pos_and_quat(self.agent.site_names)
        self.agent.set_init_state(
            joint_qpos=init_joint_qpos,
            init_site_pos_quat=init_site_pos_quat,
        )
        
        self.agent.set_action_space() 
        self.generate_action_scale_array(self._query_ctrl_info())
        self._init_playable(env)


    @property
    def agent(self) -> LeggedRobot:
        return self._legged_agent

    def on_close(self):
        pass

    def set_joint_neutral(self, env: LeggedSimEnv) -> None:
        joint_names = self.agent.joint_names
        qpos_offset, qvel_offset, qacc_offset = env.query_joint_offsets(joint_names)
        qpos_length, qvel_length, qacc_length = env.query_joint_lengths(joint_names)
        self.agent.init_joint_index(qpos_offset, qvel_offset, qacc_offset, qpos_length, qvel_length, qacc_length)        
        
        agent_joint_qpos, agent_joint_qvel = self.agent.reset(env.np_random, height_map=env.height_map)
        # print("Reset joint qpos: ", joint_qpos)
        env.set_joint_qpos(agent_joint_qpos)
        env.set_joint_qvel(agent_joint_qvel)

        env.mj_forward()
        env.update_data()      
    
    def on_step(self, env: LeggedSimEnv, action: np.ndarray) -> None:
        self.agent.step(action, update_mocap=False)

    def set_acatuator_ctrl(self, env : LeggedSimEnv, actuator_ctrl: np.ndarray) -> None:
        for i, actuator_name in enumerate(self.agent.actuator_names):
            actuator_id = env.model.actuator_name2id(actuator_name)
            env.ctrl[actuator_id] = actuator_ctrl[i]
    
    def get_obs(self, env: LeggedSimEnv) -> dict:
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

    def _query_sensor_data(self, env: LeggedSimEnv) -> dict[str, np.ndarray]:
        if self.agent._use_imu_sensor:
            return env.query_sensor_data(self.agent.sensor_names)
        else:
            return env.query_sensor_data(self.agent._foot_touch_sensor_names)



    def on_reset_model(self, env: LeggedSimEnv) -> None:
        pass
    
    def on_close(self) -> None:
        pass

    
    def generate_action_scale_array(self, ctrl_info: dict) -> np.ndarray:
        self._action_scale = next(iter(ctrl_info.values()))["action_scale"]             # shape = (1)

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
    
    def _init_playable(self, env: LeggedSimEnv) -> None:
        self.agent.init_playable()
        self.agent.player_control = True

class Lite3Agent(AgentBase):
    def __init__(self, env: LeggedSimEnv, id: int, name: str) -> None:
        super().__init__(env, id, name)
        
        self.init_agent(env, id)

class Go2Agent(AgentBase):
    def __init__(self, env: LeggedSimEnv, id: int, name: str) -> None:
        super().__init__(env, id, name)
        
        self.init_agent(env, id)


class G1Agent(AgentBase):
    def __init__(self, env: LeggedSimEnv, id: int, name: str) -> None:
        super().__init__(env, id, name)
        
        self.init_agent(env, id)