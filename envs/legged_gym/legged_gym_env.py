import numpy as np
from gymnasium.core import ObsType
from orca_gym.multi_agent import OrcaGymMultiAgentEnv
from orca_gym.utils import rotations
from typing import Optional, Any, SupportsFloat
from gymnasium import spaces
import datetime
from orca_gym.devices.keyboard import KeyboardInput
import gymnasium as gym

from .legged_robot import LeggedRobot
from .legged_config import LeggedEnvConfig, LeggedRobotConfig

class LeggedGymEnv(OrcaGymMultiAgentEnv):
    metadata = {'render_modes': ['human', 'none'], 'version': '0.0.1', 'render_fps': 30}

    def __init__(
        self,
        frame_skip: int,        
        orcagym_addr: str,
        agent_names: list,
        time_step: float,    
        max_episode_steps: int,
        render_mode: str,
        render_remote: bool,
        height_map_file: str,
        run_mode: str,
        env_id: str,
        task: str,
        **kwargs,
    ):
        self._init_height_map(height_map_file)
        self._run_mode = run_mode       
        
        super().__init__(
            frame_skip = frame_skip,
            orcagym_addr = orcagym_addr,
            agent_names = agent_names,
            time_step = time_step,    
            agent_engry="envs.legged_gym.legged_robot:LeggedRobot",        
            max_episode_steps = max_episode_steps,
            render_mode = render_mode,
            render_remote = render_remote,
            env_id = env_id,
            task = task,
            **kwargs,
        )
        
        self._randomize_agent_foot_friction()
        self._init_playable()
        self._reset_phy_config()
     
    @property
    def agents(self) -> list[LeggedRobot]:
        return self._agents

    def step_agents(self, action: np.ndarray, actuator_ctrl: np.ndarray) -> None:
        if self._task == "no_action":
            self._update_no_action_ctrl()
            return
                    
        # print("Step agents: ", action)
        self._update_playable()

        # 性能优化：在Env中批量更新所有agent的控制量
        if len(self.ctrl) == len(actuator_ctrl):
            self.ctrl = actuator_ctrl
        else:
            assert len(self.ctrl) > len(actuator_ctrl)
            actuator_ctrl = np.array(actuator_ctrl).reshape(len(self.agents), -1)
            for i in range(len(actuator_ctrl)):
                self.ctrl[self._ctrl_start[i]:self._ctrl_end[i]] = actuator_ctrl[i]

        # 切分action 给每个 agent
        action = action.reshape(len(self.agents), -1)

        # mocap 的作用是用来显示目标位置，不影响仿真，这里处理一下提升性能
        mocaps = {}
        joint_qvels = {}
        for i in range(len(self.agents)):
            agent : LeggedRobot = self.agents[i]
            act = action[i]

            agent.update_command(self.data.qpos)
            agent_ctrl, agent_mocap = agent.step(act, update_mocap=(self.render_mode == "human" and self.render_remote and self._run_mode != "play"))
            joint_qvel_dict = agent.push_robot(self.data.qvel)
            # self.ctrl[agent.ctrl_start : agent.ctrl_start + len(act)] = agent_ctrl
            mocaps.update(agent_mocap)
            joint_qvels.update(joint_qvel_dict)

        
        self.set_mocap_pos_and_quat(mocaps)
        self.set_joint_qvel(joint_qvels)


    def compute_reward(self, achieved_goal, desired_goal, info) -> SupportsFloat:
        assert achieved_goal.shape == desired_goal.shape
        if achieved_goal.ndim == 1:
            return self.agents[0].compute_reward(achieved_goal, desired_goal)
        elif achieved_goal.ndim == 2:
            agent_num = len(self.agents)
            rewards = np.zeros(agent_num)
            for i in range(len(achieved_goal)):
                rewards[i] = self.agents[i % agent_num].compute_reward(achieved_goal[i], desired_goal[i])
            return rewards
        else:
            raise ValueError("Unsupported achieved_goal shape")

    def get_obs(self) -> tuple[dict[str, np.ndarray], list[dict[str, np.ndarray]], np.ndarray, np.ndarray]:
        # get_obs_start = datetime.datetime.now()
        # print("query joint qpos: ", self._agent_joint_names)

        sensor_data = self.query_sensor_data(self._agent_sensor_names)
        # get_obs_sensor = (datetime.datetime.now() - get_obs_start).total_seconds() * 1000
        contact_dict = self._generate_contact_dict()
        # get_obs_contact = (datetime.datetime.now() - get_obs_start).total_seconds() * 1000
        site_pos_quat = self.query_site_pos_and_quat(self._agent_site_names)
        # get_obs_site = (datetime.datetime.now() - get_obs_start).total_seconds() * 1000

        # print("Sensor data: ", sensor_data)
        # print("Joint qpos: ", joint_qpos)

        # obs 形状： {key: np.ndarray(agent_num, agent_obs_len)}
        env_obs_list : list[dict[str, np.ndarray]] = []
        agent_obs : list[dict[str, np.ndarray]] = []
        achieved_goals = []
        desired_goals = []
        for agent in self.agents:
            obs = agent.get_obs(sensor_data, self.data.qpos, self.data.qvel, self.data.qacc, contact_dict, site_pos_quat)
            achieved_goals.append(obs["achieved_goal"])
            desired_goals.append(obs["desired_goal"])
            env_obs_list.append(obs["observation"])
            agent_obs.append(obs)

        achieved_goals = np.array(achieved_goals)
        desired_goals = np.array(desired_goals)
        env_obs = {}
        env_obs["observation"] = np.array(env_obs_list)
        env_obs["achieved_goal"] = achieved_goals
        env_obs["desired_goal"] = desired_goals

        # get_obs_process = (datetime.datetime.now() - get_obs_start).total_seconds() * 1000
        # get_obs_end = (datetime.datetime.now() - get_obs_start).total_seconds() * 1000

        # print("\tGet obs time, ",
        #         "\n\t\tsensor: ", get_obs_sensor,
        #         "\n\t\tcontact: ", get_obs_contact - get_obs_sensor,
        #         "\n\t\tsite: ", get_obs_site - get_obs_contact,
        #         "\n\t\tprocess: ", get_obs_process - get_obs_site,
        #         "\n\ttotal: ", get_obs_end)
        
        # print("Env obs: ", env_obs, "Agent obs: ", agent_obs, "Achieved goals: ", achieved_goals, "Desired goals: ", desired_goals)
        return env_obs, agent_obs, achieved_goals, desired_goals
        
    def reset_agents(self, agents : list[LeggedRobot]) -> None:
        if len(agents) == 0:
            return
        self._update_curriculum_level(agents)
        self._reset_agent_joint_qpos(agents)
        self._reset_command_indicators(agents)

    def _generate_contact_dict(self) -> dict[str, list[str]]:
        contacts = self.query_contact_simple()
        # print("Contacts: ", contacts)
        contact_dict : dict[str, list[str]] = {}
        for contact in contacts:
            body_name1 = self.model.get_geom_body_name(contact["Geom1"])
            body_name2 = self.model.get_geom_body_name(contact["Geom2"])
            if body_name1 not in contact_dict:
                contact_dict[body_name1] = []
            if body_name2 not in contact_dict:
                contact_dict[body_name2] = []
            contact_dict[body_name1].append(body_name2)
            contact_dict[body_name2].append(body_name1)

        return contact_dict

    def _randomize_agent_foot_friction(self) -> None:
        if self._run_mode == "testing" or self._run_mode == "play":
            print("Skip randomize foot friction in testing or play mode")
            return
        
        random_friction = self.np_random.uniform(0, 1.0)
        geom_friction_dict = {}
        for i in range(len(self.agents)):
            agent : LeggedRobot = self.agents[i]
            agent_geom_friction_dict = agent.randomize_foot_friction(random_friction, self.model.get_geom_dict())
            geom_friction_dict.update(agent_geom_friction_dict)

        # print("Set geom friction: ", geom_friction_dict)
        self.set_geom_friction(geom_friction_dict)
        
    def _init_height_map(self, height_map_file: str) -> None:
        if height_map_file is not None:
            try:
                self._height_map = np.load(height_map_file)
            except:
                gym.logger.warn("Height map file loading failed!, use default height map 200m x 200m")
                self._height_map = np.zeros((2000, 2000))  # default height map, 200m x 200m
        else:
            raise ValueError("Height map file is not provided")

    def _reset_agent_joint_qpos(self, agents: list[LeggedRobot]) -> None:
        joint_qpos = {}
        for agent in agents:
            agent_joint_qpos = agent.reset(self.np_random, height_map=self._height_map)
            joint_qpos.update(agent_joint_qpos)

        # print("Reset joint qpos: ", joint_qpos)
        self.set_joint_qpos(joint_qpos)
        self.mj_forward()
        self.update_data()        

    def _reset_command_indicators(self, agents: list[LeggedRobot]) -> None:
        mocap_dict = {}
        for agent in agents:
            agent_cmd_mocap = agent.reset_command_indicator(self.data.qpos)
            mocap_dict.update(agent_cmd_mocap)
            
        self.set_mocap_pos_and_quat(mocap_dict)
        
    def _update_curriculum_level(self, agents: list[LeggedRobot]) -> None:
        for agent in agents:
            agent.update_curriculum_level(self.data.qpos)
            
    
    def _init_playable(self) -> None:
        if self._run_mode != "play":
            return
        
        self._keyboard_controller = KeyboardInput()
        self._key_status = {"W": 0, "A": 0, "S": 0, "D": 0, "Space": 0, "Up": 0, "Down": 0, "LShift": 0, "RShift": 0}   
        
        for agent in self.agents:
            if agent.playable:
                self._player_agent = agent
                agent.init_playable()
                agent.player_control = True
                break
            
        if "go2" in self._player_agent.name:
            robot_config = LeggedRobotConfig["go2"]
        elif "A01B" in self._player_agent.name:
            robot_config = LeggedRobotConfig["A01B"]
            
        self._player_agent_lin_vel_x = robot_config["curriculum_commands"]["flat_plane"]["command_lin_vel_range_x"] / 3
        self._player_agent_lin_vel_y = robot_config["curriculum_commands"]["flat_plane"]["command_lin_vel_range_y"] / 3
    
    def _update_playable(self) -> None:
        if self._run_mode != "play":
            return
        
        lin_vel, turn_angel, reborn = self._update_keyboard_control()
        self._player_agent.update_playable(lin_vel, turn_angel)
        agent_cmd_mocap = self._player_agent.reset_command_indicator(self.data.qpos)
        self.set_mocap_pos_and_quat(agent_cmd_mocap)      
        
        if reborn:
            self.reset_agents([self._player_agent])
    
    def _update_keyboard_control(self) -> tuple[np.ndarray, float, bool]:
        self._keyboard_controller.update()
        key_status = self._keyboard_controller.get_state()
        lin_vel = np.zeros(3)
        turn_angel = 0.0
        reborn = False
        
        if key_status["W"] == 1:
            lin_vel[0] = self._player_agent_lin_vel_x
        if key_status["Q"] == 1:
            lin_vel[1] = self._player_agent_lin_vel_y
        if key_status["E"] == 1:
            lin_vel[1] += -self._player_agent_lin_vel_y
        if key_status["A"] == 1:
            turn_angel += np.pi / 2 * self.dt
        if key_status["D"] == 1:
            turn_angel += -np.pi / 2 * self.dt
        if self._key_status["Space"] == 0 and key_status["Space"] == 1:
            reborn = True
        if key_status["LShift"] == 1:
            lin_vel[:2] *= 3

        self._key_status = key_status.copy()
        # print("Lin vel: ", lin_vel, "Turn angel: ", turn_angel, "Reborn: ", reborn)
        
        return lin_vel, turn_angel, reborn

    def _reset_phy_config(self) -> None:
        phy_config = LeggedEnvConfig["phy_config"]
        self.gym.opt.iterations = LeggedEnvConfig[phy_config]["iterations"]
        self.gym.opt.noslip_iterations = LeggedEnvConfig[phy_config]["noslip_iterations"]
        self.gym.opt.mpr_iterations = LeggedEnvConfig[phy_config]["mpr_iterations"]
        self.gym.opt.sdf_iterations = LeggedEnvConfig[phy_config]["sdf_iterations"]
        self.gym.set_opt_config()

        print("Phy config: ", phy_config, "Iterations: ", self.gym.opt.iterations, "Noslip iterations: ", self.gym.opt.noslip_iterations, "MPR iterations: ", self.gym.opt.mpr_iterations, "SDF iterations: ", self.gym.opt.sdf_iterations)

    def _update_no_action_ctrl(self) -> None:
        ctrl = []
        for agent in self.agents:
            joint_neutral = agent.get_joint_neutral()
            if joint_neutral is not None:
                ctrl.extend(joint_neutral.values())

        self.ctrl = np.array(ctrl).flatten()
        # raise KeyError("Test no action mode")
            