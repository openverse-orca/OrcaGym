import numpy as np
from gymnasium.core import ObsType
from orca_gym.multi_agent import OrcaGymMultiAgentEnv
from orca_gym.utils import rotations
from typing import Optional, Any, SupportsFloat
from gymnasium import spaces
import datetime
from orca_gym.devices.keyboard import KeyboardInput
import gymnasium as gym
import time
from collections import defaultdict

import requests
from examples.vln.imgrec import RecAction
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
        is_subenv: bool,
        height_map_file: str,
        run_mode: str,
        env_id: str,
        task: str,
        port:int = 15632, 
        ip:str="localhost",
        **kwargs,
    ):
        self._init_height_map(height_map_file)
        self._run_mode = run_mode       
        
        if self._run_mode == "nav":
            self.url = "http://"+ip+f":{port}/posyaw"  
            self.rec_action = RecAction(ip=ip)
        print("------------------------------")
        print("ip:", ip, "port:", port)
        
        super().__init__(
            frame_skip = frame_skip,
            orcagym_addr = orcagym_addr,
            agent_names = agent_names,
            time_step = time_step,    
            agent_engry="envs.legged_gym.legged_robot:LeggedRobot",        
            max_episode_steps = max_episode_steps,
            render_mode = render_mode,
            is_subenv = is_subenv,
            env_id = env_id,
            task = task,
            **kwargs,
        )

        self._randomize_agent_foot_friction()
        self._add_randomized_weight()
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
            agent_ctrl, agent_mocap = agent.step(act, update_mocap=(self.render_mode == "human" and not self.is_subenv and self._run_mode != "play"))
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

        sensor_data = self._query_sensor_data()
        # get_obs_sensor = (datetime.datetime.now() - get_obs_start).total_seconds() * 1000
        contact_dict = self._generate_contact_dict()
        # get_obs_contact = (datetime.datetime.now() - get_obs_start).total_seconds() * 1000
        site_pos_quat = self._query_site_pos_and_quat()
        # get_obs_site = (datetime.datetime.now() - get_obs_start).total_seconds() * 1000

        # print("Sensor data: ", sensor_data)
        # print("Joint qpos: ", joint_qpos)

        # obs 形状： {key: np.ndarray(agent_num, agent_obs_len)}
        env_obs_list : list[dict[str, np.ndarray]] = []
        agent_obs : list[dict[str, np.ndarray]] = []
        achieved_goals = []
        desired_goals = []
        for agent in self.agents:
            obs = agent.get_obs(sensor_data, self.data.qpos, self.data.qvel, self.data.qacc, contact_dict, site_pos_quat, self._height_map)
            achieved_goals.append(obs["achieved_goal"])
            desired_goals.append(obs["desired_goal"])
            env_obs_list.append(obs["observation"])
            agent_obs.append(obs)

        if self._run_mode == "nav":
            infofeedback = self.gym.query_body_xpos_xmat_xquat([self.body("base")])
            quat = infofeedback['go2_000_base']['Quat']
            pos = infofeedback['go2_000_base']['Pos']

            mat = infofeedback['go2_000_base']['Mat']
            mat = mat.reshape(3, 3)
            # 将mat和pos结合成4x4的矩阵
            mat = np.concatenate((mat, pos.reshape(3, 1)), axis=1)
            mat = np.concatenate((mat, np.array([[0, 0, 0, 1]])), axis=0)
            # pos_noise = np.random.normal(0, 0.05, size=pos.shape)
            # pos += pos_noise
            w, x, y, z = quat

            # 计算yaw角 （ZYX顺序）
            numerator = 2 * (w * z + x * y)
            denominator = 1 - 2 * (y**2 + z**2)
            yaw_radians = np.arctan2(numerator, denominator)

            mocap_quat = np.array([np.cos(yaw_radians / 2), 0, 0, np.sin(yaw_radians / 2)])
            # mocap_pos = np.array([pos[0], pos[1], 0.7])
            mocap_pos = (mat @np.array([0.325, 0, 0.4, 1])).flatten()[:3]
            mocap_pos[2] = 0.43
            mocap_pos_and_quat_dict = {self.mocap("mocap"): {'pos': mocap_pos, 'quat': mocap_quat}}
            self.set_mocap_pos_and_quat(mocap_pos_and_quat_dict)


            # 转换为度数
            yaw_degrees = np.degrees(yaw_radians)
            # print("Mat: ", mat)
            # print("Yaw :", yaw_degrees, "Pos: ", pos)
            # print("------------")
            # print(infofeedback['go2_000_base']['Pos'])
            # print(infofeedback['go2_000_base']['Mat'].dtype)
            # print(infofeedback['go2_000_base']['Quat'].dtype)
            # print("------------")
                
            data = {
                "pos": pos.tolist(),
                "yaw": yaw_degrees.tolist(),
                "mat": mat.flatten().tolist()
            }
            try:
                response = requests.post(self.url, json=data)
            except Exception as e:
                pass
                
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

    def _generate_contact_dict(self) -> dict[str, set[str]]:
        contacts = self.query_contact_simple()

        contact_dict: dict[str, set[str]] = defaultdict(set)
        for contact in contacts:
            body_name1 = self.model.get_geom_body_name(contact["Geom1"])
            body_name2 = self.model.get_geom_body_name(contact["Geom2"])
            contact_dict[body_name1].add(body_name2)
            contact_dict[body_name2].add(body_name1)

        # print("Contact dict: ", contact_dict)

        return contact_dict

    def _randomize_agent_foot_friction(self) -> None:
        if self._run_mode == "testing" or self._run_mode == "play" or self._run_mode == "nav":
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

    def _add_randomized_weight(self) -> None:
        print("Add randomized weight")
        if self._run_mode == "testing" or self._run_mode == "play":
            print("Skip randomized weight load in testing or play mode")
            return   

        pos_scale = 0.01
        weight_load_dict = {}
        all_joint_dict = self.model.get_joint_dict()        
        for i in range(len(self.agents)):
            agent : LeggedRobot = self.agents[i]
            random_weight = self.np_random.uniform(agent.added_mass_range[0], agent.added_mass_range[1])
            random_weight_pos = [self.np_random.uniform(-pos_scale, pos_scale), self.np_random.uniform(-pos_scale, pos_scale), 0]

            base_body_id = all_joint_dict[agent.base_joint_name]["BodyID"]
            weight_load_tmp = {base_body_id : {"weight": random_weight, "pos": random_weight_pos}}
            weight_load_dict.update(weight_load_tmp)

        # print("Set weight load: ", weight_load_dict)
        self.add_extra_weight(weight_load_dict)
        
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
        joint_qvel = {}
        for agent in agents:
            agent_joint_qpos, agent_joint_qvel = agent.reset(self.np_random, height_map=self._height_map)
            joint_qpos.update(agent_joint_qpos)
            joint_qvel.update(agent_joint_qvel)

        # print("Reset joint qpos: ", joint_qpos)
        self.set_joint_qpos(joint_qpos)
        self.set_joint_qvel(joint_qvel)
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
        if self._run_mode != "play" and self._run_mode != "nav":
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
        elif "AzureLoong" in self._player_agent.name:
            robot_config = LeggedRobotConfig["AzureLoong"]
        elif "Lite3" in self._player_agent.name:
            robot_config = LeggedRobotConfig["Lite3"]
            
        self._player_agent_lin_vel_x = np.array(robot_config["curriculum_commands"]["flat_plane"]["command_lin_vel_range_x"]) / 2
        self._player_agent_lin_vel_y = np.array(robot_config["curriculum_commands"]["flat_plane"]["command_lin_vel_range_y"]) / 2
    
    def _update_playable(self) -> None:
        if self._run_mode != "play" and self._run_mode != "nav":
            return
        
        lin_vel, turn_angel, reborn = self._update_keyboard_control()
        if self._run_mode == "nav":
            # lin_vel=lin_vel/1.3
            # turn_angel=turn_angel/2
            step = self.rec_action.get_step()
            mode = self.rec_action.get_mode()
            action = self.rec_action.get_action()
            # print(f"step:{step} | mode:{mode} | action:{action} ")
            if lin_vel.any() == 0.0 and turn_angel == 0.0:
                # mode = initialize / explore / navigate
                if mode == "initialize" and self.rec_action.trigger:
                    self.rec_action.trigger = False
                    # turn_angel += np.pi / 2 * self.dt
                    turn_angel += np.deg2rad(15)
                    # print("action: ", action,mode)

                if (mode == "navigate" or mode == "explore") :
                    self.rec_action.trigger = False
                    if action is not None:
                        change_angel = 15
                        if action == 1:#前進
                            # 在realtimeGI时候，处以3
                            lin_vel[0] = self._player_agent_lin_vel_x[1]/1.6
                        elif action == 3:#右轉
                            # 在realtimeGI时候，除以12
                            turn_angel -= np.pi / 10 * self.dt
                            # turn_angel -= np.deg2rad(change_angel)
                        elif action == 2:#左轉
                            turn_angel += np.pi / 10 * self.dt
                            # turn_angel += np.deg2rad(change_angel)
                    # print("action: ", action,mode)
        # print("Lin vel: ", lin_vel, "Turn angel: ", turn_angel, "Reborn: ", reborn)

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
            lin_vel[0] = self._player_agent_lin_vel_x[1]
        if key_status["S"] == 1:
            lin_vel[0] = self._player_agent_lin_vel_x[0]
        if key_status["Q"] == 1:
            lin_vel[1] = self._player_agent_lin_vel_y[1]
        if key_status["E"] == 1:
            lin_vel[1] = self._player_agent_lin_vel_y[0]
        if key_status["A"] == 1:
            turn_angel += np.pi / 2 * self.dt
        if key_status["D"] == 1:
            turn_angel += -np.pi / 2 * self.dt
        if self._key_status["Space"] == 0 and key_status["Space"] == 1:
            reborn = True
        if key_status["LShift"] == 1:
            lin_vel[:2] *= 2

        self._key_status = key_status.copy()
        # print("Lin vel: ", lin_vel, "Turn angel: ", turn_angel, "Reborn: ", reborn)
        
        return lin_vel, turn_angel, reborn

    def _reset_phy_config(self) -> None:
        phy_config = LeggedEnvConfig["phy_config"]
        self.gym.opt.iterations = LeggedEnvConfig[phy_config]["iterations"]
        self.gym.opt.noslip_iterations = LeggedEnvConfig[phy_config]["noslip_iterations"]
        self.gym.opt.ccd_iterations = LeggedEnvConfig[phy_config]["ccd_iterations"]
        self.gym.opt.sdf_iterations = LeggedEnvConfig[phy_config]["sdf_iterations"]
        self.gym.set_opt_config()

        print("Phy config: ", phy_config, "Iterations: ", self.gym.opt.iterations, "Noslip iterations: ", self.gym.opt.noslip_iterations, "MPR iterations: ", self.gym.opt.ccd_iterations, "SDF iterations: ", self.gym.opt.sdf_iterations)

    def _update_no_action_ctrl(self) -> None:
        ctrl = []
        for agent in self.agents:
            joint_neutral = agent.get_joint_neutral()
            if joint_neutral is not None:
                ctrl.extend(joint_neutral.values())

        self.ctrl = np.array(ctrl).flatten()
        # raise KeyError("Test no action mode")
            
    def _query_sensor_data(self) -> dict[str, np.ndarray]:
        if self.agents[0]._use_imu_sensor:
            return self.query_sensor_data(self._agent_sensor_names)
        else:
            if not hasattr(self, "_agent_foot_touch_sensor_names"):
                self._agent_foot_touch_sensor_names = [sensor_name for agent in self.agents for sensor_name in agent._foot_touch_sensor_names]
        
            return self.query_sensor_data(self._agent_foot_touch_sensor_names)
        
    def _query_site_pos_and_quat(self) -> dict[str, np.ndarray]:
        if self.agents[0]._use_imu_sensor:
            return self.query_site_pos_and_quat(self._agent_site_names)
        else:
            if not hasattr(self, "_agent_contact_site_names"):
                self._agent_contact_site_names = [site_name for agent in self.agents for site_name in agent._contact_site_names]
            
            return self.query_site_pos_and_quat(self._agent_contact_site_names)