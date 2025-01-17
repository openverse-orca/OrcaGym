
import importlib
from os import path
import time
from typing import Any, Dict, Optional, Tuple, Union, SupportsFloat
import datetime
import torch

import numpy as np
from numpy.typing import NDArray

import gymnasium as gym
from gymnasium import error, spaces
from gymnasium.spaces import Space
from gymnasium.core import ObsType

from orca_gym.environment import OrcaGymLocalEnv
from .orca_gym_agent import OrcaGymAgent

class OrcaGymMultiAgentEnv(OrcaGymLocalEnv):
    def __init__(
        self,
        frame_skip: int,
        orcagym_addr: str,
        agent_names: list[str],
        time_step: float,    
        agent_engry: str,
        max_episode_steps: int,
        render_mode: str,
        is_subenv: bool,
        env_id: str,
        task: str,            
        **kwargs        
    ):

        self._render_mode = render_mode
        self._is_subenv = is_subenv
        self._env_id = env_id
        self._task = task
        
        super().__init__(
            frame_skip = frame_skip,
            orcagym_addr = orcagym_addr,
            agent_names = agent_names,
            time_step = time_step,            
            **kwargs
        )

        self._agents : list[OrcaGymAgent] = []

        self.initialize_agents(entry=agent_engry, 
                               task=task, 
                               max_episode_steps=max_episode_steps,
                               dt=self.dt)

        self._agent_joint_names = [joint_name for agent in self._agents for joint_name in agent.joint_names ]
        self._agent_actuator_names = [actuator_name for agent in self._agents for actuator_name in agent.actuator_names]
        self._agent_site_names = [site_name for agent in self._agents for site_name in agent.site_names]
        self._agent_sensor_names = [sensor_name for agent in self._agents for sensor_name in agent.sensor_names]

        self.nu = self.model.nu
        self.nq = self.model.nq
        self.nv = self.model.nv

        # control info, this is about the actuators, not about the actoins. 
        # in some cases, agent may not use all actuators for actions.
        all_actuator = self.model.get_actuator_dict()
        [agent.init_ctrl_info(all_actuator) for agent in self._agents]
        self.ctrl = np.zeros(self.nu)

        # TODO: mujoco bugs? 
        # Do mj_forward to apply site xpos and quat to the init status.
        # Otherwise, the site pos and quat will be zero in the first step.
        self.mj_forward()

        # Initialize the joints' state before the simulation starts.
        init_joint_qpos = self.query_joint_qpos(self._agent_joint_names)
        init_site_pos_quat = self.query_site_pos_and_quat(self._agent_site_names)
        # print("all inti site pos quat: ", init_site_pos_quat)
        [agent.set_init_state(init_joint_qpos, init_site_pos_quat) for agent in self._agents]

        # For performance issue, we use the qpos, qvel, qacc index, and read the data from env.data.qpos, qvel, qacc directly. 
        # The qpos, qvel, qacc buffer in env.data.qpos, qvel, qacc will be updated by once each Step.
        # Directly query the qpos, qvel, qacc will read data cross the c++ and python boundary, which is slow.
        self.init_agent_joint_index()

        self.reset_agents(self._agents)

        # Run generate_observation_space after initialization to ensure that the observation object's name is defined.
        self.set_obs_space()

        # Run set_action_space after initialization to ensure that the action size is defined.
        self.set_action_space()

        # Reorder the agents by the actuator control start index.
        # This will align the agents' actuator control index in the multi-agent environment.
        # And also align the agents' observation and action space.
        self._agents = self.reorder_agents()

        # Generate the action scale array for the multi-agent environment.
        # Do liner interpolation for the action space of each agent, map the action space to the control space.
        # For performance issue, calculate all the agents' actions together by once.
        self.generate_action_scale_array(self._query_ctrl_info())


    def set_obs_space(self):
        obs, _, _, _ = self.get_obs()
        agent_0_obs = {k: v[0] for k, v in obs.items()}
        # print("Agent 0 obs: ", agent_0_obs, "obs :", obs)
        self.observation_space = self.generate_observation_space(agent_0_obs)

    def set_action_space(self) -> None:
        action_size = self._agents[0].get_action_size()
        self.action_space = spaces.Box(
            low=np.array([-1.0] * action_size, dtype=np.float32),
            high=np.array([1.0] * action_size, dtype=np.float32),
            dtype=np.float32,
        )
        [agent.set_action_space(self.action_space) for agent in self._agents]        

    def initialize_agents(self, entry, *args, **kwargs):
        module_name, class_name = entry.rsplit(":", 1)
        module = importlib.import_module(module_name)
        class_type = getattr(module, class_name)
        for agent_name in self._agent_names:
            agent = class_type(self._env_id, agent_name, *args, **kwargs)
            self._agents.append(agent)
        return
    
    
    def get_obs(self) -> tuple[dict[str, np.ndarray], list[dict[str, np.ndarray]], np.ndarray, np.ndarray]:
        """
        Observation is environment specific and is defined in the subclass.
        """
        raise NotImplementedError
    
    def reset_agents(self, agents : list[OrcaGymAgent]):
        """
        Do specific reset operations for each agent. It is defined in the subclass.
        """
        raise NotImplementedError
    

    def step_agents(self, action: np.ndarray, actuator_ctrl: np.ndarray) -> None:
        """
        Do specific operations each step in the environment. It is defined in the subclass.
        """
        raise NotImplementedError

    def step(self, action) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        # step_start = datetime.datetime.now()

        # print("Step action: ", action)
        if len(action) != len(self._agents) * self.action_space.shape[0]:
            raise ValueError("Action dimension mismatch")
        
        actuator_ctrl = self._action2ctrl(action)
        self.step_agents(action, actuator_ctrl)
        # step_action = (datetime.datetime.now() - step_start).total_seconds() * 1000

        self.do_simulation(self.ctrl, self.frame_skip)
        # step_sim = (datetime.datetime.now() - step_start).total_seconds() * 1000

        if self.render_mode == "human" and not self.is_subenv:
            self.render()
        # step_render = (datetime.datetime.now() - step_start).total_seconds() * 1000
        
        # raise error.Error("Test robot height!")

        env_obs, agent_obs, achieved_goals, desired_goals = self.get_obs()
        # step_obs = (datetime.datetime.now() - step_start).total_seconds() * 1000
        # achieved_goal_shape = len(obs["achieved_goal"]) // len(self._agents)
        # desired_goal_shape = len(obs["desired_goal"]) // len(self._agents)

        info = {"env_obs": env_obs,
                "agent_obs": agent_obs,
                "is_success": np.zeros(len(self._agents)),
                "reward" : np.zeros(len(self._agents)),
                "terminated" : [False for _ in range(len(self._agents))],
                "truncated" : [False for _ in range(len(self._agents))]}

        agents_to_reset : list[OrcaGymAgent] = []
        for i, agent in enumerate(self._agents):
            # achieved_goal = obs["achieved_goal"][i * achieved_goal_shape : (i + 1) * achieved_goal_shape]
            # desired_goal = obs["desired_goal"][i * desired_goal_shape : (i + 1) * desired_goal_shape]
            achieved_goal = achieved_goals[i]
            desired_goal = desired_goals[i]
            info["is_success"][i] = agent.is_success(achieved_goal, desired_goal)
            info["reward"][i] = agent.compute_reward(achieved_goal, desired_goal)
            info["terminated"][i] = agent.is_terminated(achieved_goal, desired_goal)
            info["truncated"][i] = agent.truncated

            # if info["is_success"][i] > 0.0:
            #     print("Env: ", self._env_id, "Agent: ", agent.name, "Task Success: achieved goal: ", achieved_goal, "desired goal: ", desired_goal)
            # elif info["terminated"][i]:
            #     print("Env: ", self._env_id, "Agent: ", agent.name, "Task Failed: achieved goal: ", achieved_goal, "desired goal: ", desired_goal)

            if (info["terminated"][i] or info["truncated"][i]):
                # print(f"{self._env_id} Reset agent {agent.name} terminated: {info['terminated'][i]}, truncated: {info['truncated'][i]}, achieved goal: {achieved_goal}, desired goal: {desired_goal}")
                agents_to_reset.append(agent)
        # step_process = (datetime.datetime.now() - step_start).total_seconds() * 1000

        self.reset_agents(agents_to_reset)
        # step_reset = (datetime.datetime.now() - step_start).total_seconds() * 1000

        # print("Reward: ", reward)
        # print("Is success: ", info["is_success"])
        # print("Terminated: ", terminated)
        # print("Truncated: ", truncated)
        # print("Obs: ", obs)
        # print("Info: ", info)

        # step_total = (datetime.datetime.now() - step_start).total_seconds() * 1000
        # print("\tStep time, ",
        #         "\n\t\taction: ", step_action, 
        #         "\n\t\tsim: ", step_sim - step_action, 
        #         "\n\t\trender: ", step_render - step_sim, 
        #         "\n\t\tobs: ", step_obs - step_render, 
        #         "\n\t\tprocess: ", step_process - step_obs,
        #         "\n\t\treset: ", step_reset - step_process,
        #       "\n\ttotal: ", step_total)

        # 兼容 stable-baselines3 标准接口，obs 只取第一个 agent 的观测数据，实际所有 agent 的观测数据在 info 中
        # subproc_vec_env 不需要从新拼接，在这里就按照agent打好包作为dict发过去
        agent_0_obs = {k: v[0] for k, v in env_obs.items()}
        return agent_0_obs, 0.0, False, False, info


    def reset_model(self) -> tuple[ObsType, dict[str, np.ndarray]]:
        print("Reset model")

        # 依次 reset 每个agent
        self.reset_agents(self._agents)
        env_obs, agent_obs, achieved_goals, desired_goals = self.get_obs()
        reset_info = {
            "env_obs": env_obs
        }
        agent_0_obs = {k: v[0] for k, v in env_obs.items()}
        return agent_0_obs, reset_info


    def get_observation(self, obs=None):
        if obs is not None:
            return obs
        else:
            return self.get_obs()

    def init_agent_joint_index(self):
        for agent in self._agents:
            joint_names = agent.joint_names
            qpos_offset, qvel_offset, qacc_offset = self.query_joint_offsets(joint_names)
            qpos_length, qvel_length, qacc_length = self.query_joint_lengths(joint_names)
            agent.init_joint_index(qpos_offset, qvel_offset, qacc_offset, qpos_length, qvel_length, qacc_length)

    # def _build_joint_qpos_qvel_qacc_buffer(self):
    #     self._qpos_offset, self._qvel_offset, self._qacc_offset = self.query_joint_offsets(self._agent_joint_names)
    #     self._qpos_length, self._qvel_length, self._qacc_length = self.query_joint_lengths(self._agent_joint_names)

    #     # Initialize the qpos, qvel, qacc buffer
    #     self._joint_qpos_buffer = {}
    #     self._joint_qvel_buffer = {}
    #     self._joint_qacc_buffer = {}
    #     for i, joint_name in enumerate(self._agent_joint_names):
    #         self._joint_qpos_buffer[joint_name] = np.zeros(self._qpos_length[i])
    #         self._joint_qvel_buffer[joint_name] = np.zeros(self._qvel_length[i])
    #         self._joint_qacc_buffer[joint_name] = np.zeros(self._qacc_length[i])

    # def _update_joint_qpos_qvel_qacc_buffer(self):
    #     for i, joint_name in enumerate(self._agent_joint_names):
    #         self._joint_qpos_buffer[joint_name][:] = self.data.qpos[self._qpos_offset[i] : self._qpos_offset[i] + self._qpos_length[i]]
    #         self._joint_qvel_buffer[joint_name][:] = self.data.qvel[self._qvel_offset[i] : self._qvel_offset[i] + self._qvel_length[i]]
    #         self._joint_qacc_buffer[joint_name][:] = self.data.qacc[self._qacc_offset[i] : self._qacc_offset[i] + self._qacc_length[i]]

    def reorder_agents(self):
        ctrl_info = self._query_ctrl_info()

        # 按照 ctrl_start 排序
        ctrl_info = {k: v for k, v in sorted(ctrl_info.items(), key=lambda item: item[1]["ctrl_start"])}

        # print("Agent before reorder: ", [agent.name for agent in self._agents])
        reordered_agents = []
        for agent_name in ctrl_info.keys():
            for agent in self._agents:
                if agent.name == agent_name:
                    reordered_agents.append(agent)
                    break
        # print("Agent after reorder: ", [agent.name for agent in reordered_agents])

        assert len(reordered_agents) == len(self._agents)
        return reordered_agents
    
    # def get_action_scale_array(self):
    #     action_scale_array = {
    #         "actuator_type": self._actuator_type,
    #         "action_scale": self._action_scale,
    #         "action_space_range": self._action_space_range,
    #         "ctrl_start": self._ctrl_start,
    #         "ctrl_end": self._ctrl_end,
    #         "ctrl_range": self._ctrl_range,
    #         "ctrl_delta_range": self._ctrl_delta_range,
    #         "neutral_joint_values": self._neutral_joint_values,
    #     }
    #     return action_scale_array

    def _query_ctrl_info(self):
        ctrl_info = {}
        for agent in self._agents:
            ctrl_info[agent.name] = agent.get_ctrl_info()
        return ctrl_info

    def generate_action_scale_array(self, ctrl_info: dict) -> np.ndarray:
        self._actuator_type = next(iter(ctrl_info.values()))["actuator_type"]           # shape = (1)
        self._action_scale = next(iter(ctrl_info.values()))["action_scale"]             # shape = (1)
        self._action_space_range = next(iter(ctrl_info.values()))["action_space_range"] # shape = (2)

        self._ctrl_start = np.array([ctrl["ctrl_start"] for key, ctrl in ctrl_info.items()]) # shape = (agent_num)
        self._ctrl_end = np.array([ctrl["ctrl_end"] for key, ctrl in ctrl_info.items()])     # shape = (agent_num)
        self._ctrl_range = np.array([ctrl["ctrl_range"] for key, ctrl in ctrl_info.items()]).reshape(-1, 2)   # shape = (agent_num x actor_num, 2) 
        self._ctrl_delta_range = np.array([ctrl["ctrl_delta_range"] for key, ctrl in ctrl_info.items()]).reshape(-1, 2)  # shape = (agent_num x actor_num, 2)
        self._neutral_joint_values = np.array([ctrl["neutral_joint_values"] for key, ctrl in ctrl_info.items()]).reshape(-1) # shape = (agent_num x actor_num)

    # def _build_action_scale_array(self, action_scale_array: dict) -> np.ndarray:
    #     """
    #     Build the action scale array for the multi-agent environment.
    #     """
    #     remote_num = len(self.remotes)

    #     # print("Build action scale array: ", action_scale_array, "remote_num: ", remote_num)
    #     # print("Ctrl range before: ", len(action_scale_array["ctrl_range"]))
        
    #     self._actuator_type = action_scale_array["actuator_type"]
    #     self._action_scale = action_scale_array["action_scale"]
    #     self._action_space_range = np.concatenate([action_scale_array["action_space_range"]] * remote_num, axis=0)
    #     self._ctrl_start = np.concatenate([action_scale_array["ctrl_start"]] * remote_num, axis=0)
    #     self._ctrl_end = np.concatenate([action_scale_array["ctrl_end"]] * remote_num, axis=0)
    #     self._ctrl_range = np.concatenate([action_scale_array["ctrl_range"]] * remote_num, axis=0)
    #     self._ctrl_delta_range = np.concatenate([action_scale_array["ctrl_delta_range"]] * remote_num, axis=0)
    #     self._neutral_joint_values = np.concatenate([action_scale_array["neutral_joint_values"]] * remote_num, axis=0)

    #     # print("Ctrl range after: ", len(self._ctrl_range))

    #     if remote_num > 4:
    #         self._action2ctrl_use_tensor = True
    #     else:
    #         self._action2ctrl_use_tensor = False

    #     if not self._action2ctrl_use_tensor:
    #         return

    #     # 初始化阶段，将不变数据移动到 GPU
    #     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #     # device = torch.device("cpu")
    #     self._action_tensor = torch.zeros(len(self._ctrl_range), device=device)  # 每个 action 的 tensor
    #     self._action_scale_tensor = torch.ones(len(self._ctrl_range), device=device) * self._action_scale  # 每个 action 的缩放因子
    #     self._scaled_action_tensor = torch.zeros(len(self._ctrl_range), device=device)  # 缩放后的 action
    #     self._action_space_range_tensor = torch.tensor(self._action_space_range, device=device)
    #     self._clipped_action_tensor = torch.zeros(len(self._ctrl_range), device=device)  # 限制在有效范围内的 action

    #     self._ctrl_range_tensor = torch.tensor(self._ctrl_range, device=device)
    #     self._ctrl_delta_range_tensor = torch.tensor(self._ctrl_delta_range, device=device)
    #     self._neutral_joint_values_tensor = torch.tensor(self._neutral_joint_values, device=device)
    #     self._ctrl_delta_tensor = torch.zeros(len(self._ctrl_range), device=device)  # 插值后的控制量
    #     self._actuator_ctrl_tensor = torch.zeros(len(self._ctrl_range), device=device)  # actuator 控制量


    # def _action2ctrl_tensor(self, action: np.ndarray) -> np.ndarray:
    #     # 将 action 传输到 GPU
    #     self._action_tensor[:] = torch.from_numpy(action).to(self._action_tensor.device)

    #     # 缩放后的 action
    #     self._scaled_action_tensor = self._action_tensor * self._action_scale_tensor

    #     # 限制 scaled_action 在有效范围内
    #     self._clipped_action_tensor = torch.clamp(self._scaled_action_tensor, self._action_space_range_tensor[0], self._action_space_range_tensor[1])

    #     # 批量计算插值
    #     if (self._actuator_type == "position"):
    #         self._ctrl_delta_tensor = (
    #             self._ctrl_delta_range_tensor[:, 0] +  # fp1
    #             (self._ctrl_delta_range_tensor[:, 1] - self._ctrl_delta_range_tensor[:, 0]) *  # (fp2 - fp1)
    #             (self._clipped_action_tensor - self._action_space_range_tensor[0]) /  # (x - xp1)
    #             (self._action_space_range_tensor[1] - self._action_space_range_tensor[0])  # (xp2 - xp1)
    #         )

    #         self._actuator_ctrl_tensor = self._ctrl_delta_tensor + self._neutral_joint_values_tensor
    #     elif (self._actuator_type == "torque"):
    #         self._actuator_ctrl_tensor = (
    #             self._ctrl_range_tensor[:, 0] +  # fp1
    #             (self._ctrl_range_tensor[:, 1] - self._ctrl_range_tensor[:, 0]) *  # (fp2 - fp1)
    #             (self._clipped_action_tensor - self._action_space_range_tensor[0]) /  # (x - xp1)
    #             (self._action_space_range_tensor[1] - self._action_space_range_tensor[0])  # (xp2 - xp1)
    #         )
    #     else:
    #         raise ValueError(f"Unsupported actuator type: {self._actuator_type}")
    
    #     return self._actuator_ctrl_tensor.cpu().numpy()
        

    def _action2ctrl(self, action: np.ndarray) -> np.ndarray:
        # 缩放后的 action
        scaled_action = action * self._action_scale

        # 限制 scaled_action 在有效范围内
        clipped_action = np.clip(scaled_action, self._action_space_range[0], self._action_space_range[1])

        # 批量计算插值
        if (self._actuator_type == "position"):
            ctrl_delta = (
                self._ctrl_delta_range[:, 0] +  # fp1
                (self._ctrl_delta_range[:, 1] - self._ctrl_delta_range[:, 0]) *  # (fp2 - fp1)
                (clipped_action - self._action_space_range[0]) /  # (x - xp1)
                (self._action_space_range[1] - self._action_space_range[0])  # (xp2 - xp1)
            )

            actuator_ctrl = self._neutral_joint_values + ctrl_delta
        elif (self._actuator_type == "torque"):
            actuator_ctrl = (
                self._ctrl_range[:, 0] +  # fp1
                (self._ctrl_range[:, 1] - self._ctrl_range[:, 0]) *  # (fp2 - fp1)
                (clipped_action - self._action_space_range[0]) /  # (x - xp1)
                (self._action_space_range[1] - self._action_space_range[0])  # (xp2 - xp1)
            )
        else:
            raise ValueError(f"Unsupported actuator type: {self._actuator_type}")
        
        return actuator_ctrl

    # @property
    # def joint_qpos_buffer(self):
    #     return self._joint_qpos_buffer
    
    # @property
    # def joint_qvel_buffer(self):
    #     return self._joint_qvel_buffer
    
    # @property
    # def joint_qacc_buffer(self):
    #     return self._joint_qacc_buffer