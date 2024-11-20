
import importlib
from os import path
from typing import Any, Dict, Optional, Tuple, Union, SupportsFloat

import numpy as np
from numpy.typing import NDArray

import gymnasium as gym
from gymnasium import error, spaces
from gymnasium.spaces import Space
from gymnasium.core import ObsType

from . import OrcaGymAgent, OrcaGymLocalEnv

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
        render_remote: bool,
        env_id: str,
        task: str,            
        **kwargs        
    ):

        self._render_mode = render_mode
        self._render_remote = render_remote
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
                               max_episode_steps=max_episode_steps)

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
        [agent.set_ctrl_info(all_actuator) for agent in self._agents]
        self.ctrl = np.zeros(self.nu)

        # Initialize the joints' state before the simulation starts.
        init_joint_qpos = self.query_joint_qpos(self._agent_joint_names)
        [agent.set_init_state(init_joint_qpos) for agent in self._agents]

        self.reset_agents(self._agents)

        # Run generate_observation_space after initialization to ensure that the observation object's name is defined.
        self.set_obs_space()

        # Run set_action_space after initialization to ensure that the action size is defined.
        self.set_action_space()


    def set_obs_space(self):
        self.observation_space = self.generate_observation_space(self.get_obs([self._agents[0]]))

    def set_action_space(self) -> None:
        action_size = self._agents[0].get_action_size()
        self.action_space = spaces.Box(
            low=np.array([-1.0] * action_size),
            high=np.array([1.0] * action_size),
            dtype=np.float32,
        )
        [agent.set_action_space(self.action_space) for agent in self._agents]        

    def initialize_agents(self, entry, *args, **kwargs):
        module_name, class_name = entry.rsplit(":", 1)
        module = importlib.import_module(module_name)
        class_type = getattr(module, class_name)
        for agent_name in self._agent_names:
            agent = class_type(agent_name, *args, **kwargs)
            self._agents.append(agent)
        return
    
    
    def get_obs(self, agents : list[OrcaGymAgent]) -> dict[str, np.ndarray]:
        """
        Observation is environment specific and is defined in the subclass.
        """
        raise NotImplementedError
    
    def reset_agents(self, agents : list[OrcaGymAgent]):
        """
        Do specific reset operations for each agent. It is defined in the subclass.
        """
        raise NotImplementedError
    

    def do_step(self, action: np.ndarray) -> None:
        """
        Do specific operations each step in the environment. It is defined in the subclass.
        """
        raise NotImplementedError

    def step(self, action) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        # print("Step action: ", action)
        if len(action) != len(self._agents) * self.action_space.shape[0]:
            raise ValueError("Action dimension mismatch")
        
        # 切分action 给每个 agent
        action = np.array(action).reshape(len(self._agents), -1)
        self.do_step(action)
        self.do_simulation(self.ctrl, self.frame_skip)

        if self.render_mode == "human" and self._render_remote:
            self.render()

        obs = self.get_obs(self._agents).copy()
        achieved_goal_shape = len(obs["achieved_goal"]) // len(self._agents)
        desired_goal_shape = len(obs["desired_goal"]) // len(self._agents)

        info = {"is_success": np.zeros(len(self._agents))}
        reward = np.zeros(len(self._agents))
        terminated = [False for _ in range(len(self._agents))]
        truncated = [False for _ in range(len(self._agents))]
        agents_to_reset : list[OrcaGymAgent] = []
        for i, agent in enumerate(self._agents):
            achieved_goal = obs["achieved_goal"][i * achieved_goal_shape : (i + 1) * achieved_goal_shape]
            desired_goal = obs["desired_goal"][i * desired_goal_shape : (i + 1) * desired_goal_shape]
            info["is_success"][i] = agent.is_success(achieved_goal, desired_goal, self._env_id)
            reward[i] = agent.compute_reward(achieved_goal, desired_goal)
            terminated[i] = bool(info["is_success"][i] > 0.0)
            truncated[i] = agent.truncated

            if (terminated[i] or truncated[i]):
                # print(f"{self._env_id} Reset agent {agent.name} terminated: {terminated[i]}, truncated: {truncated[i]}")
                agents_to_reset.append(agent)

        self.reset_agents(agents_to_reset)

        # print("Reward: ", reward)
        # print("Is success: ", info["is_success"])
        # print("Terminated: ", terminated)
        # print("Truncated: ", truncated)
        # print("Obs: ", obs)
        # print("Info: ", info)

        return obs, reward, terminated, truncated, info


    def reset_model(self) -> list[dict]:
        print("Reset model")

        # 依次 reset 每个agent
        self.reset_agents(self._agents)
        obs = self.get_obs(self._agents).copy()
        return obs


    def get_observation(self, obs=None):
        if obs is not None:
            return obs
        else:
            return self.get_obs(self._agents).copy()