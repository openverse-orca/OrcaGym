import numpy as np
from gymnasium.core import ObsType
from envs import OrcaGymLocalEnv, OrcaGymRemoteEnv
from orca_gym.utils import rotations
from typing import Optional, Any, SupportsFloat
from gymnasium import spaces

from .legged_robot import LeggedRobot
from .go2_robot import Go2Robot


class LeggedGymEnv(OrcaGymLocalEnv):
    metadata = {'render_modes': ['human', 'none'], 'version': '0.0.1'}

    def __init__(
        self,
        frame_skip: int,        
        orcagym_addr: str,
        agent_names: list,
        time_step: float,    
        max_episode_steps: int,
        render_mode: str,
        render_remote: bool,
        env_id: str,
        task: str,
        **kwargs,
    ):

        self._render_mode = render_mode
        self._render_remote = render_remote
        self._env_id = env_id

        super().__init__(
            frame_skip = frame_skip,
            orcagym_addr = orcagym_addr,
            agent_names = agent_names,
            time_step = time_step,            
            **kwargs,
        )

        self._task = task

        self._agents: list[LeggedRobot] = []
        for agent_name in agent_names:
            agent = Go2Robot(agent_name, task, max_episode_steps)
            self._agents.append(agent)

        self._agent_joint_names = [joint_name for agent in self._agents for joint_name in agent.joint_names ]
        self._agent_actuator_names = [actuator_name for agent in self._agents for actuator_name in agent.actuator_names]
        self._agent_site_names = [site_name for agent in self._agents for site_name in agent.site_names]
        self._agent_sensor_names = [sensor_name for agent in self._agents for sensor_name in agent.sensor_names]

        self.nu = self.model.nu
        self.nq = self.model.nq
        self.nv = self.model.nv

        # control range
        all_actuator = self.model.get_actuator_dict()
        [agent.set_ctrl_info(all_actuator) for agent in self._agents]

        self.ctrl = np.zeros(self.nu)

        init_joint_qpos = self.query_joint_qpos(self._agent_joint_names)
        [agent.set_init_state(init_joint_qpos) for agent in self._agents]

        self._reset_agents(self._agents)

        # Run generate_observation_space after initialization to ensure that the observation object's name is defined.
        self._set_obs_space()
        self._set_action_space()

    def _set_obs_space(self):
        self.observation_space = self.generate_observation_space(self._get_obs([self._agents[0]]))
        # print("Observation space: ", self.observation_space)

    def _set_action_space(self):
        action_size = 12    # legged robot action size
        self.action_space = spaces.Box(
            low=np.array([-1.0] * action_size),
            high=np.array([1.0] * action_size),
            dtype=np.float32,
        )
        [agent.set_action_space(self.action_space) for agent in self._agents]
        

    def step(self, action) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        # print("Step action: ", action)
        if len(action) != len(self._agents) * self.action_space.shape[0]:
            raise ValueError("Action dimension mismatch")
        
        # 切分action 给每个 agent
        action = np.array(action).reshape(len(self._agents), -1)

        # step 每个 agent，输入 ee_pos_quat，计算 mocap 和 gripper 控制
        for i in range(len(self._agents)):
            agent = self._agents[i]
            act = action[i]
            self.ctrl[agent.ctrl_start : agent.ctrl_start + len(act)] = agent.step(act)

        self.do_simulation(self.ctrl, self.frame_skip)

        if self.render_mode == "human" and self._render_remote:
            self.render()

        obs = self._get_obs(self._agents).copy()
        achieved_goal_shape = len(obs["achieved_goal"]) // len(self._agents)
        desired_goal_shape = len(obs["desired_goal"]) // len(self._agents)

        info = {"is_success": np.zeros(len(self._agents))}
        reward = np.zeros(len(self._agents))
        terminated = [False for _ in range(len(self._agents))]
        truncated = [False for _ in range(len(self._agents))]
        agents_to_reset : list[LeggedRobot] = []
        for i, agent in enumerate(self._agents):
            achieved_goal = obs["achieved_goal"][i * achieved_goal_shape : (i + 1) * achieved_goal_shape]
            desired_goal = obs["desired_goal"][i * desired_goal_shape : (i + 1) * desired_goal_shape]
            info["is_success"][i] = agent.is_success(achieved_goal, desired_goal, self._env_id)
            reward[i] = agent.compute_reward(achieved_goal, desired_goal)
            terminated[i] = bool(info["is_success"][i] == 1.0)
            truncated[i] = agent.is_truncated(achieved_goal, desired_goal) or agent.truncated

            if (terminated[i] or truncated[i]):
                # print(f"{self._env_id} Reset agent {agent.name} terminated: {terminated[i]}, truncated: {truncated[i]}")
                agents_to_reset.append(agent)

        self._reset_agents(agents_to_reset)

        # print("Reward: ", reward)
        # print("Is success: ", info["is_success"])
        # print("Terminated: ", terminated)
        # print("Truncated: ", truncated)
        # print("Obs: ", obs)
        # print("Info: ", info)

        return obs, reward, terminated, truncated, info


    def compute_reward(self, achieved_goal, desired_goal, info) -> SupportsFloat:
        assert achieved_goal.shape == desired_goal.shape
        if achieved_goal.ndim == 1:
            return self._agents[0].compute_reward(achieved_goal, desired_goal)
        elif achieved_goal.ndim == 2:
            agent_num = len(self._agents)
            rewards = np.zeros(agent_num)
            for i in range(len(achieved_goal)):
                rewards[i] = self._agents[i % agent_num].compute_reward(achieved_goal[i], desired_goal[i])
            return rewards
        else:
            raise ValueError("Unsupported achieved_goal shape")

    def _get_obs(self, agents : list[LeggedRobot]) -> list[dict]:
        # print("query joint qpos: ", self._agent_joint_names)

        joint_qpos = self.query_joint_qpos(self._agent_joint_names)
        sensor_data = self.query_sensor_data(self._agent_sensor_names)

        # print("Sensor data: ", sensor_data)
        # print("Joint qpos: ", joint_qpos)

        # 这里，每个process将多个agent的obs拼接在一起，在 subproc_vec_env 再展开成 m x n 份
        obs = agents[0].get_obs(sensor_data, joint_qpos, self.dt)
        for i in range(1, len(agents)):
            agent_obs = agents[i].get_obs(sensor_data, joint_qpos, self.dt)
            obs = {key: np.concatenate([obs[key], agent_obs[key]]) for key in obs.keys()}
        
        return obs
        
    def _reset_agents(self, agents : list[LeggedRobot]) -> None:
        if len(agents) == 0:
            return

        joint_qpos = {}
        for agent in agents:
            agent_joint_qpos = agent.reset(self.np_random)
            joint_qpos.update(agent_joint_qpos)

        # print("Reset joint qpos: ", joint_qpos)

        self.set_joint_qpos(joint_qpos)
        self.mj_forward()


    def reset_model(self) -> list[dict]:
        print("Reset model")

        # 依次 reset 每个agent
        self._reset_agents(self._agents)
        obs = self._get_obs(self._agents).copy()
        return obs


    def get_observation(self, obs=None):
        if obs is not None:
            return obs
        else:
            return self._get_obs(self._agents).copy()