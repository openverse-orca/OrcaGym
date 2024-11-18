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
        distance_threshold: float,
        goal_xy_range: float,
        obj_xy_range: float,
        goal_x_offset: float,
        goal_z_range: float,
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
        self._distance_threshold = distance_threshold

        self._agents: list[LeggedRobot] = []
        for agent_name in agent_names:
            agent = LeggedRobot(agent_name, task, max_episode_steps, distance_threshold, 
                                goal_xy_range, obj_xy_range, goal_x_offset, goal_z_range)
            self._agents.append(agent)

        self._agent_ee_names = [agent.ee_name for agent in self._agents]
        self._agent_obj_joint_names = [agent.obj_joint_name for agent in self._agents]
        self._agent_obj_site_names = [agent.obj_site_name for agent in self._agents]
        self._agent_mocap_names = [agent.mocap_name for agent in self._agents]
        self._agent_gripper_joint_names = [gripper_joint_name for agent in self._agents for gripper_joint_name in agent.gripper_joint_names]
        self._agent_joint_names = [joint_name for agent in self._agents for joint_name in agent.joint_names]

        self.nu = self.model.nu
        self.nq = self.model.nq
        self.nv = self.model.nv

        # control range
        all_actuator = self.model.get_actuator_dict()
        [agent.set_gripper_ctrl_range(all_actuator) for agent in self._agents]

        self.ctrl = np.zeros(self.nu)
        [agent.set_init_obj_state(self.query_joint_qpos([agent.obj_joint_name])[agent.obj_joint_name], 
                                    self.query_site_pos_and_mat([agent.obj_site_name])[agent.obj_site_name]['xpos'])
                                    for agent in self._agents]
        self._set_init_state()

        [agent.set_initial_grasp_site_xpos(self.query_site_pos_and_quat([agent.ee_name])) for agent in self._agents]


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

    def _set_init_state(self) -> None:
        # print("Set initial state")
        self._set_joint_neutral(self._agents)

        self._update_ctrl()

        self._reset_mocap_welds()

        self.mj_forward()

    def _update_ctrl(self) -> None:
        joint_qpos = self.query_joint_qpos(self._agent_joint_names)
        ctrl = np.array([joint_qpos[joint_name] for joint_name in self._agent_joint_names]).flat.copy()
        self.ctrl[:len(ctrl)] = ctrl
        

    def step(self, action) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        # print("Step action: ", action)
        if len(action) != len(self._agents) * self.action_space.shape[0]:
            raise ValueError("Action dimension mismatch")
        
        # 切分action 给每个 agent
        action = np.array(action).reshape(len(self._agents), -1)

        # step 每个 agent，输入 ee_pos_quat，计算 mocap 和 gripper 控制
        ee_pos_quat = self.query_site_pos_and_quat(self._agent_ee_names)
        grasp_mocap, gripper_qpos = {}, {}
        for agent, act in zip(self._agents, action):
            agent_grasp_mocap, agent_gripper_qpos = agent.step(act, ee_pos_quat)
            grasp_mocap.update(agent_grasp_mocap)
            gripper_qpos.update(agent_gripper_qpos)

        self.set_mocap_pos_and_quat(grasp_mocap)
        self.set_joint_qpos(gripper_qpos)
        self.mj_forward()

        self._update_ctrl()

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
            info["is_success"][i] = agent.is_success(obs["achieved_goal"][i * achieved_goal_shape : (i + 1) * achieved_goal_shape], 
                                                    obs["desired_goal"][i * desired_goal_shape : (i + 1) * desired_goal_shape], self._env_id)
            reward[i] = agent.compute_reward(obs["achieved_goal"][i * achieved_goal_shape : (i + 1) * achieved_goal_shape],
                                                obs["desired_goal"][i * desired_goal_shape : (i + 1) * desired_goal_shape])
            terminated[i] = bool(info["is_success"][i] != 0)
            truncated[i] = agent.truncated

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
        # print("Compute reward : ", len(achieved_goal), len(desired_goal))
        return self._agents[0].compute_reward(achieved_goal, desired_goal)

    def _get_obs(self, agents : list[LeggedRobot]) -> list[dict]:
        site_pos_quat = self.query_site_pos_and_quat(self._agent_ee_names)
        site_pos_mat = self.query_site_pos_and_mat(self._agent_obj_site_names)
        site_xvalp, site_xvalr = self.query_site_xvalp_xvalr(self._agent_obj_site_names)
        joint_qpos = self.query_joint_qpos(self._agent_gripper_joint_names)

        # 这里，每个process将多个agent的obs拼接在一起，在 subproc_vec_env 再展开成mxn份
        obs = agents[0].get_obs(site_pos_quat, site_pos_mat, site_xvalp, site_xvalr, joint_qpos, self.dt)
        for i in range(1, len(agents)):
            agent_obs = agents[i].get_obs(site_pos_quat, site_pos_mat, site_xvalp, site_xvalr, joint_qpos, self.dt)
            obs = {key: np.concatenate([obs[key], agent_obs[key]]) for key in obs.keys()}
        
        return obs
        
    def _reset_agents(self, agents : list[LeggedRobot]) -> None:
        if len(agents) == 0:
            return

        mocap_pos_quat, joint_qpos = {}, {}
        for agent in agents:
            agent_mocap, agent_joint_qpos = agent.reset(self.np_random)
            mocap_pos_quat.update(agent_mocap)
            joint_qpos.update(agent_joint_qpos)

        self.set_mocap_pos_and_quat(mocap_pos_quat)
        self.set_joint_qpos(joint_qpos)
        self.mj_forward()


    def reset_model(self) -> list[dict]:
        print("Reset model")

        # 依次 reset 每个agent
        self._set_init_state()
        self._reset_agents(self._agents)
        obs = self._get_obs(self._agents).copy()
        return obs

    # custom methods
    # -----------------------------
    def _reset_mocap_welds(self) -> None:
        if self.model.nmocap > 0 and self.model.neq > 0:
            eq_list = self.model.get_eq_list()
            for eq in eq_list:
                if eq['eq_type'] == self.model.mjEQ_WELD:
                    obj1_id = eq['obj1_id']
                    obj2_id = eq['obj2_id']
                    eq_data = eq['eq_data'].copy()
                    # print("org eq_data: ", eq_data)
                    eq_data[3:10] = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
                    self.update_equality_constraints([{"obj1_id": obj1_id, "obj2_id": obj2_id, "eq_data": eq_data}])
        self.mj_forward()

    def _set_goal_mocap(self, goal_name, position, orientation) -> None:
        mocap_pos_and_quat_dict = {goal_name: {'pos': position, 'quat': orientation}}
        self.set_mocap_pos_and_quat(mocap_pos_and_quat_dict)

    def _set_joint_neutral(self, agents : list[LeggedRobot]) -> None:
        joint_qpos = {}
        for agent in agents:
            joint_qpos.update(agent.get_joint_neutral())

        self.set_joint_qpos(joint_qpos)


    def get_observation(self, obs=None):
        if obs is not None:
            return obs
        else:
            return self._get_obs(self._agents).copy()