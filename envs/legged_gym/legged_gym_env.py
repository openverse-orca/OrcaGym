import numpy as np
from gymnasium.core import ObsType
from orca_gym.multi_agent import OrcaGymMultiAgentEnv
from orca_gym.utils import rotations
from typing import Optional, Any, SupportsFloat
from gymnasium import spaces

from .legged_robot import LeggedRobot

class LeggedGymEnv(OrcaGymMultiAgentEnv):
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


    def step_agents(self, action: np.ndarray) -> None:
        # print("Step agents: ", action)

        mocaps = {}
        for i in range(len(self._agents)):
            agent = self._agents[i]
            act = action[i]

            # 将每个agent的ctrl拼接在一起，然后传递给仿真环境
            agent_ctrl, agent_mocap = agent.step(act)
            self.ctrl[agent.ctrl_start : agent.ctrl_start + len(act)] = agent_ctrl
            mocaps.update(agent_mocap)

        # print("env ctrl: ", self.ctrl)

        self.set_mocap_pos_and_quat(mocaps)



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

    def get_obs(self, agents : list[LeggedRobot]) -> dict[str, np.ndarray]:
        # print("query joint qpos: ", self._agent_joint_names)

        joint_qpos = self.query_joint_qpos(self._agent_joint_names)
        joint_qacc = self.query_joint_qacc(self._agent_joint_names)
        joint_qvel = self.query_joint_qvel(self._agent_joint_names)
        sensor_data = self.query_sensor_data(self._agent_sensor_names)
        contact_dict = self._generate_contact_dict()
        site_pos_quat = self.query_site_pos_and_quat(self._agent_site_names)

        # print("Sensor data: ", sensor_data)
        # print("Joint qpos: ", joint_qpos)

        # 这里，每个process将多个agent的obs拼接在一起，在 subproc_vec_env 再展开成 m x n 份
        obs = agents[0].get_obs(sensor_data, joint_qpos, joint_qacc, joint_qvel, contact_dict, site_pos_quat, self.dt)
        for i in range(1, len(agents)):
            agent_obs = agents[i].get_obs(sensor_data, joint_qpos, joint_qacc, joint_qvel, contact_dict, site_pos_quat, self.dt)
            obs = {key: np.concatenate([obs[key], agent_obs[key]]) for key in obs.keys()}
        
        return obs
        
    def reset_agents(self, agents : list[LeggedRobot]) -> None:
        if len(agents) == 0:
            return

        joint_qpos = {}
        mocaps = {}
        for agent in agents:
            agent_joint_qpos, agent_mocaps = agent.reset(self.np_random)
            joint_qpos.update(agent_joint_qpos)
            mocaps.update(agent_mocaps)

        # print("Reset joint qpos: ", joint_qpos)

        self.set_joint_qpos(joint_qpos)
        self.set_mocap_pos_and_quat(mocaps)
        self.mj_forward()

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
