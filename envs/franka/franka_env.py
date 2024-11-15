import numpy as np
from gymnasium.core import ObsType
from envs import OrcaGymLocalEnv, OrcaGymRemoteEnv
from orca_gym.utils import rotations
from typing import Optional, Any, SupportsFloat
from gymnasium import spaces


class FrankaRobot:
    def __init__(self, 
                 agent_name: str, 
                 task: str,
                 max_episode_steps: int,
                 distance_threshold: float,
                 goal_xy_range: float,
                 obj_xy_range: float,
                 goal_x_offset: float,
                 goal_z_range: float):
        
        self._agent_name = agent_name
        self._task = task
        self._max_episode_steps = max_episode_steps
        self._distance_threshold = distance_threshold
        self._goal_xy_range = goal_xy_range
        self._obj_xy_range = obj_xy_range
        self._goal_x_offset = goal_x_offset
        self._goal_z_range = goal_z_range

        self._neutral_joint_values = np.array([0.00, 0.41, 0.00, -1.85, 0.00, 2.26, 0.79, 0.00, 0.00])
        self._obj_joint_name = self.name_space('obj_joint')
        self._obj_site_name = self.name_space('obj_site')
        self._ee_name = self.name_space('ee_center_site')
        self._mocap_name = self.name_space('panda_mocap')
        self._goal_name = self.name_space('goal')
        self._arm_joint_names = self.name_space_list(["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7"])
        self._gripper_joint_names = self.name_space_list(["finger_joint1", "finger_joint2"])

        self._goal_range_low = np.array([-self._goal_xy_range / 2 + goal_x_offset, -self._goal_xy_range / 2, 0])
        self._goal_range_high = np.array([self._goal_xy_range / 2 + goal_x_offset, self._goal_xy_range / 2, self._goal_z_range])
        self._obj_range_low = np.array([-self._obj_xy_range / 2, -self._obj_xy_range / 2, 0])
        self._obj_range_high = np.array([self._obj_xy_range / 2, self._obj_xy_range / 2, 0])     

        self._ctrl = self._neutral_joint_values.copy()
        self._current_episode_step = 0

    @property
    def name(self) -> str:
        return self._agent_name

    def name_space(self, name : str) -> str:
        return f"{self._agent_name}_{name}"
    
    def name_space_list(self, names : list[str]) -> list[str]:
        return [self.name_space(name) for name in names]
    
    @property
    def obj_joint_name(self) -> str:
        return self._obj_joint_name
    
    @property
    def obj_site_name(self) -> str:
        return self._obj_site_name
    
    @property
    def ee_name(self) -> str:
        return self._ee_name
    
    @property
    def gripper_joint_names(self) -> list[str]:
        return self._gripper_joint_names
    
    @property
    def arm_joint_names(self) -> list[str]:
        return self._arm_joint_names
    
    @property
    def joint_names(self) -> list[str]:
        return self._arm_joint_names + self._gripper_joint_names
    
    @property
    def neutral_joint_values(self) -> np.ndarray:
        return self._neutral_joint_values
    
    @property
    def initial_grasp_site_xpos(self) -> np.ndarray:
        return self._initial_grasp_site_xpos
    
    @property
    def initial_grasp_site_xquat(self) -> np.ndarray:
        return self._initial_grasp_site_xquat
    
    @property
    def mocap_name(self) -> str:
        return self._mocap_name
    
    @property
    def goal_name(self) -> str:
        return self._goal_name
    
    @property
    def goal(self) -> np.ndarray:
        return self._goal
    
    @property
    def truncated(self) -> bool:
        return self._current_episode_step >= self._max_episode_steps
    
    @property
    def gripper_ctrl_range(self) -> tuple[float]:
        a = self._gripper_ctrl_range[0][0]
        b = self._gripper_ctrl_range[0][1]
        return a, b

    def set_gripper_ctrl_range(self, actuator_dict) -> None:
        if not hasattr(self, "_gripper_ctrl_range"):
            self._gripper_ctrl_range = []
            
        for key, item in actuator_dict.items():
            if item["JointName"] in self._gripper_joint_names:
                self._gripper_ctrl_range.append(item["CtrlRange"])

        # print("Gripper control range: ", self._gripper_ctrl_range)


    def set_init_obj_state(self, obj_qpos, obj_site_xpos) -> None:
        self._initial_object_qpos = obj_qpos.copy()
        self._initial_object_qpos[2] = 0.02  # 放置于地面  

    def set_initial_grasp_site_xpos(self, site_pos_quat) -> None:
        self._initial_grasp_site_xpos = site_pos_quat[self.ee_name]['xpos']
        self._initial_grasp_site_xquat = site_pos_quat[self.ee_name]['xquat']

    def get_joint_neutral(self) -> np.ndarray:
        joint_qpos = {}
        for name, value in zip(self.arm_joint_names, self.neutral_joint_values[0:7]):
            joint_qpos[name] = np.array([value])
        for name, value in zip(self.gripper_joint_names, self.neutral_joint_values[7:9]):
            joint_qpos[name] = np.array([value])

        return joint_qpos

    def sample_goal(self, obj_sampled, np_random) -> np.ndarray:
        # 训练reach时，任务是移动抓夹，goal以抓夹为原点采样
        if self._task == "reach":
            ee_position = self.initial_grasp_site_xpos.copy()
            goal = ee_position.copy()
        elif self._task == "pick_and_place":
            goal = obj_sampled[self.obj_joint_name][0:3].copy()
        else:
            raise ValueError("Unsupport task type: ", self._task)

        noise = np_random.uniform(self._goal_range_low, self._goal_range_high)

        # pick and place task 保证goal在物体上方
        if self._task == "pick_and_place":
            # noise[2] = abs(noise[2])
            noise[:] = 0.0

        # 避免与obj过度接近的情况
        for i in range(3):
            if noise[i] < self._distance_threshold + 0.01 and noise[i] >= 0:
                noise[i] = self._distance_threshold + 0.01
            if noise[i] > -self._distance_threshold - 0.01 and noise[i] < 0:
                noise[i] = -self._distance_threshold - 0.01

        # for the pick and place task
        if self._task == "pick_and_place" and self._goal_z_range > 0.0:
            if np_random.random() < 0.9999:
                # 置于目标位置的上方
                # noise[0] = 0.0
                # noise[1] = 0.0
                # 置于目标附近的地面
                noise[2] = 0.00


        
        goal += noise
        goal[2] = max(0.00, goal[2])  # 确保在地面以上，考虑方块的高度，最低为0.02
        
        self._goal = goal.copy()
        return goal

    def get_obs(self, site_pos_quat, site_pos_mat, site_xvalp, site_xvalr, joint_qpos, dt) -> dict:
        # robot
        ee_position = site_pos_quat[self.ee_name]['xpos'].copy()
        ee_velocity = site_xvalp[self.obj_site_name].copy() * dt


        fingers_qpos = np.array(self._get_fingers_qpos(joint_qpos)).flatten()

        # object
        # object cartesian position: 3
        object_position = site_pos_mat[self.obj_site_name]['xpos'].copy()
        object_mat = site_pos_mat[self.obj_site_name]['xmat'].reshape(3, 3)

        # object rotations: 3
        object_rotation = rotations.mat2euler(object_mat)


        # object linear velocities
        object_velp = site_xvalp[self.obj_site_name].copy() * dt
        object_velr = site_xvalr[self.obj_site_name].copy() * dt

        if self._task == "reach":
            achieved_goal = ee_position.copy()
            desired_goal = self.goal.copy()
        elif self._task == "pick_and_place":
            achieved_goal = np.concatenate([object_position, ee_position])
            desired_goal = np.concatenate([ee_position, self.goal])
        else:
            raise ValueError("Unsupport task type: ", self._task)   

        obs = np.concatenate(
                [
                    ee_position,
                    ee_velocity,
                    fingers_qpos,
                    object_position,
                    object_rotation,
                    object_velp,
                    object_velr,
                ]).copy()     
               
        result = {
            "observation": obs,
            "achieved_goal": achieved_goal,
            "desired_goal": desired_goal,
        }

        # print("Agent Obs: ", result)

        return result

    def _get_fingers_qpos(self, joint_qpos) -> np.ndarray:
        return [joint_qpos[finger_name] for finger_name in self._gripper_joint_names]

    def set_action(self, action, ee_pos) -> None:
        # for the pick and place task

        pos_ctrl, gripper_ctrl = action[:3].copy(), action[3].copy()

        a, b = self.gripper_ctrl_range
        scaled_gripper_ctrl = a + (b - a) * (gripper_ctrl + 1) / 2
        # gripper_ctrl = np.clip(gripper_ctrl, self._gripper_ctrl_range[0][0], self._gripper_ctrl_range[0][1])
        # fingers_half_width = gripper_ctrl / 2


        # control the gripper
        self._ctrl[-2:] = scaled_gripper_ctrl

        # control the end-effector with mocap body
        pos_offset = pos_ctrl * 0.05    # the maximum distance the end-effector can move in one step (5cm)
        mocap_xpos = ee_pos[self.ee_name]["xpos"] + pos_offset
        mocap_xpos[2] = np.max((0, mocap_xpos[2]))

        return mocap_xpos, self._initial_grasp_site_xquat, self._ctrl
    
    def set_action_space(self, action_space) -> None:
        self._action_space = action_space

    def sample_object(self, np_random) -> dict:
        object_position = self._initial_object_qpos[0:3].copy()
        noise = np_random.uniform(self._obj_range_low, self._obj_range_high)
        object_position += noise
        object_qpos = np.concatenate([object_position, self._initial_object_qpos[3:7].copy()])
        # self.set_joint_qpos({"obj_joint": object_qpos})
        return {self.obj_joint_name: object_qpos}
    
    def step(self, action, ee_pos_quat) -> tuple[dict, dict]:
        self._current_episode_step += 1

        action = np.clip(action, self._action_space.low, self._action_space.high)
        mocap_xpos, mocap_xquat, ctrl = self.set_action(action, ee_pos_quat)
        grasp_mocap = {self.mocap_name: {'pos': mocap_xpos, 'quat': mocap_xquat}}
        gripper_qpos = {name: np.array([ctrl[i-2]]) for i, name in enumerate(self.gripper_joint_names)}
        return grasp_mocap, gripper_qpos
    
    def reset(self, np_random) -> tuple[dict, dict]:
        self._current_episode_step = 0

        obj_sampled = self.sample_object(np_random)
        grasp_mocap = {self.mocap_name: {'pos': self.initial_grasp_site_xpos, 'quat': self.initial_grasp_site_xquat}}
        goal_mocap = {self.goal_name: {'pos': self.sample_goal(obj_sampled, np_random), 
                                       'quat': self.initial_grasp_site_xquat}}
        joint_neutral = self.get_joint_neutral()

        mocap_pos_quat = {**grasp_mocap, **goal_mocap}
        joint_qpos = {**obj_sampled, **joint_neutral}

        return mocap_pos_quat, joint_qpos

    def _goal_distance(self, goal_a, goal_b) -> SupportsFloat:
        assert goal_a.shape == goal_b.shape
        if goal_a.ndim == 1:
            diff = goal_a[:3] - goal_b[:3]
        elif goal_a.ndim == 2:
            diff = goal_a[:, :3] - goal_b[:, :3]
        else:
            raise ValueError("Unsupport goal shape: ", goal_a.shape)
        
        return np.linalg.norm(diff, axis=-1 if goal_a.ndim == 2 else None)
            

    def is_success(self, achieved_goal, desired_goal, env_id) -> np.float32:
        d = self._goal_distance(achieved_goal, desired_goal)
        if d < self._distance_threshold:
            print(f"{env_id} Agent {self.name} Task Sussecced: achieved goal: ", achieved_goal, "desired goal: ", desired_goal)
        return (d < self._distance_threshold).astype(np.float32)


    def _compute_reward_ndim1(self, achieved_goal, desired_goal) -> SupportsFloat:
        d = self._goal_distance(achieved_goal, desired_goal)
        if self._task == "reach":
            if d < self._distance_threshold:
                # is_success
                reward = 1.0
            else:
                # sparse reward
                reward = -1
            return reward
        elif self._task == "pick_and_place":
            if d < self._distance_threshold:
                # is_success
                reward = 1.0
            else:
                reward = self._compute_pick_and_place_reward(achieved_goal, desired_goal)

            return reward
        else:
            raise ValueError("Unsupport task type: ", self._task)
        
    def _compute_reward_ndim2(self, achieved_goal, desired_goal) -> SupportsFloat:
        d = self._goal_distance(achieved_goal, desired_goal)
        if self._task == "reach":
            rewards = np.zeros(len(achieved_goal))
            for i in range(len(achieved_goal)):
                if d[i] < self._distance_threshold:
                    # is_success
                    rewards[i] = 1.0
                else:
                    rewards[i] = -1.0
            return rewards
        elif self._task == "pick_and_place":
            rewards = np.zeros(len(achieved_goal))
            for i in range(len(achieved_goal)):
                if d[i] < self._distance_threshold:
                    # is_success
                    rewards[i] = 1.0
                else:
                    rewards[i] = self._compute_pick_and_place_reward(achieved_goal[i], desired_goal[i])

            return rewards
        else:
            raise ValueError("Unsupport task type: ", self._task)
        
    def compute_reward(self, achieved_goal, desired_goal) -> SupportsFloat:
        if achieved_goal.ndim == 1:
            return self._compute_reward_ndim1(achieved_goal, desired_goal)
        else:
            return self._compute_reward_ndim2(achieved_goal, desired_goal)
        

    def _compute_pick_and_place_reward(self, achieved_goal, desired_goal) -> SupportsFloat:
        assert achieved_goal.shape == desired_goal.shape
        ee_position = achieved_goal[:3]
        object_position = achieved_goal[3:6]
        goal_position = desired_goal[:3]

        reward = 0

        # 1. ee to object distance
        ee_to_obj_distance = np.linalg.norm(ee_position - object_position)
        reward += -ee_to_obj_distance

        # 2. object to goal distance
        obj_to_goal_distance = np.linalg.norm(object_position - goal_position)
        reward += -obj_to_goal_distance

        return reward
        


class FrankaEnv(OrcaGymLocalEnv):
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

        self._agents: list[FrankaRobot] = []
        for agent_name in agent_names:
            agent = FrankaRobot(agent_name, task, max_episode_steps, distance_threshold, 
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
        action_size = 4
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
        agents_to_reset : list[FrankaRobot] = []
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

    def _get_obs(self, agents : list[FrankaRobot]) -> list[dict]:
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
        
    def _reset_agents(self, agents : list[FrankaRobot]) -> None:
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

    def _set_joint_neutral(self, agents : list[FrankaRobot]) -> None:
        joint_qpos = {}
        for agent in agents:
            joint_qpos.update(agent.get_joint_neutral())

        self.set_joint_qpos(joint_qpos)


    def get_observation(self, obs=None):
        if obs is not None:
            return obs
        else:
            return self._get_obs(self._agents).copy()