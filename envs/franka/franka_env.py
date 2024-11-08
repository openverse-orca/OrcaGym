import numpy as np
from gymnasium.core import ObsType
from envs import OrcaGymLocalEnv, OrcaGymRemoteEnv
from orca_gym.utils import rotations
from typing import Optional, Any, SupportsFloat
from gymnasium import spaces


class FrankaRobot:
    def __init__(self, 
                 agent_name: str, 
                 reward_type: str,
                 has_object: bool,
                 block_gripper: bool,
                 distance_threshold: float,
                 goal_xy_range: float,
                 obj_xy_range: float,
                 goal_x_offset: float,
                 goal_z_range: float):
        
        self._agent_name = agent_name
        self._reward_type = reward_type
        self._has_object = has_object
        self._block_gripper = block_gripper
        self._distance_threshold = distance_threshold
        self._goal_xy_range = goal_xy_range
        self._obj_xy_range = obj_xy_range
        self._goal_x_offset = goal_x_offset
        self._goal_z_range = goal_z_range

        self._neutral_joint_values = np.array([0.00, 0.41, 0.00, -1.85, 0.00, 2.26, 0.79, 0.00, 0.00])
        self._obj_joint_name = self.name('obj_joint')
        self._obj_site_name = self.name('obj_site')
        self._ee_name = self.name('ee_center_site')
        self._mocap_name = self.name('panda_mocap')
        self._arm_joint_names = self.names(["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7"])
        self._gripper_joint_names = self.names(["finger_joint1", "finger_joint2"])

        self._goal_range_low = np.array([-self._goal_xy_range / 2 + goal_x_offset, -self._goal_xy_range / 2, 0])
        self._goal_range_high = np.array([self._goal_xy_range / 2 + goal_x_offset, self._goal_xy_range / 2, self._goal_z_range])
        self._obj_range_low = np.array([-self._obj_xy_range / 2, -self._obj_xy_range / 2, 0])
        self._obj_range_high = np.array([self._obj_xy_range / 2, self._obj_xy_range / 2, 0])     

        self._ctrl = self._neutral_joint_values.copy()


    def name(self, name : str) -> str:
        return f"{self._agent_name}_{name}"
    
    def names(self, names : list[str]) -> list[str]:
        return [self.name(name) for name in names]
    
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
    def goal(self) -> np.ndarray:
        return self._goal
    
    def set_gripper_ctrl_range(self, actuator_dict) -> None:
        if not hasattr(self, "_gripper_ctrl_range"):
            self._gripper_ctrl_range = []
            
        for key, item in actuator_dict.items():
            if key in self._gripper_joint_names:
                self._gripper_ctrl_range.append(["CtrlRange"])

    def set_init_joint_state(self, obj_qpos, obj_site_xpos) -> None:
        self._initial_object_qpos = obj_qpos.copy()
        self._initial_object_qpos[2] = 0.02  # 放置于地面
        self._initial_object_xpos = obj_site_xpos.copy()
        self._initial_object_xpos[2] = 0.02        

    def set_initial_grasp_site_xpos(self, site_pos_quat) -> None:
        self._initial_grasp_site_xpos = site_pos_quat[self.ee_name]['xpos']
        self._initial_grasp_site_xquat = site_pos_quat[self.ee_name]['xquat']

    def sample_goal(self, ee_pos_quat, np_random) -> np.ndarray:
        # 训练reach时，任务是移动抓夹，goal以抓夹为原点采样
        if not self._has_object:
            ee_position = ee_pos_quat['xpos'].copy()
            goal = ee_position.copy()
        else:
            goal = self._initial_object_xpos.copy()

        noise = np_random.uniform(self._goal_range_low, self._goal_range_high)

        # 避免与obj过度接近的情况
        for i in range(2):
            if noise[i] < self._distance_threshold + 0.01 and noise[i] > 0:
                noise[i] = self._distance_threshold + 0.01
            if noise[i] > -self._distance_threshold - 0.01 and noise[i] < 0:
                noise[i] = -self._distance_threshold - 0.01

        # for the pick and place task
        if not self._block_gripper and self._goal_z_range > 0.0:
            if np_random.random() < 0.3:
                noise[2] = 0.0
        
        goal += noise
        goal[2] = max(0.02, goal[2])  # 确保在地面以上，考虑方块的高度，最低为0.02
        
        self._goal = goal.copy()
        return goal

    def get_obs(self, site_pos_quat, site_pos_mat, site_xvalp, site_xvalr, joint_qpos) -> dict:
        # robot
        ee_position = site_pos_quat[self.ee_name]['xpos'].copy()
        ee_velocity = site_xvalp[self.ee_name].copy() * self.dt

        if not self._block_gripper:
            fingers_qpos = self._get_fingers_qpos(joint_qpos)

        # object
        # object cartesian position: 3
        object_position = site_pos_mat[self.obj_site_name]['xpos'].copy()
        object_mat = site_pos_mat[self.obj_site_name]['xmat'].reshape(3, 3)

        # object rotations: 3
        object_rotation = rotations.mat2euler(object_mat)


        # object linear velocities
        object_velp = site_xvalp[self.obj_site_name].copy() * self.dt
        object_velr = site_xvalr[self.obj_site_name].copy() * self.dt

        if not self._has_object:
            achieved_goal = ee_position.copy()
            desired_goal = self.goal.copy()
        else:
            achieved_goal = object_position.copy()
            desired_goal = self.goal.copy()    

        if not self._block_gripper:
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
        else:
            obs = np.concatenate(
                    [
                        ee_position,
                        ee_velocity,
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
        return result

    def _get_fingers_qpos(self, joint_qpos) -> np.ndarray:
        return [joint_qpos[finger_name] for finger_name in self._gripper_joint_names]

    def set_action(self, action, ee_pos) -> None:
        # for the pick and place task
        if not self._block_gripper:
            pos_ctrl, gripper_ctrl = action[:3].copy(), action[3].copy()
            gripper_ctrl = np.clip(gripper_ctrl, self._gripper_ctrl_range[0, 0], self._gripper_ctrl_range[0, 1])
            fingers_half_width = gripper_ctrl / 2
        else:
            pos_ctrl = action.copy()
            fingers_half_width = 0

        # control the gripper
        self._ctrl[-2:] = fingers_half_width

        # control the end-effector with mocap body
        pos_offset = pos_ctrl * 0.05    # the maximum distance the end-effector can move in one step (5cm)
        # ee_pos = self.get_ee_position()
        mocap_xpos = ee_pos[self.ee_name]["xpos"] + pos_offset
        mocap_xpos[2] = np.max((0, mocap_xpos[2]))

        # print(f"Action: {action}, pos_offset: {pos_offset}, ee_pos: {ee_pos}, mocap_xpos: {mocap_xpos}")

        # self.set_grasp_mocap(mocap_xpos, self.initial_grasp_site_xquat)

        # 直接根据新的qpos位置设置控制量，类似于牵引示教
        # self.mj_forward()
        # joint_qpos = self.query_joint_qpos(self.arm_joint_names)
        # self.ctrl[:7] = np.array([joint_qpos[joint_name] for joint_name in self.arm_joint_names]).flat.copy()
        return mocap_xpos, self._initial_grasp_site_xquat, self._ctrl


class FrankaEnv(OrcaGymLocalEnv):
    metadata = {'render_modes': ['human', 'none'], 'version': '0.0.1'}

    def __init__(
        self,
        frame_skip: int,        
        orcagym_addr: str,
        agent_names: list,
        time_step: float,    
        render_mode: str,
        reward_type: str,
        has_object: bool,
        block_gripper: bool,
        distance_threshold: float,
        goal_xy_range: float,
        obj_xy_range: float,
        goal_x_offset: float,
        goal_z_range: float,
        **kwargs,
    ):

        # self.block_gripper = block_gripper
        # self.has_object = has_object
        self._render_mode = render_mode
        # print("Render mode: ", self._render_mode)

        super().__init__(
            frame_skip = frame_skip,
            orcagym_addr = orcagym_addr,
            agent_names = agent_names,
            time_step = time_step,            
            **kwargs,
        )

        self._reward_type = reward_type

        # self.neutral_joint_values = np.array([0.00, 0.41, 0.00, -1.85, 0.00, 2.26, 0.79, 0.00, 0.00])

        self._distance_threshold = distance_threshold

        # sample areas for the object and goal target
        # self.obj_xy_range = obj_xy_range
        # self.goal_xy_range = goal_xy_range
        # self.goal_x_offset = goal_x_offset
        # self.goal_z_range = goal_z_range


        self._agents = list[FrankaRobot]
        for agent_name in agent_names:
            agent = FrankaRobot(agent_name, reward_type, has_object, block_gripper, distance_threshold, goal_xy_range, obj_xy_range, goal_x_offset, goal_z_range)
            self._agents.append(agent)

        self._agent_ee_names = [agent.ee_name for agent in self._agents].flat.copy()
        self._agent_obj_joint_names = [agent.obj_joint_name for agent in self._agents].flat.copy()
        self._agent_obj_site_names = [agent.obj_site_name for agent in self._agents].flat.copy()
        self._agent_mocap_names = [agent.mocap_name for agent in self._agents].flat.copy()
        self._agent_gripper_joint_names = [agent.gripper_joint_names for agent in self._agents].flat.copy()
        self._agent_joint_names = [agent.joint_names for agent in self._agents].flat.copy()

        # self.goal_range_low = np.array([-self.goal_xy_range / 2 + goal_x_offset, -self.goal_xy_range / 2, 0])
        # self.goal_range_high = np.array([self.goal_xy_range / 2 + goal_x_offset, self.goal_xy_range / 2, self.goal_z_range])
        # self.obj_range_low = np.array([-self.obj_xy_range / 2, -self.obj_xy_range / 2, 0])
        # self.obj_range_high = np.array([self.obj_xy_range / 2, self.obj_xy_range / 2, 0])

        # 在 Orca 编辑器中，可以直接设置 obj 的位置，不需要这里手动调整
        # self.goal_range_low[0] += 0.6
        # self.goal_range_high[0] += 0.6
        # self.obj_range_low[0] += 0.6
        # self.obj_range_high[0] += 0.6

        # Three auxiliary variables to understand the component of the xml document but will not be used
        # number of actuators/controls: 7 arm joints and 2 gripper joints
        self.nu = self.model.nu
        # 16 generalized coordinates: 9 (arm + gripper) + 7 (object free joint: 3 position and 4 quaternion coordinates)
        self.nq = self.model.nq
        # 9 arm joints and 6 free joints
        self.nv = self.model.nv

        # control range
        # self.ctrl_range = self.model.get_actuator_ctrlrange()
        all_actuator = self.model.get_actuator()
        [agent.set_gripper_ctrl_range(all_actuator) for agent in self._agents]

        # index used to distinguish arm and gripper joints
        # self.free_joint_names = [self.joint('obj_joint')]
        # self.arm_joint_names = [self.joint("joint1"), self.joint("joint2"), self.joint("joint3"), self.joint("joint4"), self.joint("joint5"), self.joint("joint6"), self.joint("joint7")]
        # self.gripper_joint_names = [self.joint("finger_joint1"), self.joint("finger_joint2")]

        [agent.set_init_joint_state(self.query_joint_qpos([agent.obj_joint_name])[agent.obj_joint_name], 
                                    self.query_site_pos_and_mat([agent.obj_site_name])[agent.obj_site_name]['xpos'])
                                    for agent in self._agents]

        # self.initial_object_qpos = self.query_joint_qpos(["obj_joint"])["obj_joint"].copy()
        # self.initial_object_qpos[2] = 0.02  # 放置于地面
        # self.initial_object_xpos = self.query_site_pos_and_mat(["obj_obj_site"])["obj_obj_site"]['xpos'].copy()
        # self.initial_object_xpos[2] = 0.02

        self._set_init_state()

        # EE_NAME  = self.site("ee_center_site")
        # site_dict = self.query_site_pos_and_quat([EE_NAME])
        # self.initial_grasp_site_xpos = site_dict[EE_NAME]['xpos']
        # self.initial_grasp_site_xquat = site_dict[EE_NAME]['xquat']

        [agent.set_initial_grasp_site_xpos(self.query_site_pos_and_quat([agent.ee_name])) for agent in self._agents]

        [self.set_grasp_mocap(agent.mocap_name, agent.initial_grasp_site_xpos, 
                              agent.initial_grasp_site_xquat) for agent in self._agents]

        for agent in self._agents:
            agent_goal = agent.sample_goal(self.query_site_pos_and_quat([agent.ee_name]), self.np_random)
            self.set_goal_mocap(agent_goal, agent.initial_grasp_site_xquat)

        # self.goal = self._sample_goal()

        # Run generate_observation_space after initialization to ensure that the observation object's name is defined.
        self._set_obs_space()
        self._set_action_space(block_gripper)

    def _set_obs_space(self):
        self.observation_space = self.generate_observation_space(self._get_obs()[0])

    def _set_action_space(self, block_gripper : bool):
        action_size = 3 if block_gripper else 4
        self.action_space = spaces.Box(
            low=np.array([-1.0] * action_size),
            high=np.array([1.0] * action_size),
            dtype=np.float32,
        )

    def _set_init_state(self) -> None:
        # print("Set initial state")
        self.set_joint_neutral()

        ctrl = [agent.neutral_joint_values for agent in self._agents]
        self.ctrl = np.array(ctrl).flat.copy()
        assert self.ctrl.shape == (self.model.nu,)
        # self.ctrl = np.array(self.neutral_joint_values)
        self.set_ctrl(self.ctrl)

        self.reset_mocap_welds()

        self.mj_forward()


    def step(self, action) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        if len(action) != len(self._agents):
            raise ValueError("Action dimension mismatch")

        if np.array(action[0]).shape != self.action_space.shape:
            raise ValueError("Action dimension mismatch")

        # for every agent, calculate the mocap and gripper control
        ee_pos_quat = self.query_site_pos_and_quat(self._agent_ee_names)
        for agent, act in zip(self._agents, action):
            act = np.clip(act, self.action_space.low, self.action_space.high)
            mocap_xpos, mocap_xquat, ctrl = agent.set_action(act, ee_pos_quat)
            self.set_grasp_mocap(agent.mocap_name, mocap_xpos, mocap_xquat)
            self.set_gripper_ctrl(agent.gripper_joint_names, ctrl[-2:])

        self.mj_forward()
        joint_qpos = self.query_joint_qpos(self._agent_joint_names)
        self.ctrl = np.array([joint_qpos[joint_name] for joint_name in self._agent_joint_names]).flat.copy()
        
        self.do_simulation(self.ctrl, self.frame_skip)

        if self.render_mode == "human":
            self.render()

        obs = self._get_obs().copy()

        info = [{"is_success": self._is_success(agent_obs["achieved_goal"], agent_obs["desired_goal"])} for agent_obs in obs]

        terminated = [bool(agent_info["is_success"] != 0) for agent_info in info]
        truncated = [False for _ in range(len(terminated))]
        reward = [self.compute_reward(agent_obs["achieved_goal"], agent_obs["desired_goal"], agent_info) for agent_obs, agent_info in zip(obs, info)]

        return obs, reward, terminated, truncated, info

    def compute_reward(self, achieved_goal, desired_goal, info) -> SupportsFloat:
        d = self.goal_distance(achieved_goal, desired_goal)
        if self._reward_type == "sparse":
            return -(d > self._distance_threshold).astype(np.float32)
        else:
            return -d

    # def _set_action(self, action) -> None:
    #     # for the pick and place task
    #     if not self.block_gripper:
    #         pos_ctrl, gripper_ctrl = action[:3].copy(), action[3].copy()
    #         fingers_ctrl = gripper_ctrl * 0.2
    #         fingers_width = self.get_fingers_width().copy() + fingers_ctrl
    #         fingers_half_width = np.clip(fingers_width / 2, self.ctrl_range[-1, 0], self.ctrl_range[-1, 1])
    #     else:
    #         pos_ctrl = action.copy()
    #         fingers_half_width = 0

    #     # control the gripper
    #     self.ctrl[-2:] = fingers_half_width

    #     # 机器人通过mocap引导，控制不要对抗mocap，因此以当前位置作为新的控制输入
    #     # qpos_dict = self.query_joint_qpos(self.arm_joint_names)
    #     # ctrl_list = np.array([qpos_dict[name] for name in self.arm_joint_names])
    #     # self.ctrl = ctrl_list.flat.copy()    # 每一帧更新控制输入

    #     # control the end-effector with mocap body
    #     pos_offset = pos_ctrl * 0.05    # the maximum distance the end-effector can move in one step (5cm)
    #     ee_pos = self.get_ee_position()
    #     mocap_xpos = ee_pos + pos_offset
    #     mocap_xpos[2] = np.max((0, mocap_xpos[2]))

    #     # print(f"Action: {action}, pos_offset: {pos_offset}, ee_pos: {ee_pos}, mocap_xpos: {mocap_xpos}")

    #     self.set_grasp_mocap(mocap_xpos, self.initial_grasp_site_xquat)

    #     # 直接根据新的qpos位置设置控制量，类似于牵引示教
    #     self.mj_forward()
    #     joint_qpos = self.query_joint_qpos(self.arm_joint_names)
    #     self.ctrl[:7] = np.array([joint_qpos[joint_name] for joint_name in self.arm_joint_names]).flat.copy()

    def _get_obs(self) -> dict:
        site_pos_quat = self.query_site_pos_and_quat(self._agent_ee_names)
        site_pos_mat = self.query_site_pos_and_mat(self._agent_obj_site_names)
        site_xvalp, site_xvalr = self.query_site_xvalp_xvalr(self._agent_obj_site_names)
        joint_qpos = self.query_joint_qpos(self._agent_gripper_joint_names)

        obs = [agent.get_obs(site_pos_quat, site_pos_mat, site_xvalp, site_xvalr, joint_qpos) for agent in self._agents]
        
        return obs
        
        # # robot
        # EE_NAME = self.site("ee_center_site")
        # ee_position = self.query_site_pos_and_quat([EE_NAME])[EE_NAME]['xpos'].copy()
        # ee_xvalp, _ = self.query_site_xvalp_xvalr([EE_NAME])
        # ee_velocity = ee_xvalp[EE_NAME].copy() * self.dt

        # if not self.block_gripper:
        #     fingers_width = self.get_fingers_width().copy()

        # # object
        # # object cartesian position: 3
        # object_pos_mat = self.query_site_pos_and_mat(["obj_obj_site"])["obj_obj_site"]
        # object_position = object_pos_mat['xpos'].copy()

        # # object rotations: 3
        # object_rotation = rotations.mat2euler(object_pos_mat['xmat'].reshape(3, 3)).copy()

        # # object linear velocities
        # obj_xvalp, obj_xvalr = self.query_site_xvalp_xvalr(["obj_obj_site"])
        # object_velp = obj_xvalp["obj_obj_site"].copy() * self.dt
        # object_velr = obj_xvalr["obj_obj_site"].copy() * self.dt

        # if not self.has_object:
        #     achieved_goal = ee_position.copy()
        #     desired_goal = self.goal.copy()
        # else:
        #     achieved_goal = object_position.copy()
        #     desired_goal = self.goal.copy()    

        # if not self.block_gripper:
        #     obs = np.concatenate(
        #             [
        #                 ee_position,
        #                 ee_velocity,
        #                 fingers_width,
        #                 object_position,
        #                 object_rotation,
        #                 object_velp,
        #                 object_velr,
        #             ]).copy()            
        # else:
        #     obs = np.concatenate(
        #             [
        #                 ee_position,
        #                 ee_velocity,
        #                 object_position,
        #                 object_rotation,
        #                 object_velp,
        #                 object_velr,
        #             ]).copy()

        # result = {
        #     "observation": obs,
        #     "achieved_goal": achieved_goal,
        #     "desired_goal": desired_goal,
        # }
        # return result

    def _is_success(self, achieved_goal, desired_goal) -> np.float32:
        d = self.goal_distance(achieved_goal, desired_goal)
        if d < self._distance_threshold:
            print("Task Sussecced: achieved goal: ", achieved_goal, "desired goal: ", desired_goal)
        return (d < self._distance_threshold).astype(np.float32)

    def reset_model(self):
        self._set_init_state()
        self.set_grasp_mocap(self.initial_grasp_site_xpos, self.initial_grasp_site_xquat)

        self._sample_object()
        self.goal = self._sample_goal()
        self.mj_forward()
        obs = self._get_obs().copy()
        return obs

    # custom methods
    # -----------------------------
    def reset_mocap_welds(self) -> None:
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

    def goal_distance(self, goal_a, goal_b) -> SupportsFloat:
        assert goal_a.shape == goal_b.shape
        return np.linalg.norm(goal_a - goal_b, axis=-1)

    def set_grasp_mocap(self, mocap_name, position, orientation) -> None:
        mocap_pos_and_quat_dict = {mocap_name: {'pos': position, 'quat': orientation}}
        self.set_mocap_pos_and_quat(mocap_pos_and_quat_dict)

    def set_gripper_ctrl(self, gripper_joint_names, ctrl) -> None:
        gripper_joint_qpos = {}
        for name, value in zip(gripper_joint_names, ctrl):
            gripper_joint_qpos[name] = np.array([value])
        self.set_joint_qpos(gripper_joint_qpos)

    def set_goal_mocap(self, position, orientation) -> None:
        mocap_pos_and_quat_dict = {"goal_goal": {'pos': position, 'quat': orientation}}
        self.set_mocap_pos_and_quat(mocap_pos_and_quat_dict)

    def set_joint_neutral(self) -> None:
        # assign value to arm joints
        arm_joint_qpos = {}
        for agent in self._agents:
            for name, value in zip(agent.arm_joint_names, agent.neutral_joint_values[0:7]):
                arm_joint_qpos[name] = np.array([value])
        
        self.set_joint_qpos(arm_joint_qpos)

        # assign value to finger joints
        gripper_joint_qpos = {}
        for name, value in zip(self.gripper_joint_names, self.neutral_joint_values[7:9]):
            gripper_joint_qpos[name] = np.array([value])
        self.set_joint_qpos(gripper_joint_qpos)

    def _sample_goal(self) -> np.ndarray:
        # 训练reach时，任务是移动抓夹，goal以抓夹为原点采样
        if not self.has_object:
            # goal = self.initial_grasp_site_xpos.copy()
            EE_NAME = self.site("ee_center_site")
            ee_position = self.query_site_pos_and_quat([EE_NAME])[EE_NAME]['xpos'].copy()
            goal = ee_position.copy()
        else:
            goal = self.initial_object_xpos.copy()

        noise = self.np_random.uniform(self.goal_range_low, self.goal_range_high)

        # 避免与obj过度接近的情况
        for i in range(2):
            if noise[i] < self._distance_threshold + 0.01 and noise[i] > 0:
                noise[i] = self._distance_threshold + 0.01
            if noise[i] > -self._distance_threshold - 0.01 and noise[i] < 0:
                noise[i] = -self._distance_threshold - 0.01

        # for the pick and place task
        if not self.block_gripper and self.goal_z_range > 0.0:
            if self.np_random.random() < 0.3:
                noise[2] = 0.0
        
        goal += noise
        goal[2] = max(0.02, goal[2])  # 确保在地面以上，考虑方块的高度，最低为0.02
        # print("Goal position: ", goal)
        self.set_goal_mocap(goal, self.initial_grasp_site_xquat)
        return goal

    def _sample_object(self) -> None:
        object_position = self.initial_object_qpos[0:3].copy()
        noise = self.np_random.uniform(self.obj_range_low, self.obj_range_high)
        object_position += noise
        object_qpos = np.concatenate([object_position, self.initial_object_qpos[3:7].copy()])
        self.set_joint_qpos({"obj_joint": object_qpos})
        # print("Object position: ", object_position)

    def get_ee_position(self) -> np.ndarray:
        xpos = self.query_site_pos_and_mat([self.site("ee_center_site")])[self.site("ee_center_site")]['xpos'].copy()
        # print("EE position: ", xpos)
        return xpos

    def get_fingers_width(self) -> np.ndarray:
        qpos_dict = self.query_joint_qpos([self.joint("finger_joint1"), self.joint("finger_joint2")])
        finger1 = qpos_dict[self.joint("finger_joint1")]
        finger2 = qpos_dict[self.joint("finger_joint2")]
        return finger1 + finger2

    def get_observation(self, obs=None):
        if obs is not None:
            return obs
        else:
            return self._get_obs().copy()