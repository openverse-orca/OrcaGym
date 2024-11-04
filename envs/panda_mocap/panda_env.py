import numpy as np
from gymnasium.core import ObsType
from envs.robot_env import MujocoRobotEnv
from orca_gym.utils import rotations
from typing import Optional, Any, SupportsFloat
from gymnasium import spaces

class FrankaEnv(MujocoRobotEnv):
    def __init__(
        self,
        frame_skip: int,        
        grpc_address: str,
        agent_names: list,
        time_step: float,    
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

        self.block_gripper = block_gripper
        self.has_object = has_object

        action_size = 3
        action_size += 0 if self.block_gripper else 1

        super().__init__(
            frame_skip = frame_skip,
            grpc_address = grpc_address,
            agent_names = agent_names,
            time_step = time_step,            
            n_actions=action_size,
            observation_space = None,
            **kwargs,
        )

        self.reward_type = reward_type

        self.neutral_joint_values = np.array([0.00, 0.41, 0.00, -1.85, 0.00, 2.26, 0.79, 0.00, 0.00])

        self.distance_threshold = distance_threshold

        # sample areas for the object and goal target
        self.obj_xy_range = obj_xy_range
        self.goal_xy_range = goal_xy_range
        self.goal_x_offset = goal_x_offset
        self.goal_z_range = goal_z_range

        self.goal_range_low = np.array([-self.goal_xy_range / 2 + goal_x_offset, -self.goal_xy_range / 2, 0])
        self.goal_range_high = np.array([self.goal_xy_range / 2 + goal_x_offset, self.goal_xy_range / 2, self.goal_z_range])
        self.obj_range_low = np.array([-self.obj_xy_range / 2, -self.obj_xy_range / 2, 0])
        self.obj_range_high = np.array([self.obj_xy_range / 2, self.obj_xy_range / 2, 0])

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
        self.ctrl_range = self.model.get_actuator_ctrlrange()

        # index used to distinguish arm and gripper joints
        self.free_joint_names = ['obj_joint']
        self.arm_joint_names = [self.joint("joint1"), self.joint("joint2"), self.joint("joint3"), self.joint("joint4"), self.joint("joint5"), self.joint("joint6"), self.joint("joint7")]
        self.gripper_joint_names = [self.joint("finger_joint1"), self.joint("finger_joint2")]


        self.initial_object_qpos = self.query_joint_qpos(["obj_joint"])["obj_joint"].copy()
        self.initial_object_qpos[2] = 0.02  # 放置于地面
        self.initial_object_xpos = self.query_site_pos_and_mat(["obj_obj_site"])["obj_obj_site"]['xpos'].copy()
        self.initial_object_xpos[2] = 0.02

        self._set_init_state()

        EE_NAME  = self.site("ee_center_site")
        site_dict = self.query_site_pos_and_quat([EE_NAME])
        self.initial_grasp_site_xpos = site_dict[EE_NAME]['xpos']
        self.initial_grasp_site_xquat = site_dict[EE_NAME]['xquat']

        self.set_grasp_mocap(self.initial_grasp_site_xpos, self.initial_grasp_site_xquat)

    def _set_init_state(self) -> None:
        # print("Set initial state")
        self.set_joint_neutral()

        self.ctrl = np.array(self.neutral_joint_values)
        self.set_ctrl(self.ctrl)

        self.reset_mocap_welds()

        self.mj_forward()


    def step(self, action) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        if np.array(action).shape != self.action_space.shape:
            raise ValueError("Action dimension mismatch")

        action = np.clip(action, self.action_space.low, self.action_space.high)
        self._set_action(action)

        # ctrl = self.neutral_joint_values[:7].copy()
        # ctrl = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.do_simulation(self.ctrl, self.frame_skip)

        obs = self._get_obs().copy()

        info = {"is_success": self._is_success(obs["achieved_goal"], obs["desired_goal"])}

        terminated = bool(info["is_success"] != 0)
        truncated = self.compute_truncated(obs["achieved_goal"], obs["desired_goal"], info)
        reward = self.compute_reward(obs["achieved_goal"], obs["desired_goal"], info)

        return obs, reward, terminated, truncated, info

    def compute_reward(self, achieved_goal, desired_goal, info) -> SupportsFloat:
        d = self.goal_distance(achieved_goal, desired_goal)
        if self.reward_type == "sparse":
            return -(d > self.distance_threshold).astype(np.float32)
        else:
            return -d

    def _set_action(self, action) -> None:
        # for the pick and place task
        if not self.block_gripper:
            pos_ctrl, gripper_ctrl = action[:3].copy(), action[3].copy()
            fingers_ctrl = gripper_ctrl * 0.2
            fingers_width = self.get_fingers_width().copy() + fingers_ctrl
            fingers_half_width = np.clip(fingers_width / 2, self.ctrl_range[-1, 0], self.ctrl_range[-1, 1])
        else:
            pos_ctrl = action.copy()
            fingers_half_width = 0

        # control the gripper
        self.ctrl[-2:] = fingers_half_width

        # 机器人通过mocap引导，控制不要对抗mocap，因此以当前位置作为新的控制输入
        # qpos_dict = self.query_joint_qpos(self.arm_joint_names)
        # ctrl_list = np.array([qpos_dict[name] for name in self.arm_joint_names])
        # self.ctrl = ctrl_list.flat.copy()    # 每一帧更新控制输入

        # control the end-effector with mocap body
        pos_offset = pos_ctrl * 0.05    # the maximum distance the end-effector can move in one step (5cm)
        ee_pos = self.get_ee_position()
        mocap_xpos = ee_pos + pos_offset
        mocap_xpos[2] = np.max((0, mocap_xpos[2]))

        # print(f"Action: {action}, pos_offset: {pos_offset}, ee_pos: {ee_pos}, mocap_xpos: {mocap_xpos}")

        self.set_grasp_mocap(mocap_xpos, self.initial_grasp_site_xquat)

        # 直接根据新的qpos位置设置控制量，类似于牵引示教
        self.mj_forward()
        joint_qpos = self.query_joint_qpos(self.arm_joint_names)
        self.ctrl[:7] = np.array([joint_qpos[joint_name] for joint_name in self.arm_joint_names]).flat.copy()

    def _get_obs(self) -> dict:
        # robot
        EE_NAME = self.site("ee_center_site")
        ee_position = self.query_site_pos_and_quat([EE_NAME])[EE_NAME]['xpos'].copy()
        ee_xvalp, _ = self.query_site_xvalp_xvalr([EE_NAME])
        ee_velocity = ee_xvalp[EE_NAME].copy() * self.dt

        if not self.block_gripper:
            fingers_width = self.get_fingers_width().copy()

        # object
        # object cartesian position: 3
        object_pos_mat = self.query_site_pos_and_mat(["obj_obj_site"])["obj_obj_site"]
        object_position = object_pos_mat['xpos'].copy()

        # object rotations: 3
        object_rotation = rotations.mat2euler(object_pos_mat['xmat'].reshape(3, 3)).copy()

        # object linear velocities
        obj_xvalp, obj_xvalr = self.query_site_xvalp_xvalr(["obj_obj_site"])
        object_velp = obj_xvalp["obj_obj_site"].copy() * self.dt
        object_velr = obj_xvalr["obj_obj_site"].copy() * self.dt

        if not self.has_object:
            achieved_goal = ee_position.copy()
            desired_goal = self.goal.copy()
        else:
            achieved_goal = object_position.copy()
            desired_goal = self.goal.copy()    

        if not self.block_gripper:
            obs = np.concatenate(
                    [
                        ee_position,
                        ee_velocity,
                        fingers_width,
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

    def _is_success(self, achieved_goal, desired_goal) -> np.float32:
        d = self.goal_distance(achieved_goal, desired_goal)
        if d < self.distance_threshold:
            print("Task Sussecced: achieved goal: ", achieved_goal, "desired goal: ", desired_goal)
        return (d < self.distance_threshold).astype(np.float32)

    def _render_callback(self) -> None:
        pass

    def reset_model(self):
        # Robot_env 统一处理，这里实现空函数就可以
        pass

    def _reset_sim(self) -> bool:        
        self._set_init_state()
        self.set_grasp_mocap(self.initial_grasp_site_xpos, self.initial_grasp_site_xquat)

        self._sample_object()
        self.mj_forward()
        return True

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
                    eq_data[3:10] = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
                    self.update_equality_constraints([{"obj1_id": obj1_id, "obj2_id": obj2_id, "eq_data": eq_data}])
        self.mj_forward()

    def goal_distance(self, goal_a, goal_b) -> SupportsFloat:
        assert goal_a.shape == goal_b.shape
        return np.linalg.norm(goal_a - goal_b, axis=-1)

    def set_grasp_mocap(self, position, orientation) -> None:
        mocap_pos_and_quat_dict = {self.mocap("panda_mocap"): {'pos': position, 'quat': orientation}}
        self.set_mocap_pos_and_quat(mocap_pos_and_quat_dict)

    def set_goal_mocap(self, position, orientation) -> None:
        mocap_pos_and_quat_dict = {"goal_goal": {'pos': position, 'quat': orientation}}
        self.set_mocap_pos_and_quat(mocap_pos_and_quat_dict)

    def set_joint_neutral(self) -> None:
        # assign value to arm joints
        arm_joint_qpos = {}
        for name, value in zip(self.arm_joint_names, self.neutral_joint_values[0:7]):
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
            if noise[i] < self.distance_threshold + 0.01 and noise[i] > 0:
                noise[i] = self.distance_threshold + 0.01
            if noise[i] > -self.distance_threshold - 0.01 and noise[i] < 0:
                noise[i] = -self.distance_threshold - 0.01

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
