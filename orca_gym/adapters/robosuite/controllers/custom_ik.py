import numpy as np
from scipy.spatial.transform import Rotation

import orca_gym.adapters.robosuite.utils.transform_utils as T
from orca_gym.adapters.robosuite.controllers.base_controller import Controller
from orca_gym.adapters.robosuite.utils.control_utils import *


class CustomInverseKinematicsController(Controller):
    """
    Custom inverse kinematics controller using damped least squares (DLS) method.

    Designed for MuJoCo position actuators. Outputs absolute target joint positions
    by computing IK delta on top of the current actual joint positions each step,
    which prevents drift accumulation.

    Control input: (x, y, z, rx, ry, rz) in world frame (absolute pose).
    """

    def __init__(
        self,
        sim,
        eef_name,
        joint_indexes,
        actuator_range,
        input_max=1,
        input_min=-1,
        output_max=0.05,
        output_min=-0.05,
        kp=0.25,
        policy_freq=20,
        lambda_value=1e-3,
        alpha_value=0.2,
        convergence_threshold=1e-3,
        singularity_threshold=1e-4,
        max_joint_delta=0.1,
        pos_scale=1.0,
        rot_scale=0.5,
        use_delta=False,
        max_iter=1,
        **kwargs,
    ):
        super().__init__(
            sim,
            eef_name,
            joint_indexes,
            actuator_range,
        )

        self.control_dim = 6

        self.input_max = self.nums2array(input_max, self.control_dim)
        self.input_min = self.nums2array(input_min, self.control_dim)
        self.output_max = self.nums2array(output_max, self.control_dim)
        self.output_min = self.nums2array(output_min, self.control_dim)

        self.kp = self.nums2array(kp, self.joint_dim)
        self.control_freq = policy_freq

        self.lambda_value = lambda_value
        self.alpha_value = alpha_value
        self.convergence_threshold = convergence_threshold
        self.singularity_threshold = singularity_threshold
        self.max_joint_delta = max_joint_delta
        self.pos_scale = pos_scale
        self.rot_scale = rot_scale
        self.use_delta = use_delta
        self.max_iter = max_iter

        self.goal_pos = np.array(self.initial_ee_pos)
        self.goal_ori = np.array(self.initial_ee_ori_mat)

        self.joint_limits = self._get_joint_limits()
        self.current_control = np.array(self.initial_joint)

    def _get_joint_limits(self):
        limits = self.gym.model.get_joint_qposrange(self.joint_index)
        if limits is None or len(limits) == 0:
            return None
        if np.all(limits == 0):
            return None
        return limits

    def set_goal(self, action, set_pos=None, set_ori=None):
        self.update()

        if set_pos is not None:
            self.goal_pos = np.array(set_pos)
        elif self.use_delta:
            scaled_delta = self.scale_action(action) if action is not None else np.zeros(6)
            self.goal_pos = set_goal_position(scaled_delta[:3], self.ee_pos, position_limit=None, set_pos=None)
        else:
            self.goal_pos = np.array(action[:3])

        if set_ori is not None:
            self.goal_ori = np.array(set_ori)
        elif self.use_delta:
            scaled_delta = self.scale_action(action) if action is not None else np.zeros(6)
            self.goal_ori = set_goal_orientation(scaled_delta[3:], self.ee_ori_mat, orientation_limit=None, set_ori=None)
        else:
            self.goal_ori = T.quat2mat(T.axisangle2quat(np.array(action[3:6])))

    def run_controller(self):
        self.update()

        # 预计算当前物理状态下的Jacobian（仅一次，所有迭代共用）
        jac_site_dict = self.gym.mj_jac_site([self.eef_name])
        jac_pos = jac_site_dict[self.eef_name]['jacp'].reshape((3, -1))[:, self.qvel_index]
        jac_rot = jac_site_dict[self.eef_name]['jacr'].reshape((3, -1))[:, self.qvel_index]
        J = np.vstack((jac_pos, jac_rot))
        J_JT = J @ J.T

        manipulability = np.sqrt(max(0.0, np.linalg.det(J_JT)))
        adaptive_lambda = (
            self.lambda_value * (1.0 + self.singularity_threshold / (manipulability + 1e-10))
            if manipulability < self.singularity_threshold
            else self.lambda_value
        )
        damping = adaptive_lambda ** 2 * np.eye(6)

        try:
            inv_factor = np.linalg.solve(J_JT + damping, np.eye(6))
        except np.linalg.LinAlgError:
            super().run_controller()
            return self.current_control

        # 以当前物理状态为起点，做虚拟多步迭代（固定Jacobian线性化）
        virtual_q = self.joint_pos.copy()
        virtual_ee_pos = self.ee_pos.copy()
        virtual_ee_ori = self.ee_ori_mat.copy()

        for _ in range(self.max_iter):
            delta_pos = self.goal_pos - virtual_ee_pos
            goal_rot = Rotation.from_matrix(self.goal_ori)
            ee_rot = Rotation.from_matrix(virtual_ee_ori)
            delta_rot = (goal_rot * ee_rot.inv()).as_rotvec()

            e = np.concatenate((delta_pos * self.pos_scale, delta_rot * self.rot_scale))
            if np.linalg.norm(e) < self.convergence_threshold:
                break

            delta_q = J.T @ (inv_factor @ e) * self.alpha_value
            delta_norm = np.linalg.norm(delta_q)
            if delta_norm > self.max_joint_delta:
                delta_q *= self.max_joint_delta / delta_norm

            virtual_q = virtual_q + delta_q
            # 用Jacobian线性近似更新虚拟末端位姿
            virtual_ee_pos = virtual_ee_pos + jac_pos @ delta_q
            rotvec = jac_rot @ delta_q
            if np.linalg.norm(rotvec) > 1e-10:
                virtual_ee_ori = (
                    Rotation.from_rotvec(rotvec) * Rotation.from_matrix(virtual_ee_ori)
                ).as_matrix()

        if not np.array_equal(virtual_q, self.joint_pos):
            if self.joint_limits is not None:
                virtual_q = np.clip(virtual_q, self.joint_limits[:, 0], self.joint_limits[:, 1])
            self.current_control = virtual_q

        super().run_controller()
        return self.current_control

    def _compute_inverse_kinematics(self, target_pos, target_ori):
        """单步DLS IK，用于外部调试或单次查询场景。"""
        delta_pos = target_pos - self.ee_pos
        goal_rotation = Rotation.from_matrix(target_ori)
        ee_rotation = Rotation.from_matrix(self.ee_ori_mat)
        delta_rot = (goal_rotation * ee_rotation.inv()).as_rotvec()

        e = np.concatenate((delta_pos * self.pos_scale, delta_rot * self.rot_scale))
        if np.linalg.norm(e) < self.convergence_threshold:
            return np.zeros(self.joint_dim)

        jac_site_dict = self.gym.mj_jac_site([self.eef_name])
        jac_pos = jac_site_dict[self.eef_name]['jacp'].reshape((3, -1))[:, self.qvel_index]
        jac_rot = jac_site_dict[self.eef_name]['jacr'].reshape((3, -1))[:, self.qvel_index]
        J = np.vstack((jac_pos, jac_rot))
        J_JT = J @ J.T

        manipulability = np.sqrt(max(0.0, np.linalg.det(J_JT)))
        adaptive_lambda = (
            self.lambda_value * (1.0 + self.singularity_threshold / (manipulability + 1e-10))
            if manipulability < self.singularity_threshold
            else self.lambda_value
        )
        damping = adaptive_lambda ** 2 * np.eye(6)

        try:
            delta_q = J.T @ np.linalg.solve(J_JT + damping, e)
        except np.linalg.LinAlgError:
            return np.zeros(self.joint_dim)

        return delta_q * self.alpha_value

    def reset_goal(self):
        self.goal_pos = np.array(self.ee_pos)
        self.goal_ori = np.array(self.ee_ori_mat)
        self.current_control = np.array(self.initial_joint)

    def set_lambda(self, lambda_value):
        self.lambda_value = lambda_value

    def set_alpha(self, alpha_value):
        self.alpha_value = alpha_value

    def set_convergence_threshold(self, threshold):
        self.convergence_threshold = threshold

    def set_initial_control(self, initial_control):
        self.current_control = np.array(initial_control)

    @property
    def name(self):
        return "CUSTOM_IK_POSE"
