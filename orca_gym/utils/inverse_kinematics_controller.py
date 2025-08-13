
import numpy as np
from orca_gym.adapters.robomimic.robomimic_env import RobomimicEnv
from scipy.spatial.transform import Rotation


class InverseKinematicsController:
    """
    A class to control the inverse kinematics of a robot arm.
    """

    def __init__(self, env: RobomimicEnv, site_id, dof_indices, lamba_value=1e-3, alpha_value=0.2):
        """
        Initialize the InverseKinematicsController with a robot.

        Args:
            robot: The robot object to control.
        """
        self.env = env
        self.site_id = site_id
        self.dof_indices = dof_indices

        self.goal_pos = None
        self.goal_quat = None

        self._lambda_value = lamba_value
        self._alpha_value = alpha_value


    def set_goal(self, pos, quat):
        self.goal_pos = pos
        self.goal_quat = quat

    def set_lambda(self, lambda_value):
        """
        Set the lambda value for the inverse kinematics solver.

        Args:
            lambda_: The lambda value to set.
        """
        self._lamba_value = lambda_value

    def set_alpha(self, alpha_value):
        """
        Set the alpha value for the inverse kinematics solver.

        Args:
            alpha_: The alpha value to set.
        """
        self._alpha_value = alpha_value

    def compute_inverse_kinematics(self):
        """
        Compute the inverse kinematics for the robot to reach the target position and orientation.

        Args:
            target_position: The desired position of the end effector.
            target_orientation: The desired orientation of the end effector.

        Returns:
            joint_angles: The computed joint angles to reach the target position and orientation.
        """
        site_name = self.env.model.site_id2name(self.site_id)
        site_dict = self.env.query_site_pos_and_quat([site_name])
        ee_xpos = site_dict[site_name]['xpos']
        ee_xquat = site_dict[site_name]['xquat']

        delta_pos = self.goal_pos - ee_xpos
        
        goal_rotation = Rotation.from_quat(self.goal_quat[[1, 2, 3, 0]])
        ee_rotation = Rotation.from_quat(ee_xquat[[1, 2, 3, 0]])
        error_rot = goal_rotation * ee_rotation.inv()
        delta_rot = error_rot.as_rotvec()

        e = np.concatenate((delta_pos, delta_rot))
        if (np.linalg.norm(e) < 1e-3):
            return np.zeros(self.env.model.nv)

        jac_pos = np.zeros((3, self.env.model.nv))
        jac_rot = np.zeros((3, self.env.model.nv))
        self.env.mj_jacSite(jac_pos, jac_rot, self.site_id)

        jac_pos_sub = jac_pos[:, self.dof_indices]
        jac_rot_sub = jac_rot[:, self.dof_indices]
        J_sub = np.vstack((jac_pos_sub, jac_rot_sub))

        J_JT= J_sub @ J_sub.T

        damping = self._lambda_value ** 2 * np.eye(6)
        detal_q_sub = J_sub.T @ np.linalg.solve(J_JT + damping, e)

        detal_q = np.zeros(self.env.model.nv)
        detal_q[self.dof_indices] = detal_q_sub 

        return detal_q * self._alpha_value