
import numpy as np
from gymnasium.core import ObsType
from envs import OrcaGymLocalEnv, OrcaGymRemoteEnv
from orca_gym.utils import rotations
from typing import Optional, Any, SupportsFloat
from gymnasium import spaces

from .legged_robot import LeggedRobot        


class Go2Robot(LeggedRobot):
    def __init__(self, agent_name: str, task: str, max_episode_steps: int):
        super().__init__(agent_name, task, max_episode_steps)

        
        self._leg_joint_names = self.name_space_list(["FL_hip_joint", "FL_thigh_joint", "FL_calf_joint", 
                                                      "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
                                                      "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
                                                      "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint"])
        self._base_joint_name = self.name_space("base")
        self._joint_names = [self._base_joint_name] + self._leg_joint_names
        
        self._neutral_joint_angles = {"FL_hip_joint": 0.0, "FL_thigh_joint": 0.8, "FL_calf_joint": -1.2,
                                      "FR_hip_joint": 0.0, "FR_thigh_joint": 0.8, "FR_calf_joint": -1.2,
                                      "RL_hip_joint": 0.0, "RL_thigh_joint": 0.8, "RL_calf_joint": -1.2,
                                      "RR_hip_joint": 0.0, "RR_thigh_joint": 0.8, "RR_calf_joint": -1.2}
        
        self._actuator_names = self.name_space_list(["FR_hip", "FR_thigh", "FR_calf",
                                                    "FL_hip", "FL_thigh", "FL_calf",
                                                    "RR_hip", "RR_thigh", "RR_calf",
                                                    "RL_hip", "RL_thigh", "RL_calf"])
        
        self._imu_site_name = self.name_space("imu")
        self._contact_site_names = self.name_space_list(["base_contact_box", "base_contact_cylinder", "base_contact_sphere"])
        self._site_names = [self._imu_site_name] + self._contact_site_names

        self._sensor_imu_quat_name = self.name_space("imu_quat")
        self._sensor_imu_omega_name = self.name_space("imu_omega")
        self._sensor_imu_acc_name = self.name_space("imu_acc")
        self._sensor_base_touch_box_name = self.name_space("base_touch_box")
        self._sensor_base_touch_cylinder_name = self.name_space("base_touch_cylinder")
        self._sensor_base_touch_sphere_name = self.name_space("base_touch_sphere")
        self._sensor_names = [self._sensor_imu_quat_name, self._sensor_imu_omega_name, self._sensor_imu_acc_name,
                              self._sensor_base_touch_box_name, self._sensor_base_touch_cylinder_name, self._sensor_base_touch_sphere_name]
        self._touch_sensor_names = [self._sensor_base_touch_box_name, self._sensor_base_touch_cylinder_name, self._sensor_base_touch_sphere_name]

        self._contact_force_threshold = np.array([0.5, 0.5, 0.5])
        
        self._ctrl = np.zeros(len(self._actuator_names))
        self._nu = len(self._actuator_names)
        self._nq = len(self._leg_joint_names) + (7 * len(self._base_joint_name))
        self._nv = len(self._leg_joint_names) + (6 * len(self._base_joint_name))

    def get_body_contact_force(self, sensor_data : dict) -> np.ndarray:
        contact_force = np.zeros(len(self._touch_sensor_names))
        for i, sensor_name in enumerate(self._touch_sensor_names):
            contact_force[i] = sensor_data[sensor_name]
        return contact_force.flatten()
    
    def get_imu_data(self, sensor_data: dict) -> np.ndarray:
        quat = np.array([sensor_data[self._sensor_imu_quat_name]])
        omega = np.array([sensor_data[self._sensor_imu_omega_name]])
        acc = np.array([sensor_data[self._sensor_imu_acc_name]])
        imu_data = np.concatenate([quat, omega, acc], axis=1)
        return imu_data.flatten()
