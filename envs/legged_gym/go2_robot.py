
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
        
        self._neutral_joint_angles = {"FL_hip_joint": 0.0, "FL_thigh_joint": 0.8, "FL_calf_joint": -1.2,
                                      "FR_hip_joint": 0.0, "FR_thigh_joint": 0.8, "FR_calf_joint": -1.2,
                                      "RL_hip_joint": 0.0, "RL_thigh_joint": 0.8, "RL_calf_joint": -1.2,
                                      "RR_hip_joint": 0.0, "RR_thigh_joint": 0.8, "RR_calf_joint": -1.2}
        
        self._leg_actuator_names = self.name_space_list(["FR_hip", "FR_thigh", "FR_calf",
                                                        "FL_hip", "FL_thigh", "FL_calf",
                                                        "RR_hip", "RR_thigh", "RR_calf",
                                                        "RL_hip", "RL_thigh", "RL_calf"])
        
        self._ctrl = np.zeros(len(self._leg_actuator_names))
