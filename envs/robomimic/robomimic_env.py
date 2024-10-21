"""
This file contains the gym environment wrapper that is used
to provide a standardized environment API for training policies and interacting
with metadata present in datasets.
"""
import json
import numpy as np
from copy import deepcopy

from envs.orca_gym_env import OrcaGymEnv, ActionSpaceType
import robomimic.envs.env_base as EB
import robomimic.utils.obs_utils as ObsUtils


class RobomimicEnv(OrcaGymEnv):
    """
    Base class for RoboMimic environments.
    这个类继承自Gymnasium基类，在Robomimic框架中，等同于Gym环境
    因此，在这里需要声明Robomimic调用Gym环境的API，而具体的功能环境，需要实现这些接口
    """

    def _check_success(self):
        raise NotImplementedError
    
    def reset(self):
        raise NotImplementedError
    
    def step(self, action):
        raise NotImplementedError