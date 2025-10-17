"""
This file contains the gym environment wrapper that is used
to provide a standardized environment API for training policies and interacting
with metadata present in datasets.
"""
import json
import numpy as np
from copy import deepcopy
from typing import Any, Dict, Optional, Tuple, Union

from orca_gym.environment import OrcaGymLocalEnv
from orca_gym.environment import OrcaGymLocalEnv
from orca_gym.adapters.robomimic.task.abstract_task import AbstractTask
import robomimic.envs.env_base as EB
import robomimic.utils.obs_utils as ObsUtils

import gymnasium as gym
from gymnasium import error, spaces
from gymnasium.spaces import Space


class RobomimicEnv(OrcaGymLocalEnv):
    metadata = {'render_modes': ['human', 'none'], 'version': '0.0.1', 'render_fps': 30}

    """
    Below we outline important methods that each EnvBase subclass needs to implement or override. 
    The implementation mostly follows the OpenAI-Gym convention.

    __init__(self, ...)

        Create the wrapped environment instance and assign it to self.env. 
        For example, in EnvGym it simply calls self.env = gym.make(env_name, **kwargs). 
        Refer to EnvRobosuite as an example of handling the constructor arguments.

    step(self, action)

        Take a step in the environment with an input action, return (observation, reward, done, info).

    reset(self)

        Reset the environment, return observation

    render(self, mode="human", height=None, width=None, camera_name=None, **kwargs)

        Render the environment if mode=='human'. 
        Return an RGB array if mode=='rgb_array'

    get_observation(self, obs=None)

        Return the current environment observation as a dictionary, unless obs is not None. 
        This function should process the raw environment observation to align with the input expected by the policy model. 
        For example, it should cast an image observation to float with value range 0-1 and shape format [C, H, W].

    is_success(self)

        Check if the task condition(s) is reached. 
        Should return a dictionary { str: bool } with at least a “task” key for the overall task success, 
        and additional optional keys corresponding to other task criteria.

    serialize(self)

        Aggregate and return all information needed to re-instantiate this environment in a dictionary. 
        This is the same as @env_meta - environment metadata stored in hdf5 datasets and used in robomimic/utils/env_utils.py.

    create_for_data_processing(cls, ...)

        (Optional) A class method that initializes an environment for data-postprocessing purposes, 
        which includes extracting observations, labeling dense / sparse rewards, and annotating dones in transitions. 
        This function should at least designate the list of observation modalities that are image / low-dimensional observations 
        by calling robomimic.utils.obs_utils.initialize_obs_utils_with_obs_specs().

    get_goal(self)

        (Optional) Get goal for a goal-conditional task

    set_goal(self, goal)

        (optional) Set goal with external specification

    get_state(self)

        (Optional) This function should return the underlying state of a simulated environment. 
        Should be compatible with reset_to.

    reset_to(self, state)

        (Optional) Reset to a specific simulator state. Useful for reproducing results.
    """


    def __init__(
        self,
        frame_skip: int,
        orcagym_addr: str,
        agent_names: list[str],
        time_step: float,        
        action_step: int,        
        camera_config: Dict[str, Any],
        **kwargs        
    ):
        self._action_step = action_step
        self._camera_config = camera_config
        self._task = None
        task = kwargs["task"]
        if isinstance(task, AbstractTask):
            self._task = task

        super().__init__(
            frame_skip = frame_skip,
            orcagym_addr = orcagym_addr,
            agent_names = agent_names,
            time_step = time_step,            
            **kwargs
        )

    def check_success(self):
        """
        Check if the task condition(s) is reached. Should return a dictionary
        { str: bool } with at least a "task" key for the overall task success,
        and additional optional keys corresponding to other task criteria.
        """        
        raise NotImplementedError
    
    def step(self, action) -> tuple:
        raise NotImplementedError
    
    def get_action_step(self):
        return self._action_step    
    
    def get_camera_config(self):
        return self._camera_config
    
    def get_env_version(self):
        """
        The dataset version must correspond to the version of the environment that was used to collect the data.
        """
        raise NotImplementedError
    
    def get_observation(self, obs=None):
        """
        Return the current environment observation as a dictionary, unless obs is not None.
        This function should process the raw environment observation to align with the input expected by the policy model.
        For example, it should cast an image observation to float with value range 0-1 and shape format [C, H, W].
        """
        raise NotImplementedError
    
    def normalize_action(self, action, min_action, max_action):
        """
        将原始动作归一化到 [-1, 1] 范围。
        
        :param action: 原始动作值
        :param min_action: 动作的最小值
        :param max_action: 动作的最大值
        :return: 归一化后的动作值
        """
        normalized_action = 2 * (action - min_action) / (max_action - min_action) - 1
        return np.clip(normalized_action, -1.0, 1.0)

    def denormalize_action(self, normalized_action, min_action, max_action):
        """
        将归一化的动作值反归一化回原始范围。
        
        :param normalized_action: 归一化后的动作值
        :param min_action: 动作的最小值
        :param max_action: 动作的最大值
        :return: 反归一化后的原始动作值
        """
        original_action = (normalized_action + 1) / 2 * (max_action - min_action) + min_action
        return original_action    