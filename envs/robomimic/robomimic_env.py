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
        frame_skip: int = 5,        
        grpc_address: str = 'localhost:50051',
        agent_names: list = ['Agent0'],
        time_step: float = 0.00333333,
        **kwargs,
    ):

        super().__init__(
            frame_skip = frame_skip,
            grpc_address = grpc_address,
            agent_names = agent_names,
            time_step = time_step,            
            observation_space = None,
            **kwargs,
        )

    def _check_success(self):
        raise NotImplementedError
    
    def reset(self):
        raise NotImplementedError
    
    def step(self, action):
        raise NotImplementedError
    
    def get_env_version(self):
        """
        The dataset version must correspond to the version of the environment that was used to collect the data.
        """
        raise NotImplementedError