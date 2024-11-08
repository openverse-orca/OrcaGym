import copy
import os
from typing import Optional, Union

import numpy as np
from gymnasium import error, logger, spaces
from gymnasium.spaces import Space

from envs.robot_core import GoalEnv
from envs.orca_gym_env import OrcaGymEnv, ActionSpaceType



class BaseRobotEnv(GoalEnv):
    """Superclass for all MuJoCo fetch and hand robotic environments."""

    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
        ],
        "render_fps": 25,
    }

    def __init__(
        self,
        frame_skip: int,
        grpc_address: str,
        agent_names: list[str],
        time_step: float,        
        n_actions: int,        
        observation_space: Space,
        action_space_type: Optional[ActionSpaceType],
        action_step_count: Optional[float],             
        **kwargs 
    ):
        """Initialize the hand and fetch robot superclass.

        Args:
            frame_skip (integer): skip every `frame_skip` number of frames to speed up the simulation.
            observation_space (gym.Space): observation space of the environment.
            n_actions (integer): size of the action space.
            grpc_address (string): address of the grpc server.
            agent_names (string): name of the agent.
            time_step (float): time step for the simulation.
        """

        super().__init__(
            frame_skip = frame_skip,
            grpc_address = grpc_address,
            agent_names = [agent_names],
            time_step = time_step,            
            observation_space = observation_space,
            action_space_type = action_space_type,
            action_step_count = action_step_count,
            **kwargs 
        )

        self.initial_qpos = self.data.qpos

        self.goal = np.zeros(0)
        obs = self._get_obs()

        self.action_space = spaces.Box(-1.0, 1.0, shape=(n_actions,), dtype="float32")
        self.observation_space = spaces.Dict(
            dict(
                desired_goal=spaces.Box(
                    -np.inf, np.inf, shape=obs["achieved_goal"].shape, dtype="float64"
                ),
                achieved_goal=spaces.Box(
                    -np.inf, np.inf, shape=obs["achieved_goal"].shape, dtype="float64"
                ),
                observation=spaces.Box(
                    -np.inf, np.inf, shape=obs["observation"].shape, dtype="float64"
                ),
            )
        )


    # Env methods
    # ----------------------------
    def compute_terminated(self, achieved_goal, desired_goal, info):
        """All the available environments are currently continuing tasks and non-time dependent. The objective is to reach the goal for an indefinite period of time."""
        return False

    def compute_truncated(self, achievec_goal, desired_goal, info):
        """The environments will be truncated only if setting a time limit with max_steps which will automatically wrap the environment in a gymnasium TimeLimit wrapper."""
        return False

    def step(self, action):
        """Run one timestep of the environment's dynamics using the agent actions.

        Args:
            action (np.ndarray): Control action to be applied to the agent and update the simulation. Should be of shape :attr:`action_space`.

        Returns:
            observation (dictionary): Next observation due to the agent actions .It should satisfy the `GoalEnv` :attr:`observation_space`.
            reward (integer): The reward as a result of taking the action. This is calculated by :meth:`compute_reward` of `GoalEnv`.
            terminated (boolean): Whether the agent reaches the terminal state. This is calculated by :meth:`compute_terminated` of `GoalEnv`.
            truncated (boolean): Whether the truncation condition outside the scope of the MDP is satisfied. Timically, due to a timelimit, but
            it is also calculated in :meth:`compute_truncated` of `GoalEnv`.
            info (dictionary): Contains auxiliary diagnostic information (helpful for debugging, learning, and logging). In this case there is a single
            key `is_success` with a boolean value, True if the `achieved_goal` is the same as the `desired_goal`.
        """
        if np.array(action).shape != self.action_space.shape:
            raise ValueError("Action dimension mismatch")

        action = np.clip(action, self.action_space.low, self.action_space.high)
        self._set_action(action)

        self._mujoco_step(action)

        self._step_callback()
        obs = self._get_obs()

        info = {
            "is_success": self._is_success(obs["achieved_goal"], self.goal),
        }

        terminated = self.compute_terminated(obs["achieved_goal"], self.goal, info)
        truncated = self.compute_truncated(obs["achieved_goal"], self.goal, info)

        reward = self.compute_reward(obs["achieved_goal"], self.goal, info)

        return obs, reward, terminated, truncated, info

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        """Reset MuJoCo simulation to initial state.

        Note: Attempt to reset the simulator. Since we randomize initial conditions, it
        is possible to get into a state with numerical issues (e.g. due to penetration or
        Gimbel lock) or we may not achieve an initial condition (e.g. an object is within the hand).
        In this case, we just keep randomizing until we eventually achieve a valid initial
        configuration.

        Args:
            seed (optional integer): The seed that is used to initialize the environment's PRNG (`np_random`). Defaults to None.
            options (optional dictionary): Can be used when `reset` is override for additional information to specify how the environment is reset.

        Returns:
            observation (dictionary) : Observation of the initial state. It should satisfy the `GoalEnv` :attr:`observation_space`.
            info (dictionary): This dictionary contains auxiliary information complementing ``observation``. It should be analogous to
                the ``info`` returned by :meth:`step`.
        """
        super().reset(seed=seed)
        did_reset_sim = False
        while not did_reset_sim:
            did_reset_sim = self._reset_sim()
        self.goal = self._sample_goal().copy()
        obs = self._get_obs()

        return obs, {}

    # Extension methods
    # ----------------------------
    def _mujoco_step(self, action):
        """Advance the mujoco simulation.

        Override depending on the python binginds, either mujoco or mujoco_py
        """
        raise NotImplementedError

    def _reset_sim(self):
        """Resets a simulation and indicates whether or not it was successful.

        If a reset was unsuccessful (e.g. if a randomized state caused an error in the
        simulation), this method should indicate such a failure by returning False.
        In such a case, this method will be called again to attempt a the reset again.
        """
        return True

    def _get_obs(self):
        """Returns the observation."""
        raise NotImplementedError()

    def _set_action(self, action):
        """Applies the given action to the simulation."""
        raise NotImplementedError()

    def _is_success(self, achieved_goal, desired_goal):
        """Indicates whether or not the achieved goal successfully achieved the desired goal."""
        raise NotImplementedError()

    def _sample_goal(self):
        """Samples a new goal and returns it."""
        raise NotImplementedError()

    def _render_callback(self):
        """A custom callback that is called before rendering.

        Can be used to implement custom visualizations.
        """
        pass

    def _step_callback(self):
        """A custom callback that is called after stepping the simulation.

        Can be used to enforce additional constraints on the simulation state.
        """
        pass


class MujocoRobotEnv(BaseRobotEnv):
    """Robot base class for fetch and hand environment versions that depend on new mujoco bindings from Deepmind."""

    def __init__(self, 
                 frame_skip: int,
                 grpc_address: str,
                 agent_names: list[str],
                 time_step: float,        
                 n_actions: int,                 
                 observation_space: Space,
                 action_space_type: Optional[ActionSpaceType],
                 action_step_count: Optional[float],             
                 **kwargs):
        """Initialize mujoco environment.
        """

        super().__init__(
            frame_skip = frame_skip,
            grpc_address = grpc_address,
            agent_names = agent_names,
            time_step = time_step,
            n_actions = n_actions,            
            observation_space = observation_space,
            action_space_type = action_space_type,
            action_step_count = action_step_count,
            **kwargs)
        
        self.initial_qpos = np.copy(self.data.qpos)
        self.initial_qvel = np.copy(self.data.qvel)

    def _reset_sim(self):
        self.reset_simulation()
        self.mj_forward()
        return super()._reset_sim()


    # @property
    # def dt(self):
    #     """Return the timestep of each Gymanisum step."""
    #     return self.model.opt.timestep * self.n_substeps

    def _mujoco_step(self, action):
        # self.do_simulation(action, self.frame_skip)
        pass


class MujocoPyRobotEnv(BaseRobotEnv):
    """Robot base class for fetch and hand environment versions that depend on mujoco_py bindings."""

    def __init__(self, 
                 frame_skip: int,
                 grpc_address: str,
                 agent_names: list[str],
                 time_step: float,        
                 n_actions: int,                 
                 observation_space: Optional[Space],
                 action_space_type: Optional[ActionSpaceType],
                 action_step_count: Optional[float],             
                 **kwargs):
        """Initialize mujoco_py environment.

        The mujoco_py bindings are initialized along the respective mujoco_py_utils.

        Note: Environments that depend on mujoco_py are no longer maintained, thus a warning is created to notify the user to
        bump the environment to the latest version

        Raises:
            error.DependencyNotInstalled: if mujoco_py bindings are not installed. Install with `pip install gymnasium-robotics[mujoco-py]`
        """

        self.viewer = None
        self._viewers = {}

        logger.warn(
            "This version of the mujoco environments depends "
            "on the mujoco-py bindings, which are no longer maintained "
            "and may stop working. Please upgrade to the v4 versions of "
            "the environments (which depend on the mujoco python bindings instead), unless "
            "you are trying to precisely replicate previous works)."
        )

        super().__init__(
            frame_skip = frame_skip,
            grpc_address = grpc_address,
            agent_names = agent_names,
            time_step = time_step,
            n_actions = n_actions,            
            observation_space = observation_space,
            action_space_type = action_space_type,
            action_step_count = action_step_count,
            **kwargs)


    def _reset_sim(self):
        self.reset_simulation()
        self.mj_forward()
        return super()._reset_sim()


    # def _get_viewer(
    #     self, mode
    # ) -> Union["mujoco_py.MjViewer", "mujoco_py.MjRenderContextOffscreen"]:
    #     self.viewer = self._viewers.get(mode)
    #     if self.viewer is None:
    #         if mode == "human":
    #             self.viewer = self._mujoco_py.MjViewer(self.sim)

    #         elif mode in {
    #             "rgb_array",
    #         }:
    #             self.viewer = self._mujoco_py.MjRenderContextOffscreen(self.sim, -1)
    #         self._viewer_setup()
    #         self._viewers[mode] = self.viewer
    #     return self.viewer

    # @property
    # def dt(self):
    #     """Return the timestep of each Gymanisum step."""
    #     return self.sim.model.opt.timestep * self.sim.nsubsteps

    def _mujoco_step(self, action):
        # self.do_simulation(action, self.frame_skip)
        pass

    def _viewer_setup(self):
        """Initial configuration of the viewer. Can be used to set the camera position, for example."""
        pass
