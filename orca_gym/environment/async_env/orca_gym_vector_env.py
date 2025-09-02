from gymnasium.vector.vector_env import ArrayType, VectorEnv
import uuid
from typing import TYPE_CHECKING, Any, Generic, TypeVar

import numpy as np
import json
import gymnasium as gym
from gymnasium.core import ActType, ObsType, RenderFrame
from gymnasium.utils import seeding



from gymnasium.envs.registration import EnvSpec

ArrayType = TypeVar("ArrayType")

from orca_gym.environment.async_env.orca_gym_async_env import OrcaGymAsyncEnv

class OrcaGymVectorEnv(VectorEnv):
    """
    实现基于多个Agent的向量环境。
    这样可以用一个Mujoco Instance 来同时运行多个 env
    最后转换成VectorEnv的接口格式

    # ┌────────────┬────────────────┬──────────────────────┐
    # │            │ Mujoco Env     │ Robots in Mujoco     │
    # ├────────────┼────────────────┼──────────────────────┤
    # │ Vector Env │ OrcaGymAsyncEnv│ num_envs             │
    # │ RLLib      │ num_env_runners│ num_envs_per_runner  │
    # │ LeggedGym  │ subenv_num     │ agent_num            │
    # └────────────┴────────────────┴──────────────────────┘
    
    
    Base class for vectorized environments to run multiple independent copies of the same environment in parallel.

    Vector environments can provide a linear speed-up in the steps taken per second through sampling multiple
    sub-environments at the same time. Gymnasium contains two generalised Vector environments: :class:`AsyncVectorEnv`
    and :class:`SyncVectorEnv` along with several custom vector environment implementations.
    For :func:`reset` and :func:`step` batches `observations`, `rewards`,  `terminations`, `truncations` and
    `info` for each sub-environment, see the example below. For the `rewards`, `terminations`, and `truncations`,
    the data is packaged into a NumPy array of shape `(num_envs,)`. For `observations` (and `actions`, the batching
    process is dependent on the type of observation (and action) space, and generally optimised for neural network
    input/outputs. For `info`, the data is kept as a dictionary such that a key will give the data for all sub-environment.

    For creating environments, :func:`make_vec` is a vector environment equivalent to :func:`make` for easily creating
    vector environments that contains several unique arguments for modifying environment qualities, number of environment,
    vectorizer type, vectorizer arguments.

    Note:
        The info parameter of :meth:`reset` and :meth:`step` was originally implemented before v0.25 as a list
        of dictionary for each sub-environment. However, this was modified in v0.25+ to be a dictionary with a NumPy
        array for each key. To use the old info style, utilise the :class:`DictInfoToList` wrapper.

    To avoid having to wait for all sub-environments to terminated before resetting, implementations will autoreset
    sub-environments on episode end (`terminated or truncated is True`). As a result, when adding observations
    to a replay buffer, this requires knowing when an observation (and info) for each sub-environment are the first
    observation from an autoreset. We recommend using an additional variable to store this information such as
    ``has_autoreset = np.logical_or(terminated, truncated)``.

    The Vector Environments have the additional attributes for users to understand the implementation

    - :attr:`num_envs` - The number of sub-environment in the vector environment
    - :attr:`observation_space` - The batched observation space of the vector environment
    - :attr:`single_observation_space` - The observation space of a single sub-environment
    - :attr:`action_space` - The batched action space of the vector environment
    - :attr:`single_action_space` - The action space of a single sub-environment
    """

    metadata: dict[str, Any] = {}
    spec: EnvSpec | None = None
    render_mode: str | None = None
    closed: bool = False

    observation_space: gym.Space
    action_space: gym.Space
    single_observation_space: gym.Space
    single_action_space: gym.Space

    num_envs: int

    _np_random: np.random.Generator | None = None
    _np_random_seed: int | None = None

    def __init__(self, 
        num_envs: int, 
        worker_index: int,
        entry_point: str,
        **kwargs
    ):
        self.agent_num = num_envs
        assert num_envs % 32 == 0, "num_envs must be a multiple of 32"
        self.env_num = num_envs // 32

        self.worker_index = worker_index
        env_id = kwargs.get("env_id", "")
        env_id_prefix = "-".join(env_id.split("-")[:-1])

        self.envs = []
        for i in range(self.env_num):
            worker_env_id = f"{env_id_prefix}-{i:03d}-{worker_index:03d}"
            kwargs["env_id"] = worker_env_id
            if worker_index == 1 and i == 0:
                kwargs["is_subenv"] = False
            else:
                kwargs["is_subenv"] = True

            # print("Create OrcaGymVectorEnv: env_id={}, worker_index={}, kwargs={}, entry_point={}".format(worker_env_id, worker_index, kwargs, entry_point))

            gym.register(
                id=worker_env_id,
                entry_point=entry_point,
                kwargs=kwargs
            )

            self.envs.append(gym.make(worker_env_id, **kwargs))

        # OrcaGymAsyncEnv 的 obs 是 字典形式，不符合rllib的格式，需要转换为np形式
        obs_list = []
        for env in self.envs:
            unwrapped_env : OrcaGymAsyncEnv = env.unwrapped
            env_obs, _, _, _ = unwrapped_env.get_obs()
            obs_list.append(env_obs["observation"])

        # 将所有 env 的 obs 合并成一个 np.ndarray
        obs = np.concatenate(obs_list, axis=0)
        self.observation_space = unwrapped_env.generate_observation_space(obs)
        self.single_observation_space = unwrapped_env.generate_observation_space(obs[0])

        self.single_action_space = self.envs[0].action_space
        # 将 single_action_space 扩展到 env_num 维度
        self.action_space = gym.spaces.Box(
            low=np.tile(self.single_action_space.low, (self.agent_num, 1)),
            high=np.tile(self.single_action_space.high, (self.agent_num, 1)),
            dtype=self.single_action_space.dtype,
        )

        # print("OrcaGymVectorEnv observation_space: ", self.observation_space)
        # print("OrcaGymVectorEnv single_observation_space: ", self.single_observation_space)
        # print("OrcaGymVectorEnv action_space: ", self.action_space)
        # print("OrcaGymVectorEnv single_action_space: ", self.single_action_space)

        self.num_envs = num_envs
        self.closed = False
        self._np_random = None

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[ObsType, dict[str, Any]]:  # type: ignore
        """Reset all parallel environments and return a batch of initial observations and info.

        Args:
            seed: The environment reset seed
            options: If to return the options

        Returns:
            A batch of observations and info from the vectorized environment.

        Example:
            >>> import gymnasium as gym
            >>> envs = gym.make_vec("CartPole-v1", num_envs=3, vectorization_mode="sync")
            >>> observations, infos = envs.reset(seed=42)
            >>> observations
            array([[ 0.0273956 , -0.00611216,  0.03585979,  0.0197368 ],
                   [ 0.01522993, -0.04562247, -0.04799704,  0.03392126],
                   [-0.03774345, -0.02418869, -0.00942293,  0.0469184 ]],
                  dtype=float32)
            >>> infos
            {}
        """
        if seed is not None:
            self._np_random, self._np_random_seed = seeding.np_random(seed)

        # obs 只取第一个 agent 的观测数据，实际所有 agent 的观测数据在 info 中
        obs_list = []
        for env in self.envs:
            _, async_env_info = env.reset()
            obs_list.append(async_env_info["env_obs"]["observation"])

        obs = np.concatenate(obs_list, axis=0)
        infos = [{} for _ in range(self.agent_num)]

        # print("OrcaGymVectorEnv reset obs shape: ", obs.shape)

        return obs, infos

    def step(
        self, actions: ActType
    ) -> tuple[ObsType, ArrayType, ArrayType, ArrayType, dict[str, Any]]:
        """Take an action for each parallel environment.

        Args:
            actions: Batch of actions with the :attr:`action_space` shape.

        Returns:
            Batch of (observations, rewards, terminations, truncations, infos)

        Note:
            As the vector environments autoreset for a terminating and truncating sub-environments, this will occur on
            the next step after `terminated or truncated is True`.

        Example:
            >>> import gymnasium as gym
            >>> import numpy as np
            >>> envs = gym.make_vec("CartPole-v1", num_envs=3, vectorization_mode="sync")
            >>> _ = envs.reset(seed=42)
            >>> actions = np.array([1, 0, 1], dtype=np.int32)
            >>> observations, rewards, terminations, truncations, infos = envs.step(actions)
            >>> observations
            array([[ 0.02727336,  0.18847767,  0.03625453, -0.26141977],
                   [ 0.01431748, -0.24002443, -0.04731862,  0.3110827 ],
                   [-0.03822722,  0.1710671 , -0.00848456, -0.2487226 ]],
                  dtype=float32)
            >>> rewards
            array([1., 1., 1.])
            >>> terminations
            array([False, False, False])
            >>> terminations
            array([False, False, False])
            >>> infos
            {}
        """
        # raise NotImplementedError(f"{self.__str__()} step function is not implemented.")
        # print("OrcaGymVectorEnv step actions shape: ", actions.shape, "worker_index: ", self.worker_index, "actions: ", actions)

        # 假设原数组 action_array 形状为 (N, 12)，N 是第一维大小
        N = actions.shape[0]  # 获取第一维长度

        # 计算整除64后的分组数量
        num_groups = N // 32  # 整数除法（自动舍去余数）

        # 重塑为 (num_groups, 32, 12) 形状
        reshaped_action = actions.reshape(num_groups, 32, actions.shape[1])

        obs_list = []
        reward_list = []
        terminated_list = []
        truncated_list = []
        for i, env in enumerate(self.envs):
            _, _, _, _, info = env.step(reshaped_action[i].flatten())
            obs_list.append(info["env_obs"]["observation"])
            reward_list.append(info["reward"])
            terminated_list.append(info["terminated"])
            truncated_list.append(info["truncated"])

        obs = np.concatenate(obs_list, axis=0)
        reward = np.concatenate(reward_list, axis=0)
        terminated = np.concatenate(terminated_list, axis=0)
        truncated = np.concatenate(truncated_list, axis=0)
        infos = [{} for _ in range(self.agent_num)]

        # print("OrcaGymVectorEnv step obs shape: ", obs.shape)
        # print("OrcaGymVectorEnv step reward shape: ", reward.shape)
        # print("OrcaGymVectorEnv step terminated: ", terminated)
        # print("OrcaGymVectorEnv step truncated: ", truncated)

        return obs, reward, terminated, truncated, infos

    def render(self) -> tuple[RenderFrame, ...] | None:
        """Returns the rendered frames from the parallel environments.

        Returns:
            A tuple of rendered frames from the parallel environments
        """

        # Render 不需要都调用，反正只有第一个需要执行Render
        self.envs[0].render()

    def close(self, **kwargs: Any):
        """Close all parallel environments and release resources.

        It also closes all the existing image viewers, then calls :meth:`close_extras` and set
        :attr:`closed` as ``True``.

        Warnings:
            This function itself does not close the environments, it should be handled
            in :meth:`close_extras`. This is generic for both synchronous and asynchronous
            vectorized environments.

        Note:
            This will be automatically called when garbage collected or program exited.

        Args:
            **kwargs: Keyword arguments passed to :meth:`close_extras`
        """
        if self.closed:
            return

        for env in self.envs:
            env.close()

        self.close_extras(**kwargs)
        self.closed = True

    def _unpack_reset_info(self, info: dict[str, Any]) -> dict[str, Any]:
        return info["env_obs"]

    def _unpack_step_info(self, info: dict[str, Any]) -> dict[str, Any]:
        return info["env_obs"]


    # def close_extras(self, **kwargs: Any):
    #     """Clean up the extra resources e.g. beyond what's in this base class."""
    #     pass

    # @property
    # def np_random(self) -> np.random.Generator:
    #     """Returns the environment's internal :attr:`_np_random` that if not set will initialise with a random seed.

    #     Returns:
    #         Instances of `np.random.Generator`
    #     """
    #     if self._np_random is None:
    #         self._np_random, self._np_random_seed = seeding.np_random()
    #     return self._np_random

    # @np_random.setter
    # def np_random(self, value: np.random.Generator):
    #     self._np_random = value
    #     self._np_random_seed = -1

    # @property
    # def np_random_seed(self) -> int | None:
    #     """Returns the environment's internal :attr:`_np_random_seed` that if not set will first initialise with a random int as seed.

    #     If :attr:`np_random_seed` was set directly instead of through :meth:`reset` or :meth:`set_np_random_through_seed`,
    #     the seed will take the value -1.

    #     Returns:
    #         int: the seed of the current `np_random` or -1, if the seed of the rng is unknown
    #     """
    #     if self._np_random_seed is None:
    #         self._np_random, self._np_random_seed = seeding.np_random()
    #     return self._np_random_seed

    # @property
    # def unwrapped(self):
    #     """Return the base environment."""
    #     return self

    # def _add_info(
    #     self, vector_infos: dict[str, Any], env_info: dict[str, Any], env_num: int
    # ) -> dict[str, Any]:
    #     """Add env info to the info dictionary of the vectorized environment.

    #     Given the `info` of a single environment add it to the `infos` dictionary
    #     which represents all the infos of the vectorized environment.
    #     Every `key` of `info` is paired with a boolean mask `_key` representing
    #     whether or not the i-indexed environment has this `info`.

    #     Args:
    #         vector_infos (dict): the infos of the vectorized environment
    #         env_info (dict): the info coming from the single environment
    #         env_num (int): the index of the single environment

    #     Returns:
    #         infos (dict): the (updated) infos of the vectorized environment
    #     """
    #     for key, value in env_info.items():
    #         # If value is a dictionary, then we apply the `_add_info` recursively.
    #         if isinstance(value, dict):
    #             array = self._add_info(vector_infos.get(key, {}), value, env_num)
    #         # Otherwise, we are a base case to group the data
    #         else:
    #             # If the key doesn't exist in the vector infos, then we can create an array of that batch type
    #             if key not in vector_infos:
    #                 if type(value) in [int, float, bool] or issubclass(
    #                     type(value), np.number
    #                 ):
    #                     array = np.zeros(self.num_envs, dtype=type(value))
    #                 elif isinstance(value, np.ndarray):
    #                     # We assume that all instances of the np.array info are of the same shape
    #                     array = np.zeros(
    #                         (self.num_envs, *value.shape), dtype=value.dtype
    #                     )
    #                 else:
    #                     # For unknown objects, we use a Numpy object array
    #                     array = np.full(self.num_envs, fill_value=None, dtype=object)
    #             # Otherwise, just use the array that already exists
    #             else:
    #                 array = vector_infos[key]

    #             # Assign the data in the `env_num` position
    #             #   We only want to run this for the base-case data (not recursive data forcing the ugly function structure)
    #             array[env_num] = value

    #         # Get the array mask and if it doesn't already exist then create a zero bool array
    #         array_mask = vector_infos.get(
    #             f"_{key}", np.zeros(self.num_envs, dtype=np.bool_)
    #         )
    #         array_mask[env_num] = True

    #         # Update the vector info with the updated data and mask information
    #         vector_infos[key], vector_infos[f"_{key}"] = array, array_mask

    #     return vector_infos

    # def __del__(self):
    #     """Closes the vector environment."""
    #     if not getattr(self, "closed", True):
    #         self.close()

    # def __repr__(self) -> str:
    #     """Returns a string representation of the vector environment.

    #     Returns:
    #         A string containing the class name, number of environments and environment spec id
    #     """
    #     if self.spec is None:
    #         return f"{self.__class__.__name__}(num_envs={self.num_envs})"
    #     else:
    #         return (
    #             f"{self.__class__.__name__}({self.spec.id}, num_envs={self.num_envs})"
    #         )