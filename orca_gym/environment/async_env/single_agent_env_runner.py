from collections import defaultdict
from functools import partial
import logging
import math
import time
from typing import Collection, DefaultDict, List, Optional, Union

import gymnasium as gym
import ray
from gymnasium.wrappers.vector import DictInfoToList

from ray.rllib.algorithms.algorithm_config import AlgorithmConfig
from ray.rllib.callbacks.callbacks import RLlibCallback
from ray.rllib.callbacks.utils import make_callback
from ray.rllib.core import (
    COMPONENT_ENV_TO_MODULE_CONNECTOR,
    COMPONENT_MODULE_TO_ENV_CONNECTOR,
    COMPONENT_RL_MODULE,
    DEFAULT_AGENT_ID,
    DEFAULT_MODULE_ID,
)
from ray.rllib.core.columns import Columns
from ray.rllib.core.rl_module.rl_module import RLModule, RLModuleSpec
from ray.rllib.env import INPUT_ENV_SPACES, INPUT_ENV_SINGLE_SPACES
from ray.rllib.env.env_context import EnvContext
from ray.rllib.env.env_runner import EnvRunner, ENV_STEP_FAILURE
from ray.rllib.env.single_agent_episode import SingleAgentEpisode
from ray.rllib.env.utils import _gym_env_creator
from ray.rllib.utils import force_list
from ray.rllib.utils.annotations import override
from ray.rllib.utils.checkpoints import Checkpointable
from ray.rllib.utils.deprecation import Deprecated
from ray.rllib.utils.framework import get_device
from ray.rllib.utils.metrics import (
    ENV_TO_MODULE_CONNECTOR,
    EPISODE_DURATION_SEC_MEAN,
    EPISODE_LEN_MAX,
    EPISODE_LEN_MEAN,
    EPISODE_LEN_MIN,
    EPISODE_RETURN_MAX,
    EPISODE_RETURN_MEAN,
    EPISODE_RETURN_MIN,
    MODULE_TO_ENV_CONNECTOR,
    NUM_AGENT_STEPS_SAMPLED,
    NUM_AGENT_STEPS_SAMPLED_LIFETIME,
    NUM_ENV_STEPS_SAMPLED,
    NUM_ENV_STEPS_SAMPLED_LIFETIME,
    NUM_EPISODES,
    NUM_EPISODES_LIFETIME,
    NUM_MODULE_STEPS_SAMPLED,
    NUM_MODULE_STEPS_SAMPLED_LIFETIME,
    RLMODULE_INFERENCE_TIMER,
    SAMPLE_TIMER,
    TIME_BETWEEN_SAMPLING,
    WEIGHTS_SEQ_NO,
)
from ray.rllib.utils.spaces.space_utils import unbatch
from ray.rllib.utils.typing import EpisodeID, ResultDict, StateDict
from ray.tune.registry import ENV_CREATOR, _global_registry
from ray.util.annotations import PublicAPI

logger = logging.getLogger("ray.rllib")

from ray.rllib.env.single_agent_env_runner import SingleAgentEnvRunner
from orca_gym.environment.async_env.orca_gym_vector_env import OrcaGymVectorEnv

from orca_gym.log.orca_log import get_orca_logger
_logger = get_orca_logger()


class OrcaGymAsyncSingleAgentEnvRunner(SingleAgentEnvRunner):
    """
    Wrap the SingleAgentEnvRunner to support orca gym asynchronous environments.
    """

    @override(SingleAgentEnvRunner)
    def __init__(self, *, config: AlgorithmConfig, **kwargs):
        """Initializes a OrcaGymAsyncSingleAgentEnvRunner instance.

        Args:
            config: An `AlgorithmConfig` object containing all settings needed to
                build this `EnvRunner` class.
        """
        super().__init__(config=config, **kwargs)

        _logger.info(f"env_runner.module is :  {self.module}")

       

    @override(SingleAgentEnvRunner)
    def make_env(self) -> None:
        """Creates a vectorized gymnasium env and stores it in `self.env`.

        Note that users can change the EnvRunner's config (e.g. change
        `self.config.env_config`) and then call this method to create new environments
        with the updated configuration.
        """
        # If an env already exists, try closing it first (to allow it to properly
        # cleanup).
        if self.env is not None:
            try:
                self.env.close()
            except Exception as e:
                logger.warning(
                    "Tried closing the existing env, but failed with error: "
                    f"{e.args[0]}"
                )

        env_ctx = self.config.env_config
        if not isinstance(env_ctx, EnvContext):
            env_ctx = EnvContext(
                env_ctx,
                worker_index=self.worker_index,
                num_workers=self.num_workers,
                remote=self.config.remote_worker_envs,
            )

        # No env provided -> Error.
        if not self.config.env:
            raise ValueError(
                "`config.env` is not provided! You should provide a valid environment "
                "to your config through `config.environment([env descriptor e.g. "
                "'CartPole-v1'])`."
            )
        # Register env for the local context.
        # Note, `gym.register` has to be called on each worker.
        elif isinstance(self.config.env, str) and _global_registry.contains(
            ENV_CREATOR, self.config.env
        ):
            entry_point = partial(
                _global_registry.get(ENV_CREATOR, self.config.env),
                env_ctx,
            )
        else:
            entry_point = partial(
                _gym_env_creator,
                env_descriptor=self.config.env,
                env_context=env_ctx,
            )
        gym.register("rllib-single-agent-env-v0", entry_point=entry_point)
        vectorize_mode = self.config.gym_env_vectorize_mode

        # print(f"Make vec env: config.env type={type(self.config.env)}, env= {self.config.env}, num_envs_per_env_runner= {self.config.num_envs_per_env_runner}, worker_index= {self.worker_index}")
        env_kwargs = self.config.env_config.get("env_kwargs", {})
        entry_point = self.config.env_config.get("entry_point", "")
        # print("Make vec env: env_kwargs: ", env_kwargs)
        self.env = OrcaGymVectorEnv(
            num_envs=self.config.num_envs_per_env_runner,
            worker_index=self.worker_index,
            entry_point=entry_point,
            **env_kwargs
        )


        # self.env = DictInfoToList(
        #     gym.make_vec(
        #         "rllib-single-agent-env-v0",
        #         num_envs=self.config.num_envs_per_env_runner,
        #         vectorization_mode=(
        #             vectorize_mode
        #             if isinstance(vectorize_mode, gym.envs.registration.VectorizeMode)
        #             else gym.envs.registration.VectorizeMode(vectorize_mode.lower())
        #         ),
        #     )
        # )

        self.num_envs: int = self.env.num_envs
        assert self.num_envs == self.config.num_envs_per_env_runner

        # Set the flag to reset all envs upon the next `sample()` call.
        self._needs_initial_reset = True

        # Call the `on_environment_created` callback.
        make_callback(
            "on_environment_created",
            callbacks_objects=self._callbacks,
            callbacks_functions=self.config.callbacks_on_environment_created,
            kwargs=dict(
                env_runner=self,
                metrics_logger=self.metrics,
                env=self.env.unwrapped,
                env_context=env_ctx,
            ),
        )

