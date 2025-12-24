import multiprocessing as mp
import warnings
from collections import OrderedDict
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Type, Union, Iterable
import torch
import datetime

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from orca_gym.environment import OrcaGymBaseEnv

from stable_baselines3.common.vec_env.base_vec_env import (
    CloudpickleWrapper,
    VecEnv,
    VecEnvIndices,
    VecEnvObs,
    VecEnvStepReturn,
)
from stable_baselines3.common.vec_env.patch_gym import _patch_env

from orca_gym.log.orca_log import get_orca_logger
_logger = get_orca_logger()



def _worker(
    remote: mp.connection.Connection,
    parent_remote: mp.connection.Connection,
    env_fn_wrapper: CloudpickleWrapper,
) -> None:
    # Import here to avoid a circular import
    from stable_baselines3.common.env_util import is_wrapped

    parent_remote.close()
    env = _patch_env(env_fn_wrapper.var())

    reset_info: Optional[Dict[str, Any]] = {}
    while True:
        try:
            cmd, data = remote.recv()
            if cmd == "step":
                observation, reward, terminated, truncated, info = env.step(data)

                # print("Worker step, obs: ", observation, "reward: ", reward, "terminated: ", terminated, "truncated: ", truncated, "info: ", info)
                env_obs = info["env_obs"]
                agent_obs = info["agent_obs"]
                reward = info["reward"]
                terminated = info["terminated"]
                truncated = info["truncated"]
                is_success = info["is_success"]

                remote.send((env_obs, agent_obs, reward, terminated, truncated, is_success))
            elif cmd == "reset":
                maybe_options = {"options": data[1]} if data[1] else {}
                observation, reset_info = env.reset(seed=data[0], **maybe_options)
                env_obs = reset_info["env_obs"]
                remote.send((env_obs, reset_info))
            elif cmd == "render":
                remote.send(env.render())
            elif cmd == "close":
                env.close()
                remote.close()
                break
            elif cmd == "get_spaces":
                remote.send((env.observation_space, env.action_space))
            elif cmd == "env_method":
                method = getattr(env, data[0])
                remote.send(method(*data[1], **data[2]))
            elif cmd == "get_attr":
                remote.send(getattr(env, data))
            elif cmd == "set_attr":
                remote.send(setattr(env, data[0], data[1]))  # type: ignore[func-returns-value]
            elif cmd == "is_wrapped":
                remote.send(is_wrapped(env, data))
            elif cmd == "setup_curriculum":
                if not hasattr(env, "setup_curriculum"):
                    unwraped_env = env.unwrapped
                    unwraped_env.setup_curriculum(data)
                else:
                    env.setup_curriculum(data)
                remote.send(True)
            else:
                raise NotImplementedError(f"`{cmd}` is not implemented in the worker")
        except EOFError:
            break


class OrcaGymAsyncSubprocVecEnv(VecEnv):
    """
    Creates a multiprocess vectorized wrapper for multiple environments, distributing each environment to its own
    process, allowing significant speed up when the environment is computationally complex.

    For performance reasons, if your environment is not IO bound, the number of environments should not exceed the
    number of logical cores on your CPU.

    .. warning::

        Only 'forkserver' and 'spawn' start methods are thread-safe,
        which is important when TensorFlow sessions or other non thread-safe
        libraries are used in the parent (see issue #217). However, compared to
        'fork' they incur a small start-up cost and have restrictions on
        global variables. With those methods, users must wrap the code in an
        ``if __name__ == "__main__":`` block.
        For more information, see the multiprocessing documentation.

    :param env_fns: Environments to run in subprocesses
    :param start_method: method used to start the subprocesses.
           Must be one of the methods returned by multiprocessing.get_all_start_methods().
           Defaults to 'forkserver' on available platforms, and 'spawn' otherwise.


    修改baseline3中的SubprocVecEnv，对于每个环境，创建 agent_num 个智能体，每个智能体共享一个环境
    通过拼接每个agent的action和obs，来实现异步多环境的功能
    agent_num: 每个环境中智能体的数量
    """

    def __init__(self, env_fns: List[Callable[[], OrcaGymBaseEnv]], agent_num: int, start_method: Optional[str] = None):
        self.waiting = False
        self.closed = False
        n_envs = len(env_fns)
        self.agent_num = agent_num

        if start_method is None:
            # Fork is not a thread safe method (see issue #217)
            # but is more user friendly (does not require to wrap the code in
            # a `if __name__ == "__main__":`)
            forkserver_available = "forkserver" in mp.get_all_start_methods()
            start_method = "forkserver" if forkserver_available else "spawn"
        ctx = mp.get_context(start_method)

        self.remotes, self.work_remotes = zip(*[ctx.Pipe() for _ in range(n_envs)])
        self.processes = []

        for work_remote, remote, env_fn in zip(self.work_remotes, self.remotes, env_fns):
            args = (work_remote, remote, CloudpickleWrapper(env_fn))
            # daemon=True: if the main process crashes, we should not cause things to hang
            process = ctx.Process(target=_worker, args=args, daemon=True)  # type: ignore[attr-defined]
            process.start()
            self.processes.append(process)
            work_remote.close()

        self.remotes[0].send(("get_spaces", None))
        observation_space, action_space = self.remotes[0].recv()

        super().__init__(len(env_fns) * agent_num, observation_space, action_space)


    def step_async(self, actions: np.ndarray) -> None:
        # 拼接 actions，将 remote_num * agent_num 个动作拼接成 remote_num 个动作
        remote_actions = np.reshape(actions, (len(self.remotes), -1))
        
        # self._time_step_async_begin = datetime.datetime.now()
        
        
        for remote, action in zip(self.remotes, remote_actions):
            remote.send(("step", action))
        
        # self._time_step_async_end = datetime.datetime.now()
        # print("Subproc step async time: ", (self._time_step_async_end - self._time_step_async_begin).total_seconds() * 1000, "ms")

        self.waiting = True

    def step_wait(self) -> VecEnvStepReturn:

        # self._time_step_wait_begin = datetime.datetime.now()

        results = [remote.recv() for remote in self.remotes]

        # self._time_step_wait_end = datetime.datetime.now()
        # print("Subproc step wait time: ", (self._time_step_wait_end - self._time_step_wait_begin).total_seconds() * 1000, "ms")

        self.waiting = False
        env_obs, agent_obs, reward, terminated, truncated, is_success = zip(*results)  # type: ignore[assignment]
        # print("Subproc step wait, env_obs: " , env_obs, "reward: ", reward, "terminated: ", terminated, "truncated: ", truncated, "is_success: ", is_success)
        # self._time_process_step_1 = datetime.datetime.now()
        flatten_obs = _flatten_obs(env_obs, self.observation_space, self.agent_num)
        # self._time_process_step_2 = datetime.datetime.now()
        flatten_rewards = _flatten_reward(reward)
        # self._time_process_step_3 = datetime.datetime.now()
        flatten_dones = _flatten_dones(terminated, truncated)
        # self._time_process_step_4 = datetime.datetime.now()
        flatten_info = _flatten_info(agent_obs, is_success, truncated)

        # self._time_process_step_end = datetime.datetime.now()
        # print("Subproc step process time: ", (self._time_process_step_end - self._time_step_wait_end).total_seconds() * 1000, "ms")
        # print("\tSubproc step process time zip: ", (self._time_process_step_1 - self._time_step_wait_end).total_seconds() * 1000, "ms")
        # print("\tSubproc step process time obs: ", (self._time_process_step_2 - self._time_process_step_1).total_seconds() * 1000, "ms")
        # print("\tSubproc step process time reward: ", (self._time_process_step_3 - self._time_process_step_2).total_seconds() * 1000, "ms")
        # print("\tSubproc step process time dones: ", (self._time_process_step_4 - self._time_process_step_3).total_seconds() * 1000, "ms")
        # print("\tSubproc step process time info: ", (self._time_process_step_end - self._time_process_step_4).total_seconds() * 1000, "ms")
        # print("Total step time: ", (self._time_process_step_end - self._time_step_async_begin).total_seconds() * 1000, "ms")


        return flatten_obs, flatten_rewards, flatten_dones, flatten_info  # type: ignore[return-value]

    def reset(self) -> VecEnvObs:
        for env_idx, remote in enumerate(self.remotes):
            remote.send(("reset", (self._seeds[env_idx], self._options[env_idx])))
        results = [remote.recv() for remote in self.remotes]
        env_obs, self.reset_infos = zip(*results)  # type: ignore[assignment]
        # print("subproc reset, obs: ", env_obs, "reset_infos: ", self.reset_infos)
        # Seeds and options are only used once
        self._reset_seeds()
        self._reset_options()
        return _flatten_obs(env_obs, self.observation_space, self.agent_num)

    def close(self) -> None:
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
        for remote in self.remotes:
            remote.send(("close", None))
        for process in self.processes:
            process.join()
        self.closed = True

    def get_images(self) -> Sequence[Optional[np.ndarray]]:
        if self.render_mode != "rgb_array":
            warnings.warn(
                f"The render mode is {self.render_mode}, but this method assumes it is `rgb_array` to obtain images."
            )
            return [None for _ in self.remotes]
        for pipe in self.remotes:
            # gather render return from subprocesses
            pipe.send(("render", None))
        outputs = [pipe.recv() for pipe in self.remotes]
        return outputs

    def get_attr(self, attr_name: str, indices: VecEnvIndices = None) -> List[Any]:
        """Return attribute from vectorized environment (see base class)."""
        target_remotes = self._get_target_remotes(indices)
        for remote in target_remotes:
            remote.send(("get_attr", attr_name))
        return [remote.recv() for remote in target_remotes]

    def set_attr(self, attr_name: str, value: Any, indices: VecEnvIndices = None) -> None:
        """Set attribute inside vectorized environments (see base class)."""
        target_remotes = self._get_target_remotes(indices)
        for remote in target_remotes:
            remote.send(("set_attr", (attr_name, value)))
        for remote in target_remotes:
            remote.recv()

    def env_method(self, method_name: str, *method_args, indices: VecEnvIndices = None, **method_kwargs) -> List[Any]:
        """Call instance methods of vectorized environments."""
        target_remotes = self._get_target_remotes(indices)
        for remote in target_remotes:
            remote.send(("env_method", (method_name, method_args, method_kwargs)))
        return [remote.recv() for remote in target_remotes]

    def env_is_wrapped(self, wrapper_class: Type[gym.Wrapper], indices: VecEnvIndices = None) -> List[bool]:
        """Check if worker environments are wrapped with a given wrapper"""
        target_remotes = self._get_target_remotes(indices)
        for remote in target_remotes:
            remote.send(("is_wrapped", wrapper_class))
        return [remote.recv() for remote in target_remotes]

    def _get_target_remotes(self, indices: VecEnvIndices) -> List[Any]:
        """
        Get the connection object needed to communicate with the wanted
        envs that are in subprocesses.

        :param indices: refers to indices of envs.
        :return: Connection object to communicate between processes.
        """
        indices = self._get_indices(indices)
        return [self.remotes[i] for i in indices]

    def _get_indices(self, indices: VecEnvIndices) -> Iterable[int]:
        """
        Override the base class method to handle the multi agent case.
        """
        remote_num = self.num_envs // self.agent_num
        if indices is None:
            indices = range(remote_num)
        elif isinstance(indices, int):
            indices = [indices]
            if any(i >= remote_num for i in indices):
                raise ValueError("Out of range indices")
        return indices

    def setup_curriculum(self, curriculum_name: str) -> None:
        for env_idx, remote in enumerate(self.remotes):
            remote.send(("setup_curriculum", curriculum_name))
        results = [remote.recv() for remote in self.remotes]
        _logger.info(f"Subproc setup curriculum, results:  {results}")
    
def _split_multi_agent_obs_list(obs: List[VecEnvObs], agent_num) -> List[VecEnvObs]:
    """
    Split a list of observations into a list of observations per agent.
    """
    # 将 remote_num 个 obs 中，每个 nd.array 切成 agent_num 份，最后返回 agent_num * remote_num 个 obs

    # print("Split multi agent obs list begin: ", obs)

    splited_obs = []
    for remote_obs in obs:
        for i in range(agent_num):
            splited_obs.append({})
            for key, value in remote_obs.items():
                assert len(value) >= agent_num, f"Number of agents in observation {key} is less than agent_num"
                slice_len = len(value) // agent_num
                value_slice = value[i * slice_len: (i + 1) * slice_len]
                splited_obs[-1][key] = value_slice

    # print("Split multi agent obs list end: ", splited_obs)

    return splited_obs


def _flatten_obs(env_obs: Union[List[VecEnvObs], Tuple[VecEnvObs]], space: spaces.Space, agent_num) -> VecEnvObs:
    """
    Flatten observations, depending on the observation space.

    :param env_obs: observations.
                A list or tuple of observations, one per environment.
                Each environment observation may be a NumPy array, or a dict or tuple of NumPy arrays.
    :return: flattened observations.
            A flattened NumPy array or an OrderedDict or tuple of flattened numpy arrays.
            Each NumPy array has the environment index as its first axis.
    """
    assert isinstance(env_obs, (list, tuple)), "expected list or tuple of observations per environment"
    assert len(env_obs) > 0, "need observations from at least one environment"

    # print("_flatten_obs begin, env_obs: ", env_obs)

    if isinstance(space, spaces.Dict):
        # assert isinstance(space.spaces, OrderedDict), "Dict space must have ordered subspaces"
        assert isinstance(env_obs[0], dict), "non-dict observation for environment with Dict observation space"

        # 使用 OrderedDict 保持键的顺序
        flattend_obs = OrderedDict([
            (k, np.concatenate([o[k] for o in env_obs], axis=0))
            for k in space.spaces.keys()
        ])
            
        # print("_flatten_obs end, flattend_obs: ", flattend_obs)
        return flattend_obs
    elif isinstance(space, spaces.Tuple):
        assert isinstance(env_obs[0], tuple), "non-tuple observation for environment with Tuple observation space"
        obs_len = len(space.spaces) * agent_num
        return tuple(np.stack([o[i] for o in env_obs]) for i in range(obs_len))  # type: ignore[index]
    else:
        return np.stack(env_obs)  # type: ignore[arg-type]

def _slice_multi_agent_obs(obs: VecEnvObs, agent_num, agent_index) -> VecEnvObs:
    """
    Slice a dictionary of observations into a dictionary of observations per agent.
    """
    assert isinstance(obs, dict), "expected dict of observations"
    assert agent_index < agent_num, "agent_index is out of range"
    # 从 obs 中，取出指定 agent_index 的观测数据

    # print("Slice multi agent obs begin: ", obs)

    sliced_obs = {}
    for key, value in obs.items():
        assert len(value) >= agent_num, f"Number of agents in observation {key} is less than agent_num"
        slice_len = len(value) // agent_num
        sliced_obs[key] = value[agent_index * slice_len: (agent_index + 1) * slice_len]

    # print("Slice multi agent obs end: ", sliced_obs)

    return sliced_obs



def _flatten_info(agent_obs, is_success, truncated) -> Dict[str, List[Any]]:
    """
    Flatten infos, depending on the info space.

    :param infos: infos.
                  A list of infos, one per environment.
                  Each environment info may be a dict.
    :return: flattened infos.
            A dict of lists.
            Each list has the environment index as its first axis.
    """
    assert isinstance(is_success, (tuple, list)), "expected list or tuple of is_success per environment, got {}".format(type(is_success))
    assert len(is_success) > 0, "need is_success from at least one environment"

    assert isinstance(truncated, (tuple, list)), "expected list or tuple of dones per environment, got {}".format(type(truncated))
    assert len(truncated) > 0, "need dones from at least one environment"
    assert len(is_success) == len(truncated), "is_success and truncated should have the same length"

    # print("Flatte info begin: ", obs, is_success, truncated)

    # 使用列表推导式进行扁平化
    flattened_infos = [
        {
            "is_success": s,
            "TimeLimit.truncated": t,
            "terminal_observation": obs
        }
        for env_s, env_t, env_obs in zip(is_success, truncated, agent_obs)
        for s, t, obs in zip(env_s, env_t, env_obs)
    ]

    # print("Flatte info end: ", flattened_infos)

    return flattened_infos

def _flatten_reward(rewards: Union[tuple, list]) -> np.ndarray:
    """
    Flatten rewards, depending on the reward space.

    :param rewards: rewards.
                    A list of rewards, one per environment.
                    Each environment reward may be a NumPy array.
    :return: flattened rewards.
            A flattened NumPy array.
            Each NumPy array has the environment index as its first axis.
    """
    assert isinstance(rewards, (tuple, list)), "expected list or tuple of rewards per environment, got {}".format(type(rewards))
    assert len(rewards) > 0, "need rewards from at least one environment"

    # 如果 rewards 中的元素是标量，直接转换为 NumPy 数组
    if isinstance(rewards[0], (int, float)):
        return np.array(rewards)

    # 确保所有元素都是 NumPy 数组，并具有相同的形状（除了第一个轴）
    return np.concatenate(rewards, axis=0)

def _flatten_dones(terminated, truncated) -> np.ndarray:
    """
    Flatten dones, depending on the done space.

    :param terminated, truncated: done state for each agent.

    :return: flattened dones.
            A flattened NumPy array.
            Each NumPy array has the environment index as its first axis.
    """
    assert isinstance(terminated, (tuple, list)), "expected list or tuple of dones per environment, got {}".format(type(terminated))
    assert len(terminated) > 0, "need dones from at least one environment"

    assert isinstance(truncated, (tuple, list)), "expected list or tuple of dones per environment, got {}".format(type(truncated))
    assert len(truncated) > 0, "need dones from at least one environment"
    assert len(terminated) == len(truncated), "terminated and truncated should have the same length"

    # 将 terminated 和 truncated 转换为 NumPy 数组
    terminated_np = np.array(terminated, dtype=bool)
    truncated_np = np.array(truncated, dtype=bool)

    # 确保 terminated 和 truncated 形状相同
    if terminated_np.shape != truncated_np.shape:
        raise ValueError("terminated and truncated should have the same shape per environment and agent")

    # 进行逻辑或操作：done = truncated OR terminated
    dones_np = np.logical_or(truncated_np, terminated_np)

    # 扁平化为一维数组
    dones_flat = dones_np.reshape(-1)

    return dones_flat