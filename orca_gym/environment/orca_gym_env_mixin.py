"""OrcaGym 环境公共方法 Mixin。

提供名称空间解析、动作/观测空间生成、reset 编排等方法。
不定义 __init__，不持有状态，子类自行初始化 _agent_names 等字段。
"""

from typing import Any, Dict, Optional, Union

import numpy as np
from numpy.typing import NDArray

from gymnasium import spaces
from gymnasium.spaces import Space

from orca_gym.log.orca_log import get_orca_logger

_logger = get_orca_logger()


class OrcaGymEnvMixin:
    """OrcaGym 环境公共方法 Mixin（名称空间、空间生成、reset 编排）。

    子类必须提供以下字段/方法：
        - self._agent_names: list[str]
        - self.reset_simulation() -> None
        - self.reset_model() -> tuple[dict, dict]
        - self.render() -> Any
    """

    # --- 名称空间解析 ---

    def body(self, name: str, agent_id: int = None) -> str:
        """获取带智能体前缀的 body 名称"""
        if agent_id == None:
            return self._name_with_agent0(name)
        else:
            return self._name_with_agent(agent_id, name)

    def joint(self, name: str, agent_id: int = None) -> str:
        """获取带智能体前缀的关节名称"""
        if agent_id == None:
            return self._name_with_agent0(name)
        else:
            return self._name_with_agent(agent_id, name)

    def actuator(self, name: str, agent_id: int = None) -> str:
        """获取带智能体前缀的执行器名称"""
        if agent_id == None:
            return self._name_with_agent0(name)
        else:
            return self._name_with_agent(agent_id, name)

    def site(self, name: str, agent_id: int = None) -> str:
        """获取带智能体前缀的 site 名称"""
        if agent_id == None:
            return self._name_with_agent0(name)
        else:
            return self._name_with_agent(agent_id, name)

    def mocap(self, name: str, agent_id: int = None) -> str:
        """获取带智能体前缀的 mocap 名称"""
        if agent_id == None:
            return self._name_with_agent0(name)
        else:
            return self._name_with_agent(agent_id, name)

    def sensor(self, name: str, agent_id: int = None) -> str:
        """获取带智能体前缀的传感器名称"""
        if agent_id == None:
            return self._name_with_agent0(name)
        else:
            return self._name_with_agent(agent_id, name)

    # --- 辅助 ---

    def _name_with_agent0(self, name: str) -> str:
        """为第一个智能体添加名称前缀"""
        if len(self._agent_names) > 0:
            return f"{self._agent_names[0]}_{name}"
        else:
            return name

    def _name_with_agent(self, agent_id: int, name: str) -> str:
        """为指定智能体添加名称前缀"""
        if len(self._agent_names) > 0:
            return f"{self._agent_names[agent_id]}_{name}"
        else:
            return name

    @property
    def agent_num(self) -> int:
        """获取智能体数量"""
        return len(self._agent_names)

    # --- 空间生成 ---

    def generate_action_space(self, bounds: NDArray[np.float64]) -> Space:
        """
        生成环境的动作空间

        术语说明:
            - 动作空间 (Action Space): 强化学习中智能体可以执行的所有动作的集合
            - Box Space: 连续动作空间，每个维度有上下界限制
            - 动作: 发送给执行器的控制命令，通常是扭矩、位置或速度

        使用示例:
            ```python
            # 根据执行器控制范围生成动作空间
            ctrlrange = self.model.get_actuator_ctrlrange()
            self.action_space = self.generate_action_space(ctrlrange)
            # 动作空间形状: (nu,)，每个值在 [min, max] 范围内
            ```
        """
        low, high = 0.0, 0.0
        if len(bounds.T) > 0:
            low, high = bounds.T
        # 显式转 float32 并裁剪 ±inf 到 float32 可表示范围。
        # MuJoCo actuator_ctrlrange 对 ctrllimited=false 的执行器返回 ±inf，
        # 直接 cast 到 float32 触发 gymnasium 的 overflow-in-cast 与
        # precision-lowered warning。裁剪后语义为"无实际限幅"（边界为
        # float32 最大值），对 RL 训练更友好（归一化不会除以 inf）。
        f32_max = np.finfo(np.float32).max
        low = np.clip(np.asarray(low, dtype=np.float32), -f32_max, f32_max)
        high = np.clip(np.asarray(high, dtype=np.float32), -f32_max, f32_max)
        action_space = spaces.Box(low=low, high=high, dtype=np.float32)
        return action_space

    def generate_observation_space(self, obs: Union[Dict[str, Any], np.ndarray]) -> Space:
        """
        生成环境的观测空间

        术语说明:
            - 观测空间 (Observation Space): 强化学习中智能体能够观察到的状态信息集合
            - 观测 (Observation): 智能体在每个时间步接收到的状态信息
            - Dict Space: 字典类型的观测空间，包含多个子空间
            - Box Space: 连续观测空间，每个维度有上下界限制

        使用示例:
            ```python
            # 根据观测数据生成观测空间
            obs = self._get_obs()  # 获取示例观测
            self.observation_space = self.generate_observation_space(obs)
            # 观测空间可能是 Dict 或 Box，取决于 obs 的类型
            ```
        """
        if obs is None:
            raise ValueError("obs dictionary is None")

        if isinstance(obs, np.ndarray):
            # If obs is a numpy array, create a box space for it
            low = np.full(obs.shape, -np.finfo(np.float32).max, dtype=np.float32)
            high = np.full(obs.shape, np.finfo(np.float32).max, dtype=np.float32)
            return spaces.Box(low=low, high=high, dtype=np.float32)

        obs_space_dict = {}
        for obs_key, obs_data in obs.items():
            if isinstance(obs_data, np.ndarray):
                # 创建与 obs_data 形状相同的 low 和 high，并确保它们是 float32
                # 使用 float32 的有限边界值
                finite_max = np.finfo(np.float32).max
                low = np.full(obs_data.shape, -finite_max, dtype=np.float32)
                high = np.full(obs_data.shape, finite_max, dtype=np.float32)
                _logger.debug(f"low.dtype: {low.dtype}, high.dtype: {high.dtype}")

                obs_space_dict[obs_key] = spaces.Box(
                    low=low,
                    high=high,
                    dtype=np.float32
                )
            else:
                raise ValueError(f"Unsupported observation type: {type(obs_data)}")

        observation_space = spaces.Dict(obs_space_dict)
        return observation_space

    # --- reset 编排 ---

    def _get_reset_info(self) -> Dict[str, float]:
        """Function that generates the `info` that is returned during a `reset()`."""
        return {}

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        super().reset(seed=seed)

        if seed is not None:
            self.set_seed_value(seed)

        # mujoco.mj_resetData(self.model, self.data)
        self.reset_simulation()

        obs, info = self.reset_model()

        self.render()
        return obs, info

    def set_seed_value(self, seed: int = None) -> list:
        """设置随机数种子"""
        self.seed_value = seed
        self.np_random = np.random.RandomState(seed)
        return [seed]
