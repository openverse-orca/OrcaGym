"""空白 Euler 仿真环境（在线模式，连接 OrcaStudio）。

与 scripts/sim_env.py 的 SimEnv 对应，但继承 OrcaGymEulerEnv（新一代 Facade 架构）。
用于 run_euler_loop.py 启动空白仿真循环。

约束：
    - 不穿墙访问内部对象（遵循 OrcaGymEulerEnv 公共 API 契约）
    - step/reset_model/_get_obs 是 Gymnasium MuJoCo 标准 hook，必须子类实现
"""
from __future__ import annotations

from typing import Any, Optional

import numpy as np
from gymnasium import spaces

from orca_gym.environment.euler.orca_gym_euler_env import OrcaGymEulerEnv
from orca_gym.log.orca_log import get_orca_logger

_logger = get_orca_logger()


class EulerSimEnv(OrcaGymEulerEnv):
    """空白 Euler 仿真环境：零控输入 + 基础关节观测。

    action_space / observation_space 由 model 维度自动推导，
    无需用户配置模型，模型 XML 由 OrcaStudio 端 scene 提供（在线模式）。
    """

    def __init__(
        self,
        frame_skip: int,
        orcagym_addr: str,
        agent_names: list[str],
        time_step: float,
        max_steps: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(
            frame_skip=frame_skip,
            orcagym_addr=orcagym_addr,
            agent_names=agent_names,
            time_step=time_step,
            **kwargs,
        )

        # 辅助变量（与 SimEnv 对齐，便于观测/动作空间构造）
        self.nu = self.model.nu   # 执行器数
        self.nq = self.model.nq   # 广义坐标数
        self.nv = self.model.nv   # 广义速度数

        self._set_obs_space()
        self._set_action_space()

    def _set_obs_space(self) -> None:
        self.observation_space = self.generate_observation_space(self._get_obs().copy())

    def _set_action_space(self) -> None:
        if self.nu > 0:
            scaled_action_range = np.concatenate([[[-1.0, 1.0]] for _ in range(self.nu)])
            self.action_space = self.generate_action_space(scaled_action_range)
        else:
            self.action_space = spaces.Box(
                low=np.array([]),
                high=np.array([]),
                dtype=np.float32,
            )
            _logger.info("No action space defined, nu is 0.")

    def step(self, action) -> tuple:
        ctrl = np.zeros(self.nu, dtype=np.float32)
        self.do_simulation(ctrl, self.frame_skip)
        obs = self._get_obs().copy()

        info: dict[str, Any] = {}
        terminated = False
        truncated = False
        reward = 0.0
        return obs, reward, terminated, truncated, info

    def _get_obs(self) -> dict:
        obs = {
            "joint_pos": self.data.qpos[: self.nq].copy(),
            "joint_vel": self.data.qvel[: self.nv].copy(),
            "joint_acc": self.data.qacc[: self.nv].copy(),
        }
        return obs

    def reset_model(self) -> tuple[dict, dict]:
        self.ctrl = np.zeros(self.nu, dtype=np.float32)
        obs = self._get_obs().copy()
        return obs, {}

    def get_observation(self, obs=None) -> dict:
        if obs is not None:
            return obs
        return self._get_obs().copy()
