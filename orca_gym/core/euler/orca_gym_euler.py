"""OrcaGymEuler 仿真核心 Facade。

组合 MuJoCoSimCore 等子组件，通过 __getattr__/__dir__ 实现封装隔离，
引导用户走公共 API，避免直接访问 _mjModel/_mjData。

属于 OrcaGymEuler 体系的 P1 基础设施骨架组件。
参见 docs/design/architecture/orca_gym_euler_architecture.md 第 5.2、7 节。
"""

from __future__ import annotations

import numpy as np

from orca_gym.core.euler.mujoco_sim_core import MuJoCoSimCore


class OrcaGymEuler:
    """仿真核心 Facade，组合子组件，不暴露 _mjModel/_mjData。

    设计契约:
        ┌─────────────────────────────────────────────────────────────┐
        │  用户不应直接访问 _mjData / _mjModel。                      │
        │  读取 MuJoCo 状态 → 使用 env.data（OrcaGymDataView）        │
        │  写入外力 → 使用 env.apply_body_force()                     │
        │  配置求解器 → 使用 env.sim_config                           │
        │  缺少功能时 → 扩展 OrcaGymEulerEnv 公共方法                 │
        └─────────────────────────────────────────────────────────────┘

    P1 阶段仅组合 MuJoCoSimCore；P2/P3 将逐步加入 ModelRegistry、
    SimConfig、OrcaStudioBridge 等子组件。
    """

    _BLOCKED_ATTRS = frozenset({
        "_mjData", "_mjModel", "mj_data", "mj_model",
        "_mj_data", "_mj_model", "mjData", "mjModel",
    })

    def __init__(self, stub=None) -> None:
        # P1：仅组合 MuJoCoSimCore。后续阶段加入 _studio/_registry/_opt/_euler。
        self._sim = MuJoCoSimCore()

    def __getattr__(self, name: str):
        # 仅当正常属性查找失败时触发。_sim 等真实属性不会进入此处。
        if name in self._BLOCKED_ATTRS:
            raise AttributeError(
                f"'{type(self).__name__}' 不直接暴露 '{name}'。\n"
                f"  读取 MuJoCo 状态 → 使用 env.data（OrcaGymDataView），如 env.data.qpos\n"
                f"  写入外力 → 使用 env.apply_body_force(body_name, force, torque)\n"
                f"  配置求解器 → 使用 env.sim_config\n"
                f"  查询 body 属性 → 使用 env.data.body_xpos(name) 等\n"
                f"  如果以上 API 都不满足需求，请在 OrcaGymEulerEnv 中扩展新方法，"
                f"不要直接访问内部 MuJoCo 对象。"
            )
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def __dir__(self):
        return [
            # P1 仿真控制
            "init_simulation", "mj_step", "mj_forward", "set_ctrl",
        ]

    # --- 委托到 _sim ---

    def init_simulation(self, model_xml_path: str) -> None:
        """初始化仿真，从 MJCF 文件加载模型。委托到 MuJoCoSimCore。

        Args:
            model_xml_path: MJCF 场景 XML 文件路径。
        """
        self._sim.init_simulation(model_xml_path)

    def mj_step(self, nstep: int) -> None:
        """执行 MuJoCo 步进。委托到 MuJoCoSimCore。

        Args:
            nstep: 步进次数。
        """
        self._sim.step(nstep)

    def mj_forward(self) -> None:
        """执行前向计算。委托到 MuJoCoSimCore。"""
        self._sim.forward()

    def set_ctrl(self, ctrl: np.ndarray) -> None:
        """设置控制输入。委托到 MuJoCoSimCore。

        Args:
            ctrl: 控制输入数组。
        """
        self._sim.set_ctrl(ctrl)
