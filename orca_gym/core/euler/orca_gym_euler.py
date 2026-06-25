"""OrcaGymEuler 仿真核心 Facade。

组合 MuJoCoSimCore 等子组件，通过 __getattr__/__dir__ 实现封装隔离，
引导用户走公共 API，避免直接访问 _mjModel/_mjData。

属于 OrcaGymEuler 体系的 P3 Studio 集成组件。
参见 docs/design/architecture/orca_gym_euler_architecture.md 第 5.2、7 节。
"""

from __future__ import annotations

import numpy as np

from orca_gym.core.euler.mujoco_sim_core import MuJoCoSimCore
from orca_gym.core.euler.orca_studio_bridge import OrcaStudioBridge
from orca_gym.core.euler.model_registry import ModelRegistry
from orca_gym.core.euler.sim_config import SimConfig
from orca_gym.core.euler.orca_gym_data_view import OrcaGymDataView


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

    P3 阶段组合 MuJoCoSimCore、OrcaStudioBridge、ModelRegistry、SimConfig，
    并提供 OrcaGymDataView 状态视图。
    """

    _BLOCKED_ATTRS = frozenset({
        "_mjData", "_mjModel", "mj_data", "mj_model",
        "_mj_data", "_mj_model", "mjData", "mjModel",
    })

    def __init__(self, stub=None) -> None:
        # 子组件
        self._sim = MuJoCoSimCore()
        self._studio = OrcaStudioBridge(stub)
        self._registry: ModelRegistry | None = None
        self._opt: SimConfig | None = None
        # 状态视图（init_simulation 后创建）
        self._view: OrcaGymDataView | None = None

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
            # 仿真控制
            "init_simulation", "mj_step", "mj_forward", "set_ctrl",
            "apply_body_force", "clear_body_force", "clear_all_forces",
            # 状态访问
            "data", "model", "sim_config",
            # Studio 集成
            "load_model_xml", "render", "pause_simulation",
            "sync_to_view",
        ]

    # --- 生命周期 ---

    async def init_simulation(self, model_xml_path: str) -> None:
        """初始化仿真，从 MJCF 文件加载模型。

        委托到 MuJoCoSimCore 加载模型，然后构建 ModelRegistry、SimConfig、
        OrcaGymDataView 子组件。

        Args:
            model_xml_path: MJCF 场景 XML 文件路径。
        """
        self._sim.init_simulation(model_xml_path)
        # 构建 ModelRegistry（传入 xml_path 用于 mesh 路径解析）
        self._registry = ModelRegistry(self._sim._mjModel, xml_path=model_xml_path)
        # 构建 SimConfig
        self._opt = SimConfig(self._sim._mjModel)
        # 构建状态视图
        ogm = self._registry.build_orca_gym_model()
        self._view = OrcaGymDataView(ogm)
        # 初始同步
        self._sim.forward()
        self._sim.sync_to_view(self._view)

    async def load_model_xml(self) -> str:
        """从 OrcaStudio 加载模型 XML，返回本地路径。委托到 OrcaStudioBridge。"""
        return await self._studio.load_model_xml()

    # --- 仿真控制（委托到 _sim）---

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
        """设置控制输入，应用 Studio UI 返回的 override_ctrls。

        Args:
            ctrl: 控制输入数组。
        """
        ctrl = np.asarray(ctrl).copy()
        for idx, value in self._studio.override_ctrls.items():
            if 0 <= idx < len(ctrl):
                ctrl[idx] = value
        self._sim.set_ctrl(ctrl)

    def apply_body_force(
        self, body_name: str, force: np.ndarray, torque: np.ndarray
    ) -> None:
        """在 body 上施加世界系外力/外力矩。

        Args:
            body_name: Body 名称。
            force: 力向量 [fx, fy, fz]，world frame。
            torque: 力矩向量 [tx, ty, tz]，world frame。
        """
        if self._view is None:
            raise RuntimeError("仿真未初始化，请先调用 init_simulation()")
        body_id = self._view._model.body_name2id(body_name)
        self._sim.apply_body_force(body_id, force, torque)

    def clear_body_force(self, body_name: str) -> None:
        """清零指定 body 的外力。

        Args:
            body_name: Body 名称。
        """
        if self._view is None:
            raise RuntimeError("仿真未初始化，请先调用 init_simulation()")
        body_id = self._view._model.body_name2id(body_name)
        self._sim.clear_body_force(body_id)

    def clear_all_forces(self) -> None:
        """清零所有 body 的外力。委托到 MuJoCoSimCore。"""
        self._sim.clear_all_forces()

    # --- 状态访问 ---

    @property
    def data(self) -> OrcaGymDataView:
        """MuJoCo 状态只读视图。"""
        if self._view is None:
            raise RuntimeError("仿真未初始化，请先调用 init_simulation()")
        return self._view

    @property
    def model(self):
        """OrcaGymModel 模型信息。"""
        if self._registry is None:
            raise RuntimeError("仿真未初始化，请先调用 init_simulation()")
        return self._registry.build_orca_gym_model()

    @property
    def sim_config(self) -> SimConfig:
        """求解器配置。"""
        if self._opt is None:
            raise RuntimeError("仿真未初始化，请先调用 init_simulation()")
        return self._opt

    @property
    def studio(self) -> OrcaStudioBridge:
        """OrcaStudio 集成桥接器。"""
        return self._studio

    # --- 状态同步 ---

    def sync_to_view(self) -> None:
        """将 _mjData 状态同步到 OrcaGymDataView。

        在 mj_step/mj_forward 后调用，确保 env.data 反映最新状态。
        """
        if self._view is None:
            raise RuntimeError("仿真未初始化，请先调用 init_simulation()")
        self._sim.sync_to_view(self._view)

    # --- Studio 集成 ---

    async def render(self) -> None:
        """将当前状态发送到 OrcaStudio 渲染。委托到 OrcaStudioBridge。

        依赖反转：从 _view 读取 qpos 和 time，不直接访问 _mjData。
        """
        if self._view is None:
            raise RuntimeError("仿真未初始化，请先调用 init_simulation()")
        await self._studio.render(self._view.qpos, self._view.time)

    async def pause_simulation(self) -> None:
        """暂停 OrcaStudio 仿真（被动模式）。委托到 OrcaStudioBridge。"""
        await self._studio.pause_simulation()
