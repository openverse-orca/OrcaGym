"""OrcaGymEuler — 仿真核心 Facade，组合子组件（骨架）。

本模块属于 OrcaGym Euler 体系骨架阶段（P2-Step4），是骨架阶段的核心交付物。
组合 MuJoCoSimCore/OrcaStudioBridge/ModelRegistry/SimConfig/OrcaGymDataView
子组件，实现隔离机制（K3/K5/K8/K9）。

核心设计:
    - 不暴露 _mjModel/_mjData（K3）—— 通过 __getattribute__ 拦截
    - 不暴露子组件对象（K5）—— _sim/_studio 等带下划线，__getattribute__ 拦截
    - Studio 交互通过方法 studio_bridge() 而非 property（K9）
    - Euler 耦合查询通过 has_euler()/step_with_coupling()（K8）

隔离机制说明:
    架构 §7.2 使用 __getattr__ 拦截 _mjData/_mjModel（不存储在 Gym 上）。
    本类将 _BLOCKED_ATTRS 扩展到子组件名（_sim/_studio 等，存储在 __dict__），
    因此使用 __getattribute__ 拦截（__getattr__ 仅在属性查找失败时触发，
    无法拦截已存在于 __dict__ 的属性）。内部访问通过 object.__getattribute__
    绕过拦截。
"""

import numpy as np

from orca_gym.core.euler.mujoco_sim_core import MuJoCoSimCore
from orca_gym.core.euler.orca_studio_bridge import OrcaStudioBridge
from orca_gym.core.euler.model_registry import ModelRegistry
from orca_gym.core.euler.sim_config import SimConfig
from orca_gym.core.euler.orca_gym_data_view import OrcaGymDataView


class OrcaGymEuler:
    """仿真核心 Facade，组合子组件，不暴露 _mjModel/_mjData，不暴露子组件对象。

    ┌─────────────────────────────────────────────────────────────┐
    │  API 契约：用户不应直接访问 _mjData / _mjModel / 任何子组件。│
    │  读取 MuJoCo 状态 → 使用 env.data（OrcaGymDataView）        │
    │  写入外力 → 使用 env.apply_body_force()                     │
    │  配置求解器 → 使用 env.sim_config                           │
    │  缺少功能时 → 扩展 OrcaGymEulerEnv 公共方法                 │
    └─────────────────────────────────────────────────────────────┘

    使用契约:
        读状态:     env.data.qpos / env.data.body_xpos("link1")
        写外力:     env.apply_body_force("link1", force, torque)
        配置:       env.sim_config.timestep = 0.002
        步进:       env.mj_step(nstep=1)
        Studio:     env.studio_bridge().render(...)  # 方法，非 property

    禁止:
        不要访问 gym._mjData / gym._mjModel / gym._sim / gym._studio 等。
        不要通过 @property 暴露 studio/sim/opt/view/euler 子组件。
    """

    # K3/K5: 隔离机制 — 拦截引擎内部和子组件对象
    _BLOCKED_ATTRS = frozenset({
        # L3 引擎内部
        "_mjData", "_mjModel", "mj_data", "mj_model",
        "_mj_data", "_mj_model", "mjData", "mjModel",
        # K5: 子组件对象也不对外暴露
        "_sim", "_studio", "_registry", "_opt", "_view", "_euler",
        "sim", "studio", "registry", "opt", "view", "euler",
    })

    def __init__(self, stub=None) -> None:
        """初始化仿真核心 Facade。

        组合所有子组件，全部带下划线（不在 __dir__ 暴露，被 __getattribute__ 拦截）。

        Args:
            stub: OrcaStudio gRPC stub，传递给 OrcaStudioBridge。
        """
        # 内部组件（全部带下划线，不在 __dir__ 暴露，访问被 __getattribute__ 拦截）
        self._sim = MuJoCoSimCore()
        self._studio = OrcaStudioBridge(stub=stub)
        self._registry = ModelRegistry()
        self._opt = SimConfig()
        self._view = OrcaGymDataView()
        self._euler = None    # EulerOrchestrator | None（骨架阶段恒为 None）

    # --- K3/K5: 隔离机制 ---

    def __getattribute__(self, name: str):
        """拦截 _BLOCKED_ATTRS 的外部访问，返回引导性错误。

        使用 __getattribute__（而非 __getattr__）是因为子组件名（_sim 等）
        存储在 __dict__ 中，__getattr__ 仅在属性查找失败时触发，无法拦截。
        内部访问通过 object.__getattribute__ 绕过本拦截。
        """
        blocked = object.__getattribute__(self, "_BLOCKED_ATTRS")
        if name in blocked:
            # 针对不同违规类型给出精准引导
            if name in ("_euler", "euler"):
                euler_hint = (
                    "  Euler 耦合查询 → 使用 env.has_euler() / env.step_with_coupling()\n"
                )
            elif name in ("_studio", "studio"):
                euler_hint = "  Studio 交互 → 使用 env.studio_bridge()\n"
            elif name in ("_sim", "sim"):
                euler_hint = (
                    "  仿真步进 → 使用 env.mj_step() / env.mj_forward() / env.do_simulation()\n"
                )
            elif name in ("_opt", "opt"):
                euler_hint = "  求解器配置 → 使用 env.sim_config\n"
            elif name in ("_view", "view"):
                euler_hint = "  状态读取 → 使用 env.data（OrcaGymDataView）\n"
            else:
                # L3 引擎内部 _mjData/_mjModel 等
                euler_hint = ""
            raise AttributeError(
                f"'{type(self).__name__}' 对象的属性 '{name}' 被隔离。\n"
                f"  API 契约：用户不应直接访问 _mjData / _mjModel / 任何子组件。\n"
                f"  读取 MuJoCo 状态 → 使用 env.data（OrcaGymDataView），如 env.data.qpos\n"
                f"  写入外力 → 使用 env.apply_body_force()\n"
                f"  配置求解器 → 使用 env.sim_config\n"
                f"{euler_hint}"
                f"  缺少功能时 → 扩展 OrcaGymEulerEnv 公共方法，不要直接访问内部对象。"
            )
        return object.__getattribute__(self, name)

    def __dir__(self) -> list[str]:
        """只列出公共 API，不含子组件对象或引擎内部。"""
        result = super().__dir__()
        blocked = type(self)._BLOCKED_ATTRS
        return [name for name in result if name not in blocked]

    # --- 生命周期 ---

    async def init_simulation(self, model_xml_path: str) -> None:
        """初始化仿真（加载模型、同步 Studio）。

        Args:
            model_xml_path: MuJoCo 模型 XML 文件路径。

        Raises:
            NotImplementedError: 骨架阶段未实现真实初始化。
        """
        raise NotImplementedError("init_simulation 待 P4 填充")

    async def load_model_xml(self) -> str:
        """从 OrcaStudio 加载模型 XML 字符串。

        Returns:
            MuJoCo 模型 XML 字符串。

        Raises:
            NotImplementedError: 骨架阶段未实现真实加载。
        """
        raise NotImplementedError("load_model_xml 待 P4 填充")

    # --- 仿真控制（委托 _sim）---

    def mj_step(self, nstep: int) -> None:
        """执行 nstep 步 MuJoCo 仿真。

        Args:
            nstep: 步进次数。

        Raises:
            NotImplementedError: 骨架阶段未实现真实步进。
        """
        raise NotImplementedError("mj_step 待 P4 填充")

    def mj_forward(self) -> None:
        """执行 MuJoCo 前向计算（不步进，仅更新派生量）。

        Raises:
            NotImplementedError: 骨架阶段未实现真实前向计算。
        """
        raise NotImplementedError("mj_forward 待 P4 填充")

    def set_ctrl(self, ctrl: np.ndarray) -> None:
        """设置控制输入。

        Args:
            ctrl: 控制输入数组。

        Raises:
            NotImplementedError: 骨架阶段未实现真实控制设置。
        """
        raise NotImplementedError("set_ctrl 待 P4 填充")

    # --- 状态同步 ---

    def sync_to_view(self) -> None:
        """将 MuJoCo 状态同步到 OrcaGymDataView（env.data）。

        Raises:
            NotImplementedError: 骨架阶段未实现真实同步。
        """
        raise NotImplementedError("sync_to_view 待 P4 填充")

    # --- K5/K6: 状态访问（返回 typed 对象，不返回子组件引用）---

    @property
    def data(self) -> OrcaGymDataView:
        """返回 MuJoCo 状态只读视图（OrcaGymDataView）。

        替代直接访问 _mjData。
        """
        return object.__getattribute__(self, "_view")

    @property
    def model(self):
        """返回模型结构抽象（OrcaGymModel）。

        Raises:
            NotImplementedError: 骨架阶段未构建真实模型。
        """
        raise NotImplementedError("model 待 P4 填充（需 build_orca_gym_model）")

    @property
    def sim_config(self) -> SimConfig:
        """返回求解器配置（SimConfig）。

        替代直接访问 _mjModel.opt.*。
        """
        return object.__getattribute__(self, "_opt")

    # --- K9: Studio 桥接访问（方法而非 property）---

    def studio_bridge(self) -> OrcaStudioBridge:
        """返回 OrcaStudio 桥接对象。

        K9: 通过方法访问而非 @property，防止 gym.studio 式穿墙。
        禁止: 不提供 studio 的 property 定义。
        """
        return object.__getattribute__(self, "_studio")

    # --- Studio 委托（骨架最小集）---

    async def render(self) -> None:
        """渲染当前仿真状态到 OrcaStudio。

        Raises:
            NotImplementedError: 骨架阶段未实现真实渲染。
        """
        raise NotImplementedError("render 待 P4 填充")

    async def pause_simulation(self) -> None:
        """通知 OrcaStudio 暂停仿真。

        Raises:
            NotImplementedError: 骨架阶段未实现真实暂停。
        """
        raise NotImplementedError("pause_simulation 待 P4 填充")

    # --- K8: 步进耦合查询（供 do_simulation 使用，不暴露 _euler）---

    def has_euler(self) -> bool:
        """查询是否存在 Euler 耦合编排器。

        骨架阶段恒返回 False（_euler 为 None）。

        Returns:
            False（骨架阶段无 Euler）。
        """
        return object.__getattribute__(self, "_euler") is not None

    def step_with_coupling(self, ctrl: np.ndarray, n_frames: int, dt: float) -> None:
        """执行带 Euler 耦合的步进。

        供 do_simulation 使用，替代 do_simulation 内部直接读 self._gym._euler。
        骨架阶段无 Euler，raise NotImplementedError。

        Args:
            ctrl: 控制输入数组。
            n_frames: 帧数。
            dt: 时间步长。

        Raises:
            NotImplementedError: 骨架阶段未实现 Euler 耦合。
        """
        raise NotImplementedError("step_with_coupling 待 P4 填充")
