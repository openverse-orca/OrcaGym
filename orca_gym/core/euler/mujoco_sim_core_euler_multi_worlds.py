"""MuJoCoSimCoreEulerMultiWorlds — 多世界 Euler 后端编排层（P2 轨道 1，nworld>1）。

多世界编排层对齐 design C4（编排层最小方法集）：仅委托 4 个驱动方法
（step/forward/reset/invalidate_capture）+ 暴露 solver/mj_model/nworld/
render_bridge property，**不提供**单世界的 41 个 CPU 风格委托方法
（set_ctrl/set_qpos_qvel/query_* 等）——多世界消费者是应用框架，
按 GPU-native 范式直接操作::

    sim.solver.mjf_data.<field>.assign(np_arr)   # H2D（一次写全部世界）
    sim.step(nstep)                              # 单次 launch/Graph 重放推进所有世界
    sim.solver.mjf_data.<field>.numpy()           # D2H（形状首维 nworld）

无 lazy 标记、无 host 缓冲、无隐式同步（与单世界编排层的本质区别，
对齐 design C2「GPU-native 薄封装」）。

``_solver`` 通过字符串前向引用类型注释（`from __future__ import annotations`），
避免在 CPU-only 测试环境 import `orca.euler`；真正的 import 放在
`init_simulation` 内（与单世界一致）。
"""

from __future__ import annotations

from typing import Any


class MuJoCoSimCoreEulerMultiWorlds:
    """多世界编排层（nworld>1，GPU-native 范式，对齐 design C4）。

    使用契约:
        初始化:     core.init_simulation("model.xml", device, nworld)
        步进:       core.step(nstep=1)
        重算派生量: core.forward()
        重置:       core.reset(reset_mask=None)
        Graph 失效: core.invalidate_capture()
        数据读写:   core.solver.mjf_data.<field>.assign/.numpy()

    禁止:
        外部不应直接访问本类的 `_solver` 之外的任何 `_` 前缀成员。
    """

    def __init__(self) -> None:
        self._solver = None     # SolverMujocoMultiWorld | None
        self._nworld: int | None = None
        self._render_bridge = None    # RenderBridge | None（P3 交付本体；P2 恒 None）

    def _require_solver(self) -> None:
        """确保 solver 已初始化，否则抛 RuntimeError（与单世界一致）。"""
        if self._solver is None:
            raise RuntimeError("Simulation not initialized")

    # ---- 生命周期方法 ----

    def init_simulation(
        self,
        model_xml_path: str,
        device: str = "cuda",
        nworld: int = 2,
        timestep: float | None = None,
        opt_overrides: dict[str, Any] | None = None,
        render_config: Any | None = None,
        **solver_kwargs: Any,
    ) -> None:
        """加载模型并构造 SolverMujocoMultiWorld（两段式第二段）。

        Args:
            model_xml_path: MuJoCo 模型 XML 文件路径。
            device: GPU 设备（多世界求解器仅支持 CUDA）。
            nworld: 并行世界数，必须 > 1（单世界请使用 MuJoCoSimCoreEuler）。
            timestep: 物理时间步长（秒）。None 时保留 XML 默认值；非 None
                时在构造求解器前覆盖（Euler 后端初始化后 timestep 只读）。
            opt_overrides: 构造期 opt 覆盖（与单世界一致，经
                ``prepare_host_model`` 在 put_model 前写入 host model.opt）。
            render_config: 渲染桥配置（P3 交付本体；P2 阶段传入非 None 抛
                NotImplementedError）。
            **solver_kwargs: put_data 容量参数覆盖（nconmax/njmax 等）。
                未显式传参时沿用单世界默认策略（njmax>=2000、nconmax>=500，
                防 G1 类模型接触截断发散）。
        """
        if nworld <= 1:
            raise ValueError(
                f"MuJoCoSimCoreEulerMultiWorlds 要求 nworld>1，收到 nworld={nworld}。"
                f"单世界场景请使用 MuJoCoSimCoreEuler。"
            )
        if render_config is not None:
            raise NotImplementedError(
                "render_bridge 组件 P3 交付；P2 阶段请勿传入 render_config。"
            )
        try:
            import orca.euler as euler
        except ImportError as e:
            raise RuntimeError(
                "Euler 后端不可用：orca.euler 未安装。"
                "请确认已安装 orca.euler 且 SimConfig.backend = SimBackend.EULER。"
            ) from e

        from orca_gym.core.euler.mujoco_sim_core_euler import prepare_host_model

        model = prepare_host_model(model_xml_path, timestep, opt_overrides)
        # 缓冲区上限默认对齐单世界策略（G1 等复杂模型零控瘫倒场景下接触/
        # 约束数量远超 mujoco_flow 默认启发式，沿用默认会接触截断→数值发散）。
        # 用户显式传参时以用户值为准。
        solver_kwargs.setdefault("njmax", max(model.njmax, 2000))
        solver_kwargs.setdefault("nconmax", max(model.nconmax, 500))
        self._solver = euler.SolverMujocoMultiWorld(
            source=model, device=device, nworld=nworld, **solver_kwargs
        )
        self._nworld = int(nworld)

    # ---- 驱动方法（仅 4 个，对齐 design C4.2）----

    def step(self, nstep: int = 1) -> None:
        """步进所有世界 nstep 步（委托 solver.step，Graph Capture 内部封装）。"""
        self._require_solver()
        self._solver.step(nstep)

    def forward(self) -> None:
        """重算派生量（委托 solver.forward，不积分）。"""
        self._require_solver()
        self._solver.forward()

    def reset(self, reset_mask: Any = None) -> None:
        """重置到 model 默认状态（委托 solver.reset；掩码可选，支持部分世界）。"""
        self._require_solver()
        self._solver.reset(reset_mask)

    def invalidate_capture(self) -> None:
        """强制 Graph 重录（model 字段变更后调用）。"""
        self._require_solver()
        self._solver.invalidate_capture()

    # ---- property ----

    @property
    def solver(self):
        """暴露 solver，用户直接操作 mjf_data / mjf_model（GPU-native 范式）。

        未初始化（init_simulation 前）为 None；驱动方法的未初始化守卫见
        ``_require_solver``。
        """
        return self._solver

    @property
    def mj_model(self):
        """host MjModel（委托 solver.mj_model；SimConfig/ModelRegistry 绑定点）。

        host 元数据唯一例外：name→address 解析、维度查询等纯 host 读零 D2H；
        运行期模型参数修改请走 mjf_model + invalidate_capture。
        未初始化时为 None。
        """
        return self._solver.mj_model if self._solver is not None else None

    @property
    def nworld(self) -> int | None:
        """并行世界数（构造期断言 > 1）；未初始化时为 None。"""
        return self._nworld

    @property
    def render_bridge(self):
        """渲染桥（P3 交付本体；P2 恒 None）。"""
        return self._render_bridge
