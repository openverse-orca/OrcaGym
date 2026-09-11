"""EulerSoftSim — 非耦合柔体仿真器（ESDF 双文件注入）。

非耦合双引擎契约（区别于双向耦合的 ``_euler`` 编排器）：

  * 刚体动力学归 MuJoCo（OrcaGym Euler GPU 后端 ``MuJoCoSimCoreEuler`` 推进）
  * 柔体动力学归 Euler（本类推进，``rigid_body_mode="external"``）
  * 每个耦合周期单次数据交换：
      MuJoCo 位姿 → Euler  ``sync_body_pose``（本类）
      Euler 接触力 → MuJoCo ``body_f_numpy`` → ``xfrc_applied``
      （由 ``OrcaGymEuler.step_with_coupling`` 完成 COM→origin 力矩变换）

构建流程对齐 OrcaEuler 双文件权威示例
（``examples/solver_xpbd/08_dual_file_mujoco_euler/dual_file_drop.py``）::

    builder = ModelBuilder()
    builder.add_mjcf(xml_path)          # 刚体（与 MuJoCo 共用同一 XML）
    builder.add_esdf(esdf_path)         # 柔体
    ...（solver type 分支预处理）
    builder.finalize(device)            # 一次 finalize

兼容两种 ESDF solver：

  * ``"xpbd"``：``apply_soft_contact_from_parse_result``（soft_contact
    section 必填）+ ``solver.params`` 透传 SolverXPBD 构造参数
  * ``"spring_mass_semi_implicit"``：逐 ``spring_mass_builder`` 执行
    ``compute_dt_info`` + ``apply_damping_mode``（damping_mode 生效的
    唯一路径，见 ``esdf_damping_mode_design.md``）

外部刚体模式（external）语义：Euler 跳过刚体积分，柔体接触对刚体的
力累积进 ``body_f``（[F(3), T_com(3)]，世界系、body COM 参考）供
外部引擎读取；刚体位姿由 ``sync_body_pose`` 注入。

渲染流（P5 改动点 2，对齐 05 课 ``esdf_skinning_snapshot`` 编排）::

    构造期：build_skin_bindings_from_parse_result（P3 契约：finalize 后
    、任何 step 前读 bind pose）+ 每 body 一个 SidecarRenderGraph
    （establish，不连接 gRPC）
    enable_render_stream(target)：显式 opt-in 连接 RenderClient 并逐
    body 注册 CHANNEL_TYPE_DEFORMABLE_VERTEX（fail-fast）
    render_frame()：skin_cb（每渲染帧一次，不进 step 子步）→ 逐通道
    graph.sync → flow.synchronize → 逐 body read_q → send（per-body
    sequence 单调递增）。send 失败自动断流降级，不炸物理循环。

K3/K5：本类为 core 层内部组件，由 ``OrcaGymEuler`` Facade 组合，
不进入用户 API；对外查询经 ``OrcaGymEuler.has_soft_sim()`` 等公共方法。
"""

from __future__ import annotations

from typing import Any

import numpy as np
import orca.flow as flow

# SolverXPBD 构造参数白名单（ESDF ``solver.params`` 只透传这些键，
# 避免 ESDF 携带无关键导致 TypeError）。
_XPBD_PARAM_KEYS = frozenset({
    "iterations",
    "soft_contact_relaxation",
    "rigid_contact_relaxation",
    "rigid_contact_con_weighting",
    "soft_body_relaxation",
    "mass_scale",
})

# SolverSemiImplicit 构造参数白名单（Studio EsdfExporter 的
# ``solver.params`` 含 implicit_damping 等非构造器键，只透传这些；
# contact_margin 是 solver section 顶层字段，单独取）。
_SI_PARAM_KEYS = frozenset({
    "angular_damping",
    "friction_smoothing",
    "joint_attach_ke",
    "joint_attach_kd",
    "enable_tri_contact",
})


class EulerSoftSim:
    """非耦合柔体仿真器：ESDF 双文件注入 + 每 Gym 帧分步推进。

    使用契约::

        soft = EulerSoftSim(xml_path, esdf_path, device, mj_model)
        soft.sync_body_pose(snapshot)      # 注入 MuJoCo 位姿（快照 dict）
        soft.step(n_frames, dt_macro)      # 推进 n_frames × dt_macro 秒
        body_f = soft.body_f_numpy()       # 柔体→刚体接触力（COM 参考）
    """

    def __init__(
        self,
        model_xml_path: str,
        esdf_path: str,
        device: str,
        mj_model: Any,
    ) -> None:
        """构造双注入模型、求解器、状态对。

        Args:
            model_xml_path: MuJoCo 模型 XML 路径（与 MuJoCo 侧共用）。
            esdf_path: ESDF 场文件路径。
            device: Flow 设备（如 "cuda:0" / "cpu"）。
            mj_model: host MjModel（用于 Euler↔MuJoCo body 名字映射，
                来自 ``MuJoCoSimCoreEuler.mj_model`` 公共 property）。

        Raises:
            ImportError: orca.euler 未安装。
            RuntimeError: add_esdf 改写了 add_mjcf 的刚体标签。
            ValueError: ESDF solver.type 不支持，或 XPBD 缺 soft_contact。
        """
        try:
            import orca.euler as euler
        except ImportError as e:
            raise RuntimeError(
                "Euler 后端不可用：orca.euler 未安装，无法构建 ESDF 柔体仿真。"
            ) from e

        builder = euler.ModelBuilder()
        builder.add_mjcf(model_xml_path)
        rigid_labels = list(builder.body_label)

        parse_result = builder.add_esdf(esdf_path)
        if list(builder.body_label) != rigid_labels:
            raise RuntimeError(
                "add_esdf 改写了 add_mjcf 的刚体标签（body_label），"
                f"before={rigid_labels} after={list(builder.body_label)}"
            )

        # ESDF scene.gravity → builder.gravity（对齐 dual_file_drop L103）。
        gravity = (parse_result.scene or {}).get("gravity")
        if gravity:
            builder.gravity = float(gravity[2])

        solver_type = str(parse_result.solver_config.get("type", ""))
        params = dict(parse_result.solver_config.get("params") or {})

        if solver_type == "xpbd":
            applied = euler.apply_soft_contact_from_parse_result(
                builder, parse_result
            )
            if not applied:
                raise ValueError(
                    f"XPBD ESDF 缺少 soft_contact section: {esdf_path} "
                    "(expected ke/kd/kf/mu/ka for XPBD soft contact)"
                )
            self._model = builder.finalize(device=device)
            kwargs = {k: v for k, v in params.items() if k in _XPBD_PARAM_KEYS}
            self._solver = euler.SolverXPBD(self._model, **kwargs)
        elif solver_type in ("semi_implicit", "spring_mass_semi_implicit"):
            # Studio EsdfExporter 导出 "semi_implicit"（EsdfExporter.h 默认值），
            # OrcaEuler examples 用全称 "spring_mass_semi_implicit"——同一
            # SolverSemiImplicit（弹簧质点 + 半隐式积分），两字符串等义。
            for sm in parse_result.spring_mass_builders:
                dt_info = sm.compute_dt_info()
                sm.apply_damping_mode(dt_info)
            self._model = builder.finalize(device=device)
            kwargs = {k: v for k, v in params.items() if k in _SI_PARAM_KEYS}
            contact_margin = parse_result.solver_config.get("contact_margin")
            if contact_margin is not None:
                kwargs["contact_margin"] = float(contact_margin)
            self._solver = euler.SolverSemiImplicit(self._model, **kwargs)
        else:
            raise ValueError(
                f"不支持的 ESDF solver.type: {solver_type!r} "
                f"(支持 'xpbd' / 'semi_implicit' / 'spring_mass_semi_implicit')"
                f": {esdf_path}"
            )

        self._dt = float(euler.resolve_dt(parse_result, self._solver))
        # 接触缓冲：粒子数 × 4 为经验上界（每粒子同时至多 ~4 个活跃接触），
        # 下限 8192 对齐 06_dual_engine 示例的 SOFT_CONTACT_MAX。
        self._contacts = euler.Contacts(
            soft_contact_max=max(self._model.particle_count * 4, 8192),
            device=self._model.device,
        )

        # 双 State（XPBD/SemiImplicit 双缓冲契约：step 后交换 in/out）。
        self._state_in = self._model.state()
        self._state_out = self._model.state()

        # Euler↔MuJoCo body 映射 + 静态缓存（COM/rootid，构造期一次性）。
        self._mj_to_euler = self._build_body_map(mj_model, rigid_labels)
        self._mj_rootid = np.asarray(mj_model.body_rootid, dtype=np.int32)
        self._body_com_local = (
            self._model.body_com.numpy().copy()
            if self._model.body_count > 0 else None
        )

        # 渲染流（P5 改动点 2）：构造期就绪，连接显式 opt-in。
        self._parse_result = parse_result
        self._skin_cb: Any = None
        self._render_graphs: dict[str, tuple[Any, Any]] | None = None
        self._render_client: Any = None
        self._render_sequences: dict[str, int] = {}
        self._render_warned = False
        self._setup_render_pipeline(euler)

    def _setup_render_pipeline(self, euler: Any) -> None:
        """构造期烘焙 skinning bindings + 建渲染通道（不连接 gRPC）。

        P3 契约：``build_skin_bindings_from_parse_result`` 必须在 finalize
        后、任何 step 前（bind pose 读 ``Model.particle_q`` 初始快照）——
        本方法仅在 ``__init__`` 尾部调用一次。degraded ESDF（无
        ``render.proxy_mesh``，deformable_bindings 为空）时无渲染流，
        ``render_frame`` 恒 no-op。
        """
        bindings = euler.build_skin_bindings_from_parse_result(
            self._model, self._parse_result
        )
        if not bindings:
            return
        self._skin_cb = euler.make_step_callback(bindings)
        graphs: dict[str, tuple[Any, Any]] = {}
        for binding in bindings:
            body_name = (
                self._model.deformable_body_descriptors[binding.body_index].name
            )
            descriptor = euler.ChannelDescriptor(
                name=f"{body_name}_verts",
                type="deformable_vertex",
                euler_field="state.deformable_vertex_q",
                euler_body_name=body_name,
            )
            graph = euler.SidecarRenderGraph([descriptor], model=self._model)
            graph.establish(self._state_in)
            graphs[body_name] = (graph, graph.deformable())
        self._render_graphs = graphs
        self._render_sequences = {name: 0 for name in graphs}

    # ------------------------------------------------------------------
    # 构建辅助
    # ------------------------------------------------------------------

    def _build_body_map(
        self, mj_model: Any, rigid_labels: list[str]
    ) -> dict[int, int]:
        """按 body 名匹配构建 MuJoCo→Euler 索引映射。

        对齐 ``material_comparison_dual_engine._build_mj_to_euler_body_map``：
        Euler ``body_label[i]`` 的末段（``/`` 分隔）↔ MuJoCo body 名。
        每个 Euler body 必须在 MuJoCo 中找到匹配（fail-fast）。

        Args:
            mj_model: host MjModel。
            rigid_labels: ``add_mjcf`` 后的 Euler body_label 列表。

        Returns:
            dict[mj_i -> euler_i]。floor-only MJCF（无 <body>）返回空 dict。
        """
        mj_name_to_id = {
            mj_model.body(mj_i).name: mj_i for mj_i in range(mj_model.nbody)
        }
        mj_to_euler: dict[int, int] = {}
        for euler_i, label in enumerate(rigid_labels):
            body_name = label.split("/")[-1]
            if body_name not in mj_name_to_id:
                raise ValueError(
                    f"Euler body {euler_i} label={label!r} (name={body_name!r}) "
                    f"在 MuJoCo 中无同名 body。MuJoCo body 名: "
                    f"{sorted(mj_name_to_id.keys())}"
                )
            mj_i = mj_name_to_id[body_name]
            if mj_i in mj_to_euler:
                raise ValueError(
                    f"重复 body 名 {body_name!r}: Euler body {euler_i} 与 "
                    f"Euler body {mj_to_euler[mj_i]} 均映射到 MuJoCo body "
                    f"{mj_i}。"
                )
            mj_to_euler[mj_i] = euler_i
        return mj_to_euler

    # ------------------------------------------------------------------
    # 公共查询
    # ------------------------------------------------------------------

    @property
    def model(self) -> Any:
        """返回 Euler Model（双注入：MJCF 刚体 + ESDF 柔体）。"""
        return self._model

    @property
    def dt(self) -> float:
        """返回柔体子步时间步长（ESDF solver.dt 解析值，CFL 安全）。"""
        return self._dt

    @property
    def body_map(self) -> dict[int, int]:
        """返回 MuJoCo→Euler body 索引映射（dict[mj_i -> euler_i]）。"""
        return dict(self._mj_to_euler)

    def body_f_numpy(self) -> np.ndarray | None:
        """返回当前子步后柔体对刚体的接触力（numpy，[nbody, 6]）。

        布局 ``[F(3), T_com(3)]``：世界系、body COM 参考。回写 MuJoCo
        ``xfrc_applied``（body 原点参考）时需做 ``T_origin = T_com +
        cross(r_com, F)`` 变换（由 ``OrcaGymEuler.step_with_coupling``
        完成）。无刚体（floor-only MJCF）返回 None。
        """
        body_f = self._state_in.body_f
        return body_f.numpy() if body_f is not None else None

    # ------------------------------------------------------------------
    # 位姿注入（MuJoCo → Euler）
    # ------------------------------------------------------------------

    def sync_body_pose(self, snapshot: dict[str, np.ndarray]) -> None:
        """将 MuJoCo 最新刚体位姿/速度注入双 State。

        通用版变换（对齐 deprecated ``g1_particle_simulation`` 的
        ``inject_body_state_from_mujoco``）：

          * ``body_q[i, :3]  = xpos``；``body_q[i, 3:7] = xquat``（wxyz→xyzw）
          * ``body_qd`` 参考点变换：MuJoCo ``cvel`` 是 subtree COM 参考
            [ang, lin]，Euler 期望 body COM 参考 [lin, ang]，
            ``v_body_com = v_subtree_com + cross(ang, r_offset)``

        Args:
            snapshot: ``MuJoCoSimCoreEuler.query_dual_engine_state()`` 返回
                的快照（xpos/xquat/xmat/cvel/xipos/subtree_com）。
        """
        if self._model.body_count == 0 or not self._mj_to_euler:
            return  # floor-only MJCF：无刚体可注入
        body_q, body_qd = self._build_body_state(snapshot)
        q = flow.array(body_q, dtype=flow.transform, device=self._model.device)
        qd = flow.array(
            body_qd, dtype=flow.spatial_vector, device=self._model.device
        )
        # 双 State 都写：state_out 在下一子步交换后成为输入，若不写
        # 会回退到 rest pose 污染接触检测。
        for st in (self._state_in, self._state_out):
            st.body_q.assign(q)
            st.body_qd.assign(qd)

    def _build_body_state(
        self, snapshot: dict[str, np.ndarray]
    ) -> tuple[np.ndarray, np.ndarray]:
        """从 MuJoCo 快照构建 Euler body_q/body_qd（numpy）。"""
        n = self._model.body_count
        body_q = np.zeros((n, 7), dtype=np.float32)
        body_qd = np.zeros((n, 6), dtype=np.float32)
        for mj_i, euler_i in self._mj_to_euler.items():
            body_q[euler_i, :3] = snapshot["xpos"][mj_i]
            w, x, y, z = snapshot["xquat"][mj_i]
            body_q[euler_i, 3:7] = [x, y, z, w]  # wxyz → xyzw

            # cvel: [ang(3), lin(3)]，世界系，subtree COM 参考。
            ang_world = snapshot["cvel"][mj_i, :3]
            lin_at_subtree = snapshot["cvel"][mj_i, 3:]
            # body COM 世界坐标 = xpos + R · body_com_local。
            rot = snapshot["xmat"][mj_i].reshape(3, 3)
            r_body_com_world = (
                snapshot["xpos"][mj_i]
                + rot @ self._body_com_local[euler_i]
            )
            rootid = int(self._mj_rootid[mj_i])
            r_subtree_com = snapshot["subtree_com"][rootid]
            # 参考点变换：subtree COM → body COM。
            r_offset = r_body_com_world - r_subtree_com
            body_qd[euler_i, :3] = lin_at_subtree + np.cross(
                ang_world, r_offset
            )
            body_qd[euler_i, 3:] = ang_world
        return body_q, body_qd

    # ------------------------------------------------------------------
    # 步进（Euler 柔体子步）
    # ------------------------------------------------------------------

    def step(self, n_frames: int, dt_macro: float) -> None:
        """推进柔体 ``n_frames × dt_macro`` 秒（拆分为 CFL 子步）。

        子步数 ``n_sub = max(1, round(n_frames·dt_macro / dt))``（对齐
        mattress 示例 ``_compute_frame_skip``）。整个调用期间刚体位姿
        冻结在最近一次 ``sync_body_pose`` 注入值（对齐 06_dual_engine
        每耦合周期单次 inject 的时序）。

        每子步::

            solver.step(state_in, state_out, None, contacts, dt,
                        rigid_body_mode="external")
            state_out.body_q/qd ← state_in（external 模式 SemiImplicit
                不搬运刚体状态，XPBD 已内置拷贝，此处统一重拷为幂等）
            state_in ↔ state_out

        Args:
            n_frames: MuJoCo 宏步数（与 step_with_coupling 的 n_frames 一致）。
            dt_macro: 每宏步时长（秒）。
        """
        if n_frames <= 0:
            return
        total = float(n_frames) * float(dt_macro)
        n_sub = max(1, int(round(total / self._dt)))
        has_bodies = self._model.body_count > 0
        for _ in range(n_sub):
            self._solver.step(
                state_in=self._state_in,
                state_out=self._state_out,
                control=None,
                contacts=self._contacts,
                dt=self._dt,
                rigid_body_mode="external",
            )
            if has_bodies:
                # external 模式下 solver 跳过刚体积分，SemiImplicit 不会
                # 把 body_q/body_qd 搬进 state_out；不重拷则下一子步的
                # 接触检测使用 rest pose（XPBD 内置拷贝，此处幂等）。
                self._state_out.body_q.assign(self._state_in.body_q)
                self._state_out.body_qd.assign(self._state_in.body_qd)
            self._state_in, self._state_out = self._state_out, self._state_in

    # ------------------------------------------------------------------
    # 重置
    # ------------------------------------------------------------------

    def reset(self, snapshot: dict[str, np.ndarray] | None = None) -> None:
        """重置双 State 到模型初始状态（粒子回初始位姿，body_f 清零）。

        Args:
            snapshot: 可选的 MuJoCo 快照；给出时在重置后立即注入刚体
                位姿（典型调用点：``reset_simulation`` 中
                ``MuJoCoSimCoreEuler.reset_data()`` 之后）。
        """
        self._state_in = self._model.state()
        self._state_out = self._model.state()
        if snapshot is not None:
            self.sync_body_pose(snapshot)

    # ------------------------------------------------------------------
    # 渲染流（P5 改动点 2：render() 节拍驱动，不进 step 循环）
    # ------------------------------------------------------------------

    def enable_render_stream(
        self, target: str = "127.0.0.1:50451", timeout: float = 5.0
    ) -> None:
        """连接 Studio 渲染服务并逐 body 注册可变形顶点通道。

        显式 opt-in（fail-fast）：连接失败/注册失败直接抛错（用户显式
        传入 target 即期望推流，静默降级会掩盖 Studio 未开 50451 的
        配置错误）。渲染通道在构造期已建立，本方法仅建立 gRPC 连接。

        Args:
            target: Studio DeformableChannelComponent 的 gRPC 地址
                （默认 ``127.0.0.1:50451``）。
            timeout: gRPC channel ready 超时（秒）。

        Raises:
            RuntimeError: ESDF 无可渲染柔体（deformable_bindings 为空）。
            ImportError: grpcio 未安装。
            ConnectionError: 连接超时。
            RuntimeError: 注册被服务端拒绝。
        """
        if self._render_graphs is None:
            raise RuntimeError(
                "ESDF 无可渲染柔体（deformable_bindings 为空，缺 "
                "render.proxy_mesh？），无法启用渲染流。"
            )
        # 懒加载：RenderClient 依赖 grpcio（[render] extra），未启用渲染
        # 流的环境无需安装。
        from orca.euler.render import RenderClient

        client = RenderClient(target, timeout=timeout)
        try:
            for body_name, (_, ch) in self._render_graphs.items():
                client.register_deformable_channel(
                    name=f"{body_name}_verts",
                    body_name=body_name,
                    vertex_count=ch.vertex_count,
                )
        except Exception:
            client.close()
            raise
        self._render_client = client
        self._render_warned = False

    def disable_render_stream(self) -> None:
        """断开渲染流（幂等：未连接时 no-op）。"""
        if self._render_client is not None:
            self._render_client.close()
            self._render_client = None

    def has_render_stream(self) -> bool:
        """渲染流是否已连接（degraded ESDF 恒 False）。"""
        return self._render_client is not None

    def render_frame(self) -> bool:
        """渲染节拍执行体：skinning → 通道 sync → 读槽 → gRPC 推流。

        D4 契约：由 ``OrcaGymEuler.render()``（env 层 30Hz 节流后）驱动，
        **不进 step 子步循环**。时序对齐 05 课 driver::

            skin_cb(state_in)             # 每渲染帧一次（P3 契约）
            graph.sync(state_in)          # 写槽 + 翻指针（逐通道）
            flow.synchronize()            # D2H 前置同步
            ch.read_q.numpy()             # 冻结读槽
            send_deformable_vertices(q, seq, body_name)   # 逐 body

        send 失败（Studio 中途退出/断连）时自动断流降级：打印一次警告、
        后续 no-op，不中断物理循环（渲染中断不应炸仿真/训练）。

        Returns:
            是否完成一次推流（未启用/降级断流返回 False）。
        """
        if self._render_client is None or self._render_graphs is None:
            return False
        self._skin_cb(self._state_in, self._model)
        for graph, _ in self._render_graphs.values():
            graph.sync(self._state_in)
        flow.synchronize()
        try:
            for body_name, (_, ch) in self._render_graphs.items():
                self._render_sequences[body_name] += 1
                self._render_client.send_deformable_vertices(
                    ch.read_q.numpy(),
                    sequence=self._render_sequences[body_name],
                    body_name=body_name,
                )
        except Exception as e:  # noqa: BLE001 — 渲染中断不炸物理循环
            if not self._render_warned:
                print(
                    f"[EulerSoftSim] 渲染流推送失败，自动断流降级（仿真不受"
                    f"影响；Studio 恢复后可重新 enable）：{e!r}"
                )
                self._render_warned = True
            self.disable_render_stream()
            return False
        return True
