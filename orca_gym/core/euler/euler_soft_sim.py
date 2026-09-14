"""EulerSoftSim — 旧的 CPU 快照柔体仿真器。已禁止接到 OrcaGym。

OrcaGymEuler 不再持有 ``_soft`` 槽。生产 ESDF 走 Euler
``CoupledGpuSim``。本文件只给隔离单测对照旧路径，不要从 Gym 构造。

隔离单测仍覆盖：``add_mjcf`` + ``add_esdf``、``rigid_body_mode="external"``、
``sync_body_pose``、渲染流世界→局部。那些路径不是 Gym 生产耦合。
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


def _rotation_matrix_from_esdf_wxyz(wxyz) -> np.ndarray:
    """把 ESDF 的 ``initial.rotation``（wxyz）变成 3×3 旋转矩阵。

    做什么：单位化四元数后写出列向量约定的旋转矩阵，使
    ``p_world = R @ p_local + T``。零长度四元数退回单位矩阵。

    为什么：推流要把 Euler 世界坐标变回网格局部，必须和
    ``add_cloth_mesh`` / ``import_esdf`` 用的同一套 R，两边才能互逆。
    """
    values = [float(v) for v in wxyz]
    if len(values) != 4:
        return np.eye(3, dtype=np.float64)
    w, x, y, z = values
    norm = (w * w + x * x + y * y + z * z) ** 0.5
    if norm < 1.0e-12:
        return np.eye(3, dtype=np.float64)
    w, x, y, z = w / norm, x / norm, y / norm, z / norm
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _load_esdf_body_initial_poses(
    esdf_path: str,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """从 ESDF 读每个 body 的初始平移和旋转。

    做什么：解析 ``bodies[].initial.position`` 与 ``initial.rotation``
    （缺 rotation 时回退 ``transform.rotation``，再缺则单位四元数），
    得到 ``body_name → (T, R)``。

    为什么：``EsdfParseResult`` 没有按 body 缓存初始位姿；推流变回局部
    必须用和物理导入同一份 ESDF 初值，不能去问 Studio 实体。
    """
    import json5

    with open(esdf_path, encoding="utf-8") as handle:
        data = json5.loads(handle.read())
    poses: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for body in data.get("bodies") or []:
        if not isinstance(body, dict):
            continue
        name = body.get("body_name") or body.get("name")
        if not name:
            continue
        initial = body.get("initial") or {}
        position = np.asarray(
            initial.get("position") or (0.0, 0.0, 0.0),
            dtype=np.float64,
        ).reshape(3)
        rotation = initial.get("rotation")
        if rotation is None:
            transform = body.get("transform") or {}
            rotation = transform.get("rotation")
        if rotation is None:
            rotation = (1.0, 0.0, 0.0, 0.0)
        poses[str(name)] = (
            position,
            _rotation_matrix_from_esdf_wxyz(rotation),
        )
    return poses


def _world_vertices_to_mesh_local(
    positions: np.ndarray,
    origin: np.ndarray,
    rotation: np.ndarray,
) -> np.ndarray:
    """把 Euler 世界坐标顶点变回网格局部坐标。

    做什么：``p_local = R.T @ (p_world - T)``，写成行向量是
    ``(p_world - T) @ R``。输入可以是 ``(N, 3)`` 或扁平 ``(N*3,)``，
    输出形状、dtype 与输入相同。

    为什么：物理和 ``State.deformable_vertex_q`` 是世界坐标；Studio
    高模 buffer / L2 MLS 吃网格局部，ATOM 再乘实体世界变换。发送前
    变回局部，视口才不会把平移和旋转套两次。只在 Gym 做一次。
    """
    source = np.asarray(positions)
    points = np.reshape(source, (-1, 3)).astype(np.float64, copy=False)
    local = (
        points - np.asarray(origin, dtype=np.float64).reshape(1, 3)
    ) @ np.asarray(rotation, dtype=np.float64)
    return np.reshape(local.astype(source.dtype, copy=False), source.shape)


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
        self._esdf_path = esdf_path
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
        self._render_body_poses: dict[str, tuple[np.ndarray, np.ndarray]] = {}
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
        esdf_poses = _load_esdf_body_initial_poses(self._esdf_path)
        identity_r = np.eye(3, dtype=np.float64)
        zero_t = np.zeros(3, dtype=np.float64)
        self._render_body_poses = {
            name: esdf_poses.get(name, (zero_t, identity_r))
            for name in graphs
        }

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
        冻结在最近一次 ``sync_body_pose`` 注入值。

        第一版耦合比 M=N=1 时，``step_with_coupling`` 每个耦合窗调用
        本方法一次，``n_frames=1``、``dt_macro=物理步长``（0.001），
        于是 ``n_sub=1``：一窗走 1 个 XPBD 子步。

        每子步::

            solver.step(state_in, state_out, None, contacts, dt,
                        rigid_body_mode="external")
            state_out.body_q/qd ← state_in（external 模式 SemiImplicit
                不搬运刚体状态，XPBD 已内置拷贝，此处统一重拷为幂等）
            state_in ↔ state_out

        Args:
            n_frames: 本耦合窗的柔体步数 N（不是一次 env.step 的 frame_skip）。
            dt_macro: 每一步的物理时长（秒），与 MuJoCo timestep 相同。
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
            ch.read_q.numpy()             # 冻结读槽（世界坐标）
            变回网格局部后再 send_deformable_vertices

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
                origin, rotation = self._render_body_poses[body_name]
                self._render_client.send_deformable_vertices(
                    _world_vertices_to_mesh_local(
                        ch.read_q.numpy(), origin, rotation
                    ),
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
