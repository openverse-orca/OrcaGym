"""EulerSoftSim 验收测试（P5 Segment A：ESDF 双文件注入）。

覆盖范围：
  1. 双文件构建（CPU）：XPBD cloth + SemiImplicit box 两套 OrcaEuler
     权威双文件资产，验证 model 字段 / body_map / dt / 双 State。
  2. sync_body_pose 数值（CPU）：真 MuJoCo MjModel/MjData，验证位姿
     注入的坐标变换（xpos/xquat wxyz→xyzw/cvel subtree-COM→body-COM）。
  3. step 编排（CPU）：RecordingSolver mock，验证子步数、external 模式、
     双 State 交换、external 模式 body_q 手动拷贝、body_f 读取槽位。
  4. reset（CPU）：双 State 回初始 + 快照注入。
  5. 错误路径：不支持的 solver.type / XPBD 缺 soft_contact。
  6. 渲染流（P5 改动点 2，mock RenderClient）：构造期烘焙/建通道、
     enable 注册、render_frame payload==读槽 + sequence + 节拍不进
     step、send 失败断流降级、degraded ESDF 无渲染流。
  7. OrcaGymEuler 集成（mock）：_soft 槽隔离（K3/K5）、has_soft_sim、
     step_with_coupling 四步时序编排、render() 挂 render_frame、
     euler_render_target 透传。
  8. GPU 真链路（RTX 4070，skipUnless）：XPBD solver.step 冒烟 + body_f 有限。

资产来源（ORCA_EULER_EXAMPLES 可覆盖）：
  solver_xpbd/08_dual_file_mujoco_euler/{drop_sphere.xml, cloth_drop.esdf}
  solver_semi_implicit/08_dual_file_e2e_validation/{ground.xml, 01_box_full_link.esdf}

运行方式:
    <conda-base>/envs/OrcaFlow_Flow/bin/python -m pytest \
        tests/orca_gym/core/euler/test_euler_soft_sim.py -v
"""

import os
import tempfile
import unittest
from unittest import mock

import numpy as np

from orca_gym.core.euler.euler_soft_sim import EulerSoftSim

# ---------------------------------------------------------------------------
# 资产定位（OrcaEuler examples；跨仓库引用，允许 env 覆盖 + 存在性 skip）
# ---------------------------------------------------------------------------

_EULER_EXAMPLES = os.environ.get(
    "ORCA_EULER_EXAMPLES",
    "/home/orcacaiji/OrcaEngine/OrcaEuler/examples",
)

_XPBD_DIR = os.path.join(
    _EULER_EXAMPLES, "solver_xpbd", "08_dual_file_mujoco_euler"
)
_XPBD_XML = os.path.join(_XPBD_DIR, "drop_sphere.xml")
_XPBD_ESDF = os.path.join(_XPBD_DIR, "cloth_drop.esdf")

_SI_DIR = os.path.join(
    _EULER_EXAMPLES,
    "solver_semi_implicit",
    "08_dual_file_e2e_validation",
)
_SI_XML = os.path.join(_SI_DIR, "ground.xml")
_SI_ESDF = os.path.join(_SI_DIR, "01_box_full_link.esdf")

_ASSETS_OK = all(
    os.path.isfile(p)
    for p in (_XPBD_XML, _XPBD_ESDF, _SI_XML, _SI_ESDF)
)

_HAS_ORCA_EULER = True
try:
    import orca.euler  # noqa: F401
except ImportError:
    _HAS_ORCA_EULER = False

# 渲染流 patch RenderClient 需要真实 import orca.euler.render.client
# （其模块级 import grpc），grpcio 缺失时跳过相关用例。
_HAS_GRPC = True
try:
    import grpc  # noqa: F401
except ImportError:
    _HAS_GRPC = False

_SKIP_EULER = "orca.euler not installed (OrcaFlow_Flow env required)"
_SKIP_ASSETS = (
    f"OrcaEuler dual-file assets not found under {_EULER_EXAMPLES} "
    "(set ORCA_EULER_EXAMPLES to override)"
)


def _gpu_available() -> bool:
    try:
        import orca.flow as flow

        flow.init()
        return any(d.is_gpu for d in flow.get_devices())
    except Exception:  # noqa: BLE001 - 探测失败视为不可用
        return False


def _get_gpu_device() -> str:
    import orca.flow as flow

    flow.init()
    for d in flow.get_devices():
        if d.is_gpu:
            return d.alias
    raise RuntimeError("No GPU device available")


_SKIP_GPU = "GPU device not available (OrcaFlow_Flow interpreter required)"


# ---------------------------------------------------------------------------
# 1. 双文件构建（CPU）
# ---------------------------------------------------------------------------


@unittest.skipUnless(_HAS_ORCA_EULER and _ASSETS_OK, _SKIP_ASSETS)
class TestEulerSoftSimBuildXPBD(unittest.TestCase):
    """XPBD cloth 双文件注入构建（对齐 dual_file_drop 验收断言）。"""

    def _build(self, device: str = "cpu") -> tuple[EulerSoftSim, object]:
        import mujoco

        mj_model = mujoco.MjModel.from_xml_path(_XPBD_XML)
        soft = EulerSoftSim(
            model_xml_path=_XPBD_XML,
            esdf_path=_XPBD_ESDF,
            device=device,
            mj_model=mj_model,
        )
        return soft, mj_model

    def test_model_fields(self):
        """双注入模型字段：刚体 + 柔体 + 弹簧 + shape（dual_file 验收集）。"""
        soft, mj_model = self._build()
        model = soft.model
        self.assertGreaterEqual(model.body_count, 1)
        self.assertGreater(model.particle_count, 0)
        self.assertGreater(model.spring_count, 0)
        self.assertGreaterEqual(model.shape_count, 1)

    def test_body_map_matches_mujoco(self):
        """body_map：drop_sphere（mj_i=1）↔ Euler 唯一刚体。"""
        soft, mj_model = self._build()
        body_map = soft.body_map
        self.assertEqual(len(body_map), 1)
        mj_i, euler_i = next(iter(body_map.items()))
        self.assertEqual(mj_model.body(mj_i).name, "drop_sphere")
        self.assertEqual(euler_i, 0)

    def test_dt_resolved_from_esdf(self):
        """dt = ESDF solver.dt（cloth_drop 声明 0.0005）。"""
        soft, _ = self._build()
        self.assertAlmostEqual(soft.dt, 5.0e-4, places=8)

    def test_solver_is_xpbd(self):
        """XPBD ESDF → SolverXPBD。"""
        import orca.euler as euler

        soft, _ = self._build()
        self.assertIsInstance(soft._solver, euler.SolverXPBD)

    def test_dual_state_initialized(self):
        """双 State 构造 + 初始 body_f 为零。"""
        soft, _ = self._build()
        body_f = soft.body_f_numpy()
        self.assertIsNotNone(body_f)
        self.assertEqual(body_f.shape, (soft.model.body_count, 6))
        np.testing.assert_array_equal(body_f, 0.0)


@unittest.skipUnless(_HAS_ORCA_EULER and _ASSETS_OK, _SKIP_ASSETS)
class TestEulerSoftSimBuildSemiImplicit(unittest.TestCase):
    """SemiImplicit box 双文件注入构建（floor-only MJCF：空 body_map）。"""

    def _build(self) -> EulerSoftSim:
        import mujoco

        mj_model = mujoco.MjModel.from_xml_path(_SI_XML)
        return EulerSoftSim(
            model_xml_path=_SI_XML,
            esdf_path=_SI_ESDF,
            device="cpu",
            mj_model=mj_model,
        )

    def test_model_fields(self):
        """box 双注入：粒子/弹簧/可变形体就绪，刚体列表为空（floor-only）。"""
        soft = self._build()
        model = soft.model
        self.assertEqual(model.body_count, 0)
        self.assertEqual(soft.body_map, {})
        self.assertGreater(model.particle_count, 0)
        self.assertGreater(model.spring_count, 0)
        self.assertGreaterEqual(model.deformable_body_count, 1)

    def test_solver_is_semi_implicit(self):
        """SemiImplicit ESDF → SolverSemiImplicit。"""
        import orca.euler as euler

        soft = self._build()
        self.assertIsInstance(soft._solver, euler.SolverSemiImplicit)

    def test_body_f_numpy_none_when_no_bodies(self):
        """floor-only：body_f_numpy() 返回 None（柔体力无刚体可回流）。"""
        soft = self._build()
        self.assertIsNone(soft.body_f_numpy())


# ---------------------------------------------------------------------------
# 2. sync_body_pose 数值（CPU，真 MuJoCo）
# ---------------------------------------------------------------------------


@unittest.skipUnless(_HAS_ORCA_EULER and _ASSETS_OK, _SKIP_ASSETS)
class TestEulerSoftSimSyncBodyPose(unittest.TestCase):
    """位姿注入数值验证（drop_sphere 自由体：COM=origin 退化直传）。"""

    @classmethod
    def setUpClass(cls):
        import mujoco

        cls.mj_model = mujoco.MjModel.from_xml_path(_XPBD_XML)
        cls.mj_data = mujoco.MjData(cls.mj_model)
        mujoco.mj_forward(cls.mj_model, cls.mj_data)
        # 推进数步让 cvel 非零（球下落）。
        for _ in range(20):
            mujoco.mj_step(cls.mj_model, cls.mj_data)
        cls.soft = EulerSoftSim(
            model_xml_path=_XPBD_XML,
            esdf_path=_XPBD_ESDF,
            device="cpu",
            mj_model=cls.mj_model,
        )
        cls.snapshot = {
            "xpos": np.array(cls.mj_data.xpos),
            "xquat": np.array(cls.mj_data.xquat),
            "xmat": np.array(cls.mj_data.xmat),
            "cvel": np.array(cls.mj_data.cvel),
            "xipos": np.array(cls.mj_data.xipos),
            "subtree_com": np.array(cls.mj_data.subtree_com),
        }

    def test_body_q_pose_injected(self):
        """body_q: xpos 直传 + xquat wxyz→xyzw。"""
        self.soft.sync_body_pose(self.snapshot)
        body_q = self.soft._state_in.body_q.numpy()
        euler_i = self.soft.body_map[1]  # drop_sphere 的 mj_i=1
        np.testing.assert_allclose(
            body_q[euler_i, :3], self.mj_data.xpos[1], atol=1e-6
        )
        w, x, y, z = self.mj_data.xquat[1]
        np.testing.assert_allclose(
            body_q[euler_i, 3:7], [x, y, z, w], atol=1e-6
        )

    def test_body_qd_free_body_passthrough(self):
        """自由体（COM=origin）退化：subtree-COM→body-COM 变换 r_offset=0，
        body_qd == [cvel.lin, cvel.ang] 直传。"""
        self.soft.sync_body_pose(self.snapshot)
        body_qd = self.soft._state_in.body_qd.numpy()
        euler_i = self.soft.body_map[1]
        np.testing.assert_allclose(
            body_qd[euler_i, :3], self.mj_data.cvel[1, 3:], atol=1e-5
        )
        np.testing.assert_allclose(
            body_qd[euler_i, 3:], self.mj_data.cvel[1, :3], atol=1e-5
        )

    def test_both_states_updated(self):
        """双 State 均被注入（state_out 在下一子步交换后成为输入）。"""
        self.soft.sync_body_pose(self.snapshot)
        q_in = self.soft._state_in.body_q.numpy()
        q_out = self.soft._state_out.body_q.numpy()
        np.testing.assert_array_equal(q_in, q_out)
        np.testing.assert_allclose(q_in[0, :3], self.mj_data.xpos[1], atol=1e-6)


# ---------------------------------------------------------------------------
# 3. step 编排（CPU，RecordingSolver mock）
# ---------------------------------------------------------------------------


class _RecordingSolver:
    """mock solver：记录 step 调用，模拟 SemiImplicit external 行为。

    只搬 body_f/particle_f（对齐 SolverSemiImplicit external 分支），
    故意不搬 body_q/body_qd —— 用于验证 EulerSoftSim 的手动拷贝。
    """

    def __init__(self):
        self.calls: list[dict] = []

    def step(self, state_in, state_out, control, contacts, dt, rigid_body_mode):
        self.calls.append(
            {
                "state_in": state_in,
                "state_out": state_out,
                "control": control,
                "contacts": contacts,
                "dt": dt,
                "mode": rigid_body_mode,
            }
        )
        state_out.body_f.assign(state_in.body_f)
        state_out.particle_f.assign(state_in.particle_f)


@unittest.skipUnless(_HAS_ORCA_EULER and _ASSETS_OK, _SKIP_ASSETS)
class TestEulerSoftSimStepOrchestration(unittest.TestCase):
    """step 子步编排：子步数 / external / 交换 / body_q 手动拷贝。"""

    @classmethod
    def setUpClass(cls):
        import mujoco

        mj_model = mujoco.MjModel.from_xml_path(_XPBD_XML)
        cls.soft = EulerSoftSim(
            model_xml_path=_XPBD_XML,
            esdf_path=_XPBD_ESDF,
            device="cpu",
            mj_model=mj_model,
        )
        cls.recorder = _RecordingSolver()
        cls.soft._solver = cls.recorder

    def test_substep_count_and_mode(self):
        """n_frames=2, dt_macro=0.001, dt=0.0005 → 4 子步 external。"""
        n_before = len(self.recorder.calls)
        self.soft.step(n_frames=2, dt_macro=0.001)
        new_calls = self.recorder.calls[n_before:]
        self.assertEqual(len(new_calls), 4)
        for call in new_calls:
            self.assertEqual(call["mode"], "external")
            self.assertAlmostEqual(call["dt"], 5.0e-4, places=8)
            self.assertIsNone(call["control"])
            self.assertIs(call["contacts"], self.soft._contacts)

    def test_state_pingpong_swap(self):
        """双缓冲交换：上一子步 out 是下一子步 in。"""
        n_before = len(self.recorder.calls)
        self.soft.step(n_frames=1, dt_macro=0.0005)
        calls = self.recorder.calls[n_before:]
        self.assertEqual(len(calls), 1)
        # 下一次调用验证交换：新的 in 是上次的 out。
        self.soft.step(n_frames=1, dt_macro=0.0005)
        calls = self.recorder.calls[n_before:]
        self.assertIs(calls[0]["state_out"], calls[1]["state_in"])
        self.assertIs(calls[0]["state_in"], calls[1]["state_out"])

    def test_external_body_q_copied_across_substeps(self):
        """external 模式 body_q 手动拷贝：注入位姿在子步间不丢。

        RecordingSolver 故意不搬 body_q（模拟 SemiImplicit external），
        若 EulerSoftSim 缺手动拷贝，后续子步的 state_in.body_q 会回退
        rest pose。
        """
        # 注入一个显著偏离 rest pose 的位姿。
        pose = np.zeros((self.soft.model.body_count, 7), dtype=np.float32)
        pose[0, :3] = [9.9, 8.8, 7.7]
        pose[0, 6] = 1.0
        import orca.flow as flow

        arr = flow.array(pose, dtype=flow.transform, device="cpu")
        self.soft._state_in.body_q.assign(arr)
        self.soft._state_out.body_q.assign(arr)

        self.soft.step(n_frames=1, dt_macro=0.0005)  # 1 子步
        # 交换后 _state_in 是刚写出的 state；其 body_q 必须保持注入值。
        q_after = self.soft._state_in.body_q.numpy()
        np.testing.assert_allclose(q_after[0, :3], [9.9, 8.8, 7.7], atol=1e-5)

        # 连续 3 子步后再验（拷贝链跨子步保持）。
        self.soft.step(n_frames=3, dt_macro=0.0005)
        q_after = self.soft._state_in.body_q.numpy()
        np.testing.assert_allclose(q_after[0, :3], [9.9, 8.8, 7.7], atol=1e-5)

    def test_body_f_read_slot(self):
        """body_f_numpy 读当前（交换后）state_in.body_f。"""
        wrench = np.array([[1.0, 2.0, 3.0, 0.1, 0.2, 0.3]], dtype=np.float32)
        self.soft._state_in.body_f.assign(wrench)
        np.testing.assert_allclose(
            self.soft.body_f_numpy(), wrench, atol=1e-6
        )

    def test_zero_frames_noop(self):
        """n_frames=0 / 负数：不推进（无 solver 调用）。"""
        n_before = len(self.recorder.calls)
        self.soft.step(n_frames=0, dt_macro=0.001)
        self.soft.step(n_frames=-3, dt_macro=0.001)
        self.assertEqual(len(self.recorder.calls), n_before)


# ---------------------------------------------------------------------------
# 4. reset（CPU）
# ---------------------------------------------------------------------------


@unittest.skipUnless(_HAS_ORCA_EULER and _ASSETS_OK, _SKIP_ASSETS)
class TestEulerSoftSimReset(unittest.TestCase):
    """reset：双 State 回初始 + 可选快照注入。"""

    @classmethod
    def setUpClass(cls):
        import mujoco

        mj_model = mujoco.MjModel.from_xml_path(_XPBD_XML)
        cls.soft = EulerSoftSim(
            model_xml_path=_XPBD_XML,
            esdf_path=_XPBD_ESDF,
            device="cpu",
            mj_model=mj_model,
        )
        cls.initial_q = cls.soft._state_in.particle_q.numpy().copy()
        cls.initial_body_q = cls.soft._state_in.body_q.numpy().copy()

    def test_reset_restores_particle_state(self):
        """改动粒子状态后 reset 回初始位姿。"""
        disturbed = self.initial_q + np.float32(0.5)
        self.soft._state_in.particle_q.assign(
            disturbed.astype(np.float32)
        )
        self.soft.reset()
        np.testing.assert_allclose(
            self.soft._state_in.particle_q.numpy(),
            self.initial_q,
            atol=1e-6,
        )

    def test_reset_with_snapshot_injects_pose(self):
        """reset(snapshot) 后 body_q 为快照位姿（非 rest pose）。"""
        import mujoco

        self.mj_model = mujoco.MjModel.from_xml_path(_XPBD_XML)
        mj_data = mujoco.MjData(self.mj_model)
        mujoco.mj_forward(self.mj_model, mj_data)
        snapshot = {
            "xpos": np.array(mj_data.xpos),
            "xquat": np.array(mj_data.xquat),
            "xmat": np.array(mj_data.xmat),
            "cvel": np.array(mj_data.cvel),
            "xipos": np.array(mj_data.xipos),
            "subtree_com": np.array(mj_data.subtree_com),
        }
        self.soft.reset(snapshot)
        body_q = self.soft._state_in.body_q.numpy()
        np.testing.assert_allclose(
            body_q[0, :3], mj_data.xpos[1], atol=1e-6
        )
        # body_f 清零。
        np.testing.assert_array_equal(self.soft.body_f_numpy(), 0.0)


# ---------------------------------------------------------------------------
# 5. 错误路径（CPU）
# ---------------------------------------------------------------------------


@unittest.skipUnless(_HAS_ORCA_EULER and _ASSETS_OK, _SKIP_ASSETS)
class TestEulerSoftSimErrorPaths(unittest.TestCase):
    """构造期 fail-fast：坏 solver.type / XPBD 缺 soft_contact。"""

    def _make_soft(self, esdf_path: str):
        import mujoco

        mj_model = mujoco.MjModel.from_xml_path(_XPBD_XML)
        return EulerSoftSim(
            model_xml_path=_XPBD_XML,
            esdf_path=esdf_path,
            device="cpu",
            mj_model=mj_model,
        )

    def test_unsupported_solver_type(self):
        """solver.type 非法 → ValueError（fail-fast，不静默降级）。"""
        with open(_XPBD_ESDF, encoding="utf-8") as f:
            content = f.read()
        bad = content.replace('"type": "xpbd"', '"type": "bogus_solver"')
        self.assertNotEqual(bad, content, "替换失败：源 ESDF 格式变化")
        # 写到原 ESDF 同目录：ESDF 的 asset_root/mesh 为相对路径，
        # add_esdf 解析时需能找到 assets/*.obj（相对 ESDF 文件目录）。
        with tempfile.NamedTemporaryFile(
            "w", suffix=".esdf", delete=False, encoding="utf-8",
            dir=os.path.dirname(_XPBD_ESDF),
        ) as f:
            f.write(bad)
            tmp = f.name
        try:
            with self.assertRaises(ValueError) as ctx:
                self._make_soft(tmp)
            self.assertIn("bogus_solver", str(ctx.exception))
        finally:
            os.unlink(tmp)

    def test_xpbd_missing_soft_contact(self):
        """XPBD ESDF 删 soft_contact section → ValueError。"""
        with open(_XPBD_ESDF, encoding="utf-8") as f:
            content = f.read()
        start = content.index('"soft_contact"')
        # section 到下一个顶层键（"globals"）之前。
        end = content.index('"globals"')
        bad = content[:start] + content[end:]
        with tempfile.NamedTemporaryFile(
            "w", suffix=".esdf", delete=False, encoding="utf-8",
            dir=os.path.dirname(_XPBD_ESDF),
        ) as f:
            f.write(bad)
            tmp = f.name
        try:
            with self.assertRaises(ValueError) as ctx:
                self._make_soft(tmp)
            self.assertIn("soft_contact", str(ctx.exception))
        finally:
            os.unlink(tmp)


# ---------------------------------------------------------------------------
# 6. 渲染流（P5 改动点 2；mock RenderClient，CPU）
# ---------------------------------------------------------------------------

_RENDER_TARGET = "127.0.0.1:50451"


@unittest.skipUnless(
    _HAS_ORCA_EULER and _ASSETS_OK and _HAS_GRPC,
    "orca.euler / dual-file assets / grpcio required",
)
class TestEulerSoftSimRenderStream(unittest.TestCase):
    """渲染流三件套：构造期烘焙 → enable 注册 → render_frame 推流。"""

    def _build(self) -> EulerSoftSim:
        import mujoco

        mj_model = mujoco.MjModel.from_xml_path(_SI_XML)
        return EulerSoftSim(
            model_xml_path=_SI_XML,
            esdf_path=_SI_ESDF,
            device="cpu",
            mj_model=mj_model,
        )

    def _channel(self, soft: EulerSoftSim):
        """白盒取 (body_name, channel)：单 body 资产（soft_block）。"""
        self.assertEqual(len(soft._render_graphs), 1)  # noqa: SLF001
        body_name, (_, ch) = next(iter(soft._render_graphs.items()))
        return body_name, ch

    def test_pipeline_baked_at_construction(self):
        """构造期烘焙 bindings + 建 1 个 per-body 通道（未连接）。

        01_box_full_link.esdf 带 render.proxy_mesh → deformable_bindings
        非空 → _render_graphs 有 soft_block 一项；未 enable 时
        render_frame no-op。
        """
        soft = self._build()
        self.assertIsNotNone(soft._render_graphs)  # noqa: SLF001
        body_name, ch = self._channel(soft)
        self.assertEqual(body_name, "soft_block")
        # proxy 盒子 8 顶点（对齐 05 课 driver 输出 proxy_verts=8）。
        self.assertEqual(ch.vertex_count, 8)
        # 未连接：has_render_stream False，render_frame no-op 返回 False。
        self.assertFalse(soft.has_render_stream())
        self.assertFalse(soft.render_frame())

    def test_enable_registers_per_body(self):
        """enable：连接一次 + 逐 body register_deformable_channel。"""
        soft = self._build()
        body_name, ch = self._channel(soft)
        with mock.patch("orca.euler.render.RenderClient") as client_cls:
            soft.enable_render_stream(_RENDER_TARGET, timeout=1.5)
            client_cls.assert_called_once_with(
                _RENDER_TARGET, timeout=1.5
            )
            client = client_cls.return_value
            client.register_deformable_channel.assert_called_once_with(
                name=f"{body_name}_verts",
                body_name=body_name,
                vertex_count=ch.vertex_count,
            )
        self.assertTrue(soft.has_render_stream())
        soft.disable_render_stream()
        self.assertFalse(soft.has_render_stream())
        # disable 幂等。
        soft.disable_render_stream()

    def test_render_frame_payload_and_sequence(self):
        """render_frame：payload==读槽、sequence 单调、body_name 路由。

        CPU 路径：skinning（bind 恒等）→ sync → 读槽。推流两次验证
        sequence 1→2 与读槽逐元素一致。
        """
        soft = self._build()
        body_name, ch = self._channel(soft)
        with mock.patch("orca.euler.render.RenderClient") as client_cls:
            soft.enable_render_stream(_RENDER_TARGET)
            client = client_cls.return_value
            self.assertTrue(soft.render_frame())
            self.assertTrue(soft.render_frame())
            self.assertEqual(client.send_deformable_vertices.call_count, 2)
            for i, call in enumerate(
                client.send_deformable_vertices.call_args_list
            ):
                args, kwargs = call
                payload = args[0] if args else kwargs["positions"]
                np.testing.assert_allclose(
                    payload, ch.read_q.numpy(), atol=1e-6
                )
                seq = (
                    kwargs["sequence"]
                    if "sequence" in kwargs else args[1]
                )
                self.assertEqual(seq, i + 1)
                routed = (
                    kwargs.get("body_name")
                    if "body_name" in kwargs
                    else (args[2] if len(args) > 2 else None)
                )
                self.assertEqual(routed, body_name)
            # 通道 sync 计数 == render_frame 次数（节拍：每渲染帧一次）。
            self.assertEqual(ch.sync_count, 2)

    def test_step_does_not_push_render(self):
        """节拍契约（D4）：step 子步循环不触发 skin/推流。

        solver 换 mock（CPU 无法 launch kernel，对齐 02/05 课
        "solver 步进仅 CUDA" 策略；step 编排本身在 CPU 可跑）。
        """
        soft = self._build()
        soft._solver = mock.MagicMock()  # noqa: SLF001  CPU 无 kernel
        with mock.patch("orca.euler.render.RenderClient") as client_cls:
            soft.enable_render_stream(_RENDER_TARGET)
            client = client_cls.return_value
            soft.step(n_frames=2, dt_macro=0.002)
            self.assertGreaterEqual(soft._solver.step.call_count, 1)
            client.send_deformable_vertices.assert_not_called()
            _, (_, ch) = next(iter(soft._render_graphs.items()))  # noqa: SLF001
            self.assertEqual(ch.sync_count, 0)
            # render_frame 才推流。
            self.assertTrue(soft.render_frame())
            self.assertEqual(client.send_deformable_vertices.call_count, 1)

    def test_send_failure_degrades_and_recovers_semantics(self):
        """send 失败：自动断流（close + has=False）、后续 no-op 不抛。"""
        soft = self._build()
        soft._solver = mock.MagicMock()  # noqa: SLF001  CPU 无 kernel
        with mock.patch("orca.euler.render.RenderClient") as client_cls:
            soft.enable_render_stream(_RENDER_TARGET)
            client = client_cls.return_value
            client.send_deformable_vertices.side_effect = RuntimeError(
                "UpdateChannelData failed: server down"
            )
            self.assertFalse(soft.render_frame())
            client.close.assert_called_once()
            self.assertFalse(soft.has_render_stream())
            # 后续 render_frame 不再抛（降级 no-op）。
            self.assertFalse(soft.render_frame())
            # 物理循环不受影响（step 编排继续跑，按 CFL 拆子步）。
            soft.step(n_frames=1, dt_macro=0.001)
            self.assertGreaterEqual(soft._solver.step.call_count, 1)  # noqa: SLF001

    def test_register_failure_closes_client(self):
        """enable 注册失败：client.close 后异常冒泡（fail-fast）。"""
        soft = self._build()
        with mock.patch("orca.euler.render.RenderClient") as client_cls:
            client = client_cls.return_value
            client.register_deformable_channel.side_effect = RuntimeError(
                "RegisterChannel rejected"
            )
            with self.assertRaises(RuntimeError):
                soft.enable_render_stream(_RENDER_TARGET)
            client.close.assert_called_once()
            self.assertFalse(soft.has_render_stream())

    def test_degraded_esdf_no_render_stream(self):
        """删 render section 的 ESDF：无通道，enable 抛 RuntimeError。"""
        with open(_SI_ESDF, encoding="utf-8") as f:
            content = f.read()
        start = content.index('"render"')
        end = content.index('"skin_binding"')
        bad = content[:start] + content[end:]
        self.assertNotEqual(bad, content, "替换失败：源 ESDF 格式变化")
        # 写到原 ESDF 同目录（相对路径 assets 解析依赖）。
        with tempfile.NamedTemporaryFile(
            "w", suffix=".esdf", delete=False, encoding="utf-8",
            dir=os.path.dirname(_SI_ESDF),
        ) as f:
            f.write(bad)
            tmp = f.name
        try:
            soft = self._build_esdf(tmp)
            self.assertIsNone(soft._render_graphs)  # noqa: SLF001
            self.assertFalse(soft.has_render_stream())
            self.assertFalse(soft.render_frame())
            with self.assertRaises(RuntimeError) as ctx:
                soft.enable_render_stream(_RENDER_TARGET)
            self.assertIn("无可渲染柔体", str(ctx.exception))
        finally:
            os.unlink(tmp)

    def _build_esdf(self, esdf_path: str) -> EulerSoftSim:
        import mujoco

        mj_model = mujoco.MjModel.from_xml_path(_SI_XML)
        return EulerSoftSim(
            model_xml_path=_SI_XML,
            esdf_path=esdf_path,
            device="cpu",
            mj_model=mj_model,
        )


# ---------------------------------------------------------------------------
# 7. OrcaGymEuler 集成（mock，无 orca.euler 依赖）
# ---------------------------------------------------------------------------


class TestOrcaGymEulerSoftSlot(unittest.TestCase):
    """_soft 槽隔离（K3/K5）+ has_soft_sim + step_with_coupling 编排。"""

    def test_soft_attr_blocked(self):
        """访问 gym._soft/soft 抛 AttributeError，dir 不列出（K3/K5）。"""
        from orca_gym.core.euler.orca_gym_euler import OrcaGymEuler

        gym = OrcaGymEuler()
        for name in ("_soft", "soft"):
            with self.assertRaises(AttributeError):
                getattr(gym, name)
            self.assertNotIn(name, dir(gym))

    def test_soft_in_instance_dict(self):
        """_soft 槽存在（初始 None）。"""
        from orca_gym.core.euler.orca_gym_euler import OrcaGymEuler

        gym = OrcaGymEuler()
        self.assertIn("_soft", gym.__dict__)
        self.assertIsNone(gym.__dict__["_soft"])
        self.assertFalse(gym.has_soft_sim())

    def test_step_with_coupling_orchestration(self):
        """四步时序：sim.step → snapshot → soft.sync/step → 力回流。

        用 mock _sim/_soft 验证编排与 COM→origin 力矩变换，不依赖
        orca.euler / GPU。
        """
        from orca_gym.core.euler.orca_gym_euler import OrcaGymEuler

        gym = OrcaGymEuler()

        fake_snapshot = {
            "xpos": np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 1.0]]),
            "xipos": np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 1.2]]),
        }
        # body_map={1: 0}：MuJoCo body 1 ↔ Euler body 0。
        # body_f[0]: F=(3,0,0)，T_com=(0,0,0)；r_com = xipos-xpos = (0,0,0.2)。
        # 期望 T_origin = T_com + cross(r_com, F) = (0, 0.6, 0)。
        # （叉积 (0,0,0.2)×(3,0,0) = (0, +0.6, 0)）
        fake_body_f = np.array(
            [[3.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0] * 6], dtype=np.float64
        )

        sim_mock = mock.MagicMock()
        sim_mock.query_dual_engine_state.return_value = fake_snapshot

        soft_mock = mock.MagicMock()
        soft_mock.body_map = {1: 0}
        soft_mock.body_f_numpy.return_value = fake_body_f

        object.__setattr__(gym, "_sim", sim_mock)
        object.__setattr__(gym, "_soft", soft_mock)

        ctrl = np.zeros(4)
        gym.step_with_coupling(ctrl, n_frames=2, dt=0.001)

        # Step 1: MuJoCo 推进（set_ctrl + step(2)）。
        sim_mock.set_ctrl.assert_called_once_with(ctrl)
        sim_mock.step.assert_called_once_with(2)
        # Step 2/3: 快照 → 注入 → 柔体推进。
        sim_mock.query_dual_engine_state.assert_called_once()
        soft_mock.sync_body_pose.assert_called_once_with(fake_snapshot)
        soft_mock.step.assert_called_once_with(2, 0.001)
        # Step 4: 力回流（COM→origin 力矩变换）。
        sim_mock.apply_body_force.assert_called_once()
        args, _ = sim_mock.apply_body_force.call_args
        body_id, force, torque = args
        self.assertEqual(body_id, 1)
        np.testing.assert_allclose(force, [3.0, 0.0, 0.0], atol=1e-9)
        np.testing.assert_allclose(torque, [0.0, 0.6, 0.0], atol=1e-9)

    def test_step_with_coupling_no_soft_pure_mujoco(self):
        """_soft=None：退化为 set_ctrl + step（P1/P2 行为不变）。"""
        from orca_gym.core.euler.orca_gym_euler import OrcaGymEuler

        gym = OrcaGymEuler()
        sim_mock = mock.MagicMock()
        object.__setattr__(gym, "_sim", sim_mock)
        ctrl = np.zeros(3)
        gym.step_with_coupling(ctrl, n_frames=1, dt=0.002)
        sim_mock.set_ctrl.assert_called_once_with(ctrl)
        sim_mock.step.assert_called_once_with(1)
        sim_mock.query_dual_engine_state.assert_not_called()
        sim_mock.apply_body_force.assert_not_called()

    def test_step_with_coupling_no_body_f(self):
        """body_f=None（floor-only）：跳过力回流但不报错。"""
        from orca_gym.core.euler.orca_gym_euler import OrcaGymEuler

        gym = OrcaGymEuler()
        sim_mock = mock.MagicMock()
        soft_mock = mock.MagicMock()
        soft_mock.body_map = {}
        soft_mock.body_f_numpy.return_value = None
        object.__setattr__(gym, "_sim", sim_mock)
        object.__setattr__(gym, "_soft", soft_mock)
        gym.step_with_coupling(np.zeros(1), n_frames=1, dt=0.001)
        soft_mock.sync_body_pose.assert_called_once()
        soft_mock.step.assert_called_once()
        sim_mock.apply_body_force.assert_not_called()

    def test_reset_coupling_state_resets_soft(self):
        """reset_coupling_state：_soft 存在时 reset(快照)。"""
        from orca_gym.core.euler.orca_gym_euler import OrcaGymEuler

        gym = OrcaGymEuler()
        sim_mock = mock.MagicMock()
        soft_mock = mock.MagicMock()
        object.__setattr__(gym, "_sim", sim_mock)
        object.__setattr__(gym, "_soft", soft_mock)
        gym.reset_coupling_state()
        soft_mock.reset.assert_called_once_with(
            sim_mock.query_dual_engine_state.return_value
        )

    # --- P5 改动点 2：render() 挂 render_frame ---

    def test_render_calls_soft_render_frame(self):
        """render()：_soft 存在时先 soft.render_frame()（推流先于
        studio.render）。"""
        import asyncio

        from orca_gym.core.euler.orca_gym_euler import OrcaGymEuler

        gym = OrcaGymEuler()
        sim_mock = mock.MagicMock()
        sim_mock.query_contact_simple.return_value = []
        soft_mock = mock.MagicMock()
        object.__setattr__(gym, "_sim", sim_mock)
        object.__setattr__(gym, "_soft", soft_mock)

        asyncio.run(gym.render())
        soft_mock.render_frame.assert_called_once()

    def test_render_without_soft_noop_regression(self):
        """render()：_soft=None（纯刚体）行为不变（render_frame 不被调，
        也不抛）。"""
        import asyncio

        from orca_gym.core.euler.orca_gym_euler import OrcaGymEuler

        gym = OrcaGymEuler()
        asyncio.run(gym.render())  # 不抛即通过（既有离线 no-op 语义）

    # --- P5 改动点 2：euler_render_target 透传 ---

    def _init_euler_backend_with_mocks(self, esdf_path, render_target):
        """直接调 _init_euler_backend（patch 掉两个 core 组件类）。"""
        from orca_gym.core.euler.orca_gym_euler import OrcaGymEuler

        gym = OrcaGymEuler()
        opt_mock = mock.MagicMock()
        opt_mock.nworld = 1
        opt_mock.device = "cpu"
        object.__setattr__(gym, "_opt", opt_mock)
        object.__setattr__(gym, "_registry", mock.MagicMock())

        with mock.patch(
            "orca_gym.core.euler.mujoco_sim_core_euler.MuJoCoSimCoreEuler"
        ) as sim_cls, mock.patch(
            "orca_gym.core.euler.euler_soft_sim.EulerSoftSim"
        ) as soft_cls:
            gym._init_euler_backend(  # noqa: SLF001  白盒：透传接线验证
                "x.xml", esdf_path, None, render_target
            )
        return sim_cls, soft_cls

    def test_render_target_passthrough_to_soft(self):
        """esdf + target：EulerSoftSim 构造后 enable_render_stream(target)。"""
        sim_cls, soft_cls = self._init_euler_backend_with_mocks(
            esdf_path="y.esdf", render_target="127.0.0.1:50451"
        )
        soft_cls.assert_called_once_with(
            model_xml_path="x.xml",
            esdf_path="y.esdf",
            device="cpu",
            mj_model=sim_cls.return_value.mj_model,
        )
        soft_cls.return_value.enable_render_stream.assert_called_once_with(
            "127.0.0.1:50451"
        )

    def test_render_target_ignored_without_esdf(self):
        """esdf=None + target：忽略（不构造 _soft、不连接），仅告警。"""
        sim_cls, soft_cls = self._init_euler_backend_with_mocks(
            esdf_path=None, render_target="127.0.0.1:50451"
        )
        soft_cls.assert_not_called()
        soft_cls.return_value.enable_render_stream.assert_not_called()

    def test_no_render_target_no_enable(self):
        """esdf + target=None：构造 _soft 但不连接渲染流（默认降级）。"""
        sim_cls, soft_cls = self._init_euler_backend_with_mocks(
            esdf_path="y.esdf", render_target=None
        )
        soft_cls.assert_called_once()
        soft_cls.return_value.enable_render_stream.assert_not_called()


# ---------------------------------------------------------------------------
# 7. GPU 真链路（skipUnless GPU）
# ---------------------------------------------------------------------------


@unittest.skipUnless(
    _HAS_ORCA_EULER and _ASSETS_OK and _gpu_available(), _SKIP_GPU
)
class TestEulerSoftSimGPU(unittest.TestCase):
    """XPBD 真链路冒烟：sync → step → body_f 有限。"""

    def test_xpbd_step_smoke(self):
        """GPU 上 20 子步推进：粒子有限、body_f 有限、位姿注入生效。"""
        import mujoco

        mj_model = mujoco.MjModel.from_xml_path(_XPBD_XML)
        mj_data = mujoco.MjData(mj_model)
        mujoco.mj_forward(mj_model, mj_data)

        soft = EulerSoftSim(
            model_xml_path=_XPBD_XML,
            esdf_path=_XPBD_ESDF,
            device=_get_gpu_device(),
            mj_model=mj_model,
        )

        def make_snapshot():
            return {
                "xpos": np.array(mj_data.xpos),
                "xquat": np.array(mj_data.xquat),
                "xmat": np.array(mj_data.xmat),
                "cvel": np.array(mj_data.cvel),
                "xipos": np.array(mj_data.xipos),
                "subtree_com": np.array(mj_data.subtree_com),
            }

        # 10 个耦合周期：MuJoCo（CPU host）推进 → 注入 → 柔体推进。
        for _ in range(10):
            for _ in range(10):
                mujoco.mj_step(mj_model, mj_data)
            soft.sync_body_pose(make_snapshot())
            soft.step(n_frames=10, dt_macro=0.001)

            pq = soft._state_in.particle_q.numpy()
            self.assertTrue(np.all(np.isfinite(pq)), "particle_q NaN/Inf")
            body_f = soft.body_f_numpy()
            self.assertIsNotNone(body_f)
            self.assertTrue(np.all(np.isfinite(body_f)), "body_f NaN/Inf")
            # external 模式：注入位姿不被柔体步进改写。
            body_q = soft._state_in.body_q.numpy()
            np.testing.assert_allclose(
                body_q[0, :3], mj_data.xpos[1], atol=1e-5
            )


if __name__ == "__main__":
    unittest.main()
