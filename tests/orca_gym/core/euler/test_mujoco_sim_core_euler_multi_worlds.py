"""P2 轨道 1 单元测试：MuJoCoSimCoreEulerMultiWorlds 编排层。

覆盖实施指导（orca_gym_euler_multi_world_implementation_guide.md）：

Mock 部分（Phase A 验收，无需 GPU）:
- 构造断言：nworld<=1 抛 ValueError（消息含单世界类指引）。
- render_config 非 None 抛 NotImplementedError（P3 占位）。
- 类型级 API 面：4 驱动方法 + solver/mj_model/nworld/render_bridge property
  存在；CPU 风格 API（set_ctrl/query_*/host/flush 等）不存在（对齐 C4.2）。
- 容量默认：未传 solver_kwargs 时 njmax>=2000 / nconmax>=500；显式传参
  以用户值为准（决策 D2）。
- 未初始化守卫：4 个驱动方法抛 RuntimeError。

GPU 部分（Phase C 验收，见 TestMuJoCoSimCoreEulerMultiWorldsGPU）:
- broadcast 一致性（atol=0）、H2D/D2H 形状契约、mj_model 绑定、
  nworld=64 冒烟、Facade 集成。

运行方式（GPU 用例需白名单解释器直调，勿加 shell 管道）:
    <conda-base>/envs/OrcaFlow_Flow/bin/python -m unittest \
        tests.orca_gym.core.euler.test_mujoco_sim_core_euler_multi_worlds -v
"""

import os
import unittest
from unittest import mock

import numpy as np

_FIXTURES_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "..", "environment", "euler", "fixtures",
)
_PENDULUM_XML = os.path.join(_FIXTURES_DIR, "simple_pendulum.xml")

_HAS_ORCA_EULER = True
try:
    import orca.euler  # noqa: F401
except ImportError:
    _HAS_ORCA_EULER = False

_SKIP_EULER = "orca.euler not installed (OrcaFlow_Flow env required)"

# CPU 风格 API 名单（单世界 41 委托方法的代表集 + host 缓冲系），
# 多世界编排层必须类型级不存在（对齐 design C4.2 / 适配设计 §10.1 判据 4）。
_CPU_STYLE_APIS = (
    "set_ctrl",
    "set_qpos_qvel",
    "query_joint_qpos",
    "query_joint_qvel",
    "query_joint_qacc",
    "query_sensor_data",
    "query_actuator_torques",
    "query_contact_simple",
    "mj_jacBody",
    "apply_body_force",
    "set_mocap_pos_and_quat",
    "sync_to_view",
    "flush_batch",
    "sync_batch_to_numpy",
    "host",
    "sync_to_host",
    "notify_model_changed",
)

_DRIVE_APIS = (
    "step",
    "forward",
    "reset",
    "invalidate_capture",
)

_PROPERTY_APIS = (
    "solver",
    "mj_model",
    "nworld",
    "render_bridge",
)


@unittest.skipUnless(_HAS_ORCA_EULER, _SKIP_EULER)
class TestMuJoCoSimCoreEulerMultiWorldsMock(unittest.TestCase):
    """Phase A 验收：mock 单测（构造断言 / API 面 / 容量默认）。"""

    def _make_sim(self):
        from orca_gym.core.euler import MuJoCoSimCoreEulerMultiWorlds

        return MuJoCoSimCoreEulerMultiWorlds()

    # ---- 构造断言 ----

    def test_nworld_le_1_rejected(self):
        """nworld<=1 抛 ValueError，消息指引单世界类。"""
        sim = self._make_sim()
        for bad in (1, 0, -3):
            with self.assertRaises(ValueError) as ctx:
                sim.init_simulation(_PENDULUM_XML, nworld=bad)
            self.assertIn("MuJoCoSimCoreEuler", str(ctx.exception))

    def test_render_config_rejected(self):
        """render_config 非 None 抛 NotImplementedError（P3 占位）。"""
        sim = self._make_sim()
        with self.assertRaises(NotImplementedError):
            sim.init_simulation(
                _PENDULUM_XML, nworld=2, render_config=object()
            )

    # ---- 类型级 API 面 ----

    def test_api_surface_positive(self):
        """4 驱动方法 + 4 property 存在。"""
        sim = self._make_sim()
        for attr in _DRIVE_APIS + _PROPERTY_APIS:
            self.assertTrue(hasattr(sim, attr), f"missing public API: {attr}")

    def test_api_surface_no_cpu_style(self):
        """CPU 风格 API 类型级不存在（design C4.2 范式隔离）。"""
        sim = self._make_sim()
        for attr in _CPU_STYLE_APIS:
            self.assertFalse(
                hasattr(sim, attr), f"CPU-style API leaked: {attr}"
            )

    def test_uninitialized_properties_none(self):
        """未初始化时 property 返回 None（不抛异常）。"""
        sim = self._make_sim()
        self.assertIsNone(sim.solver)
        self.assertIsNone(sim.mj_model)
        self.assertIsNone(sim.nworld)
        self.assertIsNone(sim.render_bridge)

    def test_uninitialized_drive_methods_raise(self):
        """未初始化时 4 个驱动方法抛 RuntimeError。"""
        sim = self._make_sim()
        with self.assertRaises(RuntimeError):
            sim.step(1)
        with self.assertRaises(RuntimeError):
            sim.forward()
        with self.assertRaises(RuntimeError):
            sim.reset()
        with self.assertRaises(RuntimeError):
            sim.invalidate_capture()

    # ---- 容量默认（决策 D2）----

    def test_capacity_defaults(self):
        """未传 solver_kwargs 时 njmax>=2000 / nconmax>=500。"""
        sim = self._make_sim()
        with mock.patch("orca.euler.SolverMujocoMultiWorld") as factory:
            sim.init_simulation(_PENDULUM_XML, device="cuda:0", nworld=2)
        kwargs = factory.call_args.kwargs
        self.assertGreaterEqual(kwargs["njmax"], 2000)
        self.assertGreaterEqual(kwargs["nconmax"], 500)
        self.assertEqual(kwargs["nworld"], 2)

    def test_capacity_explicit_override(self):
        """显式传 njmax/nconmax 时以用户值为准。"""
        sim = self._make_sim()
        with mock.patch("orca.euler.SolverMujocoMultiWorld") as factory:
            sim.init_simulation(
                _PENDULUM_XML, device="cuda:0", nworld=2, njmax=300, nconmax=64
            )
        kwargs = factory.call_args.kwargs
        self.assertEqual(kwargs["njmax"], 300)
        self.assertEqual(kwargs["nconmax"], 64)

    def test_model_prepared_before_solver(self):
        """solver 收到的 source 是 prepare_host_model 的产物（host MjModel）。"""
        import mujoco

        sim = self._make_sim()
        with mock.patch("orca.euler.SolverMujocoMultiWorld") as factory:
            sim.init_simulation(
                _PENDULUM_XML, device="cuda:0", nworld=3, timestep=0.001
            )
        model = factory.call_args.kwargs["source"]
        self.assertIsInstance(model, mujoco.MjModel)
        self.assertAlmostEqual(model.opt.timestep, 0.001)
        # timestep 经 prepare_host_model 下发（Euler 后端初始化后只读）
        self.assertEqual(sim.nworld, 3)


def _gpu_available() -> bool:
    try:
        import orca.flow as flow

        flow.init()
        return any(d.is_gpu for d in flow.get_devices())
    except Exception:
        return False


def _get_gpu_device() -> str:
    import orca.flow as flow

    flow.init()
    for d in flow.get_devices():
        if d.is_gpu:
            return d.alias
    raise RuntimeError("No GPU device available")


_SKIP_GPU = "GPU device not available (whitelisted OrcaFlow_Flow interpreter required)"


@unittest.skipUnless(_HAS_ORCA_EULER and _gpu_available(), _SKIP_GPU)
class TestMuJoCoSimCoreEulerMultiWorldsGPU(unittest.TestCase):
    """Phase C 验收：GPU 集成（广播一致性 / H2D-D2H / 绑定 / 冒烟）。"""

    def _make_sim(self, nworld: int, xml: str = _PENDULUM_XML):
        from orca_gym.core.euler import MuJoCoSimCoreEulerMultiWorlds

        sim = MuJoCoSimCoreEulerMultiWorlds()
        sim.init_simulation(xml, device=_get_gpu_device(), nworld=nworld)
        return sim

    def _broadcast_qpos(self, sim, value: float) -> None:
        nq = sim.solver.mjf_model.nq
        base = np.zeros(nq, dtype=np.float64)
        base[0] = value
        sim.solver.mjf_data.qpos.assign(
            np.broadcast_to(base, (sim.nworld, nq)).copy()
        )

    def test_opt_registry_bind(self):
        """判据 5：opt._bind / registry._bind 到 sim.mj_model 成功。"""
        import mujoco

        sim = self._make_sim(2)
        self.assertIsInstance(sim.mj_model, mujoco.MjModel)
        # 绑定语义冒烟：元数据可解析（消费方模式）
        self.assertEqual(sim.mj_model.nq, sim.solver.mjf_data.qpos.shape[1])

    def test_broadcast_consistency_atol0(self):
        """判据 7：nworld=8 广播一致，step(50) 逐位一致（atol=0）。"""
        sim = self._make_sim(8)
        self._broadcast_qpos(sim, 0.3)
        sim.step(50)
        qpos = sim.solver.mjf_data.qpos.numpy()
        np.testing.assert_array_equal(qpos, np.broadcast_to(qpos[0], qpos.shape))

    def test_h2d_write_then_step(self):
        """判据 2：qpos.assign((nworld, nq)) 一次 H2D，step 后状态演化正确。"""
        sim = self._make_sim(4)
        self._broadcast_qpos(sim, 0.3)
        qpos_before = sim.solver.mjf_data.qpos.numpy().copy()
        sim.step(10)
        qpos_after = sim.solver.mjf_data.qpos.numpy()
        self.assertFalse(np.allclose(qpos_after, qpos_before))
        # 4 世界同输入同演化（广播一致性附带验证）
        np.testing.assert_array_equal(
            qpos_after, np.broadcast_to(qpos_after[0], qpos_after.shape)
        )

    def test_d2h_shape_contract(self):
        """判据 3：xpos.numpy() 形状 (nworld, nbody, 3)。"""
        sim = self._make_sim(4)
        sim.forward()
        xpos = sim.solver.mjf_data.xpos.numpy()
        self.assertEqual(xpos.shape, (4, sim.mj_model.nbody, 3))

    def test_nworld64_smoke(self):
        """判据 2（64 世界）：构造 + step(10) 冒烟（不要求逐位）。"""
        sim = self._make_sim(64)
        self.assertEqual(sim.nworld, 64)
        sim.step(10)
        qpos = sim.solver.mjf_data.qpos.numpy()
        self.assertTrue(np.all(np.isfinite(qpos)))

    def test_mjf_jac_direct_call(self):
        """判据 4：mjf.jac 经 sim.solver 直调跑通（不封装）。"""
        import orca.flow as flow
        import orca.mujoco_flow as mujoco_flow

        sim = self._make_sim(2)
        self._broadcast_qpos(sim, 0.3)
        sim.forward()
        m = sim.solver.mjf_model
        d = sim.solver.mjf_data
        with flow.ScopedDevice(sim.solver.device):
            jacp = flow.zeros((d.nworld, 3, m.nv), dtype=float)
            jacr = flow.zeros((d.nworld, 3, m.nv), dtype=float)
            point = flow.array(
                np.zeros((d.nworld, 3), dtype=np.float32), dtype=flow.vec3
            )
            body = flow.array(
                np.zeros(d.nworld, dtype=np.int32), dtype=int
            )
            mujoco_flow.jac(m, d, jacp=jacp, jacr=jacr, point=point, body=body)
        jacp_np = jacp.numpy()
        self.assertEqual(jacp_np.shape, (2, 3, m.nv))
        self.assertTrue(np.all(np.isfinite(jacp_np)))

    def test_facade_integration(self):
        """判据 1：SimConfig(nworld=4, EULER) 经 OrcaGymEuler.init_simulation 构造成功。"""
        import asyncio

        from orca_gym.core.euler.orca_gym_euler import OrcaGymEuler
        from orca_gym.core.euler.sim_config import SimBackend

        gym = OrcaGymEuler()
        opt = object.__getattribute__(gym, "_opt")
        opt.backend = SimBackend.EULER
        opt.device = _get_gpu_device()
        opt.nworld = 4

        asyncio.run(
            gym.init_simulation(_PENDULUM_XML, esdf_path=None)
        )
        sim = object.__getattribute__(gym, "_sim")
        self.assertIsInstance(
            sim,
            __import__(
                "orca_gym.core.euler.mujoco_sim_core_euler_multi_worlds",
                fromlist=["MuJoCoSimCoreEulerMultiWorlds"],
            ).MuJoCoSimCoreEulerMultiWorlds,
        )
        # DataView 边界（决策 D5）：多世界跳过 OrcaGymModel 构建与首同步
        self.assertIsNone(object.__getattribute__(gym, "_orca_model"))
        self.assertTrue(object.__getattribute__(gym, "_multi_world"))
        # 步进冒烟
        sim.step(5)
        self.assertTrue(
            np.all(np.isfinite(sim.solver.mjf_data.qpos.numpy()))
        )


if __name__ == "__main__":
    unittest.main()
