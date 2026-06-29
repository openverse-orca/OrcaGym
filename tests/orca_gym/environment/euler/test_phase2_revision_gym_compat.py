"""阶段二变更修订 — Step 3: Gym 层与子组件填充兼容性验证。

验证 Gym 层 K3/K5 隔离机制保留、_bind 延迟绑定、委托链路完整。

对齐文档: docs/design/development/orca_gym_euler_phase2_revision_development.md Step 3
"""

import asyncio
import os
import unittest

import numpy as np

from orca_gym.core.euler.orca_gym_euler import OrcaGymEuler
from orca_gym.core.euler.orca_gym_data_view import OrcaGymDataView
from orca_gym.core.euler.sim_config import SimConfig


_PENDULUM_XML = os.path.join(
    os.path.dirname(__file__),
    "..", "..", "..", "..", "..",
    "OrcaPlayground", "envs", "euler", "scenes", "simple_pendulum.xml",
)
_PENDULUM_XML = os.path.abspath(_PENDULUM_XML)


def _make_gym():
    """构造离线模式 OrcaGymEuler 并完成 init_simulation。"""
    gym = OrcaGymEuler()   # 无参构造（stub=None 默认）
    asyncio.run(gym.init_simulation(_PENDULUM_XML))
    return gym


class TestPhase2GymK3K5IsolationRetained(unittest.TestCase):
    """Gym 层 K3/K5 隔离机制保留（未被骨架迁移删除）。"""

    def test_gym_has_blocked_attrs(self):
        """OrcaGymEuler 类定义 _BLOCKED_ATTRS。"""
        self.assertIn("_BLOCKED_ATTRS", vars(OrcaGymEuler))

    def test_gym_has_getattribute(self):
        """OrcaGymEuler 类定义 __getattribute__（拦截外部访问）。"""
        self.assertIn("__getattribute__", vars(OrcaGymEuler))

    def test_gym_blocked_attrs_contains_mjdata_mjmodel(self):
        """_BLOCKED_ATTRS 含 _mjData/_mjModel（K3 L3 引擎内部）。"""
        blocked = OrcaGymEuler._BLOCKED_ATTRS
        self.assertIn("_mjData", blocked)
        self.assertIn("_mjModel", blocked)

    def test_gym_blocked_attrs_contains_subcomponents(self):
        """_BLOCKED_ATTRS 含 _sim/_studio/_registry/_opt/_view/_euler（K5 子组件）。"""
        blocked = OrcaGymEuler._BLOCKED_ATTRS
        for name in ["_sim", "_studio", "_registry", "_opt", "_view", "_euler"]:
            with self.subTest(attr=name):
                self.assertIn(name, blocked)

    def test_gym_external_access_blocked_with_guidance(self):
        """Gym 外部访问 _mjData 抛 AttributeError 含引导消息（K3）。"""
        gym = _make_gym()
        with self.assertRaises(AttributeError) as ctx:
            _ = gym._mjData
        # Gym 层 __getattribute__ 提供引导消息（与 Env 层 M0 原生 AttributeError 不同）
        self.assertIn("公共", str(ctx.exception))

    def test_gym_external_access_sim_blocked(self):
        """Gym 外部访问 _sim 抛 AttributeError（K5）。"""
        gym = _make_gym()
        with self.assertRaises(AttributeError):
            _ = gym._sim

    def test_gym_external_access_studio_blocked(self):
        """Gym 外部访问 _studio 抛 AttributeError（K5）。"""
        gym = _make_gym()
        with self.assertRaises(AttributeError):
            _ = gym._studio

    def test_gym_external_access_opt_blocked(self):
        """Gym 外部访问 _opt 抛 AttributeError（K5）。"""
        gym = _make_gym()
        with self.assertRaises(AttributeError):
            _ = gym._opt


class TestPhase2GymBindDeferredBinding(unittest.TestCase):
    """Gym _bind 延迟绑定模式。"""

    def test_sim_config_bound_after_init(self):
        """init_simulation 后 SimConfig._mj_model 已绑定。"""
        gym = _make_gym()
        # 通过 sim_config.timestep setter 不抛 RuntimeError 验证已绑定
        gym.sim_config.timestep = 0.003
        self.assertAlmostEqual(gym.sim_config.timestep, 0.003)

    def test_model_registry_bound_after_init(self):
        """init_simulation 后 ModelRegistry 已绑定，model property 返回 OrcaGymModel。"""
        gym = _make_gym()
        model = gym.model
        self.assertEqual(model.nq, 1)   # pendulum nq=1
        self.assertEqual(model.nv, 1)
        self.assertEqual(model.nu, 1)

    def test_sim_config_returns_simconfig_instance(self):
        """sim_config property 返回 SimConfig 实例。"""
        gym = _make_gym()
        self.assertIsInstance(gym.sim_config, SimConfig)


class TestPhase2GymDelegationChain(unittest.TestCase):
    """委托链路 Env → Gym → SimCore 完整。"""

    def test_gym_step_with_coupling_works(self):
        """gym.step_with_coupling 步进后 time 增加。"""
        gym = _make_gym()
        gym.sync_to_view()
        time_before = float(gym.data.time)
        gym.step_with_coupling(np.array([0.0]), 5, 0.002 * 4)
        gym.sync_to_view()
        time_after = float(gym.data.time)
        self.assertAlmostEqual(time_after - time_before, 5 * 0.002, places=5)

    def test_gym_set_qpos_qvel_writes_state(self):
        """gym.set_qpos_qvel 写入 qpos/qvel。"""
        gym = _make_gym()
        gym.set_qpos_qvel(np.array([0.4]), np.array([0.2]))
        gym.mj_forward()
        gym.sync_to_view()
        self.assertAlmostEqual(float(gym.data.qpos[0]), 0.4)
        self.assertAlmostEqual(float(gym.data.qvel[0]), 0.2)

    def test_gym_sync_to_view_populates_dataview(self):
        """gym.sync_to_view 后 DataView 基本字段已填充。"""
        gym = _make_gym()
        gym.sync_to_view()
        self.assertEqual(gym.data.qpos.shape, (1,))
        self.assertEqual(gym.data.qvel.shape, (1,))

    def test_gym_data_returns_dataview(self):
        """gym.data 返回 OrcaGymDataView 实例。"""
        gym = _make_gym()
        self.assertIsInstance(gym.data, OrcaGymDataView)

    def test_gym_mj_step_advances_time(self):
        """gym.mj_step(1) + sync_to_view 后 data.time > 0。"""
        gym = _make_gym()
        gym.mj_step(1)
        gym.sync_to_view()
        self.assertGreater(float(gym.data.time), 0.0)


class TestPhase2GymHasEulerFalse(unittest.TestCase):
    """K8: has_euler() 恒返回 False（骨架阶段无 Euler）。"""

    def test_has_euler_returns_false(self):
        gym = _make_gym()
        self.assertFalse(gym.has_euler())

    def test_has_euler_callable(self):
        """has_euler 方法可调用。"""
        self.assertTrue(callable(getattr(OrcaGymEuler, "has_euler", None)))


class TestPhase2GymK9StudioBridgeAccess(unittest.TestCase):
    """K9: studio_bridge 是方法（非 property），gym.studio 抛 AttributeError。"""

    def test_studio_bridge_is_method(self):
        """studio_bridge 是方法（callable），非 property。"""
        self.assertTrue(callable(getattr(OrcaGymEuler, "studio_bridge", None)))

    def test_gym_studio_blocked(self):
        """gym.studio 被 __getattribute__ 拦截（K9）。"""
        gym = _make_gym()
        with self.assertRaises(AttributeError):
            _ = gym.studio

    def test_studio_bridge_returns_bridge(self):
        """studio_bridge() 返回 OrcaStudioBridge 实例。"""
        gym = _make_gym()
        bridge = gym.studio_bridge()
        self.assertIsNotNone(bridge)


if __name__ == "__main__":
    unittest.main()
