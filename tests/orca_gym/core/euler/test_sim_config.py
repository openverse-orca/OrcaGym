"""阶段二 Step 2: SimConfig 功能验收测试。

验证 SimConfig 的 typed 接口和委托机制（架构 §5.6, §12.2）。
未绑定时 property 走缓存占位字段；绑定后委托真实 `_mj_model.opt.*`。
`_bind` 仅切换引用不同步缓存值——绑定后 mj_model.opt 保留 XML 原值，
Env 层负责显式重新应用需生效的缓存配置。

运行方式:
    <conda-base>/envs/orca/bin/python tests/run_tests.py --component core/euler
"""

import os
import unittest

import mujoco
import numpy as np

from orca_gym.core.euler.sim_config import SimBackend, SimConfig


# 测试用 XML 模型：单铰链倒立摆（timestep=0.002, integrator=RK4, gravity=0 0 -9.81）
# 使用本仓 fixtures 目录，无外部依赖（见 AGENTS.md 测试独立性要求）
_PENDULUM_XML = os.path.join(
    os.path.dirname(__file__),
    "..", "..", "environment", "euler", "fixtures", "simple_pendulum.xml",
)
_PENDULUM_XML = os.path.abspath(_PENDULUM_XML)


class TestSimConfigSkeleton(unittest.TestCase):
    """SimConfig 结构验收：property 签名、方法存在、docstring 契约。"""

    def test_sim_config_constructable(self):
        """SimConfig() 可无参构造（未绑定时使用缓存默认值）。"""
        config = SimConfig()
        self.assertIsInstance(config, SimConfig)

    def test_sim_config_has_timestep_property(self):
        """timestep 是 property，有 getter 和 setter。"""
        self.assertIsInstance(SimConfig.timestep, property)
        self.assertIsNotNone(SimConfig.timestep.fget)
        self.assertIsNotNone(SimConfig.timestep.fset)

    def test_sim_config_has_integrator_property(self):
        """integrator 是 property，有 getter 和 setter。"""
        self.assertIsInstance(SimConfig.integrator, property)
        self.assertIsNotNone(SimConfig.integrator.fget)
        self.assertIsNotNone(SimConfig.integrator.fset)

    def test_sim_config_has_iterations_property(self):
        """iterations 是 property，有 getter 和 setter。"""
        self.assertIsInstance(SimConfig.iterations, property)
        self.assertIsNotNone(SimConfig.iterations.fget)
        self.assertIsNotNone(SimConfig.iterations.fset)

    def test_sim_config_has_gravity_property(self):
        """gravity 是 property，有 getter 和 setter。"""
        self.assertIsInstance(SimConfig.gravity, property)
        self.assertIsNotNone(SimConfig.gravity.fget)
        self.assertIsNotNone(SimConfig.gravity.fset)

    def test_sim_config_has_load_from_dict(self):
        """load_from_dict 方法存在且可调用。"""
        self.assertTrue(callable(getattr(SimConfig, "load_from_dict", None)))

    def test_sim_config_has_to_dict(self):
        """to_dict 方法存在且可调用。"""
        self.assertTrue(callable(getattr(SimConfig, "to_dict", None)))

    def test_sim_config_has_bind_method(self):
        """_bind 方法存在且可调用（阶段二新增）。"""
        self.assertTrue(callable(getattr(SimConfig, "_bind", None)))

    def test_sim_config_docstring_has_contract(self):
        """docstring 含「使用契约」和「禁止」关键词（K12）。"""
        doc = SimConfig.__doc__ or ""
        self.assertIn("使用契约", doc)
        self.assertIn("禁止", doc)


class TestSimConfigUnboundCache(unittest.TestCase):
    """未绑定时 property 走缓存占位字段（对应阶段二 Step 2 验收标准）。"""

    def test_timestep_round_trip_unbound(self):
        """未绑定时 timestep setter 写入缓存，getter 读缓存。"""
        config = SimConfig()
        config.timestep = 0.005
        self.assertAlmostEqual(config.timestep, 0.005)

    def test_integrator_round_trip_unbound(self):
        config = SimConfig()
        config.integrator = 1
        self.assertEqual(config.integrator, 1)

    def test_iterations_round_trip_unbound(self):
        config = SimConfig()
        config.iterations = 200
        self.assertEqual(config.iterations, 200)

    def test_gravity_round_trip_unbound(self):
        config = SimConfig()
        new_g = np.array([0.0, 0.0, -10.0])
        config.gravity = new_g
        np.testing.assert_array_equal(config.gravity, new_g)

    def test_unbound_defaults(self):
        """未绑定时返回合理默认值（timestep=0.002, gravity=[0,0,-9.81]）。"""
        config = SimConfig()
        self.assertAlmostEqual(config.timestep, 0.002)
        self.assertEqual(config.integrator, 0)
        self.assertEqual(config.iterations, 100)
        np.testing.assert_array_equal(config.gravity, np.array([0.0, 0.0, -9.81]))

    def test_load_from_dict_sets_values_unbound(self):
        config = SimConfig()
        config.load_from_dict({"timestep": 0.001, "iterations": 50})
        self.assertAlmostEqual(config.timestep, 0.001)
        self.assertEqual(config.iterations, 50)

    def test_to_dict_returns_all_keys_unbound(self):
        config = SimConfig()
        d = config.to_dict()
        self.assertIn("timestep", d)
        self.assertIn("integrator", d)
        self.assertIn("iterations", d)
        self.assertIn("gravity", d)


class TestSimConfigBoundDelegation(unittest.TestCase):
    """绑定后 property 委托真实 _mj_model.opt.*（对应阶段二 Step 2 验收标准）。

    _bind 仅切换引用不同步缓存值——绑定后 mj_model.opt 保留 XML 原值，
    Env 层负责显式重新应用需生效的缓存配置（如 timestep）。
    """

    def setUp(self):
        """每个测试前加载真实 mjModel。"""
        self.mj_model = mujoco.MjModel.from_xml_path(_PENDULUM_XML)

    def test_bind_preserves_xml_opt_values(self):
        """_bind 后 mj_model.opt 保留 XML 原值（不同步缓存默认值）。"""
        config = SimConfig()  # 未绑定，缓存有默认值（integrator=0）
        # 绑定前 mj_model.opt 保持 XML 原值
        self.assertAlmostEqual(self.mj_model.opt.timestep, 0.002)

        config._bind(self.mj_model)

        # 绑定后 mj_model.opt 仍为 XML 原值，未被缓存默认值覆盖
        self.assertAlmostEqual(self.mj_model.opt.timestep, 0.002)
        self.assertEqual(int(self.mj_model.opt.integrator), 1)  # RK4 from XML

    def test_bound_getter_reads_mjmodel_opt(self):
        """绑定后 getter 读取 mj_model.opt.*（XML 原值）。"""
        config = SimConfig()
        config._bind(self.mj_model)

        # pendulum.xml: timestep=0.002, integrator=RK4(1), gravity=[0,0,-9.81]
        self.assertAlmostEqual(config.timestep, 0.002)
        self.assertEqual(int(config.integrator), 1)

    def test_bound_setter_writes_mjmodel_opt(self):
        """绑定后 setter 写入 mj_model.opt.*。"""
        config = SimConfig()
        config._bind(self.mj_model)

        config.timestep = 0.01
        self.assertAlmostEqual(self.mj_model.opt.timestep, 0.01)
        self.assertAlmostEqual(config.timestep, 0.01)

        config.iterations = 300
        self.assertEqual(int(self.mj_model.opt.iterations), 300)

    def test_bound_gravity_round_trip(self):
        """绑定后 gravity setter/getter 委托 mj_model.opt.gravity。"""
        config = SimConfig()
        config._bind(self.mj_model)

        new_g = np.array([0.0, 0.0, -5.0])
        config.gravity = new_g
        np.testing.assert_array_almost_equal(
            np.array(self.mj_model.opt.gravity), new_g
        )
        np.testing.assert_array_almost_equal(config.gravity, new_g)

    def test_bound_load_from_dict(self):
        """绑定后 load_from_dict 委托 mj_model.opt.*。"""
        config = SimConfig()
        config._bind(self.mj_model)

        config.load_from_dict({"timestep": 0.001, "iterations": 50})
        self.assertAlmostEqual(self.mj_model.opt.timestep, 0.001)
        self.assertEqual(int(self.mj_model.opt.iterations), 50)

    def test_bound_to_dict(self):
        """绑定后 to_dict 读取 mj_model.opt.*。"""
        config = SimConfig()
        config._bind(self.mj_model)

        d = config.to_dict()
        self.assertAlmostEqual(d["timestep"], 0.002)
        self.assertIn("integrator", d)
        self.assertIn("iterations", d)
        self.assertIn("gravity", d)

    def test_env_reapplies_cached_timestep_after_bind(self):
        """模拟 Env 层流程：未绑定设 timestep → bind → 重新应用 timestep。"""
        config = SimConfig()  # 未绑定
        config.timestep = 0.005  # Env 在 init_simulation 前设置

        # init_simulation 后绑定
        config._bind(self.mj_model)
        # 绑定后 getter 读 XML 原值（0.002），缓存值未自动生效
        self.assertAlmostEqual(config.timestep, 0.002)

        # Env 层显式重新应用（Step 6 模式）
        config.timestep = 0.005
        self.assertAlmostEqual(self.mj_model.opt.timestep, 0.005)
        self.assertAlmostEqual(config.timestep, 0.005)

    def test_construct_with_mj_model_directly(self):
        """SimConfig(mj_model) 直接绑定，property 立即委托。"""
        config = SimConfig(self.mj_model)
        self.assertAlmostEqual(config.timestep, 0.002)
        self.assertEqual(int(config.integrator), 1)


class TestSimConfigBackend(unittest.TestCase):
    """SimBackend 枚举与 backend/device/nworld 字段（Phase A 验收）。"""

    def test_backend_default_mujoco(self):
        """默认后端为 MUJOCO（对齐 design §2.1 二选一默认项）。"""
        self.assertIs(SimConfig().backend, SimBackend.MUJOCO)

    def test_backend_accepts_str(self):
        """backend setter 接受字符串 "euler" 并规范化为枚举。"""
        config = SimConfig()
        config.backend = "euler"
        self.assertIs(config.backend, SimBackend.EULER)

    def test_backend_accepts_enum(self):
        """backend setter 接受 SimBackend 枚举。"""
        config = SimConfig()
        config.backend = SimBackend.EULER
        self.assertIs(config.backend, SimBackend.EULER)

    def test_device_nworld_fields(self):
        """默认 device == 'cuda'、nworld == 1（对齐 design §4.3）。"""
        config = SimConfig()
        self.assertEqual(config.device, "cuda")
        self.assertEqual(config.nworld, 1)

    def test_backend_device_nworld_round_trip(self):
        """backend/device/nworld getter/setter 直存往返。"""
        config = SimConfig()
        config.backend = SimBackend.EULER
        config.device = "cuda:0"
        config.nworld = 2
        self.assertIs(config.backend, SimBackend.EULER)
        self.assertEqual(config.device, "cuda:0")
        self.assertEqual(config.nworld, 2)


class TestSimConfigEulerGuard(unittest.TestCase):
    """Euler 后端 init_simulation 后只读守卫（Phase A A3 验收）。"""

    def setUp(self):
        """每个测试前加载真实 mjModel。"""
        self.mj_model = mujoco.MjModel.from_xml_path(_PENDULUM_XML)

    def test_timestep_setter_guard_euler(self):
        """Euler 后端绑定后 timestep setter 抛 RuntimeError。"""
        config = SimConfig()
        config.backend = SimBackend.EULER
        config._bind(self.mj_model)
        with self.assertRaises(RuntimeError):
            config.timestep = 0.001

    def test_integrator_setter_guard_euler(self):
        """Euler 后端绑定后 integrator setter 抛 RuntimeError。"""
        config = SimConfig()
        config.backend = SimBackend.EULER
        config._bind(self.mj_model)
        with self.assertRaises(RuntimeError):
            config.integrator = 0

    def test_iterations_setter_guard_euler(self):
        """Euler 后端绑定后 iterations setter 抛 RuntimeError。"""
        config = SimConfig()
        config.backend = SimBackend.EULER
        config._bind(self.mj_model)
        with self.assertRaises(RuntimeError):
            config.iterations = 50

    def test_gravity_setter_guard_euler(self):
        """Euler 后端绑定后 gravity setter 抛 RuntimeError。"""
        config = SimConfig()
        config.backend = SimBackend.EULER
        config._bind(self.mj_model)
        with self.assertRaises(RuntimeError):
            config.gravity = np.array([0.0, 0.0, -5.0])

    def test_unbound_euler_no_guard(self):
        """Euler 后端未绑定时 setter 不抛（允许 init_simulation 前设置）。"""
        config = SimConfig()
        config.backend = SimBackend.EULER
        config.timestep = 0.001  # 不应抛
        self.assertAlmostEqual(config.timestep, 0.001)

    def test_mujoco_backend_no_guard(self):
        """MUJOCO 后端绑定后 setter 仍可写（不回归既有委托行为）。"""
        config = SimConfig()
        config._bind(self.mj_model)
        config.timestep = 0.01  # 不应抛
        self.assertAlmostEqual(config.timestep, 0.01)


if __name__ == "__main__":
    unittest.main()
