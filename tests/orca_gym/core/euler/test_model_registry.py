"""阶段二 Step 3: ModelRegistry 功能验收测试。

验证 ModelRegistry 的 build_orca_gym_model 真实构建（架构 §5.5, §12.2）。
`build_orca_gym_data` 和扩展查询方法仍 raise NotImplementedError（留待完整 P4）。

运行方式:
    <conda-base>/envs/orca/bin/python tests/run_tests.py --component core/euler
"""

import os
import unittest

import mujoco

from orca_gym.core.euler.model_registry import ModelRegistry
from orca_gym.core.orca_gym_model import OrcaGymModel


# 测试用 XML 模型：单铰链倒立摆（nq=1, nv=1, nu=1, nbody=2, njnt=1, nsite=1, ngeom=1）
_PENDULUM_XML = os.path.join(
    os.path.dirname(__file__),
    "..", "..", "..", "..", "..",
    "OrcaPlayground", "envs", "euler", "scenes", "simple_pendulum.xml",
)
_PENDULUM_XML = os.path.abspath(_PENDULUM_XML)


class TestModelRegistrySkeleton(unittest.TestCase):
    """ModelRegistry 结构验收：类结构、方法签名、docstring 契约。"""

    def test_registry_constructable(self):
        """ModelRegistry() 可无参构造。"""
        registry = ModelRegistry()
        self.assertIsInstance(registry, ModelRegistry)

    def test_registry_has_build_methods(self):
        """build_orca_gym_model / build_orca_gym_data 方法存在。"""
        self.assertTrue(callable(getattr(ModelRegistry, "build_orca_gym_model", None)))
        self.assertTrue(callable(getattr(ModelRegistry, "build_orca_gym_data", None)))

    def test_registry_has_bind_method(self):
        """_bind 方法存在且可调用（阶段二新增）。"""
        self.assertTrue(callable(getattr(ModelRegistry, "_bind", None)))

    def test_registry_has_query_methods(self):
        """body_subtree_mass / equality_data_width / equality_object_ids 方法存在。"""
        self.assertTrue(callable(getattr(ModelRegistry, "body_subtree_mass", None)))
        self.assertTrue(callable(getattr(ModelRegistry, "equality_data_width", None)))
        self.assertTrue(callable(getattr(ModelRegistry, "equality_object_ids", None)))

    def test_registry_docstring_has_contract(self):
        """docstring 含「使用契约」和「禁止」关键词（K12）。"""
        doc = ModelRegistry.__doc__ or ""
        self.assertIn("使用契约", doc)
        self.assertIn("禁止", doc)


class TestModelRegistryMethodStubs(unittest.TestCase):
    """build_orca_gym_data 和扩展查询方法仍 raise NotImplementedError（留待完整 P4）。"""

    def test_build_orca_gym_data_raises_not_implemented(self):
        registry = ModelRegistry()
        with self.assertRaises(NotImplementedError):
            registry.build_orca_gym_data()

    def test_query_methods_raise_not_implemented(self):
        registry = ModelRegistry()
        with self.assertRaises(NotImplementedError):
            registry.body_subtree_mass("dummy_body")
        with self.assertRaises(NotImplementedError):
            registry.equality_data_width()
        with self.assertRaises(NotImplementedError):
            registry.equality_object_ids(0)


class TestModelRegistryBuildModel(unittest.TestCase):
    """ModelRegistry.build_orca_gym_model 真实构建测试（对应阶段二 Step 3 验收标准）。"""

    def setUp(self):
        """每个测试前加载真实 mjModel 并绑定到 registry。"""
        self.mj_model = mujoco.MjModel.from_xml_path(_PENDULUM_XML)
        self.registry = ModelRegistry()
        self.registry._bind(self.mj_model)

    def test_build_returns_orca_gym_model(self):
        """build_orca_gym_model 返回 OrcaGymModel 实例。"""
        model = self.registry.build_orca_gym_model()
        self.assertIsInstance(model, OrcaGymModel)

    def test_model_dimensions_correct(self):
        """model.nq/nv/nu 正确（pendulum: nq=1, nv=1, nu=1）。"""
        model = self.registry.build_orca_gym_model()
        self.assertEqual(model.nq, 1)
        self.assertEqual(model.nv, 1)
        self.assertEqual(model.nu, 1)
        self.assertEqual(model.ngeom, 1)

    def test_build_raises_before_bind(self):
        """未绑定 mjModel 时 build_orca_gym_model 抛 RuntimeError。"""
        registry = ModelRegistry()
        with self.assertRaises(RuntimeError):
            registry.build_orca_gym_model()

    def test_body_dict_populated(self):
        """body 字典正确填充（pendulum 有 2 个 body：world + pendulum）。"""
        model = self.registry.build_orca_gym_model()
        body_dict = model.get_body_dict()
        self.assertEqual(len(body_dict), 2)  # world + pendulum
        self.assertIn("world", body_dict)
        self.assertIn("pendulum", body_dict)

    def test_body_name2id_works(self):
        """body_name2id 正确返回 ID（pendulum body id=1）。"""
        model = self.registry.build_orca_gym_model()
        self.assertEqual(model.body_name2id("pendulum"), 1)
        self.assertEqual(model.body_name2id("world"), 0)

    def test_body_mass_correct(self):
        """body 质量正确（pendulum body mass=1.0）。"""
        model = self.registry.build_orca_gym_model()
        pendulum = model.get_body_byname("pendulum")
        self.assertAlmostEqual(pendulum["Mass"], 1.0)

    def test_joint_dict_populated(self):
        """joint 字典正确填充（pendulum 有 1 个 hinge 关节）。"""
        model = self.registry.build_orca_gym_model()
        joint_dict = model.get_joint_dict()
        self.assertEqual(len(joint_dict), 1)
        self.assertIn("hinge", joint_dict)

    def test_joint_name2id_works(self):
        """joint_name2id 正确返回 ID。"""
        model = self.registry.build_orca_gym_model()
        self.assertEqual(model.joint_name2id("hinge"), 0)

    def test_actuator_dict_populated(self):
        """actuator 字典正确填充（pendulum 有 1 个 hinge_motor 执行器）。"""
        model = self.registry.build_orca_gym_model()
        actuator_dict = model.get_actuator_dict()
        self.assertEqual(len(actuator_dict), 1)
        self.assertIn("hinge_motor", actuator_dict)

    def test_actuator_name2id_works(self):
        """actuator_name2id 正确返回 ID。"""
        model = self.registry.build_orca_gym_model()
        self.assertEqual(model.actuator_name2id("hinge_motor"), 0)

    def test_actuator_joint_linkage(self):
        """执行器关联关节正确（hinge_motor → hinge）。"""
        model = self.registry.build_orca_gym_model()
        motor = model.get_actuator_byname("hinge_motor")
        self.assertEqual(motor["JointName"], "hinge")

    def test_site_dict_populated(self):
        """site 字典正确填充（pendulum 有 1 个 tip site）。"""
        model = self.registry.build_orca_gym_model()
        site_dict = model.get_site_dict()
        self.assertEqual(len(site_dict), 1)
        self.assertIn("tip", site_dict)

    def test_eq_list_empty(self):
        """等式约束列表为空（pendulum 无等式约束）。"""
        model = self.registry.build_orca_gym_model()
        eq_list = model.get_eq_list()
        self.assertEqual(len(eq_list), 0)

    def test_mocap_dict_empty(self):
        """mocap 字典为空（pendulum 无 mocap body）。"""
        model = self.registry.build_orca_gym_model()
        # pendulum 无 mocap body
        mocap_dict = model.get_mocap_dict() if hasattr(model, "get_mocap_dict") else {}
        self.assertEqual(len(mocap_dict), 0)

    def test_construct_with_mj_model_directly(self):
        """ModelRegistry(mj_model) 直接绑定，build 立即可用。"""
        registry = ModelRegistry(self.mj_model)
        model = registry.build_orca_gym_model()
        self.assertIsInstance(model, OrcaGymModel)
        self.assertEqual(model.nq, 1)


if __name__ == "__main__":
    unittest.main()
