"""阶段二 Step 3: ModelRegistry 功能验收测试。

验证 ModelRegistry 的 build_orca_gym_model 真实构建（架构 §5.5, §12.2）。
`build_orca_gym_data` 仍 raise NotImplementedError（留待完整 P4）。

阶段三 3.1.5 扩展：3 个扩展查询方法（body_subtree_mass/equality_data_width/
equality_object_ids）已实现，新增架构遵从性测试 + 功能单元测试。

运行方式:
    <conda-base>/envs/orca/bin/python tests/run_tests.py --component core/euler
"""

import inspect
import os
import unittest

import mujoco
import numpy as np

from orca_gym.core.euler.model_registry import ModelRegistry
from orca_gym.core.orca_gym_model import OrcaGymModel


# 测试用 XML 模型：单铰链倒立摆（nq=1, nv=1, nu=1, nbody=2, njnt=1, nsite=1, ngeom=1）
# 使用本仓 fixtures 目录，无外部依赖（见 AGENTS.md 测试独立性要求）
_PENDULUM_XML = os.path.join(
    os.path.dirname(__file__),
    "..", "..", "environment", "euler", "fixtures", "simple_pendulum.xml",
)
_PENDULUM_XML = os.path.abspath(_PENDULUM_XML)

# G1 模型 XML（阶段三 3.1.5 功能测试用，含 pelvis body + equality 约束）
# 简化版：mesh 替换为基础几何体，无外部 mesh 依赖
_G1_XML = os.path.join(
    os.path.dirname(__file__),
    "..", "..", "environment", "euler", "fixtures", "g1_29dof_camera_simplified.xml",
)
_G1_XML = os.path.abspath(_G1_XML)


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
    """build_orca_gym_data 仍 raise NotImplementedError（留待完整 P4）。

    注意：阶段三 3.1.5 已实现 body_subtree_mass/equality_data_width/
    equality_object_ids，不再 raise NotImplementedError。
    """

    def test_build_orca_gym_data_raises_not_implemented(self):
        registry = ModelRegistry()
        with self.assertRaises(NotImplementedError):
            registry.build_orca_gym_data()


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


# =============================================================================
# 阶段三 3.1.5：ModelRegistry 扩展查询方法（body_subtree_mass/equality_*）
# =============================================================================


class TestModelRegistryExtQueryArchCompliance(unittest.TestCase):
    """子步骤 3.1.5 架构遵从性测试（K11 typed 返回 + P2 不泄漏 _mj_model）。

    对应文档 §5.6 架构遵从性测试表。
    """

    def setUp(self):
        self.mj_model = mujoco.MjModel.from_xml_path(_G1_XML)
        self.registry = ModelRegistry(self.mj_model)

    def test_registry_body_subtree_mass_returns_float(self):
        """K11: body_subtree_mass 返回 Python float（非 numpy 泄漏）。"""
        result = self.registry.body_subtree_mass("pelvis")
        self.assertIsInstance(result, float)
        self.assertNotIsInstance(result, (np.floating,))

    def test_registry_equality_returns_typed(self):
        """K11: equality_data_width 返回 int，equality_object_ids 返回 tuple[int, int]。"""
        width = self.registry.equality_data_width()
        self.assertIsInstance(width, int)
        self.assertNotIsInstance(width, (np.integer,))

        # G1 有 1 个 equality 约束（weld）
        self.assertGreater(self.mj_model.neq, 0)
        ids = self.registry.equality_object_ids(0)
        self.assertIsInstance(ids, tuple)
        self.assertEqual(len(ids), 2)
        for oid in ids:
            self.assertIsInstance(oid, int)
            self.assertNotIsInstance(oid, (np.integer,))

    def test_registry_no_mjmodel_leak(self):
        """P2/K11: grep 断言 3.1.5 扩展查询区块不 return self._mj_model。"""
        source = inspect.getsource(ModelRegistry)
        start = source.find("# --- 扩展查询方法")
        self.assertGreater(start, 0, "未找到扩展查询方法区块")
        block_source = source[start:]
        self.assertNotIn(
            "return self._mj_model", block_source,
            "ModelRegistry 扩展查询方法不得 return self._mj_model（P2 泄漏）",
        )


class TestModelRegistryExtQueryFunctional(unittest.TestCase):
    """子步骤 3.1.5 功能单元测试（G1 XML 真实数据）。

    对应文档 §5.6 功能单元测试表。验证数值与 _mjModel 一致。
    """

    def setUp(self):
        self.mj_model = mujoco.MjModel.from_xml_path(_G1_XML)
        self.registry = ModelRegistry(self.mj_model)

    def test_body_subtree_mass_positive(self):
        """body_subtree_mass("pelvis") 返回正标量。"""
        mass = self.registry.body_subtree_mass("pelvis")
        self.assertGreater(mass, 0.0)

    def test_body_subtree_mass_matches_mujoco(self):
        """与 _mjModel.body_subtreemass[body_id] 一致。"""
        body_name = "pelvis"
        body_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        expected = float(self.mj_model.body_subtreemass[body_id])
        result = self.registry.body_subtree_mass(body_name)
        self.assertAlmostEqual(result, expected)

    def test_equality_data_width_matches_model(self):
        """与 _mjModel.eq_data.shape[1] 一致。"""
        expected = int(self.mj_model.eq_data.shape[1])
        result = self.registry.equality_data_width()
        self.assertEqual(result, expected)

    def test_equality_object_ids_matches_model(self):
        """与 eq_obj1id/eq_obj2id 一致。"""
        eq_idx = 0
        expected = (
            int(self.mj_model.eq_obj1id[eq_idx]),
            int(self.mj_model.eq_obj2id[eq_idx]),
        )
        result = self.registry.equality_object_ids(eq_idx)
        self.assertEqual(result, expected)


if __name__ == "__main__":
    unittest.main()
