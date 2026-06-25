"""P1-Step2/阶段二-Step4: OrcaGymDataView 验收测试。

验证 OrcaGymDataView 的字段定义、方法签名、__getattr__ 兜底机制完整
（架构 §5.7, §7.4, §12.2），以及 body/site 查询方法在绑定真实 mjData/mjModel
后返回正确结果（阶段二 Step 4 验收标准）。

运行方式:
    <conda-base>/envs/orca/bin/python tests/run_tests.py --component core/euler
"""

import os
import unittest

import mujoco
import numpy as np

from orca_gym.core.euler.orca_gym_data_view import OrcaGymDataView


# 测试用 XML 模型：单铰链倒立摆（nbody=2, nsite=1）
_PENDULUM_XML = os.path.join(
    os.path.dirname(__file__),
    "..", "..", "..", "..", "..",
    "OrcaPlayground", "envs", "euler", "scenes", "simple_pendulum.xml",
)
_PENDULUM_XML = os.path.abspath(_PENDULUM_XML)


class TestOrcaGymDataViewSkeleton(unittest.TestCase):
    """OrcaGymDataView 骨架验收测试（对应 P1-Step2 验收标准）。"""

    def test_data_view_constructable(self):
        """OrcaGymDataView() 可无参构造。"""
        view = OrcaGymDataView()
        self.assertIsInstance(view, OrcaGymDataView)

    def test_data_view_has_basic_fields(self):
        """实例有 qpos/qvel/qacc/qfrc_bias/time 五个基本字段。"""
        view = OrcaGymDataView()
        self.assertTrue(hasattr(view, "qpos"))
        self.assertTrue(hasattr(view, "qvel"))
        self.assertTrue(hasattr(view, "qacc"))
        self.assertTrue(hasattr(view, "qfrc_bias"))
        self.assertTrue(hasattr(view, "time"))

    def test_data_view_has_extended_fields(self):
        """实例有 xfrc_applied/actuator_force/contact 扩展字段。"""
        view = OrcaGymDataView()
        self.assertTrue(hasattr(view, "xfrc_applied"))
        self.assertTrue(hasattr(view, "actuator_force"))
        self.assertTrue(hasattr(view, "contact"))

    def test_data_view_has_body_query_methods(self):
        """body_xpos/body_xquat/body_xmat/body_cvel/body_subtree_mass 方法存在。"""
        self.assertTrue(callable(getattr(OrcaGymDataView, "body_xpos", None)))
        self.assertTrue(callable(getattr(OrcaGymDataView, "body_xquat", None)))
        self.assertTrue(callable(getattr(OrcaGymDataView, "body_xmat", None)))
        self.assertTrue(callable(getattr(OrcaGymDataView, "body_cvel", None)))
        self.assertTrue(callable(getattr(OrcaGymDataView, "body_subtree_mass", None)))

    def test_data_view_has_site_query_methods(self):
        """site_xpos/site_xmat 方法存在。"""
        self.assertTrue(callable(getattr(OrcaGymDataView, "site_xpos", None)))
        self.assertTrue(callable(getattr(OrcaGymDataView, "site_xmat", None)))

    def test_data_view_getattr_guidance(self):
        """访问不存在的字段（如 cvel）抛 AttributeError，消息含引导文本。

        验证 M3 兜底机制（架构 §7.4）：缺字段时引导扩展而非绕道。
        引导文本应列出可用字段/方法。
        """
        view = OrcaGymDataView()
        with self.assertRaises(AttributeError) as ctx:
            _ = view.cvel  # cvel 不是 DataView 的字段（需通过 body_cvel(name) 查询）

        msg = str(ctx.exception)
        # 引导文本应包含：被访问的字段名、可用字段/方法提示、扩展引导
        self.assertIn("cvel", msg)
        self.assertIn("可用字段", msg)
        self.assertIn("可用方法", msg)
        self.assertIn("OrcaGymDataView", msg)

    def test_data_view_docstring_has_contract(self):
        """docstring 含「使用契约」和「禁止」关键词（K12）。"""
        doc = OrcaGymDataView.__doc__ or ""
        self.assertIn("使用契约", doc)
        self.assertIn("禁止", doc)


class TestOrcaGymDataViewFieldTypes(unittest.TestCase):
    """补充：验证基本字段和扩展字段的类型符合架构 §5.7 定义。

    骨架阶段字段为空数组/默认值，但类型应与文档一致。
    """

    def test_basic_fields_types(self):
        view = OrcaGymDataView()
        self.assertIsInstance(view.qpos, np.ndarray)
        self.assertIsInstance(view.qvel, np.ndarray)
        self.assertIsInstance(view.qacc, np.ndarray)
        self.assertIsInstance(view.qfrc_bias, np.ndarray)
        self.assertIsInstance(view.time, float)

    def test_extended_fields_types(self):
        view = OrcaGymDataView()
        self.assertIsInstance(view.xfrc_applied, np.ndarray)
        self.assertIsInstance(view.actuator_force, np.ndarray)
        self.assertIsInstance(view.contact, list)

    def test_basic_fields_not_none(self):
        """基本字段不应为 None（初始化为空数组/默认值，便于用户直接读取）。"""
        view = OrcaGymDataView()
        self.assertIsNotNone(view.qpos)
        self.assertIsNotNone(view.qvel)
        self.assertIsNotNone(view.qacc)
        self.assertIsNotNone(view.qfrc_bias)
        self.assertIsNotNone(view.time)

    def test_extended_fields_not_none(self):
        """扩展字段不应为 None。"""
        view = OrcaGymDataView()
        self.assertIsNotNone(view.xfrc_applied)
        self.assertIsNotNone(view.actuator_force)
        self.assertIsNotNone(view.contact)


class TestOrcaGymDataViewRealQuery(unittest.TestCase):
    """阶段二 Step 4: body/site 查询方法真实查询测试。

    验证 _sync_from_mjdata 后基本字段零拷贝视图一致，body/site 查询方法
    返回正确结果（对应阶段二 Step 4 验收标准）。
    """

    def setUp(self):
        """每个测试前加载真实 mjModel/mjData 并同步到 view。"""
        self.mj_model = mujoco.MjModel.from_xml_path(_PENDULUM_XML)
        self.mj_data = mujoco.MjData(self.mj_model)
        # 前向计算以填充 xpos/xquat/xmat/cvel 等派生字段
        mujoco.mj_forward(self.mj_model, self.mj_data)
        self.view = OrcaGymDataView()
        self.view._sync_from_mjdata(self.mj_data, self.mj_model)

    def test_sync_makes_qpos_zero_copy_view(self):
        """sync_from_mjdata 后 view.qpos 与 mj_data.qpos 一致（零拷贝视图）。"""
        # 数值一致
        np.testing.assert_array_equal(self.view.qpos, self.mj_data.qpos)
        # 零拷贝：修改 mj_data.qpos 后 view.qpos 同步变化
        self.mj_data.qpos[0] = 0.5
        self.assertEqual(self.view.qpos[0], 0.5)

    def test_sync_makes_qvel_zero_copy_view(self):
        """sync_from_mjdata 后 view.qvel 与 mj_data.qvel 一致（零拷贝视图）。"""
        np.testing.assert_array_equal(self.view.qvel, self.mj_data.qvel)

    def test_sync_populates_time(self):
        """sync_from_mjdata 后 view.time 为 float 类型。"""
        self.assertIsInstance(self.view.time, float)
        self.assertEqual(self.view.time, float(self.mj_data.time))

    def test_body_xpos_returns_correct_shape(self):
        """body_xpos("pendulum") 返回 (3,) 数组。"""
        xpos = self.view.body_xpos("pendulum")
        self.assertEqual(xpos.shape, (3,))

    def test_body_xpos_correct_value(self):
        """body_xpos("pendulum") 与 mj_data.body(1).xpos 一致。"""
        np.testing.assert_array_equal(
            self.view.body_xpos("pendulum"),
            self.mj_data.body(1).xpos,
        )

    def test_body_xpos_world(self):
        """body_xpos("world") 返回原点 [0,0,0]。"""
        xpos = self.view.body_xpos("world")
        np.testing.assert_array_almost_equal(xpos, np.zeros(3))

    def test_body_xquat_returns_correct_shape(self):
        """body_xquat("pendulum") 返回 (4,) 数组。"""
        xquat = self.view.body_xquat("pendulum")
        self.assertEqual(xquat.shape, (4,))

    def test_body_xmat_returns_correct_shape(self):
        """body_xmat("pendulum") 返回 (9,) 数组（MuJoCo 扁平存储）。"""
        xmat = self.view.body_xmat("pendulum")
        self.assertEqual(xmat.shape, (9,))

    def test_body_cvel_returns_correct_shape(self):
        """body_cvel("pendulum") 返回 (6,) 数组。"""
        cvel = self.view.body_cvel("pendulum")
        self.assertEqual(cvel.shape, (6,))

    def test_body_subtree_mass_correct(self):
        """body_subtree_mass("pendulum") 返回正确质量（1.0）。"""
        mass = self.view.body_subtree_mass("pendulum")
        self.assertAlmostEqual(mass, 1.0)

    def test_site_xpos_returns_correct_shape(self):
        """site_xpos("tip") 返回 (3,) 数组。"""
        xpos = self.view.site_xpos("tip")
        self.assertEqual(xpos.shape, (3,))

    def test_site_xpos_correct_value(self):
        """site_xpos("tip") 与 mj_data.site(0).xpos 一致。"""
        np.testing.assert_array_equal(
            self.view.site_xpos("tip"),
            self.mj_data.site(0).xpos,
        )

    def test_site_xmat_returns_correct_shape(self):
        """site_xmat("tip") 返回 (9,) 数组（MuJoCo 扁平存储）。"""
        xmat = self.view.site_xmat("tip")
        self.assertEqual(xmat.shape, (9,))

    def test_query_before_sync_raises(self):
        """未 sync 前调用 body_xpos 抛 TypeError（_mj_model 为 None）。"""
        view = OrcaGymDataView()
        with self.assertRaises(TypeError):
            view.body_xpos("pendulum")


if __name__ == "__main__":
    unittest.main()
