"""P2-Step4/阶段二-Step5: OrcaGymEuler 验收测试。

验证 OrcaGymEuler 的隔离机制（K3/K5/K8/K9）、组合结构、公共 API 完整性
（架构 §5.2, §7.1-7.4, §12.2, §12.3），以及委托方法在 init_simulation 后
正确转发到 MuJoCoSimCore/ModelRegistry/SimConfig（阶段二 Step 5 验收标准）。

阶段三 3.1.6 扩展：查询委托链路测试（关节/Body/Site/传感器/执行器/接触/Geom
+ body_subtree_mass），验证 K3 object.__getattribute__ + K2 __dir__ + K11 typed 返回。

运行方式:
    <conda-base>/envs/orca/bin/python tests/run_tests.py --component core/euler
"""

import asyncio
import inspect
import os
import sys
import types
import unittest
from unittest import mock

import numpy as np

from orca_gym.core.euler.orca_gym_euler import OrcaGymEuler
from orca_gym.core.euler.orca_gym_data_view import OrcaGymDataView
from orca_gym.core.euler.sim_config import SimBackend, SimConfig
from orca_gym.core.orca_gym_model import OrcaGymModel


# 测试用 XML 模型：单铰链倒立摆（nq=1, nv=1, nu=1, nbody=2, nsite=1）
# 使用本仓 fixtures 目录，无外部依赖（见 AGENTS.md 测试独立性要求）
_PENDULUM_XML = os.path.join(
    os.path.dirname(__file__),
    "..", "..", "environment", "euler", "fixtures", "simple_pendulum.xml",
)
_PENDULUM_XML = os.path.abspath(_PENDULUM_XML)

# G1 模型 XML（阶段三 3.1.6 功能测试用，含传感器/执行器/equality）
# 简化版：mesh 替换为基础几何体，无外部 mesh 依赖
_G1_XML = os.path.join(
    os.path.dirname(__file__),
    "..", "..", "environment", "euler", "fixtures", "g1_29dof_camera_simplified.xml",
)
_G1_XML = os.path.abspath(_G1_XML)


class TestOrcaGymEulerSkeleton(unittest.TestCase):
    """OrcaGymEuler 骨架验收测试（对应 P2-Step4 验收标准）。"""

    def test_gym_constructable(self):
        """OrcaGymEuler() 可无参构造。"""
        gym = OrcaGymEuler()
        self.assertIsInstance(gym, OrcaGymEuler)

    def test_gym_has_private_components(self):
        """实例 __dict__ 有 _sim/_studio/_registry/_opt/_view/_euler（全部带下划线，K5）。"""
        gym = OrcaGymEuler()
        self.assertIn("_sim", gym.__dict__)
        self.assertIn("_studio", gym.__dict__)
        self.assertIn("_registry", gym.__dict__)
        self.assertIn("_opt", gym.__dict__)
        self.assertIn("_view", gym.__dict__)
        self.assertIn("_euler", gym.__dict__)

    def test_gym_blocked_attrs_include_components(self):
        """访问 gym._sim/_studio/_opt/_view/_euler/_mjData/_mjModel 抛 AttributeError（K3/K5）。"""
        gym = OrcaGymEuler()
        blocked_names = [
            "_sim", "_studio", "_registry", "_opt", "_view", "_euler",
            "_mjData", "_mjModel",
            "sim", "studio", "opt", "view", "euler",
        ]
        for name in blocked_names:
            with self.subTest(attr=name):
                with self.assertRaises(AttributeError):
                    getattr(gym, name)

    def test_gym_blocked_attrs_message_has_guidance(self):
        """AttributeError 消息含引导文本（env.data/env.apply_body_force/env.sim_config）（K3）。"""
        gym = OrcaGymEuler()
        with self.assertRaises(AttributeError) as ctx:
            _ = gym._mjData
        msg = str(ctx.exception)
        self.assertIn("env.data", msg)
        self.assertIn("env.apply_body_force", msg)
        self.assertIn("env.sim_config", msg)

    def test_gym_no_internal_property(self):
        """类不定义 studio/sim/opt/view/euler 的 property 或属性（K5）。"""
        class_attrs = vars(OrcaGymEuler)
        forbidden_properties = ["studio", "sim", "opt", "view", "euler"]
        for prop_name in forbidden_properties:
            self.assertNotIn(
                prop_name, class_attrs,
                f"类不应定义 '{prop_name}' 属性/property（K5: 不暴露子组件）",
            )

    def test_gym_dir_only_exposes_public_api(self):
        """dir(gym) 不含 _sim/_studio/_registry/_opt/_view/_euler/_mjData/_mjModel/sim/studio/opt（K3）。"""
        gym = OrcaGymEuler()
        d = dir(gym)
        forbidden_in_dir = [
            "_sim", "_studio", "_registry", "_opt", "_view", "_euler",
            "_mjData", "_mjModel",
            "sim", "studio", "opt", "view", "euler",
        ]
        for name in forbidden_in_dir:
            with self.subTest(attr=name):
                self.assertNotIn(name, d, f"dir(gym) 不应列出 '{name}'")

    def test_gym_dir_contains_public_methods(self):
        """dir(gym) 含公共 API 方法（K3）。"""
        gym = OrcaGymEuler()
        d = dir(gym)
        expected_public = [
            "data", "model", "sim_config",
            "mj_step", "mj_forward", "set_ctrl", "sync_to_view",
            "studio_bridge", "render", "pause_simulation",
            "has_euler", "step_with_coupling",
        ]
        for name in expected_public:
            with self.subTest(attr=name):
                self.assertIn(name, d, f"dir(gym) 应包含公共 API '{name}'")

    def test_gym_has_euler_returns_false(self):
        """has_euler() 返回 False（骨架阶段无 Euler，K8）。"""
        gym = OrcaGymEuler()
        self.assertFalse(gym.has_euler())

    def test_gym_step_with_coupling_callable(self):
        """step_with_coupling 方法存在且可调用（K8）。"""
        self.assertTrue(callable(getattr(OrcaGymEuler, "step_with_coupling", None)))

    def test_gym_studio_bridge_is_method_not_property(self):
        """studio_bridge 是方法（callable），gym.studio 抛 AttributeError（K9）。"""
        gym = OrcaGymEuler()
        # studio_bridge 是方法，不是 property
        self.assertTrue(callable(getattr(OrcaGymEuler, "studio_bridge", None)))
        # gym.studio 被 __getattribute__ 拦截
        with self.assertRaises(AttributeError):
            _ = gym.studio

    def test_gym_data_returns_view(self):
        """gym.data 返回 OrcaGymDataView 实例（K6）。"""
        gym = OrcaGymEuler()
        result = gym.data
        self.assertIsInstance(result, OrcaGymDataView)

    def test_gym_sim_config_returns_config(self):
        """gym.sim_config 返回 SimConfig 实例。"""
        gym = OrcaGymEuler()
        result = gym.sim_config
        self.assertIsInstance(result, SimConfig)

    def test_gym_docstring_has_contract(self):
        """docstring 含「API 契约」和「禁止」关键词（K12）。"""
        doc = OrcaGymEuler.__doc__ or ""
        self.assertIn("API 契约", doc)
        self.assertIn("禁止", doc)


class TestOrcaGymEulerStudioDelegation(unittest.TestCase):
    """阶段二 Step 2: OrcaGymEuler render/pause_simulation 委托到 studio bridge。

    load_model_xml 委托到 OrcaStudioBridge.load_model_xml，离线模式下未配置 raise。
    """

    def test_render_delegates_to_studio_offline_noop(self):
        """render 离线模式 no-op（委托到 studio.render，stub=None 不抛异常）。"""
        gym = OrcaGymEuler()
        # 离线模式 stub=None，render 读 _view.qpos/time（空 DataView）委托 studio no-op
        asyncio.run(gym.render())

    def test_pause_simulation_delegates_to_studio_offline_noop(self):
        """pause_simulation 离线模式 no-op（委托到 studio.pause_simulation）。"""
        gym = OrcaGymEuler()
        asyncio.run(gym.pause_simulation())

    def test_load_model_xml_delegates_to_studio(self):
        """load_model_xml 委托到 studio bridge（离线模式未配置 raise RuntimeError）。"""
        gym = OrcaGymEuler()
        with self.assertRaises(RuntimeError):
            asyncio.run(gym.load_model_xml())


class TestOrcaGymEulerDelegation(unittest.TestCase):
    """阶段二 Step 5: OrcaGymEuler 委托方法真实转发测试。

    验证 init_simulation 后 model/nq/nu 正确，mj_step 后 data.time > 0，
    step_with_coupling 等价于 set_ctrl + step（对应 Step 5 验收标准）。
    """

    def setUp(self):
        """每个测试前初始化仿真（加载 pendulum 模型）。"""
        self.gym = OrcaGymEuler()
        asyncio.run(self.gym.init_simulation(_PENDULUM_XML))

    def test_model_returns_orca_gym_model(self):
        """init_simulation 后 gym.model 返回 OrcaGymModel 实例。"""
        model = self.gym.model
        self.assertIsInstance(model, OrcaGymModel)

    def test_model_nq_correct(self):
        """init_simulation 后 gym.model.nq == 1（pendulum）。"""
        self.assertEqual(self.gym.model.nq, 1)

    def test_gym_nq_property(self):
        """gym.nq property 返回正确维度（1）。"""
        self.assertEqual(self.gym.nq, 1)

    def test_gym_nu_property(self):
        """gym.nu property 返回正确维度（1）。"""
        self.assertEqual(self.gym.nu, 1)

    def test_mj_step_advances_time(self):
        """mj_step(1) + sync_to_view 后 data.time > 0。

        time 是 float 标量（值拷贝，非零拷贝视图），需 sync_to_view 刷新。
        """
        self.gym.mj_step(1)
        self.gym.sync_to_view()
        self.assertGreater(self.gym.data.time, 0.0)

    def test_mj_forward_does_not_advance_time(self):
        """mj_forward 不步进，time 不变（仍为 0）。"""
        self.gym.mj_forward()
        self.assertEqual(self.gym.data.time, 0.0)

    def test_set_ctrl_affects_step(self):
        """set_ctrl 后 mj_step 使摆杆运动（qpos 变化）。"""
        qpos_before = float(self.gym.data.qpos[0])
        self.gym.set_ctrl(np.array([1.0]))
        self.gym.mj_step(1)
        qpos_after = float(self.gym.data.qpos[0])
        self.assertNotEqual(qpos_before, qpos_after)

    def test_sync_to_view_updates_data(self):
        """sync_to_view 后 data.qpos 反映最新 mjData 状态。"""
        self.gym.set_ctrl(np.array([1.0]))
        self.gym.mj_step(5)
        self.gym.sync_to_view()
        self.assertEqual(self.gym.data.qpos.shape, (1,))

    def test_reset_data_clears_state(self):
        """reset_data 后 qpos 回到初始状态（0）。"""
        self.gym.set_ctrl(np.array([1.0]))
        self.gym.mj_step(5)
        self.gym.reset_data()
        np.testing.assert_array_almost_equal(self.gym.data.qpos, np.zeros(1))

    def test_set_qpos_qvel(self):
        """set_qpos_qvel 设置广义坐标和速度。"""
        self.gym.set_qpos_qvel(np.array([0.5]), np.array([0.1]))
        self.gym.mj_forward()
        self.assertAlmostEqual(float(self.gym.data.qpos[0]), 0.5)
        self.assertAlmostEqual(float(self.gym.data.qvel[0]), 0.1)

    def test_step_with_coupling_equals_set_ctrl_plus_step(self):
        """step_with_coupling(ctrl, 5, dt) 等价于 set_ctrl(ctrl) + mj_step(5)。

        has_euler()=False 时，step_with_coupling 内部仅做 set_ctrl + step。
        通过对比两条路径的 qpos 终态验证等价性。
        """
        ctrl = np.array([0.5])
        dt = 0.002

        # 路径 A: step_with_coupling
        gym_a = OrcaGymEuler()
        asyncio.run(gym_a.init_simulation(_PENDULUM_XML))
        gym_a.step_with_coupling(ctrl, 5, dt)
        qpos_a = float(gym_a.data.qpos[0])

        # 路径 B: set_ctrl + mj_step
        gym_b = OrcaGymEuler()
        asyncio.run(gym_b.init_simulation(_PENDULUM_XML))
        gym_b.set_ctrl(ctrl)
        gym_b.mj_step(5)
        qpos_b = float(gym_b.data.qpos[0])

        self.assertAlmostEqual(qpos_a, qpos_b)

    def test_model_is_cached(self):
        """多次访问 gym.model 返回同一对象（缓存）。"""
        m1 = self.gym.model
        m2 = self.gym.model
        self.assertIs(m1, m2)

    def test_blocked_attrs_still_enforced_after_init(self):
        """init_simulation 后 __getattribute__ 拦截机制未被破坏。"""
        with self.assertRaises(AttributeError):
            _ = self.gym._sim
        with self.assertRaises(AttributeError):
            _ = self.gym._mjData
        with self.assertRaises(AttributeError):
            _ = self.gym._mjModel

    def test_set_ctrl_applies_override_ctrls(self):
        """set_ctrl 应用 studio override_ctrls（Step 2 验收标准）。

        注入 override 到 studio bridge，验证 set_ctrl 写入 _mjData.ctrl 的值被覆盖。
        """
        # 直接注入 override 到 bridge 的内部缓存
        bridge = self.gym.studio_bridge()
        bridge._override_ctrls = {0: 0.5}
        # set_ctrl 应应用 override：ctrl[0] = 0.5 而非原始 1.0
        self.gym.set_ctrl(np.array([1.0]))
        # 通过 sync_to_view 后 actuator_force 反映 override
        self.gym.mj_forward()
        self.gym.sync_to_view()
        # ctrl 在 _mjData 中应为 override 值
        # 直接读 data.ctrl 不可用，验证方式：override 影响步进结果
        # 更直接：注入 override 后 set_ctrl 应使 _mjData.ctrl[0] == 0.5
        # 由于 _mjData 被隔离，通过 actuator_force 间接验证
        actuator_force = float(self.gym.data.actuator_force[0])
        # 0.5 * gain 应与 0.5 相关（pendulum gain=1），验证 override 生效
        self.assertAlmostEqual(actuator_force, 0.5, places=5)

    def test_set_ctrl_no_override_unchanged(self):
        """无 override 时 set_ctrl 原样写入（Step 2）。"""
        # 确保无 override
        bridge = self.gym.studio_bridge()
        bridge._override_ctrls.clear()
        self.gym.set_ctrl(np.array([1.0]))
        self.gym.mj_forward()
        self.gym.sync_to_view()
        actuator_force = float(self.gym.data.actuator_force[0])
        self.assertAlmostEqual(actuator_force, 1.0, places=5)

    def test_set_ctrl_override_out_of_range_ignored(self):
        """override 索引越界时安全忽略（不抛 IndexError）。"""
        bridge = self.gym.studio_bridge()
        bridge._override_ctrls = {5: 0.5}  # pendulum nu=1，索引 5 越界
        # 不应抛异常
        self.gym.set_ctrl(np.array([1.0]))
        self.gym.mj_forward()
        self.gym.sync_to_view()
        # 原始 ctrl 值应保留
        actuator_force = float(self.gym.data.actuator_force[0])
        self.assertAlmostEqual(actuator_force, 1.0, places=5)


class TestOrcaGymEulerStudioBridgeAccess(unittest.TestCase):
    """补充：验证 studio_bridge() 返回 Bridge 对象（K9 方法访问模式）。"""

    def test_studio_bridge_returns_bridge(self):
        """studio_bridge() 返回 OrcaStudioBridge 实例。"""
        from orca_gym.core.euler.orca_studio_bridge import OrcaStudioBridge
        gym = OrcaGymEuler()
        bridge = gym.studio_bridge()
        self.assertIsInstance(bridge, OrcaStudioBridge)


class TestOrcaGymEulerViolationPatterns(unittest.TestCase):
    """违规访问拦截测试。

    对照架构文档 §6.2/§6.3/§6.5/§7.6 明确列举的违规访问模式，验证
    _BLOCKED_ATTRS 全部变体、多层穿墙路径、K8/K9 违规模式均被拦截。
    """

    def test_all_mjdata_mjmodel_variants_blocked(self):
        """_BLOCKED_ATTRS 中 _mjData/_mjModel 的全部 8 个变体都被拦截（K3）。

        覆盖: _mjData, _mjModel, mj_data, mj_model, _mj_data, _mj_model, mjData, mjModel
        """
        gym = OrcaGymEuler()
        variants = [
            "_mjData", "_mjModel", "mj_data", "mj_model",
            "_mj_data", "_mj_model", "mjData", "mjModel",
        ]
        for name in variants:
            with self.subTest(attr=name):
                with self.assertRaises(AttributeError):
                    getattr(gym, name)

    def test_all_component_variants_blocked(self):
        """_BLOCKED_ATTRS 中子组件的全部带/不带下划线变体都被拦截（K5）。

        覆盖: _sim/sim, _studio/studio, _registry/registry, _opt/opt, _view/view, _euler/euler
        """
        gym = OrcaGymEuler()
        components = ["sim", "studio", "registry", "opt", "view", "euler"]
        for comp in components:
            with self.subTest(attr=f"_{comp}"):
                with self.assertRaises(AttributeError):
                    getattr(gym, f"_{comp}")
            with self.subTest(attr=comp):
                with self.assertRaises(AttributeError):
                    getattr(gym, comp)

    def test_multilayer_tunnel_mjdata_blocked(self):
        """多层穿墙 gym._sim._mjData 在第一层 gym._sim 即被拦截（架构 §6.2 R1）。

        架构 §6.2 违规示例: env._gym._sim._mjData.qpos
        本测试验证 Gym 层第一层拦截: gym._sim 抛 AttributeError，
        使穿墙链在第一层即断裂，无法到达 _mjData。
        """
        gym = OrcaGymEuler()
        with self.assertRaises(AttributeError):
            gym._sim._mjData    # 第一层 gym._sim 即被拦截

    def test_multilayer_tunnel_mjmodel_blocked(self):
        """多层穿墙 gym._sim._mjModel 在第一层即被拦截（架构 §6.5 C1）。

        架构 §6.5 违规示例: env._gym._sim._mjModel.opt.timestep
        本测试验证 Gym 层第一层拦截: gym._sim 抛 AttributeError。
        """
        gym = OrcaGymEuler()
        with self.assertRaises(AttributeError):
            gym._sim._mjModel    # 第一层 gym._sim 即被拦截

    def test_multilayer_tunnel_xfrc_blocked(self):
        """多层穿墙 gym._sim._mjData.xfrc_applied 在第一层即被拦截（架构 §6.3 W2）。

        架构 §6.3 违规示例: env._gym._sim._mjData.xfrc_applied[body_id, :3] = force
        本测试验证 Gym 层第一层拦截: gym._sim 抛 AttributeError。
        """
        gym = OrcaGymEuler()
        with self.assertRaises(AttributeError):
            gym._sim._mjData.xfrc_applied    # 第一层 gym._sim 即被拦截

    def test_k8_euler_private_access_blocked(self):
        """K8: gym._euler 访问被拦截，引导用 has_euler()/step_with_coupling()。

        架构 §8.2 违规示例: if self._gym._euler is not None
        本测试验证 Gym 层拦截 _euler，防止 do_simulation 内穿墙。
        """
        gym = OrcaGymEuler()
        with self.assertRaises(AttributeError) as ctx:
            _ = gym._euler
        msg = str(ctx.exception)
        # 引导消息应指向 has_euler / step_with_coupling
        self.assertIn("has_euler", msg)
        self.assertIn("step_with_coupling", msg)

    def test_k9_studio_property_access_blocked(self):
        """K9: gym.studio 访问被拦截，引导用 studio_bridge()。

        架构 §7.1 M2: Studio 交互通过方法 studio_bridge() 而非 property。
        """
        gym = OrcaGymEuler()
        with self.assertRaises(AttributeError) as ctx:
            _ = gym.studio
        msg = str(ctx.exception)
        # 引导消息应指向 studio_bridge
        self.assertIn("studio_bridge", msg)

    def test_k5_sim_access_guided_to_step_methods(self):
        """K5: gym._sim 访问被拦截，引导用 mj_step/mj_forward/do_simulation。"""
        gym = OrcaGymEuler()
        with self.assertRaises(AttributeError) as ctx:
            _ = gym._sim
        msg = str(ctx.exception)
        self.assertIn("mj_step", msg)
        self.assertIn("mj_forward", msg)

    def test_k5_opt_access_guided_to_sim_config(self):
        """K5: gym._opt 访问被拦截，引导用 sim_config。"""
        gym = OrcaGymEuler()
        with self.assertRaises(AttributeError) as ctx:
            _ = gym._opt
        msg = str(ctx.exception)
        self.assertIn("sim_config", msg)

    def test_k5_view_access_guided_to_data(self):
        """K5: gym._view 访问被拦截，引导用 env.data。"""
        gym = OrcaGymEuler()
        with self.assertRaises(AttributeError) as ctx:
            _ = gym._view
        msg = str(ctx.exception)
        self.assertIn("env.data", msg)

    def test_blocked_attrs_frozenset_complete(self):
        """_BLOCKED_ATTRS 是 frozenset 且包含全部 30 个拦截名（K3/K5 + Euler 后端）。"""
        self.assertIsInstance(OrcaGymEuler._BLOCKED_ATTRS, frozenset)
        expected_blocked = {
            # L3 引擎内部 (8)
            "_mjData", "_mjModel", "mj_data", "mj_model",
            "_mj_data", "_mj_model", "mjData", "mjModel",
            # K5 子组件 (12 = 6 带下划线 + 6 不带)
            "_sim", "_studio", "_registry", "_opt", "_view", "_euler",
            "sim", "studio", "registry", "opt", "view", "euler",
            # Euler 后端 GPU 对象 (10)
            "_mjf_model", "_mjf_data", "mjf_model", "mjf_data",
            "_host_cache", "host_cache",
            "_step_graph", "step_graph",
            "_coupling", "coupling",
        }
        self.assertEqual(OrcaGymEuler._BLOCKED_ATTRS, expected_blocked)


# =============================================================================
# 阶段三 3.1.6：OrcaGymEuler 查询委托链路
# =============================================================================


class TestGymQueryDelegationArchCompliance(unittest.TestCase):
    """子步骤 3.1.6 架构遵从性测试（K3/K2/K5/K11）。

    对应文档 §5.7 架构遵从性测试表。
    """

    def test_gym_query_delegates_use_getattribute(self):
        """K3: grep 断言新增委托方法均用 object.__getattribute__(self, "_sim"/"_registry"/"_orca_model")。"""
        source = inspect.getsource(OrcaGymEuler)
        start = source.find("# --- 查询委托（阶段三 3.1.6")
        self.assertGreater(start, 0, "未找到 3.1.6 查询委托区块")
        block_source = source[start:]
        # 委托方法不应直接 self._sim（被 __getattribute__ 拦截）
        self.assertNotIn(
            "self._sim.", block_source,
            "3.1.6 委托方法不得直接 self._sim（K3 必须用 object.__getattribute__）",
        )
        self.assertNotIn(
            "self._registry.", block_source,
            "3.1.6 委托方法不得直接 self._registry（K3）",
        )
        self.assertNotIn(
            "self._orca_model", block_source,
            "3.1.6 委托方法不得直接 self._orca_model（K3）",
        )
        # 应包含 object.__getattribute__ 调用
        self.assertIn("object.__getattribute__(self, \"_sim\")", block_source)
        self.assertIn("object.__getattribute__(self, \"_registry\")", block_source)
        self.assertIn("object.__getattribute__(self, \"_orca_model\")", block_source)

    def test_gym_dir_no_internal_leak(self):
        """K2/K3: dir(gym) 不含 _sim/_studio/_registry/_mjData/_mjModel。"""
        gym = OrcaGymEuler()
        d = dir(gym)
        for name in ["_sim", "_studio", "_registry", "_mjData", "_mjModel"]:
            with self.subTest(attr=name):
                self.assertNotIn(name, d)

    def test_gym_no_new_property_for_internal(self):
        """K5: grep 断言 3.1.6 委托区块不新增 @property 暴露 _sim/_studio/_registry。"""
        source = inspect.getsource(OrcaGymEuler)
        start = source.find("# --- 查询委托（阶段三 3.1.6")
        block_source = source[start:]
        # 委托区块不应包含 @property 装饰器
        self.assertNotIn("@property", block_source)

    def test_gym_query_returns_typed(self):
        """K11: gym.query_joint_qpos 返回 dict[str, np.ndarray]，非 MjData。"""
        gym = OrcaGymEuler()
        asyncio.run(gym.init_simulation(_G1_XML))
        result = gym.query_joint_qpos(["left_hip_pitch_joint"])
        self.assertIsInstance(result, dict)
        for k, v in result.items():
            self.assertIsInstance(k, str)
            self.assertIsInstance(v, np.ndarray)

    def test_gym_sensor_delegates_assemble_sensor_info(self):
        """K3: query_sensor_data 从 _orca_model 拼装 sensor_info 传入 SimCore。"""
        gym = OrcaGymEuler()
        asyncio.run(gym.init_simulation(_G1_XML))
        sensor_names = ["left_hip_pitch_pos"]
        result = gym.query_sensor_data(sensor_names)
        self.assertIsInstance(result, dict)
        self.assertIn("left_hip_pitch_pos", result)
        self.assertIsInstance(result["left_hip_pitch_pos"], np.ndarray)


class TestGymQueryDelegationFunctional(unittest.TestCase):
    """子步骤 3.1.6 功能单元测试（G1 XML 真实数据）。

    对应文档 §5.7 功能单元测试表。验证委托链路结果与底层 SimCore/Registry 一致。
    """

    def setUp(self):
        self.gym = OrcaGymEuler()
        asyncio.run(self.gym.init_simulation(_G1_XML))
        self.gym.mj_forward()

    def test_gym_query_joint_qpos_delegates_to_simcore(self):
        """Gym 委托结果与 SimCore 直接调用结果一致。"""
        joint_names = ["left_hip_pitch_joint", "left_hip_roll_joint"]
        gym_result = self.gym.query_joint_qpos(joint_names)
        # 通过 object.__getattribute__ 取 _sim 直接调用 SimCore
        sim = object.__getattribute__(self.gym, "_sim")
        sim_result = sim.query_joint_qpos(joint_names)
        for name in joint_names:
            np.testing.assert_array_equal(gym_result[name], sim_result[name])

    def test_gym_query_sensor_data_assembles_sensor_info(self):
        """sensor_info 正确拼装，传感器数据正确。"""
        sensor_names = ["left_hip_pitch_pos"]
        gym_result = self.gym.query_sensor_data(sensor_names)
        # 直接从 SimCore + 手动拼装 sensor_info 对比
        sim = object.__getattribute__(self.gym, "_sim")
        model = object.__getattribute__(self.gym, "_orca_model")
        sensor_info = {name: model.get_sensor(name) for name in sensor_names}
        sim_result = sim.query_sensor_data(sensor_names, sensor_info)
        np.testing.assert_array_equal(
            gym_result["left_hip_pitch_pos"],
            sim_result["left_hip_pitch_pos"],
        )

    def test_gym_body_subtree_mass_delegates_to_registry(self):
        """委托结果与 Registry 直接调用一致。"""
        body_name = "pelvis"
        gym_result = self.gym.body_subtree_mass(body_name)
        registry = object.__getattribute__(self.gym, "_registry")
        registry_result = registry.body_subtree_mass(body_name)
        self.assertAlmostEqual(gym_result, registry_result)

    def test_gym_query_body_xpos_delegates_to_simcore(self):
        """query_body_xpos_xmat_xquat 委托 SimCore 结果一致。"""
        body_names = ["pelvis"]
        gym_result = self.gym.query_body_xpos_xmat_xquat(body_names)
        sim = object.__getattribute__(self.gym, "_sim")
        sim_result = sim.query_body_xpos_xmat_xquat(body_names)
        for name in body_names:
            np.testing.assert_array_equal(
                gym_result[name]["xpos"], sim_result[name]["xpos"]
            )

    def test_gym_get_goal_bounding_box_delegates_to_simcore(self):
        """get_goal_bounding_box 委托 SimCore 结果一致。"""
        geom_name = "manipulation_box_geom"
        gym_result = self.gym.get_goal_bounding_box(geom_name)
        sim = object.__getattribute__(self.gym, "_sim")
        sim_result = sim.get_goal_bounding_box(geom_name)
        np.testing.assert_array_equal(gym_result, sim_result)


class TestOrcaGymEulerEulerBackend(unittest.TestCase):
    """Phase G：Euler 后端适配（_init_euler_backend / _BLOCKED_ATTRS / reset_coupling_state）。"""

    def test_new_blocked_keys_present(self):
        blocked = OrcaGymEuler._BLOCKED_ATTRS
        for key in (
            "_mjf_model", "_mjf_data", "mjf_model", "mjf_data",
            "_host_cache", "host_cache", "_step_graph", "step_graph",
            "_coupling", "coupling",
        ):
            self.assertIn(key, blocked)

    def test_new_blocked_keys_raise_attribute_error(self):
        gym = OrcaGymEuler()
        for key in ("mjf_data", "_host_cache", "step_graph", "coupling"):
            with self.subTest(key=key):
                with self.assertRaises(AttributeError):
                    getattr(gym, key)

    def test_reset_coupling_state_noop_when_euler_none(self):
        gym = OrcaGymEuler()
        gym.reset_coupling_state()  # P1 no-op，无异常

    def test_init_euler_backend_constructs_mujoco_sim_core_euler(self):
        gym = OrcaGymEuler()
        opt = object.__getattribute__(gym, "_opt")
        opt.backend = SimBackend.EULER
        opt.device = "cuda:0"
        opt.nworld = 1

        fake_model = object()
        captured = {}

        class FakeCore:
            def init_simulation(self, model_xml_path, device="cuda", nworld=1,
                                timestep=None, opt_overrides=None):
                captured["init_args"] = (model_xml_path, device, nworld, timestep, opt_overrides)
                self.mj_model = fake_model

        fake_mod = types.ModuleType("orca_gym.core.euler.mujoco_sim_core_euler")
        fake_mod.MuJoCoSimCoreEuler = FakeCore

        with mock.patch.dict(sys.modules, {
            "orca_gym.core.euler.mujoco_sim_core_euler": fake_mod,
        }):
            gym._init_euler_backend("dummy.xml", None)

        self.assertEqual(captured["init_args"], ("dummy.xml", "cuda:0", 1, 0.002, None))
        sim = object.__getattribute__(gym, "_sim")
        self.assertIsInstance(sim, FakeCore)
        self.assertIs(sim.mj_model, fake_model)
        self.assertIsNone(object.__getattribute__(gym, "_euler"))
        self.assertIs(opt._mj_model, fake_model)

    def test_init_euler_backend_passes_opt_overrides(self):
        """Feature A：opt_overrides 透传到 MuJoCoSimCoreEuler.init_simulation。"""
        gym = OrcaGymEuler()
        opt = object.__getattribute__(gym, "_opt")
        opt.backend = SimBackend.EULER
        opt.device = "cuda:0"
        opt.nworld = 1

        fake_model = object()
        captured = {}

        class FakeCore:
            def init_simulation(self, model_xml_path, device="cuda", nworld=1,
                                timestep=None, opt_overrides=None):
                captured["init_args"] = (model_xml_path, device, nworld, timestep, opt_overrides)
                self.mj_model = fake_model

        fake_mod = types.ModuleType("orca_gym.core.euler.mujoco_sim_core_euler")
        fake_mod.MuJoCoSimCoreEuler = FakeCore

        overrides = {"integrator": 0}
        with mock.patch.dict(sys.modules, {
            "orca_gym.core.euler.mujoco_sim_core_euler": fake_mod,
        }):
            gym._init_euler_backend("dummy.xml", None, overrides)

        self.assertEqual(
            captured["init_args"], ("dummy.xml", "cuda:0", 1, 0.002, overrides)
        )
        sim = object.__getattribute__(gym, "_sim")
        self.assertIsInstance(sim, FakeCore)

    def test_init_euler_backend_multi_world_raises(self):
        gym = OrcaGymEuler()
        opt = object.__getattribute__(gym, "_opt")
        opt.backend = SimBackend.EULER
        opt.nworld = 2
        with self.assertRaises(NotImplementedError):
            gym._init_euler_backend("dummy.xml", None)


if __name__ == "__main__":
    unittest.main()
