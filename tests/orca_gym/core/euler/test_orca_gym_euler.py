"""P2-Step4: OrcaGymEuler 骨架验收测试。

验证 OrcaGymEuler 的隔离机制（K3/K5/K8/K9）、组合结构、公共 API 完整性
（架构 §5.2, §7.1-7.4, §12.2, §12.3），不验证 MuJoCo/Studio 功能正确性
（骨架阶段方法体 raise NotImplementedError）。

运行方式:
    <conda-base>/envs/orca/bin/python tests/run_tests.py --component core/euler
"""

import unittest

from orca_gym.core.euler.orca_gym_euler import OrcaGymEuler
from orca_gym.core.euler.orca_gym_data_view import OrcaGymDataView
from orca_gym.core.euler.sim_config import SimConfig


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


class TestOrcaGymEulerMethodStubs(unittest.TestCase):
    """补充：验证方法在骨架阶段按约定 raise NotImplementedError。"""

    def test_lifecycle_methods_raise_not_implemented(self):
        gym = OrcaGymEuler()
        # async methods — 用 asyncio.run 检查
        import asyncio

        async def check_async():
            with self.assertRaises(NotImplementedError):
                await gym.init_simulation("dummy.xml")
            with self.assertRaises(NotImplementedError):
                await gym.load_model_xml()
            with self.assertRaises(NotImplementedError):
                await gym.render()
            with self.assertRaises(NotImplementedError):
                await gym.pause_simulation()
        asyncio.run(check_async())

    def test_sync_methods_raise_not_implemented(self):
        gym = OrcaGymEuler()
        with self.assertRaises(NotImplementedError):
            gym.mj_step(1)
        with self.assertRaises(NotImplementedError):
            gym.mj_forward()
        with self.assertRaises(NotImplementedError):
            gym.set_ctrl(None)
        with self.assertRaises(NotImplementedError):
            gym.sync_to_view()
        with self.assertRaises(NotImplementedError):
            gym.step_with_coupling(None, 1, 0.002)

    def test_model_property_raises_not_implemented(self):
        """model property 在骨架阶段 raise NotImplementedError。"""
        gym = OrcaGymEuler()
        with self.assertRaises(NotImplementedError):
            _ = gym.model


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
        """_BLOCKED_ATTRS 是 frozenset 且包含全部 20 个拦截名（K3/K5 完整性）。"""
        self.assertIsInstance(OrcaGymEuler._BLOCKED_ATTRS, frozenset)
        expected_blocked = {
            # L3 引擎内部 (8)
            "_mjData", "_mjModel", "mj_data", "mj_model",
            "_mj_data", "_mj_model", "mjData", "mjModel",
            # K5 子组件 (12 = 6 带下划线 + 6 不带)
            "_sim", "_studio", "_registry", "_opt", "_view", "_euler",
            "sim", "studio", "registry", "opt", "view", "euler",
        }
        self.assertEqual(OrcaGymEuler._BLOCKED_ATTRS, expected_blocked)


if __name__ == "__main__":
    unittest.main()
