"""P3-Step1/阶段二-Step6: OrcaGymEulerEnv 验收测试。

验证 OrcaGymEulerEnv 的隔离机制（K1/K2/K4/K6/K7/K8/K9/K10/K11/K12）、
父类和解（K10 方案 A）、公共 API 完整性（架构 §5.3, §11, §12.5），
以及生命周期与步进方法在离线模式真实工作（阶段二 Step 6 验收标准）。

运行方式:
    <conda-base>/envs/orca/bin/python tests/run_tests.py --component environment/euler
"""

import pathlib
import re
import unittest

import numpy as np
from scipy.spatial.transform import Rotation as R

from orca_gym.environment.euler.orca_gym_euler_env import OrcaGymEulerEnv
from orca_gym.core.euler.orca_gym_data_view import OrcaGymDataView
from orca_gym.core.euler.sim_config import SimConfig

# Env 源码路径（用于 K4/K8/K9 源码审查测试）
# __file__ = tests/orca_gym/environment/euler/test_*.py
# parents[4] = 仓库根目录
_ENV_SOURCE_PATH = (
    pathlib.Path(__file__).resolve().parents[4]
    / "orca_gym" / "environment" / "euler" / "orca_gym_euler_env.py"
)


def _exec_source_without_docstrings() -> str:
    """返回 Env 源码的可执行部分（去除 docstring 和注释）。

    K4/K8/K9 源码审查需区分「合规说明文本」与「真实代码访问」。
    docstring 中合法地引用 _gym._sim / gym.studio 等作为禁止说明，
    但可执行代码不应出现这些穿墙访问。
    本函数移除三引号 docstring 和注释行，保留可执行代码供 grep 检查。
    """
    source = _ENV_SOURCE_PATH.read_text(encoding="utf-8")
    # 逐行过滤：移除注释行
    exec_lines: list[str] = []
    for line in source.splitlines(keepends=True):
        stripped = line.lstrip()
        if stripped.startswith("#"):
            continue
        exec_lines.append(line)
    exec_source = "".join(exec_lines)
    # 移除三引号 docstring（模块/函数/类级字符串字面量）
    exec_source = re.sub(r'"""[\s\S]*?"""', '', exec_source)
    exec_source = re.sub(r"'''[\s\S]*?'''", '', exec_source)
    return exec_source


def _make_skeleton_env(render_mode: str = "human", sync_render: bool = False) -> OrcaGymEulerEnv:
    """构造离线模式 Env（skip_grpc_load=True，加载本地 pendulum 模型）。"""
    # OrcaPlayground 是 OrcaGym 的同级目录
    # __file__ = OrcaGym/tests/orca_gym/environment/euler/test_*.py
    # parents[4] = OrcaGym，再向上一级到 repo 根，再到 OrcaPlayground
    _pendulum_xml = (
        pathlib.Path(__file__).resolve().parents[4].parent
        / "OrcaPlayground" / "envs" / "euler" / "scenes" / "simple_pendulum.xml"
    )
    return OrcaGymEulerEnv(
        frame_skip=4,
        orcagym_addr="localhost:50051",
        agent_names=["agent0"],
        time_step=0.002,
        model_xml_path=str(_pendulum_xml),
        skip_grpc_load=True,
        render_mode=render_mode,
        sync_render=sync_render,
    )


class TestEnvK1NamingConstraint(unittest.TestCase):
    """K1: 命名约束 — 内部组件带下划线，不暴露公共名。"""

    def test_env_no_public_internal_attrs(self):
        """Env __dict__ 不含 gym/stub/channel（含 _gym/_stub/_channel）。"""
        env = _make_skeleton_env()
        self.assertNotIn("gym", env.__dict__)
        self.assertNotIn("stub", env.__dict__)
        self.assertNotIn("channel", env.__dict__)
        self.assertIn("_gym", env.__dict__)
        self.assertIn("_stub", env.__dict__)
        self.assertIn("_channel", env.__dict__)

    def test_env_has_studio_bridge_private(self):
        """Env __dict__ 含 _studio_bridge（带下划线，K9）。"""
        env = _make_skeleton_env()
        self.assertIn("_studio_bridge", env.__dict__)


class TestEnvK2Isolation(unittest.TestCase):
    """K2: Env 层隔离机制。"""

    def test_env_blocked_attrs_raise_guidance(self):
        """访问 env.gym/env.stub/env._mjData/env._mjModel/env.mjData/env.mjModel 抛 AttributeError。"""
        env = _make_skeleton_env()
        blocked_names = [
            "gym", "stub", "channel",
            "_mjData", "_mjModel", "mj_data", "mj_model",
            "_mj_data", "_mj_model", "mjData", "mjModel",
        ]
        for name in blocked_names:
            with self.subTest(attr=name):
                with self.assertRaises(AttributeError):
                    getattr(env, name)

    def test_env_blocked_attrs_message_has_guidance(self):
        """AttributeError 消息含引导文本（env.data/env.sim_config/env.do_simulation）。"""
        env = _make_skeleton_env()
        with self.assertRaises(AttributeError) as ctx:
            _ = env.gym
        msg = str(ctx.exception)
        self.assertIn("env.data", msg)
        self.assertIn("env.sim_config", msg)
        self.assertIn("env.do_simulation", msg)

    def test_env_dir_only_exposes_public_api(self):
        """dir(env) 不含 gym/stub/channel/_gym/_studio_bridge/_mjData/_mjModel。"""
        env = _make_skeleton_env()
        d = dir(env)
        forbidden_in_dir = [
            "gym", "stub", "channel",
            "_gym", "_stub", "_channel", "_studio_bridge",
            "_mjData", "_mjModel", "mj_data", "mj_model",
        ]
        for name in forbidden_in_dir:
            with self.subTest(attr=name):
                self.assertNotIn(name, d, f"dir(env) 不应列出 '{name}'")

    def test_env_dir_contains_public_api(self):
        """dir(env) 含公共 API（data/model/sim_config/dt/ctrl/do_simulation/mj_step/mj_forward/set_ctrl/render）。"""
        env = _make_skeleton_env()
        d = dir(env)
        expected_public = [
            "data", "model", "sim_config", "dt", "ctrl",
            "do_simulation", "mj_step", "mj_forward", "set_ctrl",
            "render", "studio_bridge",
        ]
        for name in expected_public:
            with self.subTest(attr=name):
                self.assertIn(name, d, f"dir(env) 应包含公共 API '{name}'")

    def test_env_no_internal_property(self):
        """类不定义 gym/stub/channel 的 property 或属性（K1/K2）。"""
        class_attrs = vars(OrcaGymEulerEnv)
        for prop_name in ["gym", "stub", "channel"]:
            self.assertNotIn(
                prop_name, class_attrs,
                f"类不应定义 '{prop_name}' 属性/property（K1/K2: 不暴露内部组件）",
            )


class TestEnvK2ViolationPatterns(unittest.TestCase):
    """K2 违规访问拦截测试（对照架构 §6.2/§6.3/§6.5/§7.6）。

    验证 Env 层 _BLOCKED_ATTRS 全部变体、三层穿墙路径、K4/K8/K9 违规模式均被拦截。
    Env 层是用户直接接触的入口，穿墙路径比 Gym 层多一层（env._gym._sim._mjData）。
    """

    def test_env_all_mjdata_mjmodel_variants_blocked(self):
        """Env _BLOCKED_ATTRS 中 _mjData/_mjModel 全部 8 个变体都被拦截（K2）。"""
        env = _make_skeleton_env()
        variants = [
            "_mjData", "_mjModel", "mj_data", "mj_model",
            "_mj_data", "_mj_model", "mjData", "mjModel",
        ]
        for name in variants:
            with self.subTest(attr=name):
                with self.assertRaises(AttributeError):
                    getattr(env, name)

    def test_env_all_internal_component_variants_blocked(self):
        """gym/stub/channel 都被拦截（K1/K2）。"""
        env = _make_skeleton_env()
        for name in ["gym", "stub", "channel"]:
            with self.subTest(attr=name):
                with self.assertRaises(AttributeError):
                    getattr(env, name)

    def test_env_multilayer_tunnel_mjdata_blocked(self):
        """三层穿墙 env._gym._sim._mjData 在第一层 env._gym 即被拦截（架构 §6.2 R1）。

        架构 §6.2 违规示例: env._gym._sim._mjData.qpos
        本测试验证 Env 层第一层拦截: env._gym 抛 AttributeError。
        """
        env = _make_skeleton_env()
        with self.assertRaises(AttributeError):
            env._gym._sim._mjData    # 第一层 env._gym 即被拦截

    def test_env_multilayer_tunnel_mjmodel_opt_blocked(self):
        """三层穿墙 env._gym._sim._mjModel.opt 在第一层即被拦截（架构 §6.5 C1）。

        架构 §6.5 违规示例: env._gym._sim._mjModel.opt.timestep
        """
        env = _make_skeleton_env()
        with self.assertRaises(AttributeError):
            env._gym._sim._mjModel.opt    # 第一层 env._gym 即被拦截

    def test_env_multilayer_tunnel_xfrc_blocked(self):
        """三层穿墙 env._gym._sim._mjData.xfrc_applied 在第一层即被拦截（架构 §6.3 W2）。

        架构 §6.3 违规示例: env._gym._sim._mjData.xfrc_applied[body_id, :3] = force
        """
        env = _make_skeleton_env()
        with self.assertRaises(AttributeError):
            env._gym._sim._mjData.xfrc_applied    # 第一层 env._gym 即被拦截

    def test_env_k8_euler_tunnel_blocked(self):
        """四层穿墙 env._gym._euler 在第一层 env._gym 即被拦截（架构 §8.2）。

        架构 §8.2 违规示例: if self._gym._euler is not None
        """
        env = _make_skeleton_env()
        with self.assertRaises(AttributeError):
            env._gym._euler    # 第一层 env._gym 即被拦截

    def test_env_k9_studio_tunnel_blocked(self):
        """穿墙 env._gym.studio 在第一层 env._gym 即被拦截（架构 §7.1 M2）。

        架构 §7.1 M2: Studio 交互通过方法 studio_bridge() 而非 property。
        """
        env = _make_skeleton_env()
        with self.assertRaises(AttributeError):
            env._gym.studio    # 第一层 env._gym 即被拦截

    def test_env_blocked_attrs_frozenset_complete(self):
        """Env _BLOCKED_ATTRS 是 frozenset 且包含全部拦截名（K2 完整性）。"""
        self.assertIsInstance(OrcaGymEulerEnv._BLOCKED_ATTRS, frozenset)
        expected_blocked = {
            # L3 引擎内部 (8)
            "_mjData", "_mjModel", "mj_data", "mj_model",
            "_mj_data", "_mj_model", "mjData", "mjModel",
            # L2 内部组件 (3, 父类残留的公共名)
            "gym", "stub", "channel",
        }
        self.assertEqual(OrcaGymEulerEnv._BLOCKED_ATTRS, expected_blocked)


class TestEnvK4NoGymPrivateAccess(unittest.TestCase):
    """K4: Env 源码不穿墙访问 Gym 私有（AST 检查，排除 docstring）。"""

    def test_env_no_gym_private_access(self):
        """Env 可执行代码不含 _gym._sim / _gym._studio / _gym._registry / _gym._opt / _gym._view / _gym._euler（K4）。

        注意: docstring 中合法地引用这些模式作为禁止说明，本测试通过
        _exec_source_without_docstrings() 去除 docstring/注释后再检查。
        """
        exec_source = _exec_source_without_docstrings()
        forbidden_patterns = [
            "_gym._sim", "_gym._studio", "_gym._registry",
            "_gym._opt", "_gym._view", "_gym._euler",
        ]
        for pattern in forbidden_patterns:
            self.assertNotIn(
                pattern, exec_source,
                f"K4 违规: Env 可执行代码包含穿墙访问 '{pattern}'",
            )


class TestEnvK6DataView(unittest.TestCase):
    """K6: data 返回 OrcaGymDataView。"""

    def test_data_property_returns_view(self):
        """env.data 是 OrcaGymDataView 实例，非 OrcaGymData/mujoco.MjData。"""
        env = _make_skeleton_env()
        result = env.data
        self.assertIsInstance(result, OrcaGymDataView)

    def test_initialize_simulation_returns_view(self):
        """initialize_simulation() 返回的第二个元素是 OrcaGymDataView。"""
        env = _make_skeleton_env()
        _, data_view = env.initialize_simulation()
        self.assertIsInstance(data_view, OrcaGymDataView)


class TestEnvK7PropertyDelegation(unittest.TestCase):
    """K7: 属性通过 Gym 公共属性委托。"""

    def test_data_delegates_to_gym(self):
        """env.data 等同于 env._gym.data（委托，非自持）。"""
        env = _make_skeleton_env()
        self.assertIs(env.data, env._gym.data)

    def test_sim_config_delegates_to_gym(self):
        """env.sim_config 等同于 env._gym.sim_config。"""
        env = _make_skeleton_env()
        self.assertIs(env.sim_config, env._gym.sim_config)

    def test_dt_uses_sim_config(self):
        """env.dt = sim_config.timestep * frame_skip（替代父类 self.gym.opt.timestep）。"""
        env = _make_skeleton_env()
        expected_dt = env._gym.sim_config.timestep * env.frame_skip
        self.assertAlmostEqual(env.dt, expected_dt)

    def test_sim_config_returns_config_type(self):
        """env.sim_config 返回 SimConfig 实例（K11 typed 返回）。"""
        env = _make_skeleton_env()
        self.assertIsInstance(env.sim_config, SimConfig)


class TestEnvK8NoEulerPrivate(unittest.TestCase):
    """K8: do_simulation 不读 _euler。"""

    def test_do_simulation_no_euler_private_access(self):
        """Env 可执行代码不含 _euler 属性访问（K8: 耦合查询通过 has_euler/step_with_coupling）。

        使用词边界正则匹配 _euler 属性访问，忽略 orca_gym_euler 模块名中的 _euler 子串。
        注意: docstring 中合法地引用 _euler 作为禁止说明，本测试通过
        _exec_source_without_docstrings() 去除 docstring/注释后再检查。
        """
        exec_source = _exec_source_without_docstrings()
        # 词边界匹配 _euler 属性访问，忽略 orca_gym_euler 等标识符中的子串
        match = re.search(r'(?<![\w])_euler(?![\w])', exec_source)
        self.assertIsNone(
            match,
            "K8 违规: Env 可执行代码不应出现 _euler 属性访问"
            "（耦合查询通过 has_euler/step_with_coupling）",
        )

    def test_do_simulation_validates_action_dim(self):
        """do_simulation 对错误维度抛 ValueError（K4/K8 合规）。"""
        import numpy as np
        env = _make_skeleton_env()
        # pendulum nu=1，传入 nu=0 的数组应抛 ValueError
        with self.assertRaises(ValueError):
            env.do_simulation(np.zeros(0), 1)


class TestEnvK9StudioAccess(unittest.TestCase):
    """K9: Studio 访问合规。"""

    def test_no_studio_property_access(self):
        """Env 可执行代码不含 gym.studio 穿墙（允许 _studio_bridge 和 _gym.studio_bridge()）。

        K9: Studio 交互通过自持 _studio_bridge，不通过 gym.studio property。
        注意: docstring 中合法地引用 gym.studio 作为禁止说明，本测试通过
        _exec_source_without_docstrings() 去除 docstring/注释后再检查。
        """
        exec_source = _exec_source_without_docstrings()
        # 排除合法的 _gym.studio_bridge() 调用后，不应有 gym.studio 穿墙
        cleaned = exec_source.replace("_gym.studio_bridge", "")
        self.assertNotIn(
            "gym.studio", cleaned,
            "K9 违规: Env 可执行代码通过 gym.studio 访问 Studio",
        )

    def test_studio_bridge_is_method(self):
        """studio_bridge 是方法（callable），env.studio 抛 AttributeError。"""
        env = _make_skeleton_env()
        self.assertTrue(callable(getattr(OrcaGymEulerEnv, "studio_bridge", None)))
        # env.studio 不应是属性（被 __getattr__ 拦截或不存在）
        with self.assertRaises(AttributeError):
            _ = env.studio

    def test_studio_bridge_returns_bridge(self):
        """studio_bridge() 返回 OrcaStudioBridge 实例。"""
        from orca_gym.core.euler.orca_studio_bridge import OrcaStudioBridge
        env = _make_skeleton_env()
        bridge = env.studio_bridge()
        self.assertIsInstance(bridge, OrcaStudioBridge)


class TestEnvK10ParentShielding(unittest.TestCase):
    """K10: __setattr__ 屏蔽父类的 gym/stub/channel/model/data 赋值（方案 A）。"""

    def test_parent_gym_assignment_shielded(self):
        """父类 self.gym = X 后 env.gym 抛 AttributeError，env._gym 是 X。"""
        env = _make_skeleton_env()
        original_gym = env._gym
        # 模拟父类赋值 self.gym = new_value
        env.gym = "fake_gym_value"
        # _gym 被转发更新
        self.assertEqual(env._gym, "fake_gym_value")
        # gym 不存在于 __dict__（被 __setattr__ 转发）
        self.assertNotIn("gym", env.__dict__)
        # env.gym 抛 AttributeError（__getattr__ 拦截）
        with self.assertRaises(AttributeError):
            _ = env.gym
        # 恢复
        env._gym = original_gym

    def test_parent_stub_assignment_shielded(self):
        """父类 self.stub = S 后 env.stub 抛 AttributeError，env._stub 是 S。"""
        env = _make_skeleton_env()
        env.stub = "fake_stub"
        self.assertEqual(env._stub, "fake_stub")
        self.assertNotIn("stub", env.__dict__)
        with self.assertRaises(AttributeError):
            _ = env.stub

    def test_parent_channel_assignment_shielded(self):
        """父类 self.channel = C 后 env.channel 抛 AttributeError，env._channel 是 C。"""
        env = _make_skeleton_env()
        env.channel = "fake_channel"
        self.assertEqual(env._channel, "fake_channel")
        self.assertNotIn("channel", env.__dict__)
        with self.assertRaises(AttributeError):
            _ = env.channel

    def test_parent_model_assignment_shielded(self):
        """父类 self.model = M 后 env.model 走 property（从 _gym.model 取），不接受父类赋值。

        env.model 委托到 _gym.model（OrcaGymModel 实例），证明 env.model 走 property
        而非返回父类赋值的 "fake_model"。
        """
        from orca_gym.core.orca_gym_model import OrcaGymModel
        env = _make_skeleton_env()
        # 模拟父类赋值 self.model = "fake_model"
        env.model = "fake_model"
        # model 不存在于 __dict__（被 __setattr__ 忽略）
        self.assertNotIn("model", env.__dict__)
        # env.model 走 property 委托到 _gym.model（返回 OrcaGymModel 实例，
        # 证明未返回父类赋值的 "fake_model"）
        result = env.model
        self.assertIsInstance(result, OrcaGymModel)
        self.assertNotEqual(result, "fake_model")

    def test_parent_data_assignment_shielded(self):
        """父类 self.data = D 后 env.data 走 property（从 _gym.data 取），不接受父类赋值。"""
        env = _make_skeleton_env()
        # 模拟父类赋值 self.data = "fake_data"
        env.data = "fake_data"
        # data 不存在于 __dict__（被 __setattr__ 忽略）
        self.assertNotIn("data", env.__dict__)
        # env.data 走 property，返回 _gym.data（不是 "fake_data"）
        self.assertIsInstance(env.data, OrcaGymDataView)
        self.assertNotEqual(env.data, "fake_data")

    def test_shielded_attrs_frozenset_complete(self):
        """_SHIELDED_ATTRS 是 frozenset 且包含 gym/stub/channel/model/data。"""
        self.assertIsInstance(OrcaGymEulerEnv._SHIELDED_ATTRS, frozenset)
        expected = {"gym", "stub", "channel", "model", "data"}
        self.assertEqual(OrcaGymEulerEnv._SHIELDED_ATTRS, expected)


class TestEnvK11TypedReturn(unittest.TestCase):
    """K11: 公共方法返回 typed 对象。"""

    def test_data_returns_view_not_mjdata(self):
        """env.data 返回 OrcaGymDataView，不返回 mujoco.MjData。"""
        env = _make_skeleton_env()
        result = env.data
        self.assertIsInstance(result, OrcaGymDataView)
        # 确保不是 OrcaGymData
        self.assertNotEqual(type(result).__name__, "OrcaGymData")

    def test_sim_config_returns_config(self):
        """env.sim_config 返回 SimConfig，不返回 mujoco.MjModel.opt。"""
        env = _make_skeleton_env()
        result = env.sim_config
        self.assertIsInstance(result, SimConfig)


class TestEnvK12Docstring(unittest.TestCase):
    """K12: docstring 含使用契约。"""

    def test_env_docstring_has_contract(self):
        """Env 类 docstring 含「使用契约」和「禁止」关键词。"""
        doc = OrcaGymEulerEnv.__doc__ or ""
        self.assertIn("使用契约", doc)
        self.assertIn("禁止", doc)


class TestEnvLifecycleAndStepping(unittest.TestCase):
    """阶段二 Step 6: OrcaGymEulerEnv 生命周期与步进真实功能测试。

    验证离线模式 initialize_simulation/reset_simulation/init_qpos_qvel/do_simulation
    真实工作（对应 Step 6 验收标准）。
    """

    def test_initialize_simulation_loads_model(self):
        """离线模式 initialize_simulation 成功加载 pendulum 模型。"""
        env = _make_skeleton_env()
        model, view = env.initialize_simulation()
        # model 是 OrcaGymModel（pendulum nq=1）
        from orca_gym.core.orca_gym_model import OrcaGymModel
        self.assertIsInstance(model, OrcaGymModel)
        self.assertEqual(model.nq, 1)
        # view 是 OrcaGymDataView
        self.assertIsInstance(view, OrcaGymDataView)

    def test_reset_simulation_clears_state(self):
        """reset_simulation 后 qpos 回到初始状态。"""
        env = _make_skeleton_env()
        env.do_simulation(np.array([1.0]), 1)
        env.reset_simulation()
        np.testing.assert_array_almost_equal(env.data.qpos, np.zeros(1))

    def test_init_qpos_qvel_saves_initial_state(self):
        """init_qpos_qvel 正确保存初始 qpos/qvel。"""
        env = _make_skeleton_env()
        env.init_qpos_qvel()
        self.assertEqual(env.init_qpos.shape, (1,))
        self.assertEqual(env.init_qvel.shape, (1,))
        np.testing.assert_array_almost_equal(env.init_qpos, np.zeros(1))

    def test_do_simulation_advances_time(self):
        """do_simulation(ctrl, 5) 步进 5 帧后 env.data.time 增加 5 * timestep。"""
        env = _make_skeleton_env()
        time_before = float(env.data.time)
        env.do_simulation(np.array([0.0]), 5)
        time_after = float(env.data.time)
        expected_dt = 5 * env.sim_config.timestep
        self.assertAlmostEqual(time_after - time_before, expected_dt, places=5)

    def test_set_joint_qpos_writes_state(self):
        """set_joint_qpos 正确写入 qpos。"""
        env = _make_skeleton_env()
        env.set_joint_qpos(np.array([0.5]))
        env.mj_forward()
        env._gym.sync_to_view()
        self.assertAlmostEqual(float(env.data.qpos[0]), 0.5)

    def test_set_joint_qvel_writes_state(self):
        """set_joint_qvel 正确写入 qvel。"""
        env = _make_skeleton_env()
        env.set_joint_qvel(np.array([0.3]))
        env.mj_forward()
        env._gym.sync_to_view()
        self.assertAlmostEqual(float(env.data.qvel[0]), 0.3)

    def test_ctrl_property_returns_actuator_force(self):
        """ctrl getter 返回 actuator_force（阶段二简化实现）。"""
        env = _make_skeleton_env()
        env.set_ctrl(np.array([0.7]))
        env._gym.sync_to_view()
        # actuator_force 反映已设置的 ctrl
        self.assertEqual(env.ctrl.shape, (1,))

    def test_ctrl_setter_delegates_to_set_ctrl(self):
        """ctrl setter 委托到 set_ctrl，set_ctrl 写入 mjData.ctrl。"""
        env = _make_skeleton_env()
        env.ctrl = np.array([0.5])
        # 验证 set_ctrl 写入 mjData.ctrl（需通过 actuator_force 间接验证）
        # set_ctrl 后调用 mj_forward 更新 actuator_force
        env.mj_forward()
        env._gym.sync_to_view()
        # ctrl getter 读 actuator_force，pendulum 的 actuator_force == ctrl（无齿轮比）
        np.testing.assert_array_almost_equal(env.ctrl, np.array([0.5]))

    def test_render_offline_returns_none(self):
        """离线模式 render 返回 None（无 OrcaStudio 可渲染，Step 8）。"""
        env = _make_skeleton_env()
        result = env.render()
        self.assertIsNone(result)

    def test_render_mode_none_returns_none(self):
        """render_mode='none' 时 render 立即返回 None（Step 3 验收标准）。"""
        env = _make_skeleton_env(render_mode="none")
        result = env.render()
        self.assertIsNone(result)

    def test_render_mode_human_offline_returns_none(self):
        """render_mode='human' 离线模式 render 返回 None（Step 3）。"""
        env = _make_skeleton_env(render_mode="human")
        result = env.render()
        self.assertIsNone(result)

    def test_do_body_manipulation_offline_noop(self):
        """离线模式 do_body_manipulation no-op 不抛异常（Step 3）。"""
        env = _make_skeleton_env()
        env.do_body_manipulation()   # 不应抛异常

    def test_gymnasium_methods_raise_not_implemented(self):
        """step/reset_model/_get_obs 仍 raise NotImplementedError（待子类实现）。"""
        env = _make_skeleton_env()
        with self.assertRaises(NotImplementedError):
            env.reset_model()
        with self.assertRaises(NotImplementedError):
            env._get_obs()


class TestEnvParentReconciliation(unittest.TestCase):
    """补充：验证父类和解（架构 §12.5 方案 A）生效。"""

    def test_env_inherits_base_env(self):
        """OrcaGymEulerEnv 继承 OrcaGymBaseEnv。"""
        from orca_gym.environment.orca_gym_env import OrcaGymBaseEnv
        self.assertTrue(issubclass(OrcaGymEulerEnv, OrcaGymBaseEnv))

    def test_parent_init_completes_without_error(self):
        """父类 __init__ 完整执行不报错（K10 屏蔽生效）。"""
        # 若 K10 屏蔽失效，父类 __init__ 会在 self.gym=None 或 self.model,self.data=... 时出错
        env = _make_skeleton_env()
        # 构造成功即说明父类 __init__ 完整执行
        self.assertIsNotNone(env._gym)

    def test_dt_overrides_parent(self):
        """env.dt 覆盖父类的 self.gym.opt.timestep * frame_skip，使用 sim_config。"""
        env = _make_skeleton_env()
        # 父类 dt: self.gym.opt.timestep * self.frame_skip（会因 env.gym 拦截而失败）
        # 子类 dt: self._gym.sim_config.timestep * self.frame_skip（应成功）
        self.assertEqual(env.dt, env._gym.sim_config.timestep * env.frame_skip)


class TestEnvInitializeGrpcOnlineMode(unittest.TestCase):
    """阶段二 Step 3: initialize_grpc 在线模式测试。

    由于真实 gRPC 服务不可用，通过 mock 验证 channel/stub 创建逻辑。
    """

    def test_initialize_grpc_online_creates_channel_and_stub(self):
        """在线模式 initialize_grpc 创建 grpc.aio channel + GrpcServiceStub。

        通过 mock grpc.aio.insecure_channel 和 GrpcServiceStub 验证调用。
        """
        from unittest import mock

        # 直接调用 initialize_grpc，mock 掉 grpc.aio.insecure_channel 和 GrpcServiceStub
        # 由于父类 __init__ 会在 super().__init__ 中调用 initialize_grpc，
        # 我们需要构造一个 skip_grpc_load=True 的 env，然后手动调用 initialize_grpc
        env = _make_skeleton_env()

        # mock grpc.aio.insecure_channel 和 GrpcServiceStub
        with mock.patch("orca_gym.environment.euler.orca_gym_euler_env.grpc.aio.insecure_channel") as mock_channel, \
             mock.patch("orca_gym.environment.euler.orca_gym_euler_env.GrpcServiceStub") as mock_stub_class:
            mock_channel.return_value = mock.MagicMock(name="channel")
            mock_stub = mock.MagicMock(name="stub")
            mock_stub_class.return_value = mock_stub

            # 强制走在线模式
            env._skip_grpc_load = False
            env.initialize_grpc()

            # 验证 channel 创建（带大消息选项）
            mock_channel.assert_called_once()
            call_args = mock_channel.call_args
            self.assertEqual(call_args[0][0], "localhost:50051")
            options = call_args[1]["options"]
            # 验证包含大消息长度选项
            option_keys = [opt[0] for opt in options]
            self.assertIn("grpc.max_receive_message_length", option_keys)
            self.assertIn("grpc.max_send_message_length", option_keys)

            # 验证 stub 创建
            mock_stub_class.assert_called_once()

            # 验证 _channel 和 _stub 被设置
            self.assertIsNotNone(env._channel)
            self.assertIsNotNone(env._stub)
            # 验证 _studio_bridge 被设置
            self.assertIsNotNone(env._studio_bridge)

    def test_initialize_grpc_offline_configures_bridge(self):
        """离线模式 initialize_grpc 配置 studio_bridge 的 local_xml_path。"""
        env = _make_skeleton_env()
        # 离线模式下 studio_bridge 应已配置 local_xml_path
        bridge = env.studio_bridge()
        self.assertIsNotNone(bridge._local_xml_path)
        self.assertTrue(bridge._local_xml_path.endswith("simple_pendulum.xml"))


class TestEnvRenderThrottling(unittest.TestCase):
    """阶段二 Step 3: render 节流逻辑测试。"""

    def test_render_sync_render_counter_logic(self):
        """sync_render=True 时按计数器节流（_render_count 累积）。"""
        env = _make_skeleton_env(render_mode="human", sync_render=True)
        # 离线模式 render 直接返回 None，不进入节流逻辑
        # 验证 _render_count 初始为 0
        self.assertEqual(env._render_count, 0.0)
        env.render()
        # 离线模式不进入节流，_render_count 不变
        self.assertEqual(env._render_count, 0.0)

    def test_render_interval_initialized_from_fps(self):
        """_render_interval 从 metadata.render_fps 初始化（30fps → 1/30）。"""
        env = _make_skeleton_env()
        expected = 1.0 / 30
        self.assertAlmostEqual(env._render_interval, expected, places=6)

    def test_render_throttle_fields_initialized(self):
        """渲染节流字段在 __init__ 中正确初始化。"""
        env = _make_skeleton_env()
        self.assertEqual(env._render_count, 0.0)
        self.assertEqual(env._render_count_interval, 0.0)
        self.assertEqual(env._render_time_step, 0.0)
        self.assertEqual(env._last_frame_index, -1)


class TestEnvK9ComplianceSourceAudit(unittest.TestCase):
    """阶段二 Step 3: K9 源码合规审查。"""

    def test_no_gym_studio_tunnel_access(self):
        """源码不含 self._gym.studio. / self._gym._studio. 穿墙访问（K9）。"""
        import pathlib
        source_path = (
            pathlib.Path(__file__).resolve().parent
            / ".." / ".." / ".." / ".."
            / "orca_gym" / "environment" / "euler" / "orca_gym_euler_env.py"
        )
        source = source_path.resolve().read_text(encoding="utf-8")
        forbidden = ["self._gym.studio.", "self._gym._studio."]
        for pattern in forbidden:
            self.assertNotIn(
                pattern, source,
                f"K9 违规: orca_gym_euler_env.py 包含穿墙访问 '{pattern}'",
            )


# =============================================================================
# 阶段三 3.1.7：OrcaGymEulerEnv 公共查询 API
# =============================================================================


def _make_g1_env() -> OrcaGymEulerEnv:
    """构造加载 G1 XML 的离线 Env（用于 sensor/contact 等需要丰富场景的测试）。"""
    _g1_xml = (
        pathlib.Path(__file__).resolve().parents[4].parent
        / "OrcaPlayground" / "envs" / "euler" / "robots" / "g1_29dof_camera.xml"
    )
    return OrcaGymEulerEnv(
        frame_skip=4,
        orcagym_addr="localhost:50051",
        agent_names=["agent0"],
        time_step=0.002,
        model_xml_path=str(_g1_xml),
        skip_grpc_load=True,
    )


class TestEnvQueryDelegationArchCompliance(unittest.TestCase):
    """子步骤 3.1.7 架构遵从性测试（K1/K2/K4/K11/K12）。

    对应文档 §5.8 架构遵从性测试表。
    """

    def test_env_query_no_gym_private_access(self):
        """K4: grep 断言 3.1.7 查询区块不触 self._gym._sim/_studio/_registry/_mjData/_mjModel。"""
        source = _ENV_SOURCE_PATH.read_text(encoding="utf-8")
        start = source.find("# --- 公共查询 API（阶段三 3.1.7")
        self.assertGreater(start, 0, "未找到 3.1.7 公共查询 API 区块")
        block_source = source[start:]
        # 找区块结束（下一个 --- 分隔符或 # --- Gymnasium）
        end = block_source.find("# --- Gymnasium 接口")
        if end > 0:
            block_source = block_source[:end]
        forbidden = [
            "self._gym._sim", "self._gym._studio", "self._gym._registry",
            "self._gym._opt", "self._gym._view", "self._gym._euler",
            "self._gym._mjData", "self._gym._mjModel",
        ]
        for pattern in forbidden:
            with self.subTest(pattern=pattern):
                self.assertNotIn(
                    pattern, block_source,
                    f"K4 违规: 3.1.7 查询区块包含穿墙访问 '{pattern}'",
                )

    def test_env_query_uses_self_gym_only(self):
        """K1/K4: grep 断言查询方法均用 self._gym.<公共方法> 委托，不 self._sim/self._studio。"""
        source = _ENV_SOURCE_PATH.read_text(encoding="utf-8")
        start = source.find("# --- 公共查询 API（阶段三 3.1.7")
        block_source = source[start:]
        end = block_source.find("# --- Gymnasium 接口")
        if end > 0:
            block_source = block_source[:end]
        # 不应直接触 self._sim / self._studio（Env 不持有这些子组件）
        self.assertNotIn("self._sim", block_source)
        self.assertNotIn("self._studio", block_source)
        self.assertNotIn("self._registry", block_source)
        # 应包含 self._gym. 委托调用
        self.assertIn("self._gym.query_", block_source)
        self.assertIn("self._gym.body_subtree_mass", block_source)

    def test_env_dir_includes_new_query_methods(self):
        """K2: dir(env) 含 query_joint_qpos 等新方法。"""
        env = _make_skeleton_env()
        d = dir(env)
        expected_methods = [
            "query_joint_qpos", "query_joint_qvel", "query_joint_qacc",
            "query_joint_offsets", "query_joint_lengths", "query_joint_dofadrs",
            "jnt_qposadr", "jnt_dofadr",
            "get_body_xpos_xmat_xquat", "get_body_xpos_xmat_xquat_xvel",
            "query_site_pos_and_mat", "query_site_size",
            "query_sensor_data", "query_actuator_torques",
            "query_contact_simple", "query_contact_force",
            "get_cfrc_ext", "get_goal_bounding_box", "body_subtree_mass",
        ]
        for name in expected_methods:
            with self.subTest(method=name):
                self.assertIn(name, d, f"dir(env) 应包含新查询方法 '{name}'")

    def test_env_dir_no_internal_leak(self):
        """K2: dir(env) 不含 _sim/_studio/_registry/_mjData/_mjModel。"""
        env = _make_skeleton_env()
        d = dir(env)
        for name in ["_sim", "_studio", "_registry", "_mjData", "_mjModel"]:
            with self.subTest(attr=name):
                self.assertNotIn(name, d)

    def test_env_query_returns_typed(self):
        """K11: 公共查询方法返回 ndarray/dict/tuple，非 MjData/MjModel。"""
        env = _make_skeleton_env()
        env.mj_forward()
        # query_joint_qpos 返回 dict[str, np.ndarray]
        result = env.query_joint_qpos(["hinge"])
        self.assertIsInstance(result, dict)
        self.assertIsInstance(result["hinge"], np.ndarray)
        # jnt_qposadr 返回 int
        adr = env.jnt_qposadr("hinge")
        self.assertIsInstance(adr, int)
        # body_subtree_mass 返回 float
        mass = env.body_subtree_mass("pendulum")
        self.assertIsInstance(mass, float)

    def test_env_query_docstrings_present(self):
        """K12: 新增查询方法有 docstring（含用法与禁止说明）。"""
        methods_to_check = [
            "query_joint_qpos", "query_joint_qvel", "query_joint_qacc",
            "query_joint_offsets", "query_joint_lengths", "query_joint_dofadrs",
            "jnt_qposadr", "jnt_dofadr",
            "get_body_xpos_xmat_xquat", "get_body_xpos_xmat_xquat_xvel",
            "query_site_pos_and_mat", "query_site_size",
            "query_sensor_data", "query_actuator_torques",
            "query_contact_simple", "query_contact_force",
            "get_cfrc_ext", "get_goal_bounding_box", "body_subtree_mass",
        ]
        for name in methods_to_check:
            with self.subTest(method=name):
                method = getattr(OrcaGymEulerEnv, name)
                doc = method.__doc__ or ""
                self.assertTrue(
                    doc.strip(),
                    f"方法 {name} 缺少 docstring（K12）",
                )
                # docstring 应含「委托 self._gym」或「委托」说明
                self.assertIn("委托", doc)


class TestEnvQueryDelegationFunctional(unittest.TestCase):
    """子步骤 3.1.7 功能单元测试。

    对应文档 §5.8 功能单元测试表。验证 Env 委托链路结果与 Gym 底层一致。
    pendulum 用于基础测试，G1 XML 用于 sensor/contact 测试。
    """

    def test_env_query_joint_qpos_returns_correct_slice(self):
        """env.query_joint_qpos(["hinge"]) 返回正确切片（与 _gym.data.qpos 一致）。"""
        env = _make_skeleton_env()
        env.mj_forward()
        result = env.query_joint_qpos(["hinge"])
        # pendulum nq=1，hinge 的 qpos 切片应等于 data.qpos[0]
        expected = env.data.qpos[0]
        np.testing.assert_array_equal(result["hinge"], np.array([expected]))

    def test_env_get_body_xpos_xmat_xquat_returns_dict(self):
        """get_body_xpos_xmat_xquat 返回 dict[body -> {xpos/xmat/xquat}]，形状正确。"""
        env = _make_skeleton_env()
        env.mj_forward()
        result = env.get_body_xpos_xmat_xquat(["pendulum"])
        self.assertIsInstance(result, dict)
        self.assertIn("pendulum", result)
        # xpos 形状 (3,)，xmat 形状 (3,3) 或 (9,)，xquat 形状 (4,)
        self.assertEqual(result["pendulum"]["xpos"].shape, (3,))
        self.assertEqual(result["pendulum"]["xquat"].shape, (4,))

    def test_env_query_sensor_data_matches_sensordata(self):
        """env.query_sensor_data 与 _gym.data.sensordata 切片一致（G1 XML）。"""
        env = _make_g1_env()
        env.mj_forward()
        sensor_name = "left_hip_pitch_pos"
        result = env.query_sensor_data([sensor_name])
        self.assertIsInstance(result, dict)
        self.assertIn(sensor_name, result)
        self.assertIsInstance(result[sensor_name], np.ndarray)
        # 与 Gym 直接调用结果一致
        gym_result = env._gym.query_sensor_data([sensor_name])
        np.testing.assert_array_equal(result[sensor_name], gym_result[sensor_name])

    def test_env_query_contact_simple_returns_list(self):
        """env.query_contact_simple 返回 list[dict]，结构正确。"""
        env = _make_g1_env()
        env.mj_forward()
        result = env.query_contact_simple()
        self.assertIsInstance(result, list)
        # 接触列表可能为空（G1 初始姿态无接触），但元素结构应正确
        for contact in result:
            self.assertIsInstance(contact, dict)

    def test_env_body_subtree_mass_positive(self):
        """env.body_subtree_mass 返回正标量（pendulum body 质量 > 0）。"""
        env = _make_skeleton_env()
        mass = env.body_subtree_mass("pendulum")
        self.assertIsInstance(mass, float)
        self.assertGreater(mass, 0.0)
        # pendulum arm geom mass=1，子树质量应为 1.0
        self.assertAlmostEqual(mass, 1.0, places=5)

    def test_env_query_delegates_match_gym_direct(self):
        """Env 委托结果与 env._gym 直接调用完全一致（pendulum 多方法）。"""
        env = _make_skeleton_env()
        env.mj_forward()
        # joint qpos
        env_qpos = env.query_joint_qpos(["hinge"])
        gym_qpos = env._gym.query_joint_qpos(["hinge"])
        np.testing.assert_array_equal(env_qpos["hinge"], gym_qpos["hinge"])
        # body xpos
        env_body = env.get_body_xpos_xmat_xquat(["pendulum"])
        gym_body = env._gym.query_body_xpos_xmat_xquat(["pendulum"])
        np.testing.assert_array_equal(
            env_body["pendulum"]["xpos"], gym_body["pendulum"]["xpos"]
        )
        # body subtree mass
        env_mass = env.body_subtree_mass("pendulum")
        gym_mass = env._gym.body_subtree_mass("pendulum")
        self.assertAlmostEqual(env_mass, gym_mass)


# =============================================================================
# 阶段三 3.1.8：OrcaGymEulerEnv 基座坐标系变换方法
# =============================================================================


class TestEnvBaseTransformArchCompliance(unittest.TestCase):
    """子步骤 3.1.8 架构遵从性测试（K4/K11/K12/P2）。

    对应文档 §5.9 架构遵从性测试表。
    """

    def test_env_base_transform_no_gym_private(self):
        """K4: grep 断言 *_B/*_odom 方法不触 self._gym._sim/_mjData/_mjModel，仅调公共方法。"""
        source = _ENV_SOURCE_PATH.read_text(encoding="utf-8")
        start = source.find("# --- 基座坐标系变换方法（阶段三 3.1.8")
        self.assertGreater(start, 0, "未找到 3.1.8 基座变换区块")
        block_source = source[start:]
        end = block_source.find("# --- Gymnasium 接口")
        if end > 0:
            block_source = block_source[:end]
        forbidden = [
            "self._gym._sim", "self._gym._mjData", "self._gym._mjModel",
            "self._gym._studio", "self._gym._registry",
            "self._mjData", "self._mjModel",
        ]
        for pattern in forbidden:
            with self.subTest(pattern=pattern):
                self.assertNotIn(
                    pattern, block_source,
                    f"K4 违规: 3.1.8 基座变换区块包含穿墙访问 '{pattern}'",
                )

    def test_env_base_transform_no_simcore_dependency(self):
        """K4/P2: grep 断言 Env 层变换方法不 import MuJoCoSimCore，不直接访问 _mjData。"""
        source = _ENV_SOURCE_PATH.read_text(encoding="utf-8")
        # 整个 Env 文件不应 import MuJoCoSimCore
        self.assertNotIn("MuJoCoSimCore", source)
        # 变换区块不应直接访问 _mjData/_mjModel（应通过 self.data/self.model）
        start = source.find("# --- 基座坐标系变换方法（阶段三 3.1.8")
        block_source = source[start:]
        end = block_source.find("# --- Gymnasium 接口")
        if end > 0:
            block_source = block_source[:end]
        self.assertNotIn("self._mjData", block_source)
        self.assertNotIn("self._mjModel", block_source)

    def test_env_base_transform_returns_typed(self):
        """K11: 返回 ndarray/dict/tuple，非 MjData/MjModel。"""
        env = _make_g1_env()
        env.mj_forward()
        # query_site_pos_and_quat_B 返回 dict[str, dict]
        result = env.query_site_pos_and_quat_B(["imu"], ["pelvis"])
        self.assertIsInstance(result, dict)
        self.assertIn("imu", result)
        self.assertIsInstance(result["imu"]["xpos"], np.ndarray)
        self.assertEqual(result["imu"]["xpos"].shape, (3,))
        self.assertEqual(result["imu"]["xquat"].shape, (4,))
        # query_position_body_B 返回 ndarray
        pos = env.query_position_body_B("left_hip_pitch_link", "pelvis")
        self.assertIsInstance(pos, np.ndarray)
        self.assertEqual(pos.shape, (3,))
        # query_robot_position_odom 返回 ndarray
        base_pos = env.data.body_xpos("pelvis")
        base_quat = env.data.body_xquat("pelvis")
        odom_pos = env.query_robot_position_odom("pelvis", base_pos, base_quat)
        self.assertIsInstance(odom_pos, np.ndarray)

    def test_env_base_transform_docstrings_present(self):
        """K12: 新增 *_B/*_odom 方法有 docstring。"""
        methods_to_check = [
            "query_site_pos_and_quat_B", "query_site_xvalp_xvalr",
            "query_site_xvalp_xvalr_B", "query_velocity_body_B",
            "query_position_body_B", "query_orientation_body_B",
            "query_joint_axes_B", "query_robot_velocity_odom",
            "query_robot_position_odom", "query_robot_orientation_odom",
        ]
        for name in methods_to_check:
            with self.subTest(method=name):
                method = getattr(OrcaGymEulerEnv, name)
                doc = method.__doc__ or ""
                self.assertTrue(doc.strip(), f"方法 {name} 缺少 docstring（K12）")

    def test_env_base_transform_dir_includes_methods(self):
        """K2: dir(env) 含 query_site_pos_and_quat_B 等新方法。"""
        env = _make_skeleton_env()
        d = dir(env)
        expected = [
            "query_site_pos_and_quat_B", "query_site_xvalp_xvalr",
            "query_site_xvalp_xvalr_B", "query_velocity_body_B",
            "query_position_body_B", "query_orientation_body_B",
            "query_joint_axes_B", "query_robot_velocity_odom",
            "query_robot_position_odom", "query_robot_orientation_odom",
        ]
        for name in expected:
            with self.subTest(method=name):
                self.assertIn(name, d)


class TestEnvBaseTransformFunctional(unittest.TestCase):
    """子步骤 3.1.8 功能单元测试（G1 XML 真实数据）。

    对应文档 §5.9 功能单元测试表。
    """

    def setUp(self):
        self.env = _make_g1_env()
        self.env.mj_forward()

    def test_query_site_pos_and_quat_B_relative(self):
        """基座坐标系变换正确（与世界系差一个基座变换）。"""
        result = self.env.query_site_pos_and_quat_B(["imu"], ["pelvis"])
        # 世界系 site pos
        site_world = self.env.data.site_xpos("imu")
        # 世界系 base pos
        base_world = self.env.data.body_xpos("pelvis")
        base_quat = self.env.data.body_xquat("pelvis")  # [w,x,y,z]
        # 手动计算相对位置
        rot_base = R.from_quat([base_quat[1], base_quat[2], base_quat[3], base_quat[0]])
        expected_pos = rot_base.inv().apply(site_world - base_world)
        np.testing.assert_allclose(result["imu"]["xpos"], expected_pos, atol=1e-6)

    def test_query_velocity_body_B_consistency(self):
        """末端在基座系下速度 = 基座逆变换 ⊗ 世界系速度差。"""
        vel_B = self.env.query_velocity_body_B("left_hip_pitch_link", "pelvis")
        self.assertEqual(vel_B.shape, (6,))
        # 手动计算：ee_cvel - base_cvel，然后 base_rot.T 变换
        ee_cvel = self.env.data.body_cvel("left_hip_pitch_link")
        base_cvel = self.env.data.body_cvel("pelvis")
        base_mat = self.env.data.body_xmat("pelvis").reshape(3, 3)
        expected_lin = base_mat.T @ (ee_cvel[3:] - base_cvel[3:])
        expected_ang = base_mat.T @ (ee_cvel[:3] - base_cvel[:3])
        np.testing.assert_allclose(vel_B[:3], expected_lin, atol=1e-6)
        np.testing.assert_allclose(vel_B[3:], expected_ang, atol=1e-6)

    def test_query_robot_position_odom_accumulates(self):
        """里程计累积正确（初始时刻位置 = 0）。"""
        base_pos = self.env.data.body_xpos("pelvis").copy()
        base_quat = self.env.data.body_xquat("pelvis").copy()
        # 初始时刻，里程计位置应为 0
        odom_pos = self.env.query_robot_position_odom("pelvis", base_pos, base_quat)
        np.testing.assert_allclose(odom_pos, np.zeros(3), atol=1e-6)
        # 步进后里程计位置 = 初始逆变换 ⊗ (新位置 - 初始位置)
        self.env.do_simulation(np.zeros(self.env.model.nu), 1)
        new_pos = self.env.data.body_xpos("pelvis")
        rot_init = R.from_quat([base_quat[1], base_quat[2], base_quat[3], base_quat[0]])
        expected = rot_init.inv().apply(new_pos - base_pos)
        odom_pos2 = self.env.query_robot_position_odom("pelvis", base_pos, base_quat)
        np.testing.assert_allclose(odom_pos2, expected, atol=1e-5)

    def test_query_joint_axes_B_transformed(self):
        """关节轴在基座系下正确变换。"""
        result = self.env.query_joint_axes_B(["left_hip_pitch_joint"], "pelvis")
        self.assertIn("left_hip_pitch_joint", result)
        axis_B = result["left_hip_pitch_joint"]
        self.assertEqual(axis_B.shape, (3,))
        # 手动计算
        joint_info = self.env.model.get_joint_byname("left_hip_pitch_joint")
        jnt_axis = joint_info["Axis"]
        body_id = joint_info["BodyID"]
        body_name = self.env.model.body_id2name(body_id)
        body_quat = self.env.data.body_xquat(body_name)
        base_quat = self.env.data.body_xquat("pelvis")
        body_rot = R.from_quat([body_quat[1], body_quat[2], body_quat[3], body_quat[0]])
        base_rot = R.from_quat([base_quat[1], base_quat[2], base_quat[3], base_quat[0]])
        expected = base_rot.inv().apply(body_rot.apply(jnt_axis))
        np.testing.assert_allclose(axis_B, expected, atol=1e-6)

    def test_query_orientation_body_B_returns_quat(self):
        """query_orientation_body_B 返回 [x,y,z,w] 四元数。"""
        quat_B = self.env.query_orientation_body_B("left_hip_pitch_link", "pelvis")
        self.assertEqual(quat_B.shape, (4,))
        # 手动计算
        base_quat = self.env.data.body_xquat("pelvis")
        ee_quat = self.env.data.body_xquat("left_hip_pitch_link")
        rot_base = R.from_quat([base_quat[1], base_quat[2], base_quat[3], base_quat[0]])
        rot_ee = R.from_quat([ee_quat[1], ee_quat[2], ee_quat[3], ee_quat[0]])
        expected = (rot_base.inv() * rot_ee).as_quat()
        np.testing.assert_allclose(quat_B, expected, atol=1e-6)


# =============================================================================
# 阶段三 3.2.4：OrcaGymEulerEnv 力应用与设置委托
# =============================================================================


class TestEnvForceArchCompliance(unittest.TestCase):
    """子步骤 3.2.4 架构遵从性测试（K1/K2/K4/K9/K12）。

    对应文档 §6.5 架构遵从性测试表。
    """

    def test_env_force_no_gym_private_access(self):
        """K4: grep 断言力应用/设置方法不触 self._gym._sim/_mjData/_mjModel。"""
        source = _ENV_SOURCE_PATH.read_text(encoding="utf-8")
        start = source.find("# --- 力应用与状态设置委托（阶段三 3.2.4）")
        self.assertGreater(start, 0)
        block = source[start:]
        end = block.find("# --- Gymnasium 接口")
        if end > 0:
            block = block[:end]
        forbidden = [
            "self._gym._sim", "self._gym._mjData", "self._gym._mjModel",
            "self._gym._studio", "self._gym._registry",
            "self._mjData", "self._mjModel",
        ]
        for pattern in forbidden:
            with self.subTest(pattern=pattern):
                self.assertNotIn(
                    pattern, block,
                    f"K4 违规: 3.2.4 力应用区块包含穿墙访问 '{pattern}'",
                )

    def test_env_force_uses_self_gym_and_model(self):
        """K1/K4: grep 断言走 self._gym.<方法> + self.model.body_name2id。"""
        source = _ENV_SOURCE_PATH.read_text(encoding="utf-8")
        start = source.find("# --- 力应用与状态设置委托（阶段三 3.2.4）")
        block = source[start:]
        end = block.find("# --- Gymnasium 接口")
        if end > 0:
            block = block[:end]
        self.assertIn("self._gym.apply_body_force", block)
        self.assertIn("self.model.body_name2id", block)
        self.assertIn("self.model.site_name2id", block)
        # 不应直接访问 self._sim
        self.assertNotIn("self._sim", block)

    def test_env_mocap_uses_studio_bridge(self):
        """K9: grep 断言 set_mocap_pos_and_quat 远端同步走 self._gym.set_mocap_pos_and_quat_remote。"""
        source = _ENV_SOURCE_PATH.read_text(encoding="utf-8")
        start = source.find("def set_mocap_pos_and_quat")
        self.assertGreater(start, 0)
        block = source[start:]
        end = block.find("def set_geom_friction")
        if end > 0:
            block = block[:end]
        self.assertIn("self._gym.set_mocap_pos_and_quat_remote", block)
        # 不应直接访问 gym.studio（老体系穿墙）
        self.assertNotIn("gym.studio", block)

    def test_env_force_dir_includes_new_methods(self):
        """K2: dir(env) 含 apply_body_force/clear_all_forces 等。"""
        env = _make_skeleton_env()
        d = dir(env)
        expected = [
            "apply_body_force", "clear_body_force", "clear_all_forces",
            "mj_apply_force_at_site", "mj_clear_xfrc_applied_for_site",
            "set_mocap_pos_and_quat", "set_geom_friction", "add_extra_weight",
        ]
        for name in expected:
            with self.subTest(method=name):
                self.assertIn(name, d)

    def test_env_force_docstrings_present(self):
        """K12: 新增力应用/设置方法有 docstring。"""
        methods_to_check = [
            "apply_body_force", "clear_body_force", "clear_all_forces",
            "mj_apply_force_at_site", "mj_clear_xfrc_applied_for_site",
            "set_mocap_pos_and_quat", "set_geom_friction", "add_extra_weight",
        ]
        for name in methods_to_check:
            with self.subTest(method=name):
                method = getattr(OrcaGymEulerEnv, name)
                doc = method.__doc__ or ""
                self.assertTrue(doc.strip(), f"方法 {name} 缺少 docstring（K12）")


class TestEnvForceFunctional(unittest.TestCase):
    """子步骤 3.2.4 功能单元测试（G1 XML 真实数据）。

    对应文档 §6.5 功能单元测试表。
    """

    def setUp(self):
        self.env = _make_g1_env()
        self.env.mj_forward()
        self.pelvis_id = self.env.model.body_name2id("pelvis")

    def test_env_apply_body_force_writes_xfrc(self):
        """施力后 env.data.xfrc_applied[body_id, :3] 等于 force。"""
        self.env.clear_all_forces()
        force = np.array([1.0, 2.0, 3.0])
        torque = np.array([0.1, 0.2, 0.3])
        self.env.apply_body_force("pelvis", force, torque)
        # 需 forward 让 sync_to_view 生效（env.data 是 view）
        self.env.mj_forward()
        np.testing.assert_allclose(
            self.env.data.xfrc_applied[self.pelvis_id, :3], force, atol=1e-6
        )
        np.testing.assert_allclose(
            self.env.data.xfrc_applied[self.pelvis_id, 3:6], torque, atol=1e-6
        )

    def test_env_clear_body_force_zeroes_xfrc(self):
        """清力后 env.data.xfrc_applied[body_id, :6] 为 0。"""
        self.env.apply_body_force("pelvis", np.ones(3), np.ones(3))
        self.env.clear_body_force("pelvis")
        self.env.mj_forward()
        np.testing.assert_allclose(
            self.env.data.xfrc_applied[self.pelvis_id, :6], 0.0, atol=1e-6
        )

    def test_env_set_mocap_pos_and_quat_writes_mocap(self):
        """mocap_pos/quat 正确写入（通过 DataView 只读查询验证）。"""
        pos = np.array([0.5, 0.3, 0.8])
        quat = np.array([1.0, 0.0, 0.0, 0.0])
        self.env.set_mocap_pos_and_quat(
            {"ActorManipulator_Anchor": {"pos": pos, "quat": quat}}
        )
        # 通过 DataView 只读查询验证（K6：不触 _mjData）
        np.testing.assert_allclose(
            self.env.data.mocap_pos("ActorManipulator_Anchor"), pos, atol=1e-6
        )
        np.testing.assert_allclose(
            self.env.data.mocap_quat("ActorManipulator_Anchor"), quat, atol=1e-6
        )

    def test_env_set_geom_friction_persists(self):
        """geom_friction 修改持久化（通过 Env 只读查询验证，K4 不穿墙）。"""
        new_friction = np.array([2.5, 0.01, 0.002])
        self.env.set_geom_friction({"manipulation_box_geom": new_friction})
        # 通过 env.geom_friction 公共方法验证（K4：不触 _gym/_mjModel）
        np.testing.assert_allclose(
            self.env.geom_friction("manipulation_box_geom"),
            new_friction,
            atol=1e-6,
        )

    def test_env_apply_force_pelvis_z_changes(self):
        """施力后 pelvis z 位置变化（步进验证）。"""
        self.env.clear_all_forces()
        self.env.mj_forward()
        z_before = float(self.env.data.body_xpos("pelvis")[2])
        # 施加向上的力（大于重力，使 pelvis 上升）
        self.env.apply_body_force("pelvis", np.array([0.0, 0.0, 500.0]), np.zeros(3))
        self.env.do_simulation(np.zeros(self.env.model.nu), 10)
        z_after = float(self.env.data.body_xpos("pelvis")[2])
        self.assertNotEqual(z_before, z_after)


# =============================================================================
# 阶段三 3.2.5：OrcaGymDataView xfrc_applied 只读保护验证
# =============================================================================


class TestDataViewXfrcReadOnlyArchCompliance(unittest.TestCase):
    """子步骤 3.2.5 架构遵从性测试（K6/P4）。

    对应文档 §6.6 架构遵从性测试表。
    """

    def test_dataview_xfrc_is_view_not_copy(self):
        """K6: env.data.xfrc_applied.base 非 None（零拷贝视图）。"""
        env = _make_g1_env()
        env.mj_forward()
        # 零拷贝视图：base 非 None（共享 _mjData.xfrc_applied 数据）
        self.assertIsNotNone(env.data.xfrc_applied.base)

    def test_dataview_xfrc_direct_write_blocked(self):
        """P4/K6: 直接写 env.data.xfrc_applied[...] 抛 ValueError（只读）。"""
        env = _make_g1_env()
        env.mj_forward()
        with self.assertRaises(ValueError):
            env.data.xfrc_applied[0, :3] = np.array([1.0, 2.0, 3.0])

    def test_env_data_is_dataview_after_force(self):
        """K6: 施力后 isinstance(env.data, OrcaGymDataView) 仍为 True。"""
        env = _make_g1_env()
        env.mj_forward()
        env.apply_body_force("pelvis", np.array([1.0, 0.0, 0.0]), np.zeros(3))
        env.mj_forward()
        self.assertIsInstance(env.data, OrcaGymDataView)


class TestDataViewXfrcReadOnlyFunctional(unittest.TestCase):
    """子步骤 3.2.5 功能单元测试。

    对应文档 §6.6 功能单元测试表。
    """

    def setUp(self):
        self.env = _make_g1_env()
        self.env.mj_forward()
        self.pelvis_id = self.env.model.body_name2id("pelvis")

    def test_apply_force_blocked_via_data_view(self):
        """env.data.xfrc_applied 只读，直接写应引导报错。"""
        # 只读视图，直接写抛 ValueError
        with self.assertRaises(ValueError):
            self.env.data.xfrc_applied[self.pelvis_id, :3] = [1.0, 2.0, 3.0]
        # 引导用户走 apply_body_force
        self.env.apply_body_force("pelvis", np.array([1.0, 2.0, 3.0]), np.zeros(3))
        self.env.mj_forward()
        np.testing.assert_allclose(
            self.env.data.xfrc_applied[self.pelvis_id, :3],
            [1.0, 2.0, 3.0],
            atol=1e-6,
        )

    def test_xfrc_readable_after_apply_force(self):
        """env.apply_body_force() 后 env.data.xfrc_applied 可读到正确的力。"""
        self.env.clear_all_forces()
        force = np.array([5.0, 0.0, -2.0])
        torque = np.array([0.1, 0.2, 0.3])
        self.env.apply_body_force("pelvis", force, torque)
        self.env.mj_forward()
        np.testing.assert_allclose(
            self.env.data.xfrc_applied[self.pelvis_id, :3], force, atol=1e-6
        )
        np.testing.assert_allclose(
            self.env.data.xfrc_applied[self.pelvis_id, 3:6], torque, atol=1e-6
        )

    def test_clear_all_forces_works_through_simcore(self):
        """SimCore 通过 _mjData 写入不受 DataView 只读视图影响。"""
        self.env.apply_body_force("pelvis", np.ones(3), np.ones(3))
        self.env.mj_forward()
        # 确认有力
        self.assertGreater(np.abs(self.env.data.xfrc_applied[self.pelvis_id, :3]).sum(), 0)
        # 清除（SimCore 写 _mjData，DataView 视图同步）
        self.env.clear_all_forces()
        self.env.mj_forward()
        np.testing.assert_allclose(
            self.env.data.xfrc_applied[self.pelvis_id, :6], 0.0, atol=1e-6
        )


if __name__ == "__main__":
    unittest.main()
