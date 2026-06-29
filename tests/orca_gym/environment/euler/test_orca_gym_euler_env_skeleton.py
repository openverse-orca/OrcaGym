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
_GYM_SOURCE_PATH = (
    pathlib.Path(__file__).resolve().parents[4]
    / "orca_gym" / "core" / "euler" / "orca_gym_euler.py"
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


class TestEnvK14Inheritance(unittest.TestCase):
    """K14: 继承链约束 — 直接继承 gym.Env + OrcaGymEnvMixin，不继承 OrcaGymBaseEnv。"""

    def test_env_inheritance_chain(self):
        """OrcaGymEulerEnv.__bases__ 含 gym.Env 和 OrcaGymEnvMixin，不含 OrcaGymBaseEnv。"""
        from orca_gym.environment.orca_gym_env_mixin import OrcaGymEnvMixin
        from orca_gym.environment.orca_gym_env import OrcaGymBaseEnv
        import gymnasium as gym

        bases = OrcaGymEulerEnv.__bases__
        self.assertIn(OrcaGymEnvMixin, bases)
        self.assertIn(gym.Env, bases)
        self.assertNotIn(OrcaGymBaseEnv, bases)

    def test_env_gym_attr_natural_attribute_error(self):
        """env.gym 抛 AttributeError（Python 原生，属性不存在）。"""
        env = _make_skeleton_env()
        with self.assertRaises(AttributeError):
            _ = env.gym

    def test_env_stub_attr_natural_attribute_error(self):
        """env.stub 抛 AttributeError。"""
        env = _make_skeleton_env()
        with self.assertRaises(AttributeError):
            _ = env.stub

    def test_env_channel_attr_natural_attribute_error(self):
        """env.channel 抛 AttributeError。"""
        env = _make_skeleton_env()
        with self.assertRaises(AttributeError):
            _ = env.channel

    def test_env_no_blocked_attrs_classvar(self):
        """Env 类不定义 _BLOCKED_ATTRS / _SHIELDED_ATTRS / __getattr__ / __setattr__。"""
        class_attrs = vars(OrcaGymEulerEnv)
        self.assertNotIn("_BLOCKED_ATTRS", class_attrs)
        self.assertNotIn("_SHIELDED_ATTRS", class_attrs)
        self.assertNotIn("__getattr__", class_attrs)
        self.assertNotIn("__setattr__", class_attrs)

    def test_env_mixin_methods_available(self):
        """Env 通过 Mixin 继承获得 body/joint/actuator/site/mocap/sensor 等方法。"""
        env = _make_skeleton_env()
        mixin_methods = [
            "body", "joint", "actuator", "site", "mocap", "sensor",
            "generate_action_space", "generate_observation_space",
            "set_seed_value", "_get_reset_info",
        ]
        for method in mixin_methods:
            with self.subTest(method=method):
                self.assertTrue(callable(getattr(env, method, None)),
                                f"Env 缺少 Mixin 方法 '{method}'")
        # agent_num 是 property，不是 callable
        self.assertTrue(hasattr(env, "agent_num"))

    def test_env_body_namespace_works(self):
        """env.body('torso') 返回 'agent0_torso'（Mixin 方法真实工作）。"""
        env = _make_skeleton_env()
        result = env.body("torso")
        self.assertEqual(result, "agent0_torso")

    def test_env_agent_num_works(self):
        """env.agent_num 返回 1（Mixin property 真实工作）。"""
        env = _make_skeleton_env()
        self.assertEqual(env.agent_num, 1)

    def test_env_init_completes_without_error(self):
        """__init__ 完整执行不报错（自主编排生命周期生效）。"""
        env = _make_skeleton_env()
        self.assertIsNotNone(env._gym)

    def test_env_dt_uses_sim_config(self):
        """env.dt 使用 sim_config.timestep * frame_skip。"""
        env = _make_skeleton_env()
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


# =============================================================================
# 阶段三 3.3.3：OrcaGymEuler/Env 雅可比委托
# =============================================================================


class TestEnvJacArchCompliance(unittest.TestCase):
    """子步骤 3.3.3 架构遵从性测试（K1/K3/K4/K11）。

    对应文档 §7.4 架构遵从性测试表。
    """

    def setUp(self):
        self.env = _make_g1_env()
        self.env.mj_forward()

    def test_env_jac_no_gym_private_access(self):
        """K4: grep 断言雅可比方法不触 self._gym._sim/_mjData。"""
        source = _ENV_SOURCE_PATH.read_text(encoding="utf-8")
        start = source.find("# --- 雅可比计算委托（阶段三 3.3.3")
        self.assertGreater(start, 0, "未找到 3.3.3 雅可比委托区块")
        block = source[start:]
        end = block.find("# --- 只读查询委托")
        self.assertGreater(end, 0)
        block = block[:end]
        self.assertNotIn("self._gym._sim", block)
        self.assertNotIn("_mjData", block)
        self.assertNotIn("_mjModel", block)

    def test_env_jac_uses_self_gym_and_model(self):
        """K1/K4: grep 断言走 self._gym.<方法> + self.model.*_name2id。"""
        source = _ENV_SOURCE_PATH.read_text(encoding="utf-8")
        start = source.find("# --- 雅可比计算委托（阶段三 3.3.3")
        block = source[start:]
        end = block.find("# --- 只读查询委托")
        block = block[:end]
        self.assertIn("self._gym.mj_jacBody", block)
        self.assertIn("self._gym.mj_jacSite", block)
        self.assertIn("self._gym.mj_jac_site", block)
        self.assertIn("self.model.body_name2id", block)
        self.assertIn("self.model.site_name2id", block)

    def test_gym_jac_delegates_use_getattribute(self):
        """K3: grep 断言 Gym 委托用 object.__getattribute__。"""
        source = _GYM_SOURCE_PATH.read_text(encoding="utf-8")
        start = source.find("# --- 雅可比计算方法（阶段三 3.3.3")
        self.assertGreater(start, 0, "未找到 Gym 3.3.3 雅可比委托区块")
        block = source[start:]
        end = block.find("def equality_data_width")
        self.assertGreater(end, 0)
        block = block[:end]
        self.assertIn("object.__getattribute__(self, \"_sim\").mj_jacBody", block)
        self.assertIn("object.__getattribute__(self, \"_sim\").mj_jacSite", block)
        self.assertIn("object.__getattribute__(self, \"_sim\").mj_jac_site", block)

    def test_env_jac_returns_none_or_typed(self):
        """K11: mj_jacBody/mj_jacSite 返回 None，mj_jac_site 返回 dict。"""
        nv = self.env.data.qvel.shape[0]
        jacp = np.zeros((3, nv))
        jacr = np.zeros((3, nv))
        ret = self.env.mj_jacBody(jacp, jacr, "pelvis")
        self.assertIsNone(ret)
        ret = self.env.mj_jacSite(jacp, jacr, "imu")
        self.assertIsNone(ret)
        ret = self.env.mj_jac_site(["imu"])
        self.assertIsInstance(ret, dict)


class TestEnvJacFunctional(unittest.TestCase):
    """子步骤 3.3.3 功能单元测试（G1 XML 真实数据）。

    对应文档 §7.4 功能单元测试表。验证 name 解析 + 数值正确。
    """

    def setUp(self):
        self.env = _make_g1_env()
        self.env.mj_forward()

    def test_env_mj_jacBody_by_name(self):
        """env.mj_jacBody(jacp, jacr, "pelvis") 写入正确雅可比。"""
        nv = self.env.data.qvel.shape[0]
        jacp = np.zeros((3, nv))
        jacr = np.zeros((3, nv))
        self.env.mj_jacBody(jacp, jacr, "pelvis")
        # 对照 SimCore 直调（经 model body_name2id 解析 id）
        body_id = self.env.model.body_name2id("pelvis")
        expected_jacp = np.zeros((3, nv))
        expected_jacr = np.zeros((3, nv))
        # 用 Gym 委托到 SimCore 单点方法对照
        self.env._gym.mj_jacBody(expected_jacp, expected_jacr, body_id)
        np.testing.assert_array_equal(jacp, expected_jacp)
        np.testing.assert_array_equal(jacr, expected_jacr)

    def test_env_mj_jacSite_by_name(self):
        """env.mj_jacSite(jacp, jacr, "imu") 写入正确。"""
        nv = self.env.data.qvel.shape[0]
        jacp = np.zeros((3, nv))
        jacr = np.zeros((3, nv))
        self.env.mj_jacSite(jacp, jacr, "imu")
        site_id = self.env.model.site_name2id("imu")
        expected_jacp = np.zeros((3, nv))
        expected_jacr = np.zeros((3, nv))
        self.env._gym.mj_jacSite(expected_jacp, expected_jacr, site_id)
        np.testing.assert_array_equal(jacp, expected_jacp)
        np.testing.assert_array_equal(jacr, expected_jacr)

    def test_env_mj_jac_site_batch(self):
        """批量雅可比正确。"""
        result = self.env.mj_jac_site(["imu"])
        self.assertIn("imu", result)
        nv = self.env.data.qvel.shape[0]
        self.assertEqual(result["imu"]["jacp"].shape, (3, nv))
        self.assertEqual(result["imu"]["jacr"].shape, (3, nv))


# =============================================================================
# 阶段三 3.4.4：OrcaGymEuler/Env Studio 委托
# =============================================================================


class TestEnvStudioArchCompliance(unittest.TestCase):
    """子步骤 3.4.4 架构遵从性测试（K2/K4/K9/K11/K12）。

    对应文档 §8.5 架构遵从性测试表。
    """

    def test_env_studio_no_gym_private_access(self):
        """K4: grep 断言 Studio 区块不触 self._gym._sim/_studio/_mjData/_mjModel。"""
        source = _ENV_SOURCE_PATH.read_text(encoding="utf-8")
        start = source.find("# --- Studio 委托（阶段三 3.4.4")
        self.assertGreater(start, 0, "未找到 3.4.4 Studio 委托区块")
        block_source = source[start:]
        # 区块到下一个 --- 分隔符结束
        end = block_source.find("\n    # ---", 1)
        if end < 0:
            end = len(block_source)
        block = block_source[:end]
        self.assertNotIn("self._gym._sim", block)
        self.assertNotIn("self._gym._studio", block)
        self.assertNotIn("self._gym._registry", block)
        self.assertNotIn("_mjData", block)
        self.assertNotIn("_mjModel", block)

    def test_env_studio_uses_gym_not_gym_studio(self):
        """K9: grep 断言走 self._gym / self._studio_bridge，不走 gym.studio。"""
        source = _ENV_SOURCE_PATH.read_text(encoding="utf-8")
        start = source.find("# --- Studio 委托（阶段三 3.4.4")
        block_source = source[start:]
        end = block_source.find("\n    # ---", 1)
        if end < 0:
            end = len(block_source)
        block = block_source[:end]
        self.assertNotIn("gym.studio", block)
        # 应走 self._gym 或 self._studio_bridge
        self.assertTrue(
            "self._gym." in block or "self._studio_bridge" in block,
            "Studio 方法应委托 self._gym 或 self._studio_bridge",
        )

    def test_env_studio_dir_includes_methods(self):
        """K2: dir(env) 含 begin_save_video/get_current_frame 等。"""
        env = _make_g1_env()
        d = dir(env)
        for name in [
            "begin_save_video",
            "stop_save_video",
            "get_current_frame",
            "get_camera_time_stamp",
            "get_frame_png",
            "load_content_file",
        ]:
            self.assertIn(name, d, f"dir(env) 缺少 {name}")

    def test_env_studio_returns_typed(self):
        """K11: get_current_frame 返回 int，get_camera_time_stamp 返回 dict。"""
        env = _make_g1_env()
        ret = env.get_current_frame()
        self.assertIsInstance(ret, int)
        ret = env.get_camera_time_stamp(0)
        self.assertIsInstance(ret, dict)

    def test_env_studio_docstrings_present(self):
        """K12: 新增 Studio 委托方法有 docstring。"""
        env = _make_g1_env()
        import inspect

        for name in [
            "begin_save_video",
            "stop_save_video",
            "get_current_frame",
            "get_camera_time_stamp",
            "get_frame_png",
            "load_content_file",
        ]:
            method = getattr(env, name)
            doc = inspect.getdoc(method)
            self.assertIsNotNone(doc, f"{name} 缺少 docstring")
            self.assertGreater(len(doc), 0, f"{name} docstring 为空")


class TestEnvStudioFunctional(unittest.TestCase):
    """子步骤 3.4.4 功能单元测试。

    对应文档 §8.5 功能单元测试表。
    """

    def setUp(self):
        self.env = _make_g1_env()

    def test_env_begin_stop_save_video_offline_noop(self):
        """离线模式 no-op 不抛错。"""
        self.env.begin_save_video("/tmp/test.mp4")
        self.env.stop_save_video()

    def test_env_get_current_frame_offline_returns_neg1(self):
        """离线模式返回 -1。"""
        self.assertEqual(self.env.get_current_frame(), -1)

    def test_env_get_camera_time_stamp_offline_returns_empty(self):
        """离线模式返回空 dict。"""
        self.assertEqual(self.env.get_camera_time_stamp(0), {})

    def test_env_get_frame_png_offline_noop(self):
        """离线模式 no-op。"""
        self.env.get_frame_png("/tmp/test.png")

    def test_env_load_content_file_offline_noop(self):
        """离线模式 no-op。"""
        self.env.load_content_file("mesh.obj")

    def test_env_video_methods_delegate_to_bridge(self):
        """在线模式委托链路：Env -> Gym -> bridge（mock stub）。"""
        # 构造 mock bridge 替换 Gym 的 _studio
        captured = {}

        class MockBridge:
            async def begin_save_video(self, file_path, capture_mode):
                captured["begin"] = (file_path, capture_mode)

            async def stop_save_video(self):
                captured["stop"] = True

            async def get_current_frame(self):
                captured["frame"] = True
                return 99

            async def get_camera_time_stamp(self, last_frame_index):
                captured["ts"] = last_frame_index
                return {"cam0": [1, 2, 3]}

            async def get_frame_png(self, image_path):
                captured["png"] = image_path

            async def load_content_file(self, *args, **kwargs):
                captured["load"] = (args, kwargs)

        # 替换 Gym 的 _studio（用 object.__setattr__ 绕过 __setattr__ 拦截）
        object.__setattr__(self.env._gym, "_studio", MockBridge())
        self.env.begin_save_video("/tmp/x.mp4", capture_mode=1)
        self.assertEqual(captured["begin"], ("/tmp/x.mp4", 1))
        self.env.stop_save_video()
        self.assertTrue(captured["stop"])
        self.assertEqual(self.env.get_current_frame(), 99)
        self.assertTrue(captured["frame"])
        self.assertEqual(self.env.get_camera_time_stamp(5), {"cam0": [1, 2, 3]})
        self.assertEqual(captured["ts"], 5)
        self.env.get_frame_png("/tmp/x.png")
        self.assertEqual(captured["png"], "/tmp/x.png")
        self.env.load_content_file("mesh.obj", remote_file_dir="/r")
        self.assertIn("load", captured)


# =============================================================================
# 阶段三 3.5.3：OrcaGymEuler/Env 约束委托
# =============================================================================


class TestEnvEqualityArchCompliance(unittest.TestCase):
    """子步骤 3.5.3 架构遵从性测试（K1/K2/K3/K4/K11）。

    对应文档 §9.4 架构遵从性测试表。
    """

    def test_env_eq_no_gym_private_access(self):
        """K4: grep 断言约束区块不触 self._gym._sim/_mjModel/_mjData。"""
        source = _ENV_SOURCE_PATH.read_text(encoding="utf-8")
        start = source.find("# --- 等式约束委托（阶段三 3.5.3")
        self.assertGreater(start, 0, "未找到 3.5.3 等式约束区块")
        block_source = source[start:]
        end = block_source.find("\n    # ---", 1)
        if end < 0:
            end = len(block_source)
        block = block_source[:end]
        self.assertNotIn("self._gym._sim", block)
        self.assertNotIn("self._gym._studio", block)
        self.assertNotIn("_mjData", block)
        self.assertNotIn("_mjModel", block)

    def test_env_eq_uses_self_gym_and_model(self):
        """K1/K4: grep 断言走 self._gym.<方法> + self.model.body_name2id。"""
        source = _ENV_SOURCE_PATH.read_text(encoding="utf-8")
        start = source.find("# --- 等式约束委托（阶段三 3.5.3")
        block_source = source[start:]
        end = block_source.find("\n    # ---", 1)
        if end < 0:
            end = len(block_source)
        block = block_source[:end]
        self.assertIn("self._gym.update_equality_constraints", block)
        self.assertIn("self._gym.modify_equality_objects", block)
        self.assertIn("self.model.body_name2id", block)

    def test_gym_eq_delegates_use_getattribute(self):
        """K3: grep 断言 Gym 委托用 object.__getattribute__。"""
        source = _GYM_SOURCE_PATH.read_text(encoding="utf-8")
        start = source.find("# --- 等式约束委托（阶段三 3.5.3")
        self.assertGreater(start, 0, "未找到 Gym 3.5.3 等式约束区块")
        block_source = source[start:]
        end = block_source.find("\n    # ---", 1)
        if end < 0:
            end = len(block_source)
        block = block_source[:end]
        self.assertIn("object.__getattribute__(self, \"_sim\")", block)

    def test_env_eq_returns_none(self):
        """K11: 约束方法返回 None（写操作）。"""
        env = _make_g1_env()
        import mujoco

        eq_data = np.zeros(mujoco.mjNEQDATA)
        eq_list = [
            {
                "type": mujoco.mjtEq.mjEQ_WELD,
                "obj1_id": 1,
                "obj2_id": 2,
                "data": eq_data,
            }
        ]
        ret = env.update_equality_constraints(eq_list)
        self.assertIsNone(ret)
        ret = env.modify_equality_objects([0], obj1_names=["pelvis"])
        self.assertIsNone(ret)

    def test_env_eq_dir_includes_methods(self):
        """K2: dir(env) 含 update_equality_constraints 等。"""
        env = _make_g1_env()
        d = dir(env)
        for name in [
            "update_equality_constraints",
            "modify_equality_objects",
            "update_anchor_equality_constraints",
        ]:
            self.assertIn(name, d, f"dir(env) 缺少 {name}")


class TestEnvEqualityFunctional(unittest.TestCase):
    """子步骤 3.5.3 功能单元测试（G1 XML 真实数据）。

    对应文档 §9.4 功能单元测试表。验证 name 解析 + eq_* 写入。
    """

    def setUp(self):
        self.env = _make_g1_env()
        self.env.mj_forward()

    def test_env_update_equality_constraints_by_name(self):
        """用 body name 调用后 eq_* 字段正确写入。"""
        import mujoco

        eq_data = np.zeros(mujoco.mjNEQDATA)
        eq_data[0:3] = [0.1, 0.2, 0.3]
        eq_list = [
            {
                "type": mujoco.mjtEq.mjEQ_WELD,
                "obj1_name": "pelvis",
                "obj2_name": "torso_link",
                "data": eq_data,
            }
        ]
        self.env.update_equality_constraints(eq_list)
        # 验证写入：通过 model 查询
        model = self.env.model
        # pelvis 和 torso_link 的 body id
        pelvis_id = model.body_name2id("pelvis")
        torso_id = model.body_name2id("torso_link")
        obj1, obj2 = self.env._gym.equality_object_ids(0)
        self.assertEqual(obj1, pelvis_id)
        self.assertEqual(obj2, torso_id)

    def test_env_modify_equality_objects_by_name(self):
        """obj id 更新正确。"""
        import mujoco

        # 先写入初值
        eq_data = np.zeros(mujoco.mjNEQDATA)
        self.env.update_equality_constraints(
            [
                {
                    "type": mujoco.mjtEq.mjEQ_CONNECT,
                    "obj1_id": 1,
                    "obj2_id": 2,
                    "data": eq_data,
                }
            ]
        )
        # 用 name 修改
        self.env.modify_equality_objects(
            [0], obj1_names=["pelvis"], obj2_names=["torso_link"]
        )
        pelvis_id = self.env.model.body_name2id("pelvis")
        torso_id = self.env.model.body_name2id("torso_link")
        obj1, obj2 = self.env._gym.equality_object_ids(0)
        self.assertEqual(obj1, pelvis_id)
        self.assertEqual(obj2, torso_id)

    def test_env_update_anchor_equality_constraints(self):
        """锚点约束组装正确（actor_id + mocap_id）。"""
        self.env.update_anchor_equality_constraints("pelvis", "weld")
        # 验证 eq[0] 写入：obj1 = mocap_id, obj2 = pelvis_id
        mocap_names = self.env._gym.mocap_body_names()
        self.assertGreater(len(mocap_names), 0)
        mocap_id = self.env.model.body_name2id(mocap_names[0])
        pelvis_id = self.env.model.body_name2id("pelvis")
        obj1, obj2 = self.env._gym.equality_object_ids(0)
        self.assertEqual(obj1, mocap_id)
        self.assertEqual(obj2, pelvis_id)


class TestEnvAnchorActorArchCompliance(unittest.TestCase):
    """子步骤 3.5.4 架构遵从性测试（K1/K4/K11/K12）。

    对应文档 §9.5 架构遵从性测试表。
    """

    def test_env_anchor_actor_no_private_access(self):
        """K4: grep 断言 anchor_actor 区块不触 self._gym._sim/_mjData/_mjModel/_studio。"""
        source = _ENV_SOURCE_PATH.read_text(encoding="utf-8")
        start = source.find("# --- 体操作（阶段三 3.5.4")
        self.assertGreater(start, 0, "未找到 3.5.4 anchor_actor 区块")
        block_source = source[start:]
        end = block_source.find("\n    # ---", 1)
        if end < 0:
            end = len(block_source)
        block = block_source[:end]
        self.assertNotIn("self._gym._sim", block)
        self.assertNotIn("self._gym._studio", block)
        self.assertNotIn("self._gym._registry", block)
        self.assertNotIn("_mjData", block)
        self.assertNotIn("_mjModel", block)

    def test_env_anchor_actor_uses_compliance_api(self):
        """K1/K4: grep 断言走 set_mocap_pos_and_quat/update_anchor_equality_constraints/get_body_xpos_xmat_xquat 公共方法。"""
        source = _ENV_SOURCE_PATH.read_text(encoding="utf-8")
        start = source.find("# --- 体操作（阶段三 3.5.4")
        block_source = source[start:]
        end = block_source.find("\n    # ---", 1)
        if end < 0:
            end = len(block_source)
        block = block_source[:end]
        self.assertIn("self.get_body_xpos_xmat_xquat", block)
        self.assertIn("self.set_mocap_pos_and_quat", block)
        self.assertIn("self.update_anchor_equality_constraints", block)

    def test_env_anchor_actor_returns_none(self):
        """K11: anchor_actor 返回 None（写操作）。"""
        env = _make_g1_env()
        env.mj_forward()
        ret = env.anchor_actor("pelvis", "weld")
        self.assertIsNone(ret)

    def test_env_anchor_actor_docstring_present(self):
        """K12: anchor_actor 有 docstring。"""
        import inspect

        env = _make_g1_env()
        doc = inspect.getdoc(env.anchor_actor)
        self.assertIsNotNone(doc)
        self.assertGreater(len(doc), 0)


class TestEnvAnchorActorFunctional(unittest.TestCase):
    """子步骤 3.5.4 功能单元测试（G1 XML 真实数据）。

    对应文档 §9.5 功能单元测试表。验证 mocap 位姿 + weld 约束 + 状态记录。
    """

    def setUp(self):
        self.env = _make_g1_env()
        self.env.mj_forward()

    def test_anchor_actor_sets_mocap_to_actor_pose(self):
        """锚定后 mocap 位姿 = actor 初始位姿。"""
        actor_pose_before = self.env.get_body_xpos_xmat_xquat(["pelvis"])["pelvis"]
        self.env.anchor_actor("pelvis", "weld")
        # 查询 mocap body 当前的 pos/quat（通过 DataView 零拷贝视图）
        mocap_names = self.env._gym.mocap_body_names()
        self.assertGreater(len(mocap_names), 0)
        mocap_name = mocap_names[0]
        mocap_pos = self.env.data.mocap_pos(mocap_name)
        mocap_quat = self.env.data.mocap_quat(mocap_name)
        np.testing.assert_array_almost_equal(
            mocap_pos, actor_pose_before["xpos"]
        )
        np.testing.assert_array_almost_equal(
            mocap_quat, actor_pose_before["xquat"]
        )

    def test_anchor_actor_creates_weld_constraint(self):
        """锚定后 eq_type 为 weld，obj1/obj2 关联 actor 与 mocap。"""
        self.env.anchor_actor("pelvis", "weld")
        mocap_names = self.env._gym.mocap_body_names()
        mocap_id = self.env.model.body_name2id(mocap_names[0])
        pelvis_id = self.env.model.body_name2id("pelvis")
        obj1, obj2 = self.env._gym.equality_object_ids(0)
        self.assertEqual(obj1, mocap_id)
        self.assertEqual(obj2, pelvis_id)

    def test_anchor_actor_records_state(self):
        """_anchored_actor/_anchor_type 正确记录。"""
        self.env.anchor_actor("pelvis", "weld")
        self.assertEqual(self.env._anchored_actor, "pelvis")
        self.assertEqual(self.env._anchor_type, "weld")


class TestEnvReleaseBodyAnchoredArchCompliance(unittest.TestCase):
    """子步骤 3.5.5 架构遵从性测试（K1/K4/K11/K12）。

    对应文档 §9.6 架构遵从性测试表。
    """

    def test_env_release_no_private_access(self):
        """K4: grep 断言 release_body_anchored 区块不触 self._gym._sim/_mjData/_mjModel/_studio。"""
        source = _ENV_SOURCE_PATH.read_text(encoding="utf-8")
        start = source.find("# --- 体操作（阶段三 3.5.5")
        self.assertGreater(start, 0, "未找到 3.5.5 release_body_anchored 区块")
        block_source = source[start:]
        end = block_source.find("\n    # ---", 1)
        if end < 0:
            end = len(block_source)
        block = block_source[:end]
        self.assertNotIn("self._gym._sim", block)
        self.assertNotIn("self._gym._studio", block)
        self.assertNotIn("self._gym._registry", block)
        self.assertNotIn("_mjData", block)
        self.assertNotIn("_mjModel", block)

    def test_env_release_uses_compliance_api(self):
        """K1/K4: grep 断言走 self._gym.update_equality_constraints 公共方法。"""
        source = _ENV_SOURCE_PATH.read_text(encoding="utf-8")
        start = source.find("# --- 体操作（阶段三 3.5.5")
        block_source = source[start:]
        end = block_source.find("\n    # ---", 1)
        if end < 0:
            end = len(block_source)
        block = block_source[:end]
        self.assertIn("self._gym.update_equality_constraints", block)
        self.assertIn("self._gym.n_equality", block)

    def test_env_release_returns_none(self):
        """K11: release_body_anchored 返回 None。"""
        env = _make_g1_env()
        env.mj_forward()
        # 未锚定时 no-op
        ret = env.release_body_anchored()
        self.assertIsNone(ret)
        # 锚定后释放
        env.anchor_actor("pelvis", "weld")
        ret = env.release_body_anchored()
        self.assertIsNone(ret)

    def test_env_release_docstring_present(self):
        """K12: release_body_anchored 有 docstring。"""
        env = _make_g1_env()
        doc = env.release_body_anchored.__doc__
        self.assertIsNotNone(doc)
        self.assertGreater(len(doc), 0)


class TestEnvReleaseBodyAnchoredFunctional(unittest.TestCase):
    """子步骤 3.5.5 功能单元测试（G1 XML 真实数据）。

    对应文档 §9.6 功能单元测试表。验证约束清除 + 状态清除。
    """

    def setUp(self):
        self.env = _make_g1_env()
        self.env.mj_forward()

    def test_release_body_anchored_clears_constraint(self):
        """释放后锚点 eq_type 清零。"""
        self.env.anchor_actor("pelvis", "weld")
        # 锚定后 eq_obj1id/obj2id 应为 mocap_id/pelvis_id
        obj1, obj2 = self.env._gym.equality_object_ids(0)
        # 释放
        self.env.release_body_anchored()
        # 验证 eq_type 已清零（通过 update_equality_constraints 写入）
        # 重新读取 eq_obj1id/obj2id 应为 -1（释放写入）
        obj1_after, obj2_after = self.env._gym.equality_object_ids(0)
        self.assertEqual(obj1_after, -1)
        self.assertEqual(obj2_after, -1)

    def test_release_body_anchored_clears_state(self):
        """_anchored_actor/_anchor_type 为 None。"""
        self.env.anchor_actor("pelvis", "weld")
        self.env.release_body_anchored()
        self.assertIsNone(self.env._anchored_actor)
        self.assertIsNone(self.env._anchor_type)

    def test_release_without_anchor_noop(self):
        """未锚定时调用 no-op 不抛错。"""
        # 未锚定状态
        self.assertIsNone(self.env._anchored_actor)
        # 调用不应抛异常
        self.env.release_body_anchored()
        self.assertIsNone(self.env._anchored_actor)


class TestEnvDoBodyManipulationArchCompliance(unittest.TestCase):
    """子步骤 3.5.6 架构遵从性测试（K1/K4/K9/K11/K12）。

    对应文档 §9.7 架构遵从性测试表。
    """

    def test_env_do_body_manipulation_no_private_access(self):
        """K4: grep 断言 do_body_manipulation 区块不触 self._gym._sim/_mjData/_mjModel/_studio。"""
        source = _ENV_SOURCE_PATH.read_text(encoding="utf-8")
        start = source.find("# --- 体操作编排（阶段三 3.5.6")
        self.assertGreater(start, 0, "未找到 3.5.6 do_body_manipulation 区块")
        block_source = source[start:]
        end = block_source.find("\n    # ---", 1)
        if end < 0:
            end = len(block_source)
        block = block_source[:end]
        self.assertNotIn("self._gym._sim", block)
        self.assertNotIn("self._gym._studio", block)
        self.assertNotIn("self._gym._registry", block)
        self.assertNotIn("_mjData", block)
        self.assertNotIn("_mjModel", block)
        self.assertNotIn("self._studio_bridge", block)

    def test_env_do_body_manipulation_uses_compliance_api(self):
        """K1/K4/K9: grep 断言走 anchor_actor/release_body_anchored/set_mocap_pos_and_quat 等公共方法。"""
        source = _ENV_SOURCE_PATH.read_text(encoding="utf-8")
        start = source.find("# --- 体操作编排（阶段三 3.5.6")
        block_source = source[start:]
        end = block_source.find("\n    # ---", 1)
        if end < 0:
            end = len(block_source)
        block = block_source[:end]
        self.assertIn("self._gym.get_body_manipulation_state", block)
        self.assertIn("self.anchor_actor", block)
        self.assertIn("self.release_body_anchored", block)
        self.assertIn("self.set_mocap_pos_and_quat", block)

    def test_env_do_body_manipulation_returns_none(self):
        """K11: do_body_manipulation 返回 None。"""
        env = _make_g1_env()
        env.mj_forward()
        ret = env.do_body_manipulation()
        self.assertIsNone(ret)

    def test_env_do_body_manipulation_docstring_present(self):
        """K12: do_body_manipulation 有 docstring（含编排流程说明）。"""
        env = _make_g1_env()
        doc = env.do_body_manipulation.__doc__
        self.assertIsNotNone(doc)
        self.assertGreater(len(doc), 0)


class TestEnvDoBodyManipulationFunctional(unittest.TestCase):
    """子步骤 3.5.6 功能单元测试（G1 XML 真实数据 + bridge monkeypatch）。

    对应文档 §9.7 功能单元测试表。离线 no-op + 三动作编排 + 完整循环。
    """

    def setUp(self):
        self.env = _make_g1_env()
        self.env.mj_forward()
        # 离线 env 默认 _skip_grpc_load=True；编排方法需在线路径，
        # 测试中临时翻转标志 + monkeypatch bridge 返回 canned 状态。
        self._original_skip = self.env._skip_grpc_load

    def tearDown(self):
        self.env._skip_grpc_load = self._original_skip

    def _patch_bridge(self, anchored=None, anchor_type=0, pos=None, quat=None):
        """注入 bridge 体操作状态（async 桩）。"""
        if pos is None:
            pos = np.zeros(3)
        if quat is None:
            quat = np.array([1.0, 0.0, 0.0, 0.0])

        async def fake_anchored():
            return (anchored, anchor_type)

        async def fake_movement():
            return {"delta_pos": pos, "delta_quat": quat}

        bridge = self.env._gym.studio_bridge()
        bridge.get_body_manipulation_anchored = fake_anchored
        bridge.get_body_manipulation_movement = fake_movement
        self.env._skip_grpc_load = False

    def test_do_body_manipulation_offline_noop(self):
        """离线模式（_skip_grpc_load=True）no-op 不抛错。"""
        self.env._skip_grpc_load = True
        self.env.do_body_manipulation()  # 不应抛异常
        self.assertIsNone(self.env._anchored_actor)

    def test_do_body_manipulation_anchor_flow(self):
        """锚定请求触发 anchor_actor。"""
        from orca_gym.core.euler.orca_studio_bridge import AnchorType

        self._patch_bridge(anchored="pelvis", anchor_type=AnchorType.WELD)
        self.env.do_body_manipulation()
        self.assertEqual(self.env._anchored_actor, "pelvis")
        self.assertEqual(self.env._anchor_type, "weld")

    def test_do_body_manipulation_release_flow(self):
        """释放请求触发 release_body_anchored。"""
        from orca_gym.core.euler.orca_studio_bridge import AnchorType

        # 先锚定
        self._patch_bridge(anchored="pelvis", anchor_type=AnchorType.WELD)
        self.env.do_body_manipulation()
        self.assertIsNotNone(self.env._anchored_actor)
        # 再注入释放（Studio 无锚定 body）
        self._patch_bridge(anchored=None, anchor_type=AnchorType.NONE)
        self.env.do_body_manipulation()
        self.assertIsNone(self.env._anchored_actor)

    def test_do_body_manipulation_mocap_sync_flow(self):
        """已锚定时同步 mocap 位姿（UI 拖拽目标位姿写入）。"""
        from orca_gym.core.euler.orca_studio_bridge import AnchorType

        # 先锚定
        self._patch_bridge(anchored="pelvis", anchor_type=AnchorType.WELD)
        self.env.do_body_manipulation()
        # 注入拖拽目标位姿
        target_pos = np.array([0.5, 0.5, 1.0])
        target_quat = np.array([0.7071, 0.0, 0.0, 0.7071])
        self._patch_bridge(
            anchored="pelvis",
            anchor_type=AnchorType.WELD,
            pos=target_pos,
            quat=target_quat,
        )
        self.env.do_body_manipulation()
        # 验证 mocap 位姿已同步到目标
        mocap_names = self.env._gym.mocap_body_names()
        mocap_name = mocap_names[0]
        np.testing.assert_array_almost_equal(
            self.env.data.mocap_pos(mocap_name), target_pos
        )
        np.testing.assert_array_almost_equal(
            self.env.data.mocap_quat(mocap_name), target_quat
        )

    def test_do_body_manipulation_full_cycle(self):
        """锚定 → 移动 → 释放完整循环不抛错。"""
        from orca_gym.core.euler.orca_studio_bridge import AnchorType

        # 1. 锚定
        self._patch_bridge(anchored="pelvis", anchor_type=AnchorType.WELD)
        self.env.do_body_manipulation()
        self.assertEqual(self.env._anchored_actor, "pelvis")
        # 2. 移动
        self._patch_bridge(
            anchored="pelvis",
            anchor_type=AnchorType.WELD,
            pos=np.array([0.3, 0.3, 0.8]),
        )
        self.env.do_body_manipulation()
        # 3. 释放
        self._patch_bridge(anchored=None, anchor_type=AnchorType.NONE)
        self.env.do_body_manipulation()
        self.assertIsNone(self.env._anchored_actor)


if __name__ == "__main__":
    unittest.main()
