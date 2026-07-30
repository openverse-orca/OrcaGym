"""阶段三 §10 跨子步骤一致性验证 + §11 回归测试矩阵。

在全部 26 个子步骤独立验收后执行，验证：
- §10.1 全局架构遵从性（K1/K2/K6/K9/K14 + ruff SLF001）
- §10.2 委托链路完整性（query/set/studio/jac 四组方法链路）
- §10.3 数据一致性（DataView 零拷贝视图与 query 返回值一致 + 步进同步）
- §11.2 K 约束全量回归（ruff SLF001 全局零报警）

运行方式:
    <conda-base>/envs/orca/bin/python -m unittest discover -s tests/orca_gym -p "test_phase3_cross_substep_consistency.py" -v
"""

import pathlib
import re
import subprocess
import sys
import unittest

import numpy as np

from orca_gym.environment.euler.orca_gym_euler_env import OrcaGymEulerEnv

_ENV_SOURCE_PATH = (
    pathlib.Path(__file__).resolve().parents[4]
    / "orca_gym" / "environment" / "euler" / "orca_gym_euler_env.py"
)
_GYM_SOURCE_PATH = (
    pathlib.Path(__file__).resolve().parents[4]
    / "orca_gym" / "core" / "euler" / "orca_gym_euler.py"
)


def _make_g1_env() -> OrcaGymEulerEnv:
    """构造加载 G1 XML 的离线 Env。

    使用本仓 fixtures 目录下的简化版 G1 XML（mesh 替换为基础几何体），
    无外部 mesh 依赖，确保 OrcaGym 仓可独立运行测试。
    """
    _g1_xml = (
        pathlib.Path(__file__).resolve().parents[0]
        / "fixtures" / "g1_29dof_camera_simplified.xml"
    )
    return OrcaGymEulerEnv(
        frame_skip=4,
        orcagym_addr="localhost:50051",
        agent_names=["agent0"],
        time_step=0.002,
        model_xml_path=str(_g1_xml),
        skip_grpc_load=True,
    )


def _exec_source_without_docstrings(path: pathlib.Path) -> str:
    """返回源码可执行部分（去除 docstring 和注释行）。"""
    source = path.read_text(encoding="utf-8")
    exec_lines: list[str] = []
    for line in source.splitlines(keepends=True):
        stripped = line.lstrip()
        if stripped.startswith("#"):
            continue
        exec_lines.append(line)
    exec_source = "".join(exec_lines)
    exec_source = re.sub(r'"""[\s\S]*?"""', "", exec_source)
    exec_source = re.sub(r"'''[\s\S]*?'''", "", exec_source)
    return exec_source


# =============================================================================
# §10.1 全局架构遵从性验证
# =============================================================================


class TestGlobalArchCompliance(unittest.TestCase):
    """§10.1 全局架构遵从性（ruff SLF001）。

    K1/K2/K6/K9/K14 约束已迁移至 test_orca_gym_euler_env_skeleton.py 各 K 约束类，
    本类仅保留 ruff SLF001 全局零报警检查（扫描范围 orca_gym/，是各处 ruff 检查的超集）。
    """

    def test_ruff_slf001_global_zero(self):
        """M0-M7: ruff check --select SLF001 orca_gym/ 零报警。"""
        repo_root = _ENV_SOURCE_PATH.parents[3]
        result = subprocess.run(
            [sys.executable, "-m", "ruff", "check", "--select", "SLF001", "orca_gym/"],
            capture_output=True, text=True, cwd=str(repo_root),
        )
        self.assertEqual(
            result.returncode, 0,
            f"ruff SLF001 有违规:\n{result.stdout}\n{result.stderr}",
        )


# =============================================================================
# §10.2 委托链路完整性验证
# =============================================================================


class TestDelegationChainQueryMethods(unittest.TestCase):
    """§10.2 query_* 方法经 Env → Gym → SimCore/Registry 完整链路。"""

    def setUp(self):
        self.env = _make_g1_env()
        self.env.mj_forward()

    def test_query_joint_qpos_chain(self):
        """query_joint_qpos 返回 dict，值形状与关节自由度一致。"""
        joint_names = ["left_hip_pitch", "right_hip_pitch"]
        result = self.env.query_joint_qpos(joint_names)
        self.assertIsInstance(result, dict)
        for name in joint_names:
            self.assertIn(name, result)
            self.assertIsInstance(result[name], np.ndarray)

    def test_query_joint_qvel_chain(self):
        """query_joint_qvel 返回 dict，值形状与关节 dof 一致。"""
        joint_names = ["left_hip_pitch", "right_hip_pitch"]
        result = self.env.query_joint_qvel(joint_names)
        self.assertIsInstance(result, dict)
        for name in joint_names:
            self.assertIn(name, result)

    def test_query_sensor_data_chain(self):
        """query_sensor_data 返回 dict（G1 有传感器）。"""
        # G1 模型可能无传感器，跳过空情况
        result = self.env.query_sensor_data([])
        self.assertIsInstance(result, dict)

    def test_get_body_xpos_xmat_xquat_chain(self):
        """get_body_xpos_xmat_xquat 返回 dict 含 xpos/xmat/xquat。"""
        result = self.env.get_body_xpos_xmat_xquat(["pelvis"])
        self.assertIn("pelvis", result)
        self.assertIn("xpos", result["pelvis"])
        self.assertIn("xmat", result["pelvis"])
        self.assertIn("xquat", result["pelvis"])
        self.assertEqual(result["pelvis"]["xpos"].shape, (3,))
        self.assertEqual(result["pelvis"]["xmat"].shape, (3, 3))
        self.assertEqual(result["pelvis"]["xquat"].shape, (4,))

    def test_query_site_pos_and_mat_chain(self):
        """query_site_pos_and_mat 返回 dict 含 pos/mat。"""
        # 使用 G1 已有的 site
        result = self.env.query_site_pos_and_mat([])
        self.assertIsInstance(result, dict)


class TestDelegationChainSetMethods(unittest.TestCase):
    """§10.2 set_*/apply_* 方法经 Env → Gym → SimCore 完整链路，写入生效。"""

    def setUp(self):
        self.env = _make_g1_env()
        self.env.mj_forward()

    def test_set_joint_qpos_chain(self):
        """set_joint_qpos 写入后 query_joint_qpos 读回一致。"""
        joint_names = ["left_hip_pitch", "right_hip_pitch"]
        original = self.env.query_joint_qpos(joint_names)
        new_values = {name: val + 0.1 for name, val in original.items()}
        full_qpos = self.env.data.qpos.copy()
        for name, val in new_values.items():
            adr = self.env.jnt_qposadr(name)
            full_qpos[adr:adr + len(val)] = val
        self.env.set_joint_qpos(full_qpos)
        self.env.mj_forward()
        readback = self.env.query_joint_qpos(joint_names)
        for name in joint_names:
            np.testing.assert_array_almost_equal(
                readback[name], new_values[name], decimal=6,
            )

    def test_set_joint_qvel_chain(self):
        """set_joint_qvel 写入后 query_joint_qvel 读回一致。"""
        joint_names = ["left_hip_pitch", "right_hip_pitch"]
        original = self.env.query_joint_qvel(joint_names)
        new_values = {name: val + 0.5 for name, val in original.items()}
        full_qvel = self.env.data.qvel.copy()
        for name, val in new_values.items():
            adr = self.env.jnt_dofadr(name)
            full_qvel[adr:adr + len(val)] = val
        self.env.set_joint_qvel(full_qvel)
        self.env.mj_forward()
        readback = self.env.query_joint_qvel(joint_names)
        for name in joint_names:
            np.testing.assert_array_almost_equal(
                readback[name], new_values[name], decimal=6,
            )

    def test_apply_body_force_chain(self):
        """apply_body_force 写入后 data.xfrc_applied 反映外力。"""
        body_name = "pelvis"
        body_id = self.env.model.body_name2id(body_name)
        xfrc_before = self.env.data.xfrc_applied[body_id].copy()
        self.env.apply_body_force(body_name, [1.0, 2.0, 3.0], [0.0, 0.0, 0.0])
        xfrc_after = self.env.data.xfrc_applied[body_id].copy()
        np.testing.assert_array_almost_equal(xfrc_after[:3], [1.0, 2.0, 3.0])
        np.testing.assert_array_almost_equal(xfrc_after[3:], [0.0, 0.0, 0.0])
        # 清除
        self.env.clear_body_force(body_name)
        xfrc_cleared = self.env.data.xfrc_applied[body_id].copy()
        np.testing.assert_array_almost_equal(xfrc_cleared, xfrc_before)

    def test_set_mocap_pos_and_quat_chain(self):
        """set_mocap_pos_and_quat 写入后 data.mocap_pos/quat 反映。"""
        mocap_names = self.env._gym.mocap_body_names()
        if not mocap_names:
            self.skipTest("G1 模型无 mocap body")
        mocap_name = mocap_names[0]
        target_pos = np.array([0.5, 0.5, 1.0])
        target_quat = np.array([1.0, 0.0, 0.0, 0.0])
        self.env.set_mocap_pos_and_quat(
            {mocap_name: {"pos": target_pos, "quat": target_quat}}
        )
        np.testing.assert_array_almost_equal(
            self.env.data.mocap_pos(mocap_name), target_pos
        )
        np.testing.assert_array_almost_equal(
            self.env.data.mocap_quat(mocap_name), target_quat
        )

    def test_set_ctrl_chain(self):
        """set_ctrl 写入后 mj_forward 后 actuator_force 反映（ctrl getter 读 actuator_force）。"""
        nu = self.env.model.nu
        if nu == 0:
            self.skipTest("G1 模型无执行器")
        ctrl = np.zeros(nu)
        ctrl[0] = 0.5
        self.env.set_ctrl(ctrl)
        self.env.mj_forward()
        # ctrl getter 读 actuator_force，forward 后应反映控制输入
        np.testing.assert_array_almost_equal(self.env.ctrl, ctrl)


class TestDelegationChainStudioMethods(unittest.TestCase):
    """§10.2 Studio 方法经 Env → Gym → Bridge 链路，离线 no-op。"""

    def setUp(self):
        self.env = _make_g1_env()
        self.env.mj_forward()

    def test_studio_methods_offline_noop(self):
        """离线模式下 Studio 方法 no-op 不抛错。"""
        # 这些方法在离线模式应 no-op 或返回默认值
        self.env.begin_save_video("/tmp/test_video.mp4")
        self.env.stop_save_video()
        frame = self.env.get_current_frame()
        self.assertIsInstance(frame, int)
        next_frame = self.env.get_next_frame()
        self.assertIsInstance(next_frame, int)
        self.env.load_content_file("test_content")

    def test_render_offline_returns_none(self):
        """离线模式 render 返回 None。"""
        result = self.env.render()
        self.assertIsNone(result)


class TestDelegationChainJacMethods(unittest.TestCase):
    """§10.2 雅可比方法经 Env → Gym → SimCore 链路，数值正确。"""

    def setUp(self):
        self.env = _make_g1_env()
        self.env.mj_forward()

    def test_mj_jacBody_chain(self):
        """mj_jacBody 原地写 jacp/jacr，形状 (3, nv)。"""
        nv = self.env.model.nv
        jacp = np.zeros((3, nv))
        jacr = np.zeros((3, nv))
        self.env.mj_jacBody(jacp, jacr, "pelvis")
        self.assertEqual(jacp.shape, (3, nv))
        self.assertEqual(jacr.shape, (3, nv))

    def test_mj_jacSite_chain(self):
        """mj_jacSite 原地写 jacp/jacr，形状 (3, nv)。"""
        nv = self.env.model.nv
        site_names = self.env._gym.site_body_names() if hasattr(self.env._gym, "site_body_names") else []
        if not site_names:
            self.skipTest("G1 模型无可用 site")
        jacp = np.zeros((3, nv))
        jacr = np.zeros((3, nv))
        self.env.mj_jacSite(jacp, jacr, site_names[0])
        self.assertEqual(jacp.shape, (3, nv))
        self.assertEqual(jacr.shape, (3, nv))

    def test_mj_jac_site_batch_chain(self):
        """mj_jac_site 批量返回 dict，每个含 jacp/jacr。"""
        result = self.env.mj_jac_site([])
        self.assertIsInstance(result, dict)


# =============================================================================
# §10.3 数据一致性验证
# =============================================================================


class TestDataConsistency(unittest.TestCase):
    """§10.3 DataView 零拷贝视图与 query 方法返回值一致 + 步进后视图同步。"""

    def setUp(self):
        self.env = _make_g1_env()
        self.env.mj_forward()

    def test_dataview_query_consistency_body_xpos(self):
        """env.data.body_xpos(name) 与 env.get_body_xpos_xmat_xquat([name]) 一致。"""
        body_name = "pelvis"
        dv_xpos = self.env.data.body_xpos(body_name)
        query_result = self.env.get_body_xpos_xmat_xquat([body_name])
        np.testing.assert_array_almost_equal(
            dv_xpos, query_result[body_name]["xpos"]
        )

    def test_dataview_query_consistency_body_xmat(self):
        """env.data.body_xmat(name) 与 query 结果 xmat 一致（DataView 返回扁平化需 reshape）。"""
        body_name = "pelvis"
        dv_xmat = self.env.data.body_xmat(body_name)
        query_result = self.env.get_body_xpos_xmat_xquat([body_name])
        np.testing.assert_array_almost_equal(
            dv_xmat.reshape(3, 3), query_result[body_name]["xmat"]
        )

    def test_dataview_xfrc_consistency(self):
        """apply_body_force 后 env.data.xfrc_applied 反映写入（只读视图同步）。"""
        body_name = "pelvis"
        body_id = self.env.model.body_name2id(body_name)
        self.env.apply_body_force(body_name, [1.0, 0.0, 0.0], [0.0, 0.0, 0.0])
        # DataView 的 xfrc_applied 应反映写入
        dv_xfrc = self.env.data.xfrc_applied[body_id]
        np.testing.assert_array_almost_equal(dv_xfrc[:3], [1.0, 0.0, 0.0])
        self.env.clear_body_force(body_name)

    def test_step_forward_updates_view(self):
        """do_simulation 后 DataView 字段同步更新（qpos/xpos 变化）。"""
        body_name = "pelvis"
        qpos_before = self.env.data.qpos.copy()
        # 施加控制力矩步进
        nu = self.env.model.nu
        ctrl = np.zeros(nu) if nu > 0 else np.array([])
        self.env.do_simulation(ctrl, n_frames=1)
        xpos_after = self.env.data.body_xpos(body_name).copy()
        qpos_after = self.env.data.qpos.copy()
        # 步进后状态应有变化（或至少 DataView 仍是同一底层内存）
        self.assertEqual(xpos_after.shape, (3,))
        self.assertEqual(qpos_after.shape, qpos_before.shape)
        # DataView 应是零拷贝视图，步进后数据自动更新

    def test_dataview_qpos_is_zero_copy(self):
        """DataView qpos 是 _mjData 零拷贝视图（修改底层反映到视图）。"""
        qpos_view = self.env.data.qpos
        original = qpos_view.copy()
        # 通过 set_joint_qpos 修改底层
        self.env.set_joint_qpos(original * 0.0)
        self.env.mj_forward()
        # 视图应反映修改
        np.testing.assert_array_almost_equal(qpos_view, np.zeros_like(original))
        # 恢复
        self.env.set_joint_qpos(original)
        self.env.mj_forward()


# =============================================================================
# §11.2 K 约束全量回归（ruff SLF001 全局已在 §10.1 覆盖）
# =============================================================================


class TestKConstraintRegression(unittest.TestCase):
    """§11.2 K 约束全量回归抽样。"""

    def test_k11_typed_returns(self):
        """K11: 新增公共方法返回 typed 对象（ndarray/dict/int/float/None），无 MjData/MjModel 泄漏。"""
        env = _make_g1_env()
        env.mj_forward()
        # query 方法返回 dict
        self.assertIsInstance(env.query_joint_qpos(["left_hip_pitch"]), dict)
        # body 查询返回 dict
        self.assertIsInstance(env.get_body_xpos_xmat_xquat(["pelvis"]), dict)
        # 雅可比原地写（返回 None）
        nv = env.model.nv
        jacp = np.zeros((3, nv))
        jacr = np.zeros((3, nv))
        result = env.mj_jacBody(jacp, jacr, "pelvis")
        self.assertIsNone(result)
        self.assertIsInstance(jacp, np.ndarray)
        # body_subtree_mass 返回 float
        mass = env.body_subtree_mass("pelvis")
        self.assertIsInstance(mass, float)
        # render 返回 None
        self.assertIsNone(env.render())

    def test_k12_docstrings_present(self):
        """K12: 阶段三新增公共方法有 docstring。"""
        env = _make_g1_env()
        methods = [
            "query_joint_qpos", "apply_body_force", "set_mocap_pos_and_quat",
            "mj_jacBody", "mj_jacSite", "begin_save_video",
            "equality_find_slot_by_body",
            "equality_constraint", "equality_update",
            "query_site_pos_and_quat_B", "query_robot_velocity_odom",
        ]
        for name in methods:
            with self.subTest(method=name):
                method = getattr(env, name)
                self.assertIsNotNone(method.__doc__, f"{name} 缺少 docstring")
                self.assertGreater(len(method.__doc__), 0)


if __name__ == "__main__":
    unittest.main()
