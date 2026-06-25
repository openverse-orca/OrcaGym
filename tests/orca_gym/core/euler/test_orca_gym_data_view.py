"""OrcaGymDataView P2 单元测试。

验证点参见 docs/design/development/orca_gym_euler_development.md 第 3.3 节。
"""

from __future__ import annotations

import os
import unittest

import numpy as np

from orca_gym.core.euler.mujoco_sim_core import MuJoCoSimCore
from orca_gym.core.euler.orca_gym_data_view import OrcaGymDataView
from orca_gym.core.euler.model_registry import ModelRegistry

_SCENE_XML = os.path.join(os.path.dirname(__file__), "fixtures", "test_scene.xml")


class TestOrcaGymDataView(unittest.TestCase):
    def setUp(self) -> None:
        self.sim = MuJoCoSimCore()
        self.sim.init_simulation(_SCENE_XML)
        self.registry = ModelRegistry(self.sim._mjModel, xml_path=_SCENE_XML)
        self.model = self.registry.build_orca_gym_model()
        self.view = OrcaGymDataView(self.model)
        # forward 后同步，确保派生状态可用
        self.sim.forward()
        self.sim.sync_to_view(self.view)

    def test_qpos_qvel_qacc_consistent_after_sync(self) -> None:
        # sync 后 DataView 字段与 _mjData 一致
        d = self.sim._mjData
        np.testing.assert_allclose(self.view.qpos, d.qpos)
        np.testing.assert_allclose(self.view.qvel, d.qvel)
        np.testing.assert_allclose(self.view.qacc, d.qacc)
        np.testing.assert_allclose(self.view.qfrc_bias, d.qfrc_bias)

    def test_body_xpos_by_name(self) -> None:
        # body_xpos("world") 返回正确位置（world body 在原点）
        xpos = self.view.body_xpos("world")
        np.testing.assert_allclose(xpos, self.sim._mjData.xpos[0])

    def test_body_cvel_by_name(self) -> None:
        # body_cvel(body_name) 返回正确速度
        cvel = self.view.body_cvel("pendulum")
        np.testing.assert_allclose(cvel, self.sim._mjData.cvel[1])

    def test_body_subtree_mass_by_name(self) -> None:
        # body_subtree_mass(body_name) 返回正确质量
        mass = self.view.body_subtree_mass("pendulum")
        self.assertAlmostEqual(mass, float(self.sim._mjModel.body_subtreemass[1]))

    def test_site_xpos_by_name(self) -> None:
        # site_xpos(site_name) 返回正确位置
        xpos = self.view.site_xpos("tip")
        np.testing.assert_allclose(xpos, self.sim._mjData.site_xpos[0])

    def test_xfrc_applied_is_read_only_view(self) -> None:
        # xfrc_applied 是 _mjData.xfrc_applied 的视图（共享内存）
        self.assertTrue(
            np.shares_memory(self.view.xfrc_applied, self.sim._mjData.xfrc_applied)
        )

    def test_missing_field_raises_guidance(self) -> None:
        # 访问不存在的字段抛出引导性错误
        with self.assertRaises(AttributeError) as ctx:
            _ = self.view.nonexistent_field
        msg = str(ctx.exception)
        self.assertIn("OrcaGymDataView", msg)
        self.assertIn("nonexistent_field", msg)

    def test_time_field(self) -> None:
        # time 字段与 _mjData.time 一致
        self.assertAlmostEqual(self.view.time, float(self.sim._mjData.time))


if __name__ == "__main__":
    unittest.main()
