"""ModelRegistry P2 单元测试。

验证点参见 docs/design/development/orca_gym_euler_development.md 第 3.3 节。
"""

from __future__ import annotations

import os
import unittest

import mujoco

from orca_gym.core.euler.model_registry import ModelRegistry

_SCENE_XML = os.path.join(os.path.dirname(__file__), "fixtures", "test_scene.xml")


class TestModelRegistry(unittest.TestCase):
    def setUp(self) -> None:
        self.model = mujoco.MjModel.from_xml_path(_SCENE_XML)
        self.registry = ModelRegistry(self.model, xml_path=_SCENE_XML)

    def test_build_orca_gym_model(self) -> None:
        # build_orca_gym_model() 返回完整 OrcaGymModel
        ogm = self.registry.build_orca_gym_model()
        self.assertEqual(ogm.nq, self.model.nq)
        self.assertEqual(ogm.nv, self.model.nv)
        self.assertEqual(ogm.nu, self.model.nu)
        self.assertEqual(ogm.ngeom, self.model.ngeom)
        # body/site 字典已填充
        self.assertEqual(len(ogm.get_body_dict()), self.model.nbody)
        self.assertEqual(len(ogm.get_site_dict()), self.model.nsite)

    def test_build_orca_gym_data(self) -> None:
        # build_orca_gym_data() 返回 OrcaGymData
        ogd = self.registry.build_orca_gym_data()
        self.assertEqual(ogd.qpos.shape, (self.model.nq,))
        self.assertEqual(ogd.qvel.shape, (self.model.nv,))

    def test_body_subtree_mass(self) -> None:
        # body_subtree_mass(body_name) 返回正确质量
        mass = self.registry.body_subtree_mass("pendulum")
        self.assertAlmostEqual(mass, float(self.model.body_subtreemass[1]))

    def test_equality_data_width(self) -> None:
        # equality_data_width() 返回 eq_data 列数（无等式约束时为 0）
        width = self.registry.equality_data_width()
        if self.model.neq == 0:
            self.assertEqual(width, 0)
        else:
            self.assertEqual(width, self.model.eq_data.shape[1])

    def test_joint_name_by_id(self) -> None:
        # joint_name_by_id(joint_id) 返回正确关节名
        name = self.registry.joint_name_by_id(0)
        self.assertEqual(name, self.model.joint(0).name)


if __name__ == "__main__":
    unittest.main()
