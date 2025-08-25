from actor_outline_model import ActorOutlineModel, GroupActor

from PySide6.QtTest import QAbstractItemModelTester
from PySide6.QtCore import QModelIndex, Qt

import unittest

from orca_gym.orca_lab.actor import AssetActor


class TestAddFunction(unittest.TestCase):
    def test_empty_path_is_invalid(self):
        model = ActorOutlineModel()

        group1 = GroupActor("g1")
        group2 = GroupActor("g2", group1)
        group3 = GroupActor("g3", group1)
        group4 = GroupActor("g4", group3)
        asset1 = AssetActor("a1", "spw_name", group1)

        model.set_root_group(group1)

        self.assertEqual(model.rowCount(QModelIndex()), 3)
        index1 = model.index(0, 0, QModelIndex())
        self.assertEqual(index1.isValid(), True)
        self.assertEqual(index1.data(Qt.DisplayRole), "g2")
        self.assertEqual(
            model.parent(model.index(0, 0, QModelIndex())).isValid(), False
        )
        self.assertEqual(
            model.parent(model.index(1, 0, QModelIndex())).isValid(), False
        )
        self.assertEqual(
            model.parent(model.index(2, 0, QModelIndex())).isValid(), False
        )

        mode = QAbstractItemModelTester.FailureReportingMode.Fatal
        tester = QAbstractItemModelTester(model, mode)


if __name__ == "__main__":
    unittest.main()

