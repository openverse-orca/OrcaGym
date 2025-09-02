from typing import override

from PySide6.QtCore import QAbstractItemModel, QModelIndex, Qt, QMimeData, Signal

from orca_gym.orca_lab.actor import BaseActor, GroupActor
from orca_gym.orca_lab.local_scene import LocalScene
from orca_gym.orca_lab.path import Path


class ReparentData:
    def __init__(self):
        # actor to be reparented
        self.actor: BaseActor = None
        self.actor_path: Path = None

        # new parent
        self.parent: BaseActor = None
        self.parent_path: Path = None


class ActorOutlineModel(QAbstractItemModel):
    # actor path, new parent path, index to insert at (-1 means append to the end)
    request_reparent = Signal(Path, Path, int)

    def __init__(self, local_scene: LocalScene, parent=None):
        super().__init__(parent)
        self.column_count = 1
        self.m_root_group: GroupActor | None = None
        self.reparent_mime = "application/x-orca-actor-reparent"
        self.local_scene = local_scene

    def get_actor(self, index: QModelIndex) -> BaseActor | None:
        if not index.isValid():
            return self.m_root_group

        if index.model() != self:
            return None

        actor = index.internalPointer()
        if isinstance(actor, BaseActor):
            return actor

        return None

    def get_index_from_actor(self, actor: BaseActor) -> QModelIndex:
        if not isinstance(actor, BaseActor):
            raise ValueError("Invalid actor.")

        if actor == self.m_root_group:
            return QModelIndex()

        parent_actor = actor.parent
        if parent_actor is None:
            raise Exception("Actor that is not pseudo root should always has a parent.")

        index = -1
        children = parent_actor.children
        for i, child in enumerate(children):
            if child == actor:
                index = i
                break

        if index == -1:
            raise Exception("Child not found from it's parent.")

        return self.createIndex(index, 0, actor)

    @override
    def index(self, row, column, /, parent=...):
        if row < 0 or column < 0 or column >= self.column_count:
            return QModelIndex()

        if not parent.isValid():
            child = self.m_root_group.children[row]
            if child is not None:
                return self.createIndex(row, column, child)
        else:
            if parent.column() == 0:
                parent_actor = self.get_actor(parent)
                if isinstance(parent_actor, GroupActor):
                    child = parent_actor.children[row]
                    return self.createIndex(row, column, child)

        return QModelIndex()

    @override
    def parent(self, child):
        super().parent()
        if not child.isValid():
            return QModelIndex()

        actor = self.get_actor(child)
        if actor is None:
            return QModelIndex()

        parent_actor = actor.parent
        if parent_actor == self.m_root_group:
            return QModelIndex()

        return self.get_index_from_actor(parent_actor)

    @override
    def rowCount(self, /, parent=...):
        if not parent.isValid():
            if self.m_root_group is not None:
                return len(self.m_root_group.children)
        else:
            if parent.column() == 0:
                actor = self.get_actor(parent)
                if isinstance(actor, GroupActor):
                    return len(actor.children)
        return 0

    @override
    def columnCount(self, /, parent=...):
        return 1

    @override
    def data(self, index, /, role=...):
        if not index.isValid():
            return None

        actor = self.get_actor(index)
        if actor is None:
            return None

        if index.column() == 0:
            if role == Qt.DisplayRole:
                return actor.name

        return None

    @override
    def flags(self, index, /):
        ItemFlag = Qt.ItemFlag

        if not index.isValid():
            return ItemFlag.ItemIsDropEnabled

        f = (
            ItemFlag.ItemIsEnabled
            | ItemFlag.ItemIsSelectable
            | ItemFlag.ItemIsDragEnabled
            | ItemFlag.ItemIsDropEnabled
            # | ItemFlag.ItemIsEditable
        )

        return f

    def set_root_group(self, group: GroupActor):
        if not isinstance(group, GroupActor):
            raise TypeError("Root group must be an instance of GroupActor.")

        self.beginResetModel()
        self.m_root_group = group
        self.endResetModel()

    def supportedDropActions(self):
        return Qt.CopyAction

    def supportedDragActions(self):
        return Qt.CopyAction

    def dropMimeData(self, data, action, row, column, parent):

        reparent_data = ReparentData()
        if not self.prepare_reparent_data(
            reparent_data, data, action, row, column, parent
        ):
            return False

        self.request_reparent.emit(
            reparent_data.actor_path, reparent_data.parent_path, row
        )

        return True

    def canDropMimeData(self, data, action, row, column, parent):
        reparent_data = ReparentData()
        return self.prepare_reparent_data(
            reparent_data, data, action, row, column, parent
        )

    def mimeData(self, indexes):
        if len(indexes) == 0:
            return None

        actor = self.get_actor(indexes[0])
        if actor is None:
            return None

        actor_path = self.local_scene.get_actor_path(actor)
        if actor_path is None:
            return None

        mime_data = QMimeData()
        mime_data.setData(self.reparent_mime, actor_path.string().encode("utf-8"))
        return mime_data

    def mimeTypes(self):
        return [self.reparent_mime]

    def prepare_reparent_data(
        self,
        reparent_data: ReparentData,
        mime_data: QMimeData,
        action,
        row,
        column,
        parent: QModelIndex,
    ) -> bool:
        if not mime_data.hasFormat(self.reparent_mime):
            return False

        if action != Qt.CopyAction:
            return False

        if column > 0:
            return False

        parent_actor = self.get_actor(parent)
        if parent_actor is None:
            return False

        parent_actor_path = self.local_scene.get_actor_path(parent_actor)
        if parent_actor_path is None:
            return False

        actor_path_bytes = mime_data.data(self.reparent_mime)
        actor_path_str = actor_path_bytes.data().decode("utf-8")

        actor_path = Path(actor_path_str)
        actor = self.local_scene.find_actor_by_path(actor_path)

        print(f"drop {parent_actor_path}, row: {row}, col:{column}")

        ok, err = self.local_scene.can_reparent_actor(actor, parent_actor)
        if not ok:
            return False

        reparent_data.actor = actor
        reparent_data.actor_path = actor_path
        reparent_data.parent = parent_actor
        reparent_data.parent_path = parent_actor_path

        return True


if __name__ == "__main__":
    import unittest
    from orca_gym.orca_lab.actor import AssetActor
    from PySide6.QtTest import QAbstractItemModelTester
    from PySide6.QtCore import QModelIndex, Qt

    class TestAddFunction(unittest.TestCase):
        def test_empty_path_is_invalid(self):
            local_scene = LocalScene()

            local_scene.add_actor(GroupActor("g1"), Path("/"))
            local_scene.add_actor(GroupActor("g2"), Path("/"))
            local_scene.add_actor(GroupActor("g3"), Path("/"))
            local_scene.add_actor(GroupActor("g4"), Path("/g2"))
            local_scene.add_actor(AssetActor("a1", "spw_name"), Path("/g3"))

            model = ActorOutlineModel(local_scene)
            model.set_root_group(local_scene.root_actor)

            self.assertEqual(model.rowCount(QModelIndex()), 3)
            index1 = model.index(0, 0, QModelIndex())
            self.assertEqual(index1.isValid(), True)
            self.assertEqual(index1.data(Qt.DisplayRole), "g1")
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

    unittest.main()
