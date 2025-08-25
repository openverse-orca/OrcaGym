import typing
from typing import override

from PySide6.QtCore import QAbstractItemModel, QModelIndex, Qt

from orca_gym.orca_lab.actor import BaseActor, GroupActor


class ActorOutlineModel(QAbstractItemModel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.column_count = 1
        self.m_root_group: GroupActor | None = None

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
        )

        return f

    def set_root_group(self, group: GroupActor):
        if not isinstance(group, GroupActor):
            raise TypeError("Root group must be an instance of GroupActor.")

        self.beginResetModel()
        self.m_root_group = group
        self.endResetModel()
