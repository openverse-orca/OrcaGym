from PySide6 import QtCore, QtWidgets, QtGui
from enum import Enum

from orca_gym.orca_lab.actor import BaseActor
from orca_gym.orca_lab.path import Path


class ActorOutline(QtWidgets.QTreeView):
    actor_selection_changed = QtCore.Signal(list)

    def __init__(self):
        super().__init__()
        self.setHeaderHidden(True)
        self._change_from_inside = False
        self._change_from_outside = False

    def set_actor_model(self, model):
        self.setModel(model)
        self.selectionModel().selectionChanged.connect(self._on_selection_changed)

    @QtCore.Slot()
    def _on_selection_changed(self):
        if self._change_from_outside:
            return

        self._change_from_inside = True

        actors = []
        indexes = self.selectedIndexes()
        for index in indexes:
            actor = index.internalPointer()
            if not isinstance(actor, BaseActor):
                raise Exception("Invalid actor.")
            actors.append(actor)

        self.actor_selection_changed.emit(actors)

        self._change_from_inside = False

    def set_actor_selection(self, actors: list[BaseActor]):
        if self._change_from_inside:
            return

        self._change_from_outside = True

        selection_model = self.selectionModel()
        selection_model.clearSelection()

        model = self.model()
        for actor in actors:
            index = model.get_index_from_actor(actor)
            if not index.isValid():
                raise Exception("Invalid actor.")

            selection_model.select(
                index, QtCore.QItemSelectionModel.SelectionFlag.Select
            )
            self.scrollTo(index)

        self._change_from_outside = False


if __name__ == "__main__":
    import sys
    from actor_outline_model import ActorOutlineModel
    from orca_gym.orca_lab.actor import GroupActor, AssetActor

    app = QtWidgets.QApplication(sys.argv)
    actor_outline = ActorOutline()
    model = ActorOutlineModel()

    group1 = GroupActor("g1")
    group2 = GroupActor("g2", group1)
    group3 = GroupActor("g3", group1)
    group4 = GroupActor("g4", group3)
    asset1 = AssetActor("a1", "spw_name", group1)

    model.set_root_group(group1)
    actor_outline.set_actor_model(model)
    actor_outline.show()

    app.exec()
