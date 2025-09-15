from PySide6 import QtCore, QtWidgets, QtGui

from orca_gym.orca_lab.actor import BaseActor, GroupActor, AssetActor
from orca_gym.orca_lab.math import Transform
from orca_gym.orca_lab.ui.transform_edit import TransformEdit


class ActorEditor(QtWidgets.QWidget):
    transform_changed = QtCore.Signal()

    def __init__(self, parent=None):
        super().__init__(parent)

        self._layout = QtWidgets.QVBoxLayout()
        self.setLayout(self._layout)
        self.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Preferred,
            QtWidgets.QSizePolicy.Policy.Preferred,
        )

        self._actor: BaseActor | None = None
        self._transform_edit: TransformEdit | None = None

        self._refresh()

    @property
    def actor(self) -> BaseActor | None:
        return self._actor

    @actor.setter
    def actor(self, actor: BaseActor | None):
        if self._actor == actor:
            return

        self._actor = actor
        self._refresh()
    '''
    def _clear_layout(self):
        layout = self._layout
        for i in reversed(range(layout.count())):
            item = layout.itemAt(i)
            if item.widget():
                item.widget().setParent(None)
            elif item.layout():
                self._clear_layout(item.layout())
            layout.removeItem(item)
            del item
    '''
    def _clear_layout(self, layout=None):
        if layout is None:
            layout = self._layout
        while layout.count():
            item = layout.takeAt(0)  # 更安全，直接取出而不是 itemAt()
            if item.widget():
                w = item.widget()
                layout.removeWidget(w)
                w.setParent(None)
            elif item.layout():
                self._clear_layout(item.layout())
                layout.removeItem(item)


    def _refresh(self):
        # self._clear_layout()

        if self._actor is None:
            label = QtWidgets.QLabel("No actor selected")
            label.setAlignment(
                QtCore.Qt.AlignmentFlag.AlignCenter
                | QtCore.Qt.AlignmentFlag.AlignVCenter
            )
            self._layout.addWidget(label)
            return

        label = QtWidgets.QLabel(f"Actor: {self._actor.name}")
        self._layout.addWidget(label)

        line = QtWidgets.QFrame()
        line.setFrameShape(QtWidgets.QFrame.Shape.HLine)
        line.setFrameShadow(QtWidgets.QFrame.Shadow.Sunken)
        self._layout.addWidget(line)

        transform_edit = TransformEdit()
        transform_edit.value_changed.connect(self._on_transform_changed)
        transform_edit.start_drag.connect(self._on_start_drag)
        transform_edit.stop_drag.connect(self._on_stop_drag)

        transform_edit.set_transform(self._actor.transform)

        # transform_edit.set_transform(self._actor.transform)
        self._transform_edit = transform_edit
        self._layout.addWidget(transform_edit)

        self._layout.addSpacing(10)

        line = QtWidgets.QFrame()
        line.setFrameShape(QtWidgets.QFrame.Shape.HLine)
        line.setFrameShadow(QtWidgets.QFrame.Shadow.Sunken)
        self._layout.addWidget(line)

        if isinstance(self._actor, AssetActor):
            label = QtWidgets.QLabel(f"Asset Name {self._actor.spawnable_name}")
            self._layout.addWidget(label)

        self._layout.addStretch(1)

    def _on_transform_changed(self):
        self.transform_changed.emit()

    def _on_start_drag(self):
        pass

    def _on_stop_drag(self):
        pass

    @property
    def transform(self):
        if self._transform_edit is None:
            return Transform()
        return self._transform_edit.transform

    def update_ui(self):
        if self._transform_edit is not None and self._actor is not None:
            self._transform_edit.set_transform(self._actor.transform)


if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication(sys.argv)

    actor = GroupActor("group1")

    editor = ActorEditor()
    editor.resize(400, 50)
    editor.show()

    editor.actor = actor

    def cb():
        print("actor transform changed")
        print(editor.transform)

    editor.transform_changed.connect(cb)

    sys.exit(app.exec())
