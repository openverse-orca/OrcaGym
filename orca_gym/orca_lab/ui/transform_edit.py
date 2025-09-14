from PySide6 import QtCore, QtWidgets, QtGui

from enum import Enum, auto

import numpy as np

from orca_gym.orca_lab.ui.float_edit import FloatEdit
from orca_gym.orca_lab.math import Transform, as_euler

from scipy.spatial.transform import Rotation


def is_close(a: float, b: float, tol: float = 1e-6) -> bool:
    return abs(a - b) < tol


class FloatInputState(Enum):
    Idle = auto()
    MouseDown = auto()
    Typing = auto()
    Dragging = auto()


class TransformEdit(QtWidgets.QWidget):
    value_changed = QtCore.Signal()
    start_drag = QtCore.Signal()
    stop_drag = QtCore.Signal()

    def __init__(self, parent=None):
        super().__init__(parent)

        self._layout = QtWidgets.QVBoxLayout()
        self.setLayout(self._layout)

        self.labels = []

        self._pos_x = self._add_line("Position X", FloatEdit())
        self._pos_y = self._add_line("Y", FloatEdit())
        self._pos_z = self._add_line("Z", FloatEdit())

        self._layout.addSpacing(10)

        self._rot_x = self._add_line("Rotation X", FloatEdit(step =1.0))
        self._rot_y = self._add_line("Y", FloatEdit(step =1.0))
        self._rot_z = self._add_line("Z", FloatEdit(step =1.0))

        self._layout.addSpacing(10)

        self._scale_uniform = self._add_line("Scale Uniform", FloatEdit())

    def _add_line(self, label, widget):
        layout = QtWidgets.QHBoxLayout()
        self._layout.addLayout(layout)

        label_widget = QtWidgets.QLabel(label)
        label_widget.setFixedWidth(80)
        label_widget.setAlignment(
            QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter
        )
        layout.addWidget(label_widget)
        layout.addWidget(widget, 1)

        self.labels.append(label_widget)

        widget.value_changed.connect(self._on_value_changed)

        return widget

    def _on_start_drag(self):
        self.start_drag.emit()

    def _on_stop_drag(self):
        self.stop_drag.emit()

    def _on_value_changed(self):
        self.value_changed.emit()

    @property
    def transform(self):
        transform = Transform()

        transform.position = np.array(
            [self._pos_x.value(), self._pos_y.value(), self._pos_z.value()],
            dtype=np.float64,
        )

        angles = [self._rot_x.value(), self._rot_y.value(), self._rot_z.value()]
        r = Rotation.from_euler(
            "xyz",
            angles,
            degrees=True,
        )
        quat = r.as_quat(scalar_first=True)
        transform.rotation = quat

        transform.scale = self._scale_uniform.value()
        return transform

    def set_transform(self, transform: Transform):

        self._pos_x.set_value(transform.position[0])
        self._pos_y.set_value(transform.position[1])
        self._pos_z.set_value(transform.position[2])

        # r = Rotation.from_quat(transform.rotation.tolist(), scalar_first=True)
        # angles = r.as_euler("xyz", degrees=True)
        angles = as_euler(transform.rotation, "xyz", degrees=True)

        self._rot_x.set_value(angles[0])
        self._rot_y.set_value(angles[1])
        self._rot_z.set_value(angles[2])

        self._scale_uniform.set_value(transform.scale)


if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication(sys.argv)
    input = TransformEdit()
    input.show()
    input.resize(400, 50)

    def on_value_changed():
        print("value changed", input.transform())

    input.value_changed.connect(on_value_changed)

    sys.exit(app.exec())
