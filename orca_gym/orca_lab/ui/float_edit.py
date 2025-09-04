from PySide6 import QtCore, QtWidgets, QtGui

from enum import Enum, auto


def is_close(a: float, b: float, tol: float = 1e-6) -> bool:
    return abs(a - b) < tol


class FloatInputState(Enum):
    Idle = auto()
    MouseDown = auto()
    Typing = auto()
    Dragging = auto()


class FloatEdit(QtWidgets.QLineEdit):
    value_changed = QtCore.Signal(float)
    start_drag = QtCore.Signal()
    stop_drag = QtCore.Signal()

    def __init__(self, parent=None):
        super().__init__(parent)

        self.setValidator(QtGui.QDoubleValidator())
        self.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)
        self.setCursor(QtCore.Qt.CursorShape.SizeHorCursor)
        self.setMouseTracking(True)

        self._state = FloatInputState.Idle
        self._value = 0.0
        self.setText("0.0")

        self.textChanged.connect(self._text_changed)

        self.installEventFilter(self)

    def set_state(self, state: FloatInputState):
        if state == FloatInputState.Idle:
            assert self._state in [FloatInputState.Typing, FloatInputState.Dragging]
            self.setCursor(QtCore.Qt.CursorShape.SizeHorCursor)
        elif state == FloatInputState.MouseDown:
            assert self._state == FloatInputState.Idle
        elif state == FloatInputState.Typing:
            assert self._state == FloatInputState.MouseDown
            self.setCursor(QtCore.Qt.CursorShape.IBeamCursor)
        elif state == FloatInputState.Dragging:
            assert self._state == FloatInputState.MouseDown
        else:
            raise Exception(f"Invalid state transition: {self._state} -> {state}")

        self._state = state

    def eventFilter(self, watched, event):

        if event.type() == QtCore.QEvent.Type.KeyPress:
            if self._state == FloatInputState.Typing:
                event: QtGui.QKeyEvent = event
                keys = [
                    QtCore.Qt.Key.Key_Return,
                    QtCore.Qt.Key.Key_Enter,
                    QtCore.Qt.Key.Key_Escape,
                ]
                if self.hasFocus() and event.key() in keys:
                    # clearFocus will trigger FocusOut event, which will set state to Idle
                    self.clearFocus()
                    assert self._state == FloatInputState.Idle

        if event.type() == QtCore.QEvent.Type.FocusOut:
            if self._state == FloatInputState.Typing:
                self.set_state(FloatInputState.Idle)

        if event.type() == QtCore.QEvent.Type.MouseButtonPress:
            if self._state == FloatInputState.Idle:
                event: QtGui.QMouseEvent = event
                self.grabMouse()
                self.set_state(FloatInputState.MouseDown)
                self.last_mouse_pos = event.globalPosition()

        if event.type() == QtCore.QEvent.Type.MouseButtonRelease:
            if self._state == FloatInputState.MouseDown:
                self.releaseMouse()
                self.setFocus()
                self.set_state(FloatInputState.Typing)

            if self._state == FloatInputState.Dragging:
                self.releaseMouse()
                self.set_state(FloatInputState.Idle)

                self.stop_drag.emit()

        if event.type() == QtCore.QEvent.Type.MouseMove:
            if self._state == FloatInputState.MouseDown:
                self.set_state(FloatInputState.Dragging)
                self.start_drag.emit()

            if self._state == FloatInputState.Dragging:
                event: QtGui.QMouseEvent = event
                delta = event.globalPosition().x() - self.last_mouse_pos.x()
                value = self._value + delta * 0.01
                self.value = value
                self.last_mouse_pos = event.globalPosition()

        return super().eventFilter(watched, event)

    def _text_changed(self, text: str):
        if self._state != FloatInputState.Typing:
            return

        try:
            value = float(text)
            self._value = value
            self.value_changed.emit(value)
        except ValueError:
            pass

    @property
    def value(self) -> float:
        return self._value

    @value.setter
    def value(self, value: float):
        if not is_close(value, self._value):
            self._value = value
            self.setText(f"{self._value:.2f}")
            self.value_changed.emit(self._value)


if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication(sys.argv)
    # input = QtWidgets.QLineEdit()
    input = FloatEdit()
    input.show()
    input.resize(400, 50)

    # input.value_changed.connect(lambda value: print(f"Value changed: {value}"))
    input.start_drag.connect(lambda: print("Start drag"))
    input.stop_drag.connect(lambda: print("Stop drag"))

    sys.exit(app.exec())
