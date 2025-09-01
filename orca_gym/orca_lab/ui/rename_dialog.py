from PySide6 import QtCore, QtWidgets, QtGui
from typing import Callable, Tuple

from orca_gym.orca_lab.path import Path


class RenameDialog(QtWidgets.QDialog):
    def __init__(
        self,
        actor_path: Path,
        can_rename: Callable[[Path, str], Tuple[bool, str]],
        parent=None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Rename Actor")
        self.setModal(True)
        self.resize(300, 100)
        self.actor_path = actor_path
        self.can_rename = can_rename

        self.new_name = None
        self._current_name = actor_path.name()
        self._init_ui()

    def _init_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        self.setLayout(layout)

        self.line_edit = QtWidgets.QLineEdit(self)
        self.line_edit.setText(self._current_name)
        layout.addWidget(self.line_edit)

        self.error_message = QtWidgets.QLabel()
        self.error_message.setStyleSheet("QLabel { color: red; }")
        layout.addWidget(self.error_message)

        self.line_edit.selectAll()
        self.line_edit.setFocus()

        ButtonRole = QtWidgets.QDialogButtonBox.ButtonRole
        button_box = QtWidgets.QDialogButtonBox()
        button_box.addButton("Confirm", ButtonRole.AcceptRole).clicked.connect(
            self.accept
        )
        button_box.addButton("Cancel", ButtonRole.RejectRole).clicked.connect(
            self.reject
        )
        layout.addWidget(button_box)

    def accept(self) -> None:
        new_name = self.line_edit.text().strip()

        [can_rename, error] = self.can_rename(self.actor_path, new_name)
        if can_rename == False:
            self.error_message.setText(error)
            return

        self.new_name = new_name
        super().accept()

    def reject(self):
        return super().reject()


if __name__ == "__main__":
    app = QtWidgets.QApplication()

    count = 0

    def can_rename(path: Path, name: str) -> Tuple[bool, str]:
        global count
        count = count + 1
        if count == 2:
            return True, ""

        return False, "Not allowed"

    rename_dialog = RenameDialog(Path("/g1/g3/g4"), can_rename)

    result = rename_dialog.exec()
    if result == QtWidgets.QDialog.DialogCode.Accepted:
        print(f"New name: {rename_dialog.new_name}")
    else:
        print("Cancelled")
