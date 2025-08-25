from typing import List
from PySide6 import QtCore, QtWidgets, QtGui


class AssetBrowser(QtWidgets.QListView):
    def __init__(self):
        super().__init__()

        self._model = QtCore.QStringListModel()
        self._model.setStringList(
            ["Hallo Welt", "Hei maailma", "Hola Mundo", "Привет мир"]
        )
        self.setModel(self._model)

    def set_assets(self, assets: List[str]):
        pass
