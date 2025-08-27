import asyncio
from typing import List
from PySide6 import QtCore, QtWidgets, QtGui


class AssetBrowser(QtWidgets.QListView):

    add_item = QtCore.Signal(str)

    def __init__(self):
        super().__init__()

        self._model = QtCore.QStringListModel()
        
        self.setModel(self._model)
        self.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self.show_context_menu)

    def set_assets(self, assets: List[str]):
        assets = [str(asset) for asset in assets]
        self._model.setStringList(assets)
        self.setModel(self._model)

    def show_context_menu(self, pos):
        index = self.indexAt(pos)
        if not index.isValid():
            return
        selected_item_name = index.data(QtCore.Qt.DisplayRole)
        context_menu = QtWidgets.QMenu(self)
        add_action = QtGui.QAction(f"Add {selected_item_name}", self)
        add_action.triggered.connect(lambda: self.on_add_item(selected_item_name))
        context_menu.addAction(add_action)
        context_menu.exec(self.mapToGlobal(pos))

    def on_add_item(self, item_name):
        self.add_item.emit(item_name)