import asyncio
from typing import List
from PySide6 import QtCore, QtWidgets, QtGui
from PySide6.QtCore import Qt
from orca_gym.orca_lab.ui.platform import Platform
import time

class AssetListModel(QtCore.QStringListModel):
    asset_mime = "application/x-orca-asset"

    def mimeTypes(self):
        return [self.asset_mime]

    def mimeData(self, indexes):
        if not indexes:
            return None
        asset_name = indexes[0].data(Qt.DisplayRole)
        mime = QtCore.QMimeData()
        mime.setData(self.asset_mime, asset_name.encode("utf-8"))
        return mime


class AssetBrowser(QtWidgets.QListView):

    add_item = QtCore.Signal(str)
    add_item_by_drag = QtCore.Signal(str, int, int)

    def __init__(self, hwnd_target=None):
        super().__init__()

        self._model = AssetListModel()
        self.hwnd_target = hwnd_target
        self.dragging = False
        self.selected_item_name = None
        self._drag_start_pos = None
        self.setDragEnabled(False)
        self.actor_outline_hwnd = None
        
        self.setModel(self._model)
        self.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self.show_context_menu)
        self.platform = Platform()

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
    
    def mousePressEvent(self, event):
        print("in preser")
        if event.button() == QtCore.Qt.LeftButton:
            index = self.indexAt(event.pos())
            if index.isValid():
                self.selected_item_name = index.data(QtCore.Qt.DisplayRole)
                self._drag_start_pos = event.pos()
            else:
                self.selected_item_name = None
                self._drag_start_pos = None
            self.dragging = True
            super().mousePressEvent(event)
    '''
    def mouseMoveEvent(self, event):
        if event.buttons() & QtCore.Qt.LeftButton and self._drag_start_pos:
            distance = (event.pos() - self._drag_start_pos).manhattanLength()
            
            if distance >= QtWidgets.QApplication.startDragDistance():
                
                drag = QtGui.QDrag(self)
                
                mime_data = QtCore.QMimeData()
                mime_data.setData("text/plain", self.selected_item_name.encode("utf-16le"))
                drag.setMimeData(mime_data)
                
                mime_data = self._model.mimeData([self.currentIndex()])
                mime_data.setText(self.selected_item_name)
                drag.setMimeData(mime_data)
                
                drop_action = drag.exec(Qt.CopyAction | Qt.MoveAction)
                
                if drop_action == Qt.IgnoreAction:
                    self._drag_start_pos = None
                    point = self.platform.get_current_window_pos(QtGui.QCursor.pos(), self.hwnd_target)
                    if point:
                        self.add_item_by_drag.emit(self.selected_item_name, point[0], point[1])
                
            super().mouseMoveEvent(event)
    ''' 

    def mouseReleaseEvent(self, event):
        print("release")
        self._drag_start_pos = None

        if self.platform.is_in_actor_outline(self.actor_outline_hwnd, QtGui.QCursor.pos()):
            self.on_add_item(self.selected_item_name)

        point = self.platform.get_current_window_pos(QtGui.QCursor.pos(), self.hwnd_target)
        if point:
            self.add_item_by_drag.emit(self.selected_item_name, point[0], point[1])
        
        super().mouseReleaseEvent(event)