import asyncio
from typing import List
import ctypes
from ctypes import wintypes, windll
from PySide6 import QtCore, QtWidgets, QtGui
from PySide6.QtCore import Qt

from orca_gym.orca_lab.remote_scene import RemoteScene

user32 = ctypes.windll.user32
WM_COPYDATA = 0x004A

class COPYDATASTRUCT(ctypes.Structure):
    _fields_ = [
        ("dwData", wintypes.LPARAM),
        ("cbData", wintypes.DWORD),
        ("lpData", wintypes.LPVOID)
    ]


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
        self.setDragEnabled(True)
        
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

    def mousePressEvent(self, event):
        if event.button() == QtCore.Qt.LeftButton:
            index = self.indexAt(event.pos())
            
            if index.isValid():
                self.selected_item_name = index.data(QtCore.Qt.DisplayRole)
                print(self.selected_item_name)
            else:
                self.selected_item_name = None
            self.dragging = True

    def mouseReleaseEvent(self, event):
        if self.dragging and event.button() == QtCore.Qt.LeftButton:
            self.dragging = False

            if not self.selected_item_name:
                print("未选择资产，取消发送")
                return
            
            # 获取全局鼠标坐标
            pos = QtGui.QCursor.pos()
            point = wintypes.POINT(pos.x(), pos.y())

            dpiX = wintypes.UINT()
            dpiY = wintypes.UINT()
            monitor = windll.user32.MonitorFromWindow(self.hwnd_target, 2)
            windll.shcore.GetDpiForMonitor(monitor, 0, ctypes.byref(dpiX), ctypes.byref(dpiY))
            scale_x = dpiX.value / 96.0
            scale_y = dpiY.value / 96.0

            point.x = int(point.x * scale_x)
            point.y = int(point.y * scale_y)

            # 鼠标下的窗口
            hwnd_under = user32.WindowFromPoint(point)
            if hwnd_under == self.hwnd_target:
                # 转换为客户区坐标
                user32.ScreenToClient(hwnd_under, ctypes.byref(point))
                self.add_item_by_drag.emit(self.selected_item_name, point.x, point.y)
            else:
                print("释放在非目标窗口")