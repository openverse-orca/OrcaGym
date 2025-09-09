import sys
import ctypes
from ctypes import wintypes

class COPYDATASTRUCT(ctypes.Structure):
    _fields_ = [
        ("dwData", wintypes.LPARAM),
        ("cbData", wintypes.DWORD),
        ("lpData", wintypes.LPVOID)
    ]

class PlatformHandlerWin:
    def __init__(self):
        self.user32 = ctypes.windll.user32
        self.WM_COPYDATA = 0x004A
        self.copydatastruct = COPYDATASTRUCT
        self.dpiX = wintypes.UINT()
        self.dpiY = wintypes.UINT()
    
    def win_pos(pos):
        return wintypes.POINT(pos.x(), pos.y())

    def get_dpi(self, hwnd_target):
        monitor = self.user32.MonitorFromWindow(hwnd_target, 2)
        ctypes.windll.shcore.GetDpiForMonitor(monitor, 0, ctypes.byref(self.dpiX), ctypes.byref(self.dpiY))
        
    def screen_to_client(self, point, hwnd_under):
        self.user32.ScreenToClient(hwnd_under, ctypes.byref(point))
        return point