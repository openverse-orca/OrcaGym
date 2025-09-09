import sys
from PySide6 import QtCore


class Platform:
    def __init__(self):
        if sys.platform == "linux":
            from orca_gym.orca_lab.ui.platform_linux import PlatformHandlerLinux
            self.platform = PlatformHandlerLinux()
        elif sys.platform == "win32":
            from orca_gym.orca_lab.ui.platform_win import PlatformHandlerWin
            self.platform = PlatformHandlerWin()
        
    def get_current_window_pos(self, pos, hwnd_target):
        if sys.platform == "win32":
            point = self.platform.win_pos(pos)
            self.platform.get_dpi(hwnd_target)
            scale_x = self.platform.dpiX.value / 96.0
            scale_y = self.platform.dpiY.value / 96.0
            point.x = int(point.x * scale_x)
            point.y = int(point.y * scale_y)

            hwnd_under = self.platform.user32.WindowFromPoint(point)
            if hwnd_under == hwnd_target:
                point = self.platform.screen_to_client(point, hwnd_under)
                relative_pos = (point.x, point.y)
                return relative_pos
            else:
                print("释放在非目标窗口")
                return None


        elif sys.platform == "linux":
            display = self.platform.display
            root = display.screen().root
            pointer = root.query_pointer()
            mouse_x = pointer.root_x
            mouse_y = pointer.root_y

            window = display.create_resource_object('window', hwnd_target)
            geom = window.get_geometry()
            window_geometry = window.translate_coords(root, 0, 0)
            window_x = -window_geometry.x
            window_y = -window_geometry.y

            relative_x = mouse_x - window_x
            relative_y = mouse_y - window_y

            window.map()
            if 0 <= relative_x < geom.width and 0 <= relative_y < geom.height:
                return (relative_x, relative_y)
            else:
                print("释放在非目标窗口")
                return None
            
    def is_in_actor_outline(self, actor_outline_hwnd, pos):
        point = self.platform.win_pos(pos)
        self.platform.get_dpi(actor_outline_hwnd)
        scale_x = self.platform.dpiX.value / 96.0
        scale_y = self.platform.dpiY.value / 96.0
        point.x = int(point.x * scale_x)
        point.y = int(point.y * scale_y)
        return self.platform.is_in(actor_outline_hwnd, point)