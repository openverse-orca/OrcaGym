import sys

from PyQt5.QtGui import QPalette, QColor
from PyQt5.QtWidgets import QFrame, QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QDesktopWidget

import videoPlayer
from orca_gym.hdf5_viewer.videoPlayer import VideoPlayer


class HDF5Viewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("HDF5/MP4 Viewer")
        self.setGeometry(100, 100, 1600, 900)

        main_widget = QWidget()
        main_layout = QHBoxLayout()

        left_frame = VideoPlayer() #视频区域
        right_frame = QWidget() #数据区域

        #设置背景颜色为黑色
        left_palette = left_frame.palette()
        left_palette.setColor(QPalette.Window, QColor(55, 55, 55))
        left_frame.setPalette(left_palette)
        left_frame.setAutoFillBackground(True)

        divider_line = QFrame()
        divider_line.setFrameShape(QFrame.VLine)  # 垂直分割线
        divider_line.setFrameShadow(QFrame.Sunken)
        divider_line.setLineWidth(1)  # 1像素宽的分割线
        divider_line.setMidLineWidth(0)
        divider_line.setStyleSheet("background-color: #a0a0a0;")

        main_layout.addWidget(left_frame, 3)
        main_layout.addWidget(divider_line)
        main_layout.addWidget(right_frame, 1)

        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)

        self.center()
        self.show()

    def center(self):
        screen_geometry = QDesktopWidget().availableGeometry()
        screen_center = screen_geometry.center()

        window_geometry = self.frameGeometry()
        window_geometry.moveCenter(screen_center)

        self.move(window_geometry.topLeft())

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = HDF5Viewer()
    sys.exit(app.exec_())