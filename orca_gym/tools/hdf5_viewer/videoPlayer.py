import sys
import os
import numpy as np
from PyQt5.QtCore import Qt, QUrl, QTimer, QSize, QPoint
from PyQt5.QtGui import (QPalette, QColor, QKeyEvent, QImage, QPixmap,
                         QPainter, QPen, QBrush)
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget,
                             QVBoxLayout, QHBoxLayout, QFrame,
                             QSlider, QLabel, QPushButton,
                             QFileDialog, QLineEdit, QMessageBox,
                             QScrollArea, QGridLayout, QListWidget,
                             QListWidgetItem, QProgressBar)
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtMultimediaWidgets import QVideoWidget

import av
import h5py
import bisect

from orca_gym.log.orca_log import get_orca_logger
_logger = get_orca_logger()


class VideoPlayer(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.media_player = QMediaPlayer(None, QMediaPlayer.VideoSurface)
        self.video_widget = QVideoWidget()
        self.media_player.setVideoOutput(self.video_widget)

        self.init_ui()

        self.play_btn.clicked.connect(self.play)
        self.pause_btn.clicked.connect(self.pause)
        self.stop_btn.clicked.connect(self.stop)
        self.open_btn.clicked.connect(self.open_file)

        self.total_frames = 0
        self.pts_list = []
        self.frame_times = []

    def init_ui(self):
        layout  = QVBoxLayout(self)
        self.video_widget.setMinimumSize(QSize(1280, 720))

        #控制面板
        control_layout = QHBoxLayout()

        self.play_btn = QPushButton("播放")
        self.pause_btn = QPushButton("暂停")
        self.stop_btn = QPushButton("停止")
        self.open_btn = QPushButton("打开视频")

        self.frame_input = QLineEdit()
        self.frame_input.setPlaceholderText("输入帧号跳转")
        self.frame_input.setMaximumWidth(120)

        self.position_slider = QSlider(Qt.Horizontal)
        self.position_slider.setRange(0, 0)

        # 状态标签
        self.status_label = QLabel("就绪")
        self.frame_label = QLabel("帧: 0/0")
        self.time_label = QLabel("时间: 00:00/00:00")

        # 添加控制按钮
        control_layout.addWidget(self.play_btn)
        control_layout.addWidget(self.pause_btn)
        control_layout.addWidget(self.stop_btn)
        control_layout.addWidget(self.open_btn)
        control_layout.addWidget(QLabel("帧:"))
        control_layout.addWidget(self.frame_input)
        control_layout.addStretch(1)
        control_layout.addWidget(self.time_label)
        control_layout.addWidget(self.frame_label)
        control_layout.addWidget(self.status_label)

        # 添加组件到主布局
        layout.addWidget(self.video_widget)
        layout.addLayout(control_layout)
        layout.addWidget(self.position_slider)

    def open_file(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "打开视频文件", "", "*.mp4")

        if file_name:
            self.get_mp4_frame_info(file_name)
            self.load_video(file_name)

    def load_video(self, file_path):
        self.media_player.setMedia(QMediaContent(QUrl.fromLocalFile(file_path)))
        self.status_label.setText(f"已加载: {os.path.basename(file_path)}")
        self.frame_input.setText("")
        self.play()

    def get_mp4_frame_info(self, file_path) -> tuple[int, list[int], list[int]]:
        """
        获取 MP4 文件的总帧数、时间戳(PTS)和帧时间

        参数:
            file_path: MP4 文件路径

        返回:
            total_frames: 总帧数
            pts_list: 每一帧的 PTS (原始时间戳)
            frame_times: 每一帧的时间(秒)
            :param file_path:
            :return:
        """

        container = av.open(file_path)
        video_stream = next(s for s in container.streams if s.type == 'video')

        self.total_frames = 0
        self.pts_list = []
        self.frame_times = []

        time_base = video_stream.time_base

        for packet in container.demux(video_stream):
            for frame in packet.decode():
                # 只处理视频帧
                if isinstance(frame, av.VideoFrame):
                    self.pts_list.append(frame.pts)
                    self.frame_times.append(float(frame.pts * time_base))

        self.total_frames = len(self.pts_list)

        _logger.performance(time_base)
        _logger.info(self.pts_list)
        _logger.performance(self.frame_times)
        _logger.info(self.total_frames)

        container.close()
        return self.total_frames, self.pts_list, self.frame_times

    def update_frame_index(self, position):
        """
        根据当前播放位置更新帧索引
        :param position: 当前播放位置（毫秒）
        """
        if not self.frame_times or self.total_frames == 0:
            return

        # 将毫秒转换成秒
        current_time = position / 1000

        idx = bisect.bisect_left(self.frame_times, current_time)


    def play(self):
        self.media_player.play()
        self.status_label.setText("正在播放...")

    def pause(self):
        self.media_player.pause()
        self.status_label.setText("已暂停...")

    def stop(self):
        self.media_player.stop()
        self.status_label.setText("已停止")
        self.time_label.setText("00:00/00:00")
        self.frame_label.setText("帧： 0/0")

