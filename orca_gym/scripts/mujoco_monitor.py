import sys
import numpy as np
import mujoco
from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, 
                             QGroupBox, QSlider, QLabel, QWidget, QPushButton, 
                             QScrollArea, QSplitter, QSizePolicy)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QColor
import pyqtgraph as pg
from collections import deque
from orca_gym.environment import OrcaGymLocalEnv
from orca_gym.orca_gym_local import get_dof_size, get_qpos_size

class MujocoMonitor(QMainWindow):
    def __init__(self,
                 env: OrcaGymLocalEnv,
                 data_queue):
        super().__init__()
        
        # 初始化MuJoCo
        self._env = env
        self._data_queue = data_queue

        
        # 初始化数据存储
        self._max_history = 1000
        self._time_data = deque(maxlen=self._max_history)
        self._joint_data = list(deque(maxlen=self._max_history) for _ in range(30))  # 最多30个关节
        self._env_joint_dict : dict = self._env.model.get_joint_dict().copy()
        self._selected_joint_names = list(self._env_joint_dict.keys())
        # 最多选中30个关节
        self._selected_joint_names = self._selected_joint_names[:30]
        self._env_actuator_dict = self._env.model.get_actuator_dict().copy()
        self._all_actuator_names = list(self._env_actuator_dict.keys())
        self._all_actuator_ctrl_ranges = [self._env_actuator_dict[name]['CtrlRange'] for name in self._all_actuator_names]

        
        # 初始化UI
        self.init_ui()

        # 初始化定时器
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_monitor)
        self.timer.start(50)  # 20Hz更新频率
        

    def init_ui(self):
        # 主布局
        main_widget = QWidget()
        main_layout = QHBoxLayout(main_widget)
        
        # 左侧：图表区
        left_panel = self.create_chart_panel()
        
        # 右侧：控制面板
        right_panel = self.create_control_panel()
        
        # 分割器布局
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        # 修改为（添加int转换）
        splitter.setSizes([
            int(self.width() * 0.7), 
            int(self.width() * 0.3)
        ])
        
        main_layout.addWidget(splitter)
        self.setCentralWidget(main_widget)
        
        # 窗口设置
        self.setWindowTitle("MuJoCo 数据监控系统")
        self.resize(1800, 900)
        
    def create_chart_panel(self):
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # 创建图表
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setBackground('w')
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
        self.plot_widget.setLabel('left', '关节角度 (rad)')
        self.plot_widget.setLabel('bottom', '时间 (s)')
        
        # 为每个关节创建曲线
        self.curves = []
        colors = ['#FF0000', '#00FF00', '#0000FF', '#FFFF00', '#FF00FF', '#00FFFF',
                  '#FFA500', '#800080', '#008000', '#000080', '#808000', '#800000',
                  '#008080', '#FFC0CB', '#A52A2A', '#D2691E', '#6495ED', '#DC143C',
                  '#B22222', '#FF4500', '#2E8B57', '#6A5ACD', '#FF6347', '#4682B4',
                  '#9ACD32', '#FF8C00', '#9932CC', '#8B0000', '#FF1493', '#00CED1']

        for i in range(len(self._selected_joint_names)):
            color = colors[i % len(colors)]
            curve = self.plot_widget.plot(pen=pg.mkPen(color, width=2))
            joint_name = self._selected_joint_names[i]
            curve.setData(name=joint_name if joint_name else f"关节{i}")
            self.curves.append(curve)
            
        # 信息显示区
        self.info_label = QLabel()
        self.info_label.setStyleSheet("font-size: 12px; padding: 5px;")
        self.info_label.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        self.info_label.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        
        layout.addWidget(self.plot_widget, 4)
        layout.addWidget(self.info_label, 1)
        
        return panel
        
    def create_control_panel(self):
        # 滚动区域容器
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        
        # 控制面板容器
        container = QWidget()
        container_layout = QVBoxLayout(container)
        
        # 折叠/展开按钮
        self.toggle_button = QPushButton("隐藏控制面板")
        self.toggle_button.clicked.connect(self.toggle_controls)
        container_layout.addWidget(self.toggle_button)
        
        # 执行器控制组
        self.control_group = QGroupBox("驱动器控制")
        control_layout = QVBoxLayout(self.control_group)
        
        # 为每个执行器创建滑块
        self.sliders = []
        self.slider_labels = []

        for i in range(len(self._all_actuator_names)):
            group = QGroupBox(f"执行器 {i}")
            group_layout = QVBoxLayout(group)
            
            # 标签显示当前值
            label = QLabel("0.00")
            label.setAlignment(Qt.AlignCenter)
            
            # 滑块
            slider = QSlider(Qt.Horizontal)
            slider.setRange(-100, 100)
            slider.setValue(0)
            slider.valueChanged.connect(lambda value, idx=i: self.slider_changed(idx, value))
            
            group_layout.addWidget(label)
            group_layout.addWidget(slider)
            
            self.sliders.append(slider)
            self.slider_labels.append(label)
            control_layout.addWidget(group)
        
        container_layout.addWidget(self.control_group)
        
        # 其他配置值显示
        self.config_group = QGroupBox("配置参数")
        config_layout = QVBoxLayout(self.config_group)
        
        # 添加配置参数显示
        self.config_labels = {
            'time': QLabel(f"仿真时间: 0.00s"),
        }
        
        for label in self.config_labels.values():
            config_layout.addWidget(label)
            
        container_layout.addWidget(self.config_group)
        container_layout.addStretch()
        
        scroll_area.setWidget(container)
        return scroll_area
    
    def toggle_controls(self):
        if self.control_group.isVisible():
            self.control_group.hide()
            self.config_group.hide()
            self.toggle_button.setText("显示控制面板")
        else:
            self.control_group.show()
            self.config_group.show()
            self.toggle_button.setText("隐藏控制面板")
    
    def slider_changed(self, actuator_id, value):
        # 将滑块值(-100到100)映射到实际控制范围
        ctrl_range_mid = self._all_actuator_ctrl_ranges[actuator_id][0] + \
                          (self._all_actuator_ctrl_ranges[actuator_id][1] - self._all_actuator_ctrl_ranges[actuator_id][0]) / 2
        scaled_value = value / 100.0 * (self._all_actuator_ctrl_ranges[actuator_id][1] - self._all_actuator_ctrl_ranges[actuator_id][0]) + ctrl_range_mid
        ctrl = self._env.get_ctrl()
        ctrl[actuator_id] = scaled_value
        self._env.set_ctrl(ctrl)
        self.slider_labels[actuator_id].setText(f"{scaled_value:.2f}")

    def update_monitor(self):
        while not self._data_queue.empty():
            data = self._data_queue.get()

        # 更新图表数据
        self._time_data.append(self._env.data.time)

        for i in range(len(self._selected_joint_names)):
            selected_joint_name = self._selected_joint_names[i]
            joint_qpos_addr = int(self._env_joint_dict[selected_joint_name]['QposIdxStart'])
            self._joint_data[i].append(self._env.data.qpos[joint_qpos_addr])
            self.curves[i].setData(list(self._time_data), list(self._joint_data[i]))

        # 更新信息显示
        info_text = (
            f"仿真状态: 运行中\n"
            f"更新时间: {self._env.data.time:.2f}s\n"
            f"关节数量: {len(self._env_joint_dict)}\n"
            f"驱动器数量: {len(self._all_actuator_names)}\n\n"
        )

        for i in range(len(self._selected_joint_names)):
            joint_name = self._selected_joint_names[i]
            joint_qpos_addr = int(self._env_joint_dict[joint_name]['QposIdxStart'])
            info_text += f"{joint_name if joint_name else f'关节{i}'}: {self._env.data.qpos[joint_qpos_addr]:.4f} rad\n"

        self.info_label.setText(info_text)
        
        # 更新配置参数显示
        self.config_labels['time'].setText(f"仿真时间: {self._env.data.time:.2f}s")

    def closeEvent(self, event):
        self.timer.stop()
        super().closeEvent(event)

def start_monitor_process(env):
    from multiprocessing import Process, Queue
    
    class MonitorProcess(Process):
        def __init__(self, env):
            super().__init__()
            self.env = env
            self.data_queue = Queue(maxsize=100)
            
        def run(self):
            app = QApplication(sys.argv)
            window = MujocoMonitor(self.env, self.data_queue)
            window.show()
            app.exec_()
    
    process = MonitorProcess(env)
    process.start()
    return process