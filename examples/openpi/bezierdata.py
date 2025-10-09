import json
from math import pi
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation as R

class AngleSmoother:
    def __init__(self):
        self.prev_angle = None
        self.offset = 0
    
    def smooth_angle(self, current_angle):
        """实时平滑角度，避免跳跃"""
        if self.prev_angle is None:
            self.prev_angle = current_angle
            return current_angle
        
        # 计算角度差
        diff = current_angle - self.prev_angle
        print(" ", diff)
        
        # 如果角度差超过π，说明发生了跳跃
        if diff > 3.14 :#np.pi:
            if current_angle - 0 < 0.00001:
                self.offset = -3.14#np.pi
            else:
                self.offset = -2 * 3.14#np.pi
        elif diff < -3.14 :#np.pi:
            if current_angle - 0 < 0.00001:
                self.offset = 3.14#np.pi
            else:
                self.offset = 2 * 3.14#np.pi
        # 应用偏移
        smooth_angle = current_angle + self.offset
        if smooth_angle > 3.14*2:
            smooth_angle = smooth_angle - 3.14*2
        elif smooth_angle < -3.14*2:
            smooth_angle = smooth_angle + 3.14*2
        self.prev_angle = current_angle
        #self.offset = 0
        
        return smooth_angle

    def reset(self):
        self.prev_angle = None
        self.offset = 0

# 使用示例
class BezierPath:
    def __init__(self, json_path):
        """
        初始化贝塞尔路径
        
        参数:
        json_path: JSON文件路径
        """
        self.control_points = self._load_points(json_path)
        self.curve_points = None
        self.current_position = 0
        self.cumulative_lengths = None
        self.total_length = 0

        self.smoother = AngleSmoother()
        
        # 生成曲线点
        self._generate_curve()

    def reset(self):
        self.smoother.reset()
        
    def _load_points(self, json_path):
        """从JSON文件加载控制点"""
        with open(json_path, 'r') as f:
            data = json.load(f)
            
        points = []
        for point in data['points']:
            points.append([point['x'], point['y'], point['z']])
            
        return np.array(points)
    
    def _de_casteljau(self, points, t):
        """德卡斯特里奥算法计算贝塞尔曲线点"""
        points = points.copy()
        n = len(points)
        
        for r in range(1, n):
            for i in range(n - r):
                points[i] = (1 - t) * points[i] + t * points[i + 1]
        
        return points[0]
    
    def _generate_curve(self, num_points=500):
        """生成贝塞尔曲线点"""
        t_values = np.linspace(0, 1, num_points)
        self.curve_points = np.array([
            self._de_casteljau(self.control_points, t) for t in t_values
        ])
   #     print("Curve data....", self.curve_points )
        # 计算累积长度
        distances = np.sqrt(np.sum(np.diff(self.curve_points, axis=0)**2, axis=1))
        self.cumulative_lengths = np.zeros(len(self.curve_points))
        self.cumulative_lengths[1:] = np.cumsum(distances)
        self.total_length = self.cumulative_lengths[-1]
    
    def get_position(self, distance):
        """
        获取指定距离处的位置
        
        参数:
        distance: 从起点开始的距离
        
        返回:
        position: 位置坐标 [x, y, z]
        """
        if distance <= 0:
            return self.curve_points[0]
        if distance >= self.total_length:
            return self.curve_points[-1]
            
        idx = np.searchsorted(self.cumulative_lengths, distance)
        if idx == 0:
            return self.curve_points[0]
            
        # 线性插值
        t = (distance - self.cumulative_lengths[idx-1]) / (
            self.cumulative_lengths[idx] - self.cumulative_lengths[idx-1]
        )
        return self.curve_points[idx-1] + t * (
            self.curve_points[idx] - self.curve_points[idx-1]
        )
    
    def get_direction(self, distance):
        """
        获取指定距离处的方向（切线方向）
        
        参数:
        distance: 从起点开始的距离
        
        返回:
        direction: 归一化的方向向量 [dx, dy, dz]
        """
        eps = 1e-6  # 小偏移量
        pos1 = self.get_position(distance - eps)
        pos2 = self.get_position(distance + eps)
        direction = pos2 - pos1
     #   direction = [1,0,0]
     #   direction = direction / np.linalg.norm(direction)



        # 假设 direction 是目标朝向向量
    #    direction = np.array([0, 1, 0])  # 示例：Y轴方向
        direction = direction / np.linalg.norm(direction)  # 归一化

  #      print("direction44444444: ", direction)

        

        # 默认Z轴朝上，计算从Z轴到direction的旋转
        y_axis = np.array([0, 1, 0])
        rotation = R.align_vectors([direction], [y_axis])[0]  # 去掉内层方括号

        # 获取欧拉角 (ZYX顺序，即yaw-pitch-roll)
        euler_angles = rotation.as_euler('xyz', degrees=False)  # [roll, pitch, yaw]
      #  euler_angles[2] = euler_angles[2] - pi/2 
       # print(f"Roll: {euler_angles[0]:.2f}°, Pitch: {euler_angles[1]:.2f}°, Yaw: {euler_angles[2]:.2f}°")
        return euler_angles 
    
    def update_position(self, current_distance, speed, dt):
        """
        更新位置
        
        参数:
        current_distance: 当前距离
        speed: 当前速度
        dt: 时间步长
        
        返回:
        new_position: 新位置 [x, y, z]
        new_distance: 新距离
        """
        new_distance = current_distance + speed * dt
        new_distance = min(max(0, new_distance), self.total_length)
        #添加返回该位置的切线方向
        direction = self.get_direction(new_distance)
        direction = [0,0, self.smoother.smooth_angle(direction[2])]
       # print("output direction: ", direction)
        return self.get_position(new_distance), new_distance, direction
    
    def visualize(self, show_control_points=True):
        """可视化贝塞尔曲线"""
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # 绘制曲线
        ax.plot(self.curve_points[:, 0], 
                self.curve_points[:, 1], 
                self.curve_points[:, 2], 
                'b-', label='Bezier Curve')
        
        if show_control_points:
            # 绘制控制点
            ax.scatter(self.control_points[:, 0], 
                      self.control_points[:, 1], 
                      self.control_points[:, 2], 
                      color='red', s=50, label='Control Points')
            
            # 绘制控制多边形
            ax.plot(self.control_points[:, 0], 
                   self.control_points[:, 1], 
                   self.control_points[:, 2], 
                   'r--', alpha=0.5, label='Control Polygon')
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.legend()
        plt.show()
'''
def main():
    # 使用示例
    json_path = "path_to_your_json_file.json"  # 替换为实际的JSON文件路径
    
    # 创建贝塞尔路径对象
    bezier_path = BezierPath(json_path)
    
    # 可视化路径
    bezier_path.visualize()
    
    # 模拟物体运动示例
    current_distance = 0
    speed = 1.0  # 单位/秒
    dt = 0.016   # 时间步长（约60fps）
    
    # 模拟运动
    while current_distance < bezier_path.total_length:
        position, current_distance = bezier_path.update_position(
            current_distance, speed, dt
        )
        direction = bezier_path.get_direction(current_distance)
        
        print(f"Position: {position}")
        print(f"Direction: {direction}")
        print(f"Distance: {current_distance}/{bezier_path.total_length}")
        print("---")

if __name__ == "__main__":
    main()

'''