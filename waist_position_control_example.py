#!/usr/bin/env python3
"""
腰部Position控制器使用示例
演示如何使用position控制器控制机器人腰部旋转
"""

import numpy as np

def print_position_control_instructions():
    """
    打印position控制器使用说明
    """
    print("=" * 70)
    print("腰部Position控制器使用说明")
    print("=" * 70)
    print("控制方式：")
    print("1. 左手柄摇杆按下 + X轴左右移动：控制腰部旋转")
    print("2. 右手柄摇杆按下 + X轴左右移动：控制腰部旋转（备用方案）")
    print("3. 左手柄A键 + 右手柄B键：重置腰部到初始位置")
    print("")
    print("Position控制器特性：")
    print("- 使用PID控制，响应更稳定")
    print("- 支持平滑过渡控制")
    print("- 自动限制角度范围：±90度")
    print("- 可配置最大角速度")
    print("")
    print("配置参数：")
    print("- 控制灵敏度：0.1 - 2.0")
    print("- 最大角速度：0.1 - 5.0 rad/s")
    print("- 平滑过渡：开启/关闭")
    print("=" * 70)

def demonstrate_position_control_functions():
    """
    演示position控制器相关函数的使用
    """
    print("\nPosition控制器函数使用示例：")
    print("-" * 50)
    
    # 模拟机器人对象
    class MockRobot:
        def __init__(self):
            self._waist_target_angle = 0.0
            self._waist_control_enabled = False
            self._waist_control_sensitivity = 0.5
            self._waist_smooth_transition = True
            self._waist_max_velocity = 1.0
            self._env = MockEnv()
        
        def set_waist_control_sensitivity(self, sensitivity: float):
            """设置腰部控制灵敏度"""
            self._waist_control_sensitivity = np.clip(sensitivity, 0.1, 2.0)
            print(f"设置腰部控制灵敏度为: {self._waist_control_sensitivity}")
        
        def set_waist_smooth_transition(self, enabled: bool):
            """设置是否启用平滑过渡"""
            self._waist_smooth_transition = enabled
            print(f"平滑过渡控制: {'启用' if enabled else '禁用'}")
        
        def set_waist_max_velocity(self, max_velocity: float):
            """设置腰部最大角速度"""
            self._waist_max_velocity = max(0.1, max_velocity)
            print(f"设置最大角速度为: {self._waist_max_velocity} rad/s ({np.degrees(self._waist_max_velocity):.1f}°/s)")
        
        def get_waist_angle(self):
            """获取当前腰部角度"""
            return self._env.data.qpos[0]
        
        def get_waist_target_angle(self):
            """获取目标腰部角度"""
            return self._waist_target_angle
        
        def reset_waist_control(self):
            """重置腰部控制"""
            self._waist_target_angle = 0.0
            self._waist_control_enabled = False
            print("腰部控制已重置到初始位置")
    
    class MockEnv:
        def __init__(self):
            self.data = MockData()
            self.dt = 0.01
    
    class MockData:
        def __init__(self):
            self.qpos = [0.1]  # 模拟当前角度
    
    # 创建模拟机器人
    robot = MockRobot()
    
    # 演示函数使用
    print("1. 获取当前腰部角度:")
    current_angle = robot.get_waist_angle()
    print(f"   当前角度: {current_angle:.3f} 弧度 ({np.degrees(current_angle):.1f} 度)")
    
    print("\n2. 获取目标腰部角度:")
    target_angle = robot.get_waist_target_angle()
    print(f"   目标角度: {target_angle:.3f} 弧度 ({np.degrees(target_angle):.1f} 度)")
    
    print("\n3. 配置控制参数:")
    robot.set_waist_control_sensitivity(0.8)
    robot.set_waist_smooth_transition(True)
    robot.set_waist_max_velocity(1.5)
    
    print("\n4. 重置腰部控制:")
    robot.reset_waist_control()

def simulate_position_control():
    """
    模拟position控制器控制过程
    """
    print("\n模拟Position控制器控制过程：")
    print("-" * 50)
    
    # 模拟控制参数
    max_angle = np.pi / 2  # 90度
    sensitivity = 0.5
    max_velocity = 1.0  # rad/s
    dt = 0.01  # 时间步长
    
    # 模拟摇杆输入序列
    joystick_inputs = [0.0, 0.5, 0.8, -0.3, -0.7, 0.0]
    
    current_angle = 0.0
    target_angle = 0.0
    
    print("时间步 | 摇杆输入 | 角度增量 | 目标角度 | 当前角度 | 平滑目标")
    print("-" * 60)
    
    for i, joystick_input in enumerate(joystick_inputs):
        # 计算角度增量
        angle_delta = joystick_input * max_angle * sensitivity * dt
        target_angle += angle_delta
        target_angle = np.clip(target_angle, -max_angle, max_angle)
        
        # 平滑过渡控制
        angle_error = target_angle - current_angle
        max_angle_step = max_velocity * dt
        
        if abs(angle_error) > max_angle_step:
            angle_step = np.sign(angle_error) * max_angle_step
            smooth_target = current_angle + angle_step
        else:
            smooth_target = target_angle
        
        # 更新当前角度（模拟position控制器的响应）
        current_angle = smooth_target
        
        print(f"{i+1:6d} | {joystick_input:8.1f} | {np.degrees(angle_delta):8.2f}° | {np.degrees(target_angle):8.1f}° | {np.degrees(current_angle):8.1f}° | {np.degrees(smooth_target):8.1f}°")

def compare_control_methods():
    """
    比较不同控制方法
    """
    print("\n控制方法比较：")
    print("-" * 50)
    
    methods = [
        {
            "name": "Motor控制器",
            "type": "力矩控制",
            "pros": ["直接力矩控制", "响应快速", "适合精确控制"],
            "cons": ["需要手动调参", "可能不稳定", "需要PD控制实现"]
        },
        {
            "name": "Position控制器",
            "type": "位置控制",
            "pros": ["内置PID控制", "稳定可靠", "自动调参", "平滑控制"],
            "cons": ["响应稍慢", "依赖控制器参数"]
        }
    ]
    
    for method in methods:
        print(f"\n{method['name']} ({method['type']}):")
        print("  优点:")
        for pro in method['pros']:
            print(f"    - {pro}")
        print("  缺点:")
        for con in method['cons']:
            print(f"    - {con}")

if __name__ == "__main__":
    print_position_control_instructions()
    demonstrate_position_control_functions()
    simulate_position_control()
    compare_control_methods()
    
    print("\n" + "=" * 70)
    print("Position控制器优势：")
    print("1. 更稳定的控制响应")
    print("2. 内置PID控制，无需手动调参")
    print("3. 支持平滑过渡，避免剧烈运动")
    print("4. 自动限制角度范围，更安全")
    print("5. 适合遥操作应用")
    print("=" * 70)
