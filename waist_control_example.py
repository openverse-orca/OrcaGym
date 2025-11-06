#!/usr/bin/env python3
"""
腰部控制使用示例
演示如何通过Pico手柄控制机器人腰部旋转
"""

import numpy as np

def print_waist_control_instructions():
    """
    打印腰部控制说明
    """
    print("=" * 60)
    print("腰部控制说明")
    print("=" * 60)
    print("控制方式：")
    print("1. 左手柄摇杆按下 + X轴左右移动：控制腰部旋转")
    print("2. 右手柄摇杆按下 + X轴左右移动：控制腰部旋转（备用方案）")
    print("3. 左手柄A键 + 右手柄B键：重置腰部到初始位置")
    print("")
    print("控制参数：")
    print("- 最大旋转角度：±90度")
    print("- 默认灵敏度：0.5")
    print("- 控制范围：[-π/2, π/2] 弧度")
    print("")
    print("注意事项：")
    print("- 需要先按下摇杆才能控制腰部")
    print("- 松开摇杆后腰部会保持当前位置")
    print("- 使用组合键可以快速重置到初始位置")
    print("=" * 60)

def demonstrate_waist_control_functions():
    """
    演示腰部控制相关函数的使用
    """
    print("\n腰部控制函数使用示例：")
    print("-" * 40)
    
    # 模拟机器人对象（实际使用时需要真实的机器人实例）
    class MockRobot:
        def __init__(self):
            self._waist_target_angle = 0.0
            self._waist_control_enabled = False
            self._waist_control_sensitivity = 0.5
            self._env = MockEnv()
        
        def set_waist_control_sensitivity(self, sensitivity: float):
            """设置腰部控制灵敏度"""
            self._waist_control_sensitivity = np.clip(sensitivity, 0.1, 2.0)
            print(f"设置腰部控制灵敏度为: {self._waist_control_sensitivity}")
        
        def get_waist_angle(self):
            """获取当前腰部角度"""
            return self._env.data.qpos[0]  # 模拟当前角度
        
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
    
    print("\n3. 调整控制灵敏度:")
    robot.set_waist_control_sensitivity(0.8)
    
    print("\n4. 重置腰部控制:")
    robot.reset_waist_control()

def simulate_joystick_control():
    """
    模拟手柄控制过程
    """
    print("\n模拟手柄控制过程：")
    print("-" * 40)
    
    # 模拟手柄状态
    joystick_states = [
        {"leftHand": {"joystickPosition": [0.0, 0.0], "joystickPressed": False}, 
         "rightHand": {"joystickPosition": [0.0, 0.0], "joystickPressed": False}},
        {"leftHand": {"joystickPosition": [0.5, 0.0], "joystickPressed": True}, 
         "rightHand": {"joystickPosition": [0.0, 0.0], "joystickPressed": False}},
        {"leftHand": {"joystickPosition": [-0.3, 0.0], "joystickPressed": True}, 
         "rightHand": {"joystickPosition": [0.0, 0.0], "joystickPressed": False}},
        {"leftHand": {"joystickPosition": [0.0, 0.0], "joystickPressed": False}, 
         "rightHand": {"joystickPosition": [0.0, 0.0], "joystickPressed": False}},
    ]
    
    # 模拟控制过程
    waist_target_angle = 0.0
    sensitivity = 0.5
    dt = 0.01  # 模拟时间步长
    
    for i, state in enumerate(joystick_states):
        print(f"\n步骤 {i+1}:")
        left_joystick_x = state["leftHand"]["joystickPosition"][0]
        left_pressed = state["leftHand"]["joystickPressed"]
        
        if left_pressed:
            max_angle = np.pi / 2
            angle_delta = left_joystick_x * max_angle * sensitivity * dt
            waist_target_angle += angle_delta
            waist_target_angle = np.clip(waist_target_angle, -max_angle, max_angle)
            print(f"   摇杆输入: {left_joystick_x:.1f}")
            print(f"   角度增量: {np.degrees(angle_delta):.2f} 度")
            print(f"   目标角度: {np.degrees(waist_target_angle):.1f} 度")
        else:
            print("   摇杆未按下，保持当前位置")

if __name__ == "__main__":
    print_waist_control_instructions()
    demonstrate_waist_control_functions()
    simulate_joystick_control()
    
    print("\n" + "=" * 60)
    print("示例完成！")
    print("在实际使用中，请确保：")
    print("1. 机器人已正确初始化")
    print("2. Pico手柄已连接并正常工作")
    print("3. 腰部关节配置正确")
    print("=" * 60)

