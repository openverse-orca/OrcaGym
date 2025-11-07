#!/usr/bin/env python3
"""
交互式手臂姿态调试工具
可以实时查看和调整关节角度
"""

import sys
import os
current_file_path = os.path.abspath('')
project_root = os.path.dirname(os.path.dirname(current_file_path))
sys.path.insert(0, project_root)

import numpy as np

# 当前配置
current_config = {
    "right": [-1.9, 0.5, 0, 2.0, -1.5708, 0, 0],
    "left":  [1.9, -0.5, 0, 2.0, 1.5708, 0, 0],
}

joint_names = ["J1", "J2", "J3", "J4", "J5", "J6", "J7"]

def print_config():
    print("\n" + "="*60)
    print("当前配置：")
    print("="*60)
    print(f"右臂: {current_config['right']}")
    print(f"左臂: {current_config['left']}")
    print()
    for i, name in enumerate(joint_names):
        print(f"  {name}: R={current_config['right'][i]:6.2f} ({np.degrees(current_config['right'][i]):6.1f}°), "
              f"L={current_config['left'][i]:6.2f} ({np.degrees(current_config['left'][i]):6.1f}°)")
    print("="*60)

def print_python_code():
    print("\n复制到dexforce_w1_config.py:")
    print("-" * 60)
    print('CURRENT_TEST = {')
    print(f'    "right": {current_config["right"]},')
    print(f'    "left":  {current_config["left"]},')
    print('}')
    print('CURRENT_NEUTRAL = CURRENT_TEST')
    print("-" * 60)

def adjust_joint():
    print("\n选择要调整的关节:")
    print("1-7: J1-J7, 0: 完成并输出配置")
    choice = input("输入关节编号: ").strip()
    
    if choice == '0':
        return False
    
    try:
        joint_idx = int(choice) - 1
        if joint_idx < 0 or joint_idx > 6:
            print("无效的关节编号！")
            return True
    except:
        print("无效输入！")
        return True
    
    joint_name = joint_names[joint_idx]
    print(f"\n调整 {joint_name}:")
    print(f"  当前值: R={current_config['right'][joint_idx]:.3f}, L={current_config['left'][joint_idx]:.3f}")
    
    print("\n选择操作:")
    print("1: 修改右臂")
    print("2: 修改左臂")
    print("3: 修改双臂（相同值）")
    print("4: 修改双臂（对称值）")
    
    op = input("选择: ").strip()
    
    if op == '1':
        val = float(input(f"右臂{joint_name}新值（弧度）: "))
        current_config['right'][joint_idx] = val
    elif op == '2':
        val = float(input(f"左臂{joint_name}新值（弧度）: "))
        current_config['left'][joint_idx] = val
    elif op == '3':
        val = float(input(f"双臂{joint_name}新值（弧度）: "))
        current_config['right'][joint_idx] = val
        current_config['left'][joint_idx] = val
    elif op == '4':
        val = float(input(f"右臂{joint_name}新值（弧度）: "))
        current_config['right'][joint_idx] = val
        current_config['left'][joint_idx] = -val
    
    return True

def main():
    print("="*60)
    print("  手臂姿态交互式调试工具")
    print("="*60)
    print("\n根据运行结果调整关节角度")
    print("提示：1.57 rad ≈ 90°, 3.14 rad ≈ 180°")
    
    while True:
        print_config()
        if not adjust_joint():
            break
    
    print_python_code()
    
    print("\n使用方法：")
    print("1. 复制上面的代码到 dexforce_w1_config.py")
    print("2. 重新运行测试程序")
    print("3. 观察效果，继续调整")

if __name__ == "__main__":
    main()

