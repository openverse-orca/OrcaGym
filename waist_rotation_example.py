#!/usr/bin/env python3
"""
腰部转动对末端位置影响示例
演示如何让末端位置跟着腰部转动
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

def demonstrate_waist_rotation_effect():
    """
    演示腰部转动对末端位置的影响
    """
    print("=" * 70)
    print("腰部转动对末端位置影响演示")
    print("=" * 70)
    
    # 模拟末端执行器在局部坐标系中的位置
    local_pos = np.array([0.5, 0.2, 0.8])  # 末端执行器位置
    local_quat = np.array([1.0, 0.0, 0.0, 0.0])  # 末端执行器姿态
    
    # 模拟基座位置和姿态
    base_pos = np.array([0.0, 0.0, 0.0])
    base_quat = np.array([1.0, 0.0, 0.0, 0.0])  # 基座姿态
    
    # 不同的腰部角度
    waist_angles = np.linspace(0, 2*np.pi, 8)
    
    print("腰部角度变化对末端位置的影响：")
    print("腰部角度(度) | 末端X位置 | 末端Y位置 | 末端Z位置")
    print("-" * 50)
    
    positions = []
    
    for waist_angle in waist_angles:
        # 创建腰部转动的四元数 (绕Z轴旋转)
        waist_quat = np.array([
            np.cos(waist_angle / 2),  # w
            0,                        # x
            0,                        # y
            np.sin(waist_angle / 2)   # z
        ])
        
        # 组合基座和腰部的旋转
        base_rot = R.from_quat([base_quat[1], base_quat[2], base_quat[3], base_quat[0]])
        waist_rot = R.from_quat([waist_quat[1], waist_quat[2], waist_quat[3], waist_quat[0]])
        combined_rot = base_rot * waist_rot
        
        # 计算全局位置
        global_pos = base_pos + combined_rot.apply(local_pos)
        positions.append(global_pos)
        
        print(f"{np.degrees(waist_angle):8.1f}° | {global_pos[0]:8.3f} | {global_pos[1]:8.3f} | {global_pos[2]:8.3f}")
    
    # 可视化
    positions = np.array(positions)
    
    fig = plt.figure(figsize=(12, 8))
    
    # 3D轨迹图
    ax1 = fig.add_subplot(221, projection='3d')
    ax1.plot(positions[:, 0], positions[:, 1], positions[:, 2], 'b-o', linewidth=2, markersize=6)
    ax1.scatter(positions[0, 0], positions[0, 1], positions[0, 2], color='green', s=100, label='起始位置')
    ax1.scatter(positions[-1, 0], positions[-1, 1], positions[-1, 2], color='red', s=100, label='结束位置')
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('Z (m)')
    ax1.set_title('末端执行器轨迹 (3D)')
    ax1.legend()
    ax1.grid(True)
    
    # XY平面投影
    ax2 = fig.add_subplot(222)
    ax2.plot(positions[:, 0], positions[:, 1], 'b-o', linewidth=2, markersize=6)
    ax2.scatter(positions[0, 0], positions[0, 1], color='green', s=100, label='起始位置')
    ax2.scatter(positions[-1, 0], positions[-1, 1], color='red', s=100, label='结束位置')
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.set_title('末端执行器轨迹 (XY平面)')
    ax2.legend()
    ax2.grid(True)
    ax2.axis('equal')
    
    # 位置分量随时间变化
    ax3 = fig.add_subplot(223)
    ax3.plot(np.degrees(waist_angles), positions[:, 0], 'r-o', label='X位置')
    ax3.plot(np.degrees(waist_angles), positions[:, 1], 'g-o', label='Y位置')
    ax3.plot(np.degrees(waist_angles), positions[:, 2], 'b-o', label='Z位置')
    ax3.set_xlabel('腰部角度 (度)')
    ax3.set_ylabel('位置 (m)')
    ax3.set_title('位置分量随腰部角度变化')
    ax3.legend()
    ax3.grid(True)
    
    # 距离基座的距离
    distances = np.linalg.norm(positions - base_pos, axis=1)
    ax4 = fig.add_subplot(224)
    ax4.plot(np.degrees(waist_angles), distances, 'purple', linewidth=2, marker='o')
    ax4.set_xlabel('腰部角度 (度)')
    ax4.set_ylabel('距离基座距离 (m)')
    ax4.set_title('末端执行器到基座的距离')
    ax4.grid(True)
    
    plt.tight_layout()
    plt.savefig('waist_rotation_effect.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\n轨迹分析：")
    print(f"起始位置: ({positions[0, 0]:.3f}, {positions[0, 1]:.3f}, {positions[0, 2]:.3f})")
    print(f"结束位置: ({positions[-1, 0]:.3f}, {positions[-1, 1]:.3f}, {positions[-1, 2]:.3f})")
    print(f"轨迹半径: {np.linalg.norm(positions[0, :2]):.3f} m")
    print(f"Z坐标变化: {positions[-1, 2] - positions[0, 2]:.3f} m")

def demonstrate_coordinate_transformation():
    """
    演示坐标系转换的数学原理
    """
    print("\n" + "=" * 70)
    print("坐标系转换数学原理")
    print("=" * 70)
    
    # 局部坐标系中的点
    local_point = np.array([1.0, 0.0, 0.0])
    print(f"局部坐标系中的点: {local_point}")
    
    # 腰部转动角度
    waist_angle = np.pi / 4  # 45度
    print(f"腰部转动角度: {np.degrees(waist_angle):.1f}°")
    
    # 方法1: 使用四元数
    waist_quat = np.array([
        np.cos(waist_angle / 2),  # w
        0,                        # x
        0,                        # y
        np.sin(waist_angle / 2)   # z
    ])
    
    # 转换为旋转矩阵
    rot_quat = R.from_quat([waist_quat[1], waist_quat[2], waist_quat[3], waist_quat[0]])
    rotation_matrix = rot_quat.as_matrix()
    
    # 应用旋转
    rotated_point_quat = rotation_matrix @ local_point
    
    # 方法2: 直接使用旋转矩阵
    rotation_matrix_direct = np.array([
        [np.cos(waist_angle), -np.sin(waist_angle), 0],
        [np.sin(waist_angle),  np.cos(waist_angle), 0],
        [0,                     0,                   1]
    ])
    
    rotated_point_direct = rotation_matrix_direct @ local_point
    
    print(f"\n旋转矩阵 (四元数方法):")
    print(rotation_matrix)
    print(f"\n旋转矩阵 (直接方法):")
    print(rotation_matrix_direct)
    
    print(f"\n旋转后的点 (四元数方法): {rotated_point_quat}")
    print(f"旋转后的点 (直接方法): {rotated_point_direct}")
    print(f"两种方法结果一致: {np.allclose(rotated_point_quat, rotated_point_direct)}")

def demonstrate_implementation_details():
    """
    演示实现细节
    """
    print("\n" + "=" * 70)
    print("实现细节说明")
    print("=" * 70)
    
    print("1. 修改的方法：")
    print("   - _local_to_global(): 局部到全局坐标系转换")
    print("   - _global_to_local(): 全局到局部坐标系转换")
    print("   - query_site_pos_and_quat_B(): 查询末端位置和姿态")
    print("   - query_site_xvalp_xvalr_B(): 查询末端速度")
    print("   - get_obs(): 获取观测数据")
    
    print("\n2. 关键参数：")
    print("   - waist_angle: 腰部关节角度 (弧度)")
    print("   - 绕Z轴旋转: 符合腰部yaw关节的运动")
    print("   - 四元数表示: 避免万向锁问题")
    
    print("\n3. 数学原理：")
    print("   - 组合旋转: R_combined = R_base * R_waist")
    print("   - 位置转换: pos_global = pos_base + R_combined * pos_local")
    print("   - 姿态转换: quat_global = R_combined * quat_local")
    
    print("\n4. 优势：")
    print("   - 末端位置自动跟随腰部转动")
    print("   - 保持相对位置关系")
    print("   - 支持连续转动")
    print("   - 数学上正确且稳定")

if __name__ == "__main__":
    demonstrate_waist_rotation_effect()
    demonstrate_coordinate_transformation()
    demonstrate_implementation_details()
    
    print("\n" + "=" * 70)
    print("总结")
    print("=" * 70)
    print("通过修改坐标系转换方法，现在末端执行器的位置计算会考虑腰部关节的转动。")
    print("这意味着当腰部转动时，末端执行器在全局坐标系中的位置会相应地发生变化，")
    print("但在局部坐标系中保持相对稳定的位置关系。")
    print("=" * 70)
