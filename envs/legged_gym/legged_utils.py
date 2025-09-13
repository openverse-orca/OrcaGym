import math
import numpy as np
from orca_gym.utils import rotations

def local2global(q_global_to_local, v_local, v_omega_local) -> tuple[np.ndarray, np.ndarray]:
    # Vg = QVlQ*

    # 将速度向量从局部坐标系转换到全局坐标系
    q_v_local = np.array([0, v_local[0], v_local[1], v_local[2]])  # 局部坐标系下的速度向量表示为四元数
    q_v_global = rotations.quat_mul(q_global_to_local, rotations.quat_mul(q_v_local, rotations.quat_conjugate(q_global_to_local)))
    v_global = np.array(q_v_global[1:])  # 提取虚部作为全局坐标系下的线速度

    # 将角速度从局部坐标系转换到全局坐标系
    q_omega_local = np.array([0, v_omega_local[0], v_omega_local[1], v_omega_local[2]])  # 局部坐标系下的角速度向量表示为四元数
    q_omega_global = rotations.quat_mul(q_global_to_local, rotations.quat_mul(q_omega_local, rotations.quat_conjugate(q_global_to_local)))
    v_omega_local = np.array(q_omega_global[1:])  # 提取虚部作为全局坐标系下的角速度

    return v_global, q_omega_global

def global2local(q_global_to_local, v_global, v_acc_global, v_omega_global) -> tuple[np.ndarray, np.ndarray]:
    # Vl = Q*VgQ

    # 将速度向量从全局坐标系转换到局部坐标系
    q_v_global = np.array([0, v_global[0], v_global[1], v_global[2]])  # 速度向量表示为四元数
    q_v_local = rotations.quat_mul(rotations.quat_conjugate(q_global_to_local), rotations.quat_mul(q_v_global, q_global_to_local))
    v_local = np.array(q_v_local[1:])  # 提取虚部作为局部坐标系下的线速度

    # 将加速度向量从全局坐标系转换到局部坐标系
    q_acc_global = np.array([0, v_acc_global[0], v_acc_global[1], v_acc_global[2]])  # 加速度向量表示为四元数
    q_acc_local = rotations.quat_mul(rotations.quat_conjugate(q_global_to_local), rotations.quat_mul(q_acc_global, q_global_to_local))
    v_acc_local = np.array(q_acc_local[1:])  # 提取虚部作为局部坐标系下的线加速度

    # 将角速度从全局坐标系转换到局部坐标系
    # print("q_omega_global: ", v_omega_global, "q_global_to_local: ", q_global_to_local)
    # q_omega_global = np.array([0, v_omega_global[0], v_omega_global[1], v_omega_global[2]])  # 角速度向量表示为四元数
    # q_omega_local = rotations.quat_mul(rotations.quat_conjugate(q_global_to_local), rotations.quat_mul(q_omega_global, q_global_to_local))
    # v_omega_local = np.array(q_omega_local[1:])  # 提取虚部作为局部坐标系下的角速度

    return v_local, v_acc_local, v_omega_global

def quat_angular_velocity(q1, q2, dt):
    # q1 = q1 / np.linalg.norm(q1)
    # q2 = q2 / np.linalg.norm(q2)
    # 计算四元数的角速度
    q_diff = rotations.quat_mul(q2, rotations.quat_conjugate(q1))
    # print("q_diff: ", q_diff)
    
    if q_diff[0] > 1.0:
        return 0.0
    
    angle = 2 * math.acos(q_diff[0])
    if angle > math.pi:
        angle = 2 * math.pi - angle
    return angle / dt

import math

def quat_to_euler(quat):
    """
    将四元数转换为Z-Y-X顺序的欧拉角（yaw, pitch, roll）
    输入：quat = [w, x, y, z]
    输出：(yaw, pitch, roll) 单位为弧度
    """
    w, x, y, z = quat
    
    # 1. Yaw (Z轴旋转)
    sin_yaw = 2.0 * (x*y + w*z)
    cos_yaw = 1.0 - 2.0 * (y*y + z*z)
    yaw = np.arctan2(sin_yaw, cos_yaw)
    
    # 2. Pitch (Y轴旋转)
    sin_pitch = 2.0 * (x*z - w*y)
    # 处理pitch超过±90°的情况
    pitch = np.arcsin(np.clip(sin_pitch, -1.0, 1.0))
    
    # 3. Roll (X轴旋转)
    sin_roll = 2.0 * (w*x + y*z)
    cos_roll = 1.0 - 2.0 * (x*x + y*y)
    roll = np.arctan2(sin_roll, cos_roll)
    
    return yaw, pitch, roll

def smooth_sqr_wave_np(phase, phase_freq, eps):
    """
    生成一个平滑的方波信号。

    参数:
    - phase: 当前相位（0到1之间的值，标量）。
    - phase_freq: 相位频率，用于调整信号的周期（标量）。
    - eps: 一个小值，用于防止分母为零（标量）。

    返回:
    - 平滑的方波信号，值介于0和1之间（标量）。
    """
    p = 2.0 * np.pi * phase * phase_freq
    numerator = np.sin(p)
    denominator = 2.0 * np.sqrt(np.sin(p)**2 + eps**2)
    return numerator / denominator + 0.5
