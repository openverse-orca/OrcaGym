"""
Lite3观测计算辅助函数
用于将OrcaGym的观测转换为Lite3_rl_deploy格式的45维观测

参考文件:
- Lite3_rl_deploy/run_policy/lite3_test_policy_runner_onnx.hpp
"""

import numpy as np
from typing import Dict, Optional


def compute_lite3_obs(
    base_ang_vel: np.ndarray,
    projected_gravity: np.ndarray,
    commands: np.ndarray,
    dof_pos: np.ndarray,
    dof_vel: np.ndarray,
    last_actions: np.ndarray,
    omega_scale: float = 0.25,
    dof_vel_scale: float = 0.05,
    max_cmd_vel: Optional[np.ndarray] = None,
    dof_pos_default: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    计算Lite3格式的45维观测
    
    参考: Lite3TestPolicyRunnerONNX::GetRobotAction()
    
    Args:
        base_ang_vel: 基础角速度 [3] 或 [N, 3]
        projected_gravity: 投影重力 [3] 或 [N, 3]
        commands: 命令速度 [3] 或 [N, 3]
        dof_pos: 关节位置 [12] 或 [N, 12]
        dof_vel: 关节速度 [12] 或 [N, 12]
        last_actions: 上一动作 [12] 或 [N, 12]
        omega_scale: 角速度缩放 (默认0.25)
        dof_vel_scale: 关节速度缩放 (默认0.05)
        max_cmd_vel: 最大命令速度 [3] (默认[0.8, 0.8, 0.8])
        dof_pos_default: 默认关节位置 [12] (用于计算相对位置)
    
    Returns:
        obs: 45维观测 [45] 或 [N, 45]
    """
    # 处理单样本情况
    single_sample = False
    if base_ang_vel.ndim == 1:
        single_sample = True
        base_ang_vel = base_ang_vel.reshape(1, -1)
        projected_gravity = projected_gravity.reshape(1, -1)
        commands = commands.reshape(1, -1)
        dof_pos = dof_pos.reshape(1, -1)
        dof_vel = dof_vel.reshape(1, -1)
        last_actions = last_actions.reshape(1, -1)
    
    # 设置默认值
    if max_cmd_vel is None:
        max_cmd_vel = np.array([0.8, 0.8, 0.8])
    if dof_pos_default is None:
        # 使用策略中的默认值
        dof_pos_default = np.array([
            0.0, -0.8, 1.6,  # FL
            0.0, -0.8, 1.6,  # FR
            0.0, -0.8, 1.6,  # HL
            0.0, -0.8, 1.6,  # HR
        ])
    
    # 1. 基础角速度 (缩放)
    base_ang_vel_scaled = base_ang_vel * omega_scale  # [N, 3]
    
    # 2. 投影重力 (已经是正确的格式)
    # projected_gravity: [N, 3]
    
    # 3. 命令速度 (归一化后乘以最大速度)
    commands_scaled = commands * max_cmd_vel.reshape(1, -1)  # [N, 3]
    
    # 4. 关节位置 (相对默认位置)
    dof_pos_relative = dof_pos - dof_pos_default.reshape(1, -1)  # [N, 12]
    
    # 5. 关节速度 (缩放)
    dof_vel_scaled = dof_vel * dof_vel_scale  # [N, 12]
    
    # 6. 上一动作
    # last_actions: [N, 12]
    
    # 拼接观测 (45维)
    obs = np.concatenate([
        base_ang_vel_scaled,      # 3
        projected_gravity,        # 3
        commands_scaled,          # 3
        dof_pos_relative,         # 12
        dof_vel_scaled,           # 12
        last_actions              # 12
    ], axis=-1)  # [N, 45]
    
    # 如果输入是单样本，返回单样本
    if single_sample:
        obs = obs.squeeze(0)
    
    return obs


def get_dof_pos_default_policy() -> np.ndarray:
    """
    获取策略中的默认关节位置
    
    Returns:
        默认关节位置 [12]
    """
    return np.array([
        0.0, -0.8, 1.6,  # FL: HipX, HipY, Knee
        0.0, -0.8, 1.6,  # FR
        0.0, -0.8, 1.6,  # HL
        0.0, -0.8, 1.6,  # HR
    ])


def get_action_scale_original() -> np.ndarray:
    """
    获取原始动作缩放
    
    Returns:
        动作缩放 [12]
    """
    return np.array([
        0.125, 0.25, 0.25,  # FL: HipX, HipY, Knee
        0.125, 0.25, 0.25,  # FR
        0.125, 0.25, 0.25,  # HL
        0.125, 0.25, 0.25,  # HR
    ])

