#!/usr/bin/env python3
"""
腰部关节数据保存示例
演示如何在数据保存时包含waist joint的数据
"""

import numpy as np
import h5py
import json
from datetime import datetime

def demonstrate_observation_structure():
    """
    演示包含腰部关节的观测数据结构
    """
    print("=" * 70)
    print("包含腰部关节的观测数据结构")
    print("=" * 70)
    
    # 模拟观测数据
    obs_data = {
        # 末端执行器位置和姿态
        "ee_pos_l": np.array([0.5, 0.2, 0.8], dtype=np.float32),
        "ee_quat_l": np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        "ee_pos_r": np.array([0.5, -0.2, 0.8], dtype=np.float32),
        "ee_quat_r": np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        
        # 末端执行器速度
        "ee_vel_linear_l": np.array([0.1, 0.0, 0.0], dtype=np.float32),
        "ee_vel_angular_l": np.array([0.0, 0.0, 0.1], dtype=np.float32),
        "ee_vel_linear_r": np.array([0.1, 0.0, 0.0], dtype=np.float32),
        "ee_vel_angular_r": np.array([0.0, 0.0, -0.1], dtype=np.float32),
        
        # 左臂关节数据
        "arm_joint_qpos_l": np.array([0.1, -0.5, 0.8, -0.3, 0.2, 0.0, 0.0], dtype=np.float32),
        "arm_joint_qpos_sin_l": np.sin(np.array([0.1, -0.5, 0.8, -0.3, 0.2, 0.0, 0.0])).astype(np.float32),
        "arm_joint_qpos_cos_l": np.cos(np.array([0.1, -0.5, 0.8, -0.3, 0.2, 0.0, 0.0])).astype(np.float32),
        "arm_joint_vel_l": np.array([0.1, -0.2, 0.3, -0.1, 0.0, 0.0, 0.0], dtype=np.float32),
        
        # 右臂关节数据
        "arm_joint_qpos_r": np.array([0.1, 0.5, -0.8, 0.3, -0.2, 0.0, 0.0], dtype=np.float32),
        "arm_joint_qpos_sin_r": np.sin(np.array([0.1, 0.5, -0.8, 0.3, -0.2, 0.0, 0.0])).astype(np.float32),
        "arm_joint_qpos_cos_r": np.cos(np.array([0.1, 0.5, -0.8, 0.3, -0.2, 0.0, 0.0])).astype(np.float32),
        "arm_joint_vel_r": np.array([0.1, 0.2, -0.3, 0.1, 0.0, 0.0, 0.0], dtype=np.float32),
        
        # 抓取数据
        "grasp_value_l": np.array([0.5], dtype=np.float32),
        "grasp_value_r": np.array([0.3], dtype=np.float32),
        "grasp_joint_pos_l": np.array([0.5], dtype=np.float32),
        "grasp_joint_pos_r": np.array([0.3], dtype=np.float32),
        
        # 腰部关节数据（新增）
        "waist_joint_qpos": np.array([0.2], dtype=np.float32),  # 当前角度
        "waist_joint_qpos_sin": np.array([np.sin(0.2)], dtype=np.float32),  # sin值
        "waist_joint_qpos_cos": np.array([np.cos(0.2)], dtype=np.float32),  # cos值
        "waist_joint_vel": np.array([0.1], dtype=np.float32),  # 角速度
    }
    
    print("观测数据键值对：")
    for key, value in obs_data.items():
        print(f"  {key}: shape={value.shape}, dtype={value.dtype}, range=[{value.min():.3f}, {value.max():.3f}]")
    
    print(f"\n总观测维度: {sum(v.size for v in obs_data.values())}")
    print(f"腰部关节相关维度: {obs_data['waist_joint_qpos'].size + obs_data['waist_joint_qpos_sin'].size + obs_data['waist_joint_qpos_cos'].size + obs_data['waist_joint_vel'].size}")

def demonstrate_action_structure():
    """
    演示包含腰部关节的动作数据结构
    """
    print("\n" + "=" * 70)
    print("包含腰部关节的动作数据结构")
    print("=" * 70)
    
    # 模拟动作数据
    action_data = {
        "left_eef_pos": np.array([0.1, 0.05, 0.02]),  # 左臂末端位置增量
        "left_eef_rot": np.array([0.0, 0.0, 0.1]),    # 左臂末端旋转增量
        "left_arm_joints": np.array([0.1, -0.2, 0.3, -0.1, 0.0, 0.0, 0.0]),  # 左臂关节控制
        "left_grasp": np.array([0.5]),  # 左臂抓取值
        
        "right_eef_pos": np.array([0.1, -0.05, 0.02]),  # 右臂末端位置增量
        "right_eef_rot": np.array([0.0, 0.0, -0.1]),    # 右臂末端旋转增量
        "right_arm_joints": np.array([0.1, 0.2, -0.3, 0.1, 0.0, 0.0, 0.0]),  # 右臂关节控制
        "right_grasp": np.array([0.3]),  # 右臂抓取值
        
        "waist_angle": np.array([0.2]),  # 腰部关节角度（新增）
    }
    
    # 构建完整的action数组
    action_array = np.concatenate([
        action_data["left_eef_pos"],
        action_data["left_eef_rot"],
        action_data["left_arm_joints"],
        action_data["left_grasp"],
        action_data["right_eef_pos"],
        action_data["right_eef_rot"],
        action_data["right_arm_joints"],
        action_data["right_grasp"],
        action_data["waist_angle"],  # 新增的腰部关节数据
    ])
    
    print("动作数据组成：")
    print(f"  左臂末端位置 (0-2): {action_data['left_eef_pos']}")
    print(f"  左臂末端旋转 (3-5): {action_data['left_eef_rot']}")
    print(f"  左臂关节控制 (6-12): {action_data['left_arm_joints']}")
    print(f"  左臂抓取值 (13): {action_data['left_grasp']}")
    print(f"  右臂末端位置 (14-16): {action_data['right_eef_pos']}")
    print(f"  右臂末端旋转 (17-19): {action_data['right_eef_rot']}")
    print(f"  右臂关节控制 (20-26): {action_data['right_arm_joints']}")
    print(f"  右臂抓取值 (27): {action_data['right_grasp']}")
    print(f"  腰部关节角度 (28): {action_data['waist_angle']}")  # 新增
    
    print(f"\n总动作维度: {len(action_array)}")
    print(f"动作数组: {action_array}")

def demonstrate_data_saving():
    """
    演示数据保存过程
    """
    print("\n" + "=" * 70)
    print("数据保存示例")
    print("=" * 70)
    
    # 模拟多步数据
    num_steps = 10
    obs_data = []
    action_data = []
    
    for step in range(num_steps):
        # 模拟观测数据
        obs = {
            "waist_joint_qpos": np.array([0.1 * step], dtype=np.float32),
            "waist_joint_qpos_sin": np.array([np.sin(0.1 * step)], dtype=np.float32),
            "waist_joint_qpos_cos": np.array([np.cos(0.1 * step)], dtype=np.float32),
            "waist_joint_vel": np.array([0.1], dtype=np.float32),
            # ... 其他观测数据
        }
        obs_data.append(obs)
        
        # 模拟动作数据
        action = np.array([0.1 * step] + [0.0] * 27)  # 腰部关节角度 + 其他动作
        action_data.append(action)
    
    # 保存到HDF5文件
    filename = f"waist_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.h5"
    
    with h5py.File(filename, 'w') as f:
        # 创建观测数据组
        obs_group = f.create_group('observations')
        
        # 保存腰部关节数据
        waist_qpos = np.array([obs['waist_joint_qpos'][0] for obs in obs_data])
        waist_qpos_sin = np.array([obs['waist_joint_qpos_sin'][0] for obs in obs_data])
        waist_qpos_cos = np.array([obs['waist_joint_qpos_cos'][0] for obs in obs_data])
        waist_vel = np.array([obs['waist_joint_vel'][0] for obs in obs_data])
        
        obs_group.create_dataset('waist_joint_qpos', data=waist_qpos)
        obs_group.create_dataset('waist_joint_qpos_sin', data=waist_qpos_sin)
        obs_group.create_dataset('waist_joint_qpos_cos', data=waist_qpos_cos)
        obs_group.create_dataset('waist_joint_vel', data=waist_vel)
        
        # 创建动作数据组
        action_group = f.create_group('actions')
        actions_array = np.array(action_data)
        action_group.create_dataset('actions', data=actions_array)
        
        # 保存元数据
        metadata = {
            "description": "包含腰部关节数据的机器人遥操作数据",
            "waist_joint_info": {
                "angle_range": "[-π/2, π/2]",
                "velocity_range": "[-π, π]",
                "position_in_action": 28,
                "observation_keys": [
                    "waist_joint_qpos",
                    "waist_joint_qpos_sin", 
                    "waist_joint_qpos_cos",
                    "waist_joint_vel"
                ]
            },
            "total_steps": num_steps,
            "action_dimension": len(action_data[0]),
            "created_at": datetime.now().isoformat()
        }
        
        f.attrs['metadata'] = json.dumps(metadata, indent=2)
    
    print(f"数据已保存到: {filename}")
    print(f"腰部关节角度序列: {waist_qpos}")
    print(f"腰部关节角速度序列: {waist_vel}")
    
    # 验证保存的数据
    with h5py.File(filename, 'r') as f:
        print(f"\n验证保存的数据:")
        print(f"  观测数据组键: {list(f['observations'].keys())}")
        print(f"  动作数据形状: {f['actions']['actions'].shape}")
        print(f"  元数据: {json.loads(f.attrs['metadata'])}")

def demonstrate_data_loading():
    """
    演示数据加载过程
    """
    print("\n" + "=" * 70)
    print("数据加载示例")
    print("=" * 70)
    
    # 模拟加载HDF5文件
    print("加载腰部关节数据的步骤：")
    print("1. 打开HDF5文件")
    print("2. 读取观测数据组中的腰部关节数据")
    print("3. 读取动作数据中的腰部关节角度")
    print("4. 应用数据预处理和归一化")
    
    # 示例代码
    example_code = '''
# 加载数据示例代码
import h5py
import numpy as np

def load_waist_data(filename):
    with h5py.File(filename, 'r') as f:
        # 读取腰部关节观测数据
        waist_qpos = f['observations']['waist_joint_qpos'][:]
        waist_qpos_sin = f['observations']['waist_joint_qpos_sin'][:]
        waist_qpos_cos = f['observations']['waist_joint_qpos_cos'][:]
        waist_vel = f['observations']['waist_joint_vel'][:]
        
        # 读取动作数据
        actions = f['actions']['actions'][:]
        waist_actions = actions[:, 28]  # 第28个元素是腰部关节角度
        
        # 读取元数据
        metadata = json.loads(f.attrs['metadata'])
        
        return {
            'waist_observations': {
                'qpos': waist_qpos,
                'qpos_sin': waist_qpos_sin,
                'qpos_cos': waist_qpos_cos,
                'vel': waist_vel
            },
            'waist_actions': waist_actions,
            'metadata': metadata
        }
    '''
    
    print("示例代码：")
    print(example_code)

if __name__ == "__main__":
    demonstrate_observation_structure()
    demonstrate_action_structure()
    demonstrate_data_saving()
    demonstrate_data_loading()
    
    print("\n" + "=" * 70)
    print("腰部关节数据保存总结")
    print("=" * 70)
    print("新增的观测数据：")
    print("  - waist_joint_qpos: 腰部关节角度")
    print("  - waist_joint_qpos_sin: 腰部关节角度sin值")
    print("  - waist_joint_qpos_cos: 腰部关节角度cos值")
    print("  - waist_joint_vel: 腰部关节角速度")
    print("\n新增的动作数据：")
    print("  - action[28]: 腰部关节目标角度")
    print("\n数据保存优势：")
    print("  1. 完整记录腰部运动状态")
    print("  2. 支持后续数据分析和回放")
    print("  3. 便于训练包含腰部的控制策略")
    print("  4. 提供丰富的运动学信息")
    print("=" * 70)
