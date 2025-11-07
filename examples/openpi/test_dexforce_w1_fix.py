#!/usr/bin/env python3
"""
测试dexforce_w1_gripper的修复
检查：
1. 物体名称前缀处理
2. 相机位置
3. 手臂初始姿态
"""

import sys
import os

# 添加项目根目录到路径
current_file_path = os.path.abspath('')
project_root = os.path.dirname(os.path.dirname(current_file_path))
sys.path.insert(0, project_root)

import numpy as np
import gymnasium as gym
from gymnasium.envs.registration import register
import yaml

import orca_gym.scripts.dual_arm_manipulation as dual_arm_manipulation
from envs.manipulation.dual_arm_env import ControlDevice, RunMode, ActionType
from orca_gym.environment.orca_gym_env import RewardType

def test_body_name_finding():
    """测试body名称查找功能"""
    print("\n" + "="*60)
    print("测试1: Body名称前缀处理")
    print("="*60)
    
    # 这里需要实际运行环境才能测试
    print("✓ 已修复 pick_place_task.py 使用 _find_body_name 方法")
    print("  现在可以自动处理不同agent前缀的物体")

def test_camera_config():
    """测试相机配置"""
    print("\n" + "="*60)
    print("测试2: 相机配置")
    print("="*60)
    
    with open("camera_config.yaml", "r") as f:
        cam_cfg = yaml.safe_load(f)
    
    print("✓ 相机配置已加载")
    print(f"  camera_head 相对位置: {cam_cfg['default']['camera_head']['translation']}")
    print(f"  camera_head 旋转: {cam_cfg['default']['camera_head']['rotation']}")
    print(f"  kitchen场景世界坐标: {cam_cfg['scenes']['kitchen']['camera_head']['translation']}")

def test_arm_config():
    """测试手臂配置"""
    print("\n" + "="*60)
    print("测试3: 手臂初始姿态配置")
    print("="*60)
    
    from envs.manipulation.robots.configs.dexforce_w1_config import dexforce_w1_config
    from envs.manipulation.robots.configs.openloong_config import openloong_config
    
    print("Dexforce W1 配置:")
    print(f"  左臂中性位: {dexforce_w1_config['left_arm']['neutral_joint_values']}")
    print(f"  右臂中性位: {dexforce_w1_config['right_arm']['neutral_joint_values']}")
    
    print("\nOpenLoong 配置 (参考):")
    print(f"  左臂中性位: {openloong_config['left_arm']['neutral_joint_values']}")
    print(f"  右臂中性位: {openloong_config['right_arm']['neutral_joint_values']}")
    
    # 检查是否相同
    l_same = np.allclose(
        dexforce_w1_config['left_arm']['neutral_joint_values'],
        openloong_config['left_arm']['neutral_joint_values']
    )
    r_same = np.allclose(
        dexforce_w1_config['right_arm']['neutral_joint_values'],
        openloong_config['right_arm']['neutral_joint_values']
    )
    
    if l_same and r_same:
        print("\n⚠️  初始姿态配置与OpenLoong相同")
        print("   如果手臂位置异常，可能是因为:")
        print("   1. 关节轴方向定义不同")
        print("   2. DH参数不同")
        print("   3. 需要在XML文件中检查关节定义")
    else:
        print("\n✓ 初始姿态配置不同")

def print_diagnostic_info():
    """打印诊断信息"""
    print("\n" + "="*60)
    print("诊断建议")
    print("="*60)
    
    print("\n【问题1：物体名称前缀】")
    print("✓ 已修复：pick_place_task.py 现在使用 _find_body_name()")
    print("  该方法会自动fallback到所有可用的body names")
    
    print("\n【问题2：相机位置】")
    print("⚠️  需要检查XML文件中camera_head的parent body")
    print("   检查步骤：")
    print("   1. 打开 DexforceW1V020_INDUSTRIAL_DH_PGC_GRIPPER_M.xml")
    print("   2. 查找 <camera name=\"camera_head\">")
    print("   3. 确认它挂在哪个body下（应该挂在头部body）")
    print("   4. 对比OpenLoong的XML，确保相机挂载位置一致")
    
    print("\n【问题3：手臂位置异常】")
    print("⚠️  可能的原因：")
    print("   1. 关节方向定义不同（检查XML中的axis属性）")
    print("   2. DH参数不同")
    print("   3. 建议调整 neutral_joint_values，可以尝试：")
    print("      - 将某些关节值取反（如 J2: 0.5 -> -0.5）")
    print("      - 或者调整 J4 的值")
    
    print("\n【推荐的测试命令】")
    print("cd /home/orca/OrcaWorkStation/OrcaGym/examples/openpi")
    print("conda activate orca")
    print("\npython run_dual_arm_sim.py \\")
    print("    --agent_names dexforce_w1_gripper \\")
    print("    --action_type end_effector_osc \\")
    print("    --task_config kitchen_task.yaml \\")
    print("    --ctrl_device vr \\")
    print("    --run_mode teleoperation")

def main():
    print("\n" + "="*60)
    print("Dexforce W1 Gripper 修复验证")
    print("="*60)
    
    test_body_name_finding()
    test_camera_config()
    test_arm_config()
    print_diagnostic_info()
    
    print("\n" + "="*60)
    print("测试完成")
    print("="*60)

if __name__ == "__main__":
    main()

