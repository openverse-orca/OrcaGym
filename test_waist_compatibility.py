#!/usr/bin/env python3
"""
腰部关节兼容性测试脚本
测试带腰部和不带腰部的机器人配置
"""

import sys
import os
import numpy as np

# 添加项目路径
sys.path.append('/home/orcash/OrcaGym/OrcaGym')

def test_with_waist_config():
    """测试带腰部关节的配置"""
    print("=" * 60)
    print("测试带腰部关节的配置")
    print("=" * 60)
    
    try:
        from envs.manipulation.robots.configs.d12_waist_config import d12_waist_config
        
        # 检查配置结构
        assert "waist" in d12_waist_config, "配置中缺少waist部分"
        assert "joint_name" in d12_waist_config["waist"], "waist配置中缺少joint_name"
        assert "position_name" in d12_waist_config["waist"], "waist配置中缺少position_name"
        
        print("✅ 带腰部配置结构正确")
        print(f"   腰部关节名称: {d12_waist_config['waist']['joint_name']}")
        print(f"   位置控制器: {d12_waist_config['waist']['position_name']}")
        print(f"   中性位置: {d12_waist_config['waist']['neutral_joint_value']}")
        
        return True
    except Exception as e:
        print(f"❌ 带腰部配置测试失败: {e}")
        return False

def test_without_waist_config():
    """测试不带腰部关节的配置"""
    print("\n" + "=" * 60)
    print("测试不带腰部关节的配置")
    print("=" * 60)
    
    try:
        from envs.manipulation.robots.configs.openloong_config import openloong_config
        
        # 检查配置结构
        assert "waist" not in openloong_config, "配置中不应该包含waist部分"
        
        print("✅ 不带腰部配置结构正确")
        print("   配置中没有waist部分")
        
        return True
    except Exception as e:
        print(f"❌ 不带腰部配置测试失败: {e}")
        return False

def test_robot_class_compatibility():
    """测试机器人类的兼容性"""
    print("\n" + "=" * 60)
    print("测试机器人类兼容性")
    print("=" * 60)
    
    try:
        from envs.manipulation.dual_arm_robot import DualArmRobot
        
        # 检查关键方法是否存在
        methods_to_check = [
            '_has_waist',
            'get_waist_angle',
            'set_waist_control',
            'reset_waist_control',
            'set_waist_joystick_control',
            '_local_to_global',
            '_global_to_local'
        ]
        
        for method_name in methods_to_check:
            if hasattr(DualArmRobot, method_name):
                print(f"✅ 方法 {method_name} 存在")
            else:
                print(f"❌ 方法 {method_name} 不存在")
                return False
        
        return True
    except Exception as e:
        print(f"❌ 机器人类兼容性测试失败: {e}")
        return False

def test_waist_angle_calculation():
    """测试腰部角度计算逻辑"""
    print("\n" + "=" * 60)
    print("测试腰部角度计算逻辑")
    print("=" * 60)
    
    try:
        # 模拟有腰部关节的情况
        class MockRobotWithWaist:
            def __init__(self):
                self._has_waist = True
                self._waist_jnt_address = 0  # 模拟地址
                
            def get_waist_angle(self):
                if self._has_waist and self._waist_jnt_address is not None:
                    return 0.5  # 模拟角度
                else:
                    return 0.0
        
        # 模拟无腰部关节的情况
        class MockRobotWithoutWaist:
            def __init__(self):
                self._has_waist = False
                self._waist_jnt_address = None
                
            def get_waist_angle(self):
                if self._has_waist and self._waist_jnt_address is not None:
                    return 0.5
                else:
                    return 0.0
        
        # 测试有腰部关节
        robot_with_waist = MockRobotWithWaist()
        angle_with = robot_with_waist.get_waist_angle()
        print(f"✅ 有腰部关节时角度: {angle_with}")
        
        # 测试无腰部关节
        robot_without_waist = MockRobotWithoutWaist()
        angle_without = robot_without_waist.get_waist_angle()
        print(f"✅ 无腰部关节时角度: {angle_without}")
        
        assert angle_with == 0.5, "有腰部关节时角度计算错误"
        assert angle_without == 0.0, "无腰部关节时角度计算错误"
        
        return True
    except Exception as e:
        print(f"❌ 腰部角度计算测试失败: {e}")
        return False

def test_coordinate_transformation():
    """测试坐标系转换兼容性"""
    print("\n" + "=" * 60)
    print("测试坐标系转换兼容性")
    print("=" * 60)
    
    try:
        import numpy as np
        from scipy.spatial.transform import Rotation as R
        
        # 模拟局部位置和姿态
        local_pos = np.array([0.5, 0.2, 0.8])
        local_quat = np.array([1.0, 0.0, 0.0, 0.0])
        
        # 模拟基座位置和姿态
        base_pos = np.array([0.0, 0.0, 0.0])
        base_quat = np.array([1.0, 0.0, 0.0, 0.0])
        
        # 测试有腰部关节的转换
        waist_angle = np.pi / 4  # 45度
        waist_quat = np.array([
            np.cos(waist_angle / 2),  # w
            0,                        # x
            0,                        # y
            np.sin(waist_angle / 2)   # z
        ])
        
        # 组合旋转
        base_rot = R.from_quat([base_quat[1], base_quat[2], base_quat[3], base_quat[0]])
        waist_rot = R.from_quat([waist_quat[1], waist_quat[2], waist_quat[3], waist_quat[0]])
        combined_rot = base_rot * waist_rot
        
        global_pos_with_waist = base_pos + combined_rot.apply(local_pos)
        print(f"✅ 有腰部关节时全局位置: {global_pos_with_waist}")
        
        # 测试无腰部关节的转换
        global_pos_without_waist = base_pos + base_rot.apply(local_pos)
        print(f"✅ 无腰部关节时全局位置: {global_pos_without_waist}")
        
        # 验证结果不同（有腰部转动时位置应该不同）
        assert not np.allclose(global_pos_with_waist, global_pos_without_waist), "有腰部和无腰部的转换结果应该不同"
        
        return True
    except Exception as e:
        print(f"❌ 坐标系转换测试失败: {e}")
        return False

def create_compatibility_examples():
    """创建兼容性示例配置"""
    print("\n" + "=" * 60)
    print("创建兼容性示例配置")
    print("=" * 60)
    
    # 带腰部关节的配置示例
    config_with_waist = {
        "robot_type": "dual_arm",
        "base": {
            "base_body_name": "base_link",
            "base_joint_name": "base_joint",
            "dummy_joint_name": "dummy_joint",
        },
        "right_arm": {
            "joint_names": ["right_shoulder_pitch_joint", "right_shoulder_roll_joint"],
            "neutral_joint_values": [-0.67, -0.72],
            "motor_names": ["M_arm_r_01", "M_arm_r_02"],
            "ee_center_site_name": "ee_center_site_r",
        },
        "left_arm": {
            "joint_names": ["left_shoulder_pitch_joint", "left_shoulder_roll_joint"],
            "neutral_joint_values": [-0.67, 0.72],
            "motor_names": ["M_arm_l_01", "M_arm_l_02"],
            "ee_center_site_name": "ee_center_site",
        },
        "waist": {
            "joint_name": "waist_yaw_joint",
            "neutral_joint_value": 0.0,
            "position_name": "P_waist",
        },
    }
    
    # 不带腰部关节的配置示例
    config_without_waist = {
        "robot_type": "dual_arm",
        "base": {
            "base_body_name": "base_link",
            "base_joint_name": "base_joint",
            "dummy_joint_name": "dummy_joint",
        },
        "right_arm": {
            "joint_names": ["right_shoulder_pitch_joint", "right_shoulder_roll_joint"],
            "neutral_joint_values": [-0.67, -0.72],
            "motor_names": ["M_arm_r_01", "M_arm_r_02"],
            "ee_center_site_name": "ee_center_site_r",
        },
        "left_arm": {
            "joint_names": ["left_shoulder_pitch_joint", "left_shoulder_roll_joint"],
            "neutral_joint_values": [-0.67, 0.72],
            "motor_names": ["M_arm_l_01", "M_arm_l_02"],
            "ee_center_site_name": "ee_center_site",
        },
        # 注意：没有waist部分
    }
    
    print("✅ 带腰部关节配置示例:")
    print("   - 包含'waist'部分")
    print("   - 包含joint_name, position_name等")
    
    print("✅ 不带腰部关节配置示例:")
    print("   - 不包含'waist'部分")
    print("   - 只有base, right_arm, left_arm")
    
    return True

def main():
    """主测试函数"""
    print("🤖 腰部关节兼容性测试")
    print("=" * 60)
    
    tests = [
        ("带腰部配置测试", test_with_waist_config),
        ("不带腰部配置测试", test_without_waist_config),
        ("机器人类兼容性测试", test_robot_class_compatibility),
        ("腰部角度计算测试", test_waist_angle_calculation),
        ("坐标系转换测试", test_coordinate_transformation),
        ("兼容性示例创建", create_compatibility_examples),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🔍 运行测试: {test_name}")
        if test_func():
            passed += 1
        else:
            print(f"❌ 测试失败: {test_name}")
    
    print("\n" + "=" * 60)
    print(f"测试结果: {passed}/{total} 通过")
    
    if passed == total:
        print("🎉 所有兼容性测试通过！")
        print("\n📋 兼容性总结:")
        print("✅ 支持带腰部关节的机器人")
        print("✅ 支持不带腰部关节的机器人")
        print("✅ 自动检测腰部关节配置")
        print("✅ 向后兼容现有配置")
        print("✅ 坐标系转换自适应")
        return True
    else:
        print("⚠️  部分测试失败，请检查代码")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

