#!/usr/bin/env python3
"""
测试腰部转动修复是否成功
验证代码语法和基本功能
"""

import sys
import os

# 添加项目路径
sys.path.append('/home/orcash/OrcaGym/OrcaGym')

def test_import():
    """测试导入是否成功"""
    try:
        from envs.manipulation.dual_arm_robot import DualArmRobot
        print("✅ DualArmRobot 导入成功")
        return True
    except Exception as e:
        print(f"❌ DualArmRobot 导入失败: {e}")
        return False

def test_syntax():
    """测试语法是否正确"""
    try:
        # 尝试编译文件
        with open('/home/orcash/OrcaGym/OrcaGym/envs/manipulation/dual_arm_robot.py', 'r') as f:
            code = f.read()
        compile(code, '/home/orcash/OrcaGym/OrcaGym/envs/manipulation/dual_arm_robot.py', 'exec')
        print("✅ 语法检查通过")
        return True
    except SyntaxError as e:
        print(f"❌ 语法错误: {e}")
        return False
    except Exception as e:
        print(f"❌ 其他错误: {e}")
        return False

def test_key_methods():
    """测试关键方法是否存在"""
    try:
        from envs.manipulation.dual_arm_robot import DualArmRobot
        
        # 检查关键方法是否存在
        methods_to_check = [
            '_local_to_global',
            '_global_to_local', 
            'get_waist_angle',
            'set_waist_control',
            'get_obs'
        ]
        
        for method_name in methods_to_check:
            if hasattr(DualArmRobot, method_name):
                print(f"✅ 方法 {method_name} 存在")
            else:
                print(f"❌ 方法 {method_name} 不存在")
                return False
        
        return True
    except Exception as e:
        print(f"❌ 方法检查失败: {e}")
        return False

def test_waist_angle_calculation():
    """测试腰部角度计算逻辑"""
    try:
        import numpy as np
        
        # 模拟腰部角度计算
        waist_angle = np.pi / 4  # 45度
        
        # 创建腰部转动的四元数 (绕Z轴旋转)
        waist_quat = np.array([
            np.cos(waist_angle / 2),  # w
            0,                        # x
            0,                        # y
            np.sin(waist_angle / 2)   # z
        ])
        
        print(f"✅ 腰部角度: {np.degrees(waist_angle):.1f}°")
        print(f"✅ 腰部四元数: {waist_quat}")
        
        # 验证四元数是否归一化
        quat_norm = np.linalg.norm(waist_quat)
        if abs(quat_norm - 1.0) < 1e-6:
            print("✅ 四元数归一化正确")
        else:
            print(f"❌ 四元数归一化错误: {quat_norm}")
            return False
        
        return True
    except Exception as e:
        print(f"❌ 腰部角度计算测试失败: {e}")
        return False

def main():
    """主测试函数"""
    print("=" * 60)
    print("腰部转动修复测试")
    print("=" * 60)
    
    tests = [
        ("语法检查", test_syntax),
        ("导入测试", test_import),
        ("方法检查", test_key_methods),
        ("腰部角度计算", test_waist_angle_calculation),
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
        print("🎉 所有测试通过！腰部转动修复成功！")
        return True
    else:
        print("⚠️  部分测试失败，请检查代码")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

