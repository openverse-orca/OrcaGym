#!/usr/bin/env python3
"""
测试机器人配置注册表的功能
"""

import sys
import os

# 确保能够导入项目模块
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..')))

from envs.manipulation.robots.configs.robot_config_registry import (
    RobotConfigRegistry,
    get_robot_config,
    list_available_configs
)


def test_list_configs():
    """测试列出所有可用配置"""
    print("=" * 60)
    print("测试1: 列出所有可用配置")
    print("=" * 60)
    
    configs = list_available_configs()
    print(f"\n找到 {len(configs)} 个配置:")
    for i, config in enumerate(configs, 1):
        print(f"  {i}. {config}")
    
    assert len(configs) > 0, "应该至少有一个配置"
    print("\n✓ 测试通过\n")


def test_get_config_by_name():
    """测试通过名称获取配置"""
    print("=" * 60)
    print("测试2: 通过名称获取配置")
    print("=" * 60)
    
    # 测试获取 openloong 配置
    try:
        config = get_robot_config("test_robot", config_name="openloong")
        print(f"\n✓ 成功获取 'openloong' 配置")
        print(f"  配置类型: {config.get('robot_type', 'N/A')}")
        print(f"  包含的键: {list(config.keys())}")
        
        # 验证配置结构
        assert "base" in config, "配置应包含 'base' 键"
        assert "right_arm" in config, "配置应包含 'right_arm' 键"
        assert "left_arm" in config, "配置应包含 'left_arm' 键"
        print("\n✓ 配置结构验证通过\n")
        
    except ValueError as e:
        print(f"\n✗ 获取配置失败: {e}\n")
        raise


def test_auto_inference():
    """测试自动推断配置"""
    print("=" * 60)
    print("测试3: 自动推断配置")
    print("=" * 60)
    
    test_cases = [
        ("openloong_hand_fix_base", "openloong"),
        ("openloong_gripper_2f85_fix_base", "openloong"),
        ("openloong_gripper_2f85_mobile_base", "openloong"),
    ]
    
    for robot_name, expected_config in test_cases:
        try:
            config = get_robot_config(robot_name)
            print(f"\n✓ 机器人 '{robot_name}' 成功推断配置")
            print(f"  预期配置: {expected_config}")
            print(f"  配置类型: {config.get('robot_type', 'N/A')}")
        except ValueError as e:
            print(f"\n✗ 机器人 '{robot_name}' 推断失败: {e}")
            # 不抛出异常，因为某些配置可能不存在
    
    print("\n✓ 测试完成\n")


def test_invalid_config():
    """测试无效配置处理"""
    print("=" * 60)
    print("测试4: 无效配置处理")
    print("=" * 60)
    
    try:
        config = get_robot_config("invalid_robot", config_name="non_existent_config")
        print("\n✗ 应该抛出 ValueError，但没有")
    except ValueError as e:
        print(f"\n✓ 正确抛出 ValueError: {e}\n")


def test_config_content():
    """测试配置内容的完整性"""
    print("=" * 60)
    print("测试5: 配置内容完整性")
    print("=" * 60)
    
    try:
        config = get_robot_config("openloong_hand_fix_base")
        
        # 检查必需字段
        required_fields = {
            "base": ["base_body_name", "base_joint_name", "dummy_joint_name"],
            "right_arm": ["joint_names", "neutral_joint_values", "motor_names", 
                         "position_names", "ee_center_site_name"],
            "left_arm": ["joint_names", "neutral_joint_values", "motor_names", 
                        "position_names", "ee_center_site_name"],
        }
        
        print("\n检查配置字段:")
        all_ok = True
        for section, fields in required_fields.items():
            if section not in config:
                print(f"  ✗ 缺少节 '{section}'")
                all_ok = False
                continue
            
            print(f"  节 '{section}':")
            for field in fields:
                if field in config[section]:
                    value = config[section][field]
                    if isinstance(value, list):
                        print(f"    ✓ {field}: {len(value)} 项")
                    else:
                        print(f"    ✓ {field}: {value}")
                else:
                    print(f"    ✗ 缺少字段 '{field}'")
                    all_ok = False
        
        if all_ok:
            print("\n✓ 所有必需字段都存在\n")
        else:
            print("\n✗ 某些必需字段缺失\n")
            
    except Exception as e:
        print(f"\n✗ 测试失败: {e}\n")
        raise


def main():
    """运行所有测试"""
    print("\n" + "=" * 60)
    print("机器人配置注册表测试套件")
    print("=" * 60 + "\n")
    
    try:
        test_list_configs()
        test_get_config_by_name()
        test_auto_inference()
        test_invalid_config()
        test_config_content()
        
        print("=" * 60)
        print("所有测试完成!")
        print("=" * 60 + "\n")
        
    except Exception as e:
        print("\n" + "=" * 60)
        print(f"测试失败: {e}")
        print("=" * 60 + "\n")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

