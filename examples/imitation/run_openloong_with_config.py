"""
示例：通过命令行参数指定机器人配置

用法示例：
1. 使用默认配置（根据机器人名称自动推断）：
   python run_openloong_with_config.py --agent_name openloong_hand_fix_base

2. 显式指定配置：
   python run_openloong_with_config.py --agent_name openloong_hand_fix_base --robot_config openloong

3. 多个机器人使用不同配置：
   python run_openloong_with_config.py --agent_name "robot1 robot2" --robot_configs "robot1:openloong,robot2:d12"

4. 列出所有可用配置：
   python run_openloong_with_config.py --list_configs
"""

import os
import sys
import argparse

current_file_path = os.path.abspath('')
project_root = os.path.dirname(os.path.dirname(current_file_path))

import orca_gym.scripts.openloong_manipulation as openloong_manipulation
from envs.manipulation.robots.configs.robot_config_registry import list_available_configs


def parse_robot_configs(robot_configs_str: str) -> dict:
    """
    解析机器人配置字符串
    
    Args:
        robot_configs_str: 格式为 "robot1:config1,robot2:config2"
        
    Returns:
        配置字典 {robot_name: config_name}
    """
    if not robot_configs_str:
        return {}
    
    configs = {}
    for pair in robot_configs_str.split(','):
        if ':' in pair:
            robot_name, config_name = pair.split(':', 1)
            configs[robot_name.strip()] = config_name.strip()
    
    return configs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='运行 OpenLoong 机器人操作示例，支持灵活配置',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # 配置相关参数
    parser.add_argument('--list_configs', action='store_true', 
                        help='列出所有可用的机器人配置并退出')
    parser.add_argument('--robot_config', type=str, default=None,
                        help='指定所有机器人使用的配置名称（例如：openloong, d12）')
    parser.add_argument('--robot_configs', type=str, default=None,
                        help='为不同机器人指定不同配置，格式：robot1:config1,robot2:config2')
    
    # 其他参数
    parser.add_argument('--orcagym_address', type=str, default='localhost:50051', 
                        help='gRPC 服务器地址')
    parser.add_argument('--agent_name', type=str, default='OpenLoongHand', 
                        help='机器人名称，多个机器人用空格分隔')
    parser.add_argument('--run_mode', type=str, default='teleoperation', 
                        help='运行模式 (teleoperation / playback / imitation / rollout / augmentation)')
    parser.add_argument('--action_type', type=str, default='end_effector', 
                        help='动作类型 (end_effector / joint_pos)')
    parser.add_argument('--action_step', type=int, default=5, 
                        help='每个动作的仿真步数，end_effector 建议 5，joint_pos 建议 1')
    parser.add_argument('--prompt', type=str, default="Do something.", 
                        help='遥操作任务指令')
    parser.add_argument('--task_config', type=str, 
                        help='任务配置文件路径')
    parser.add_argument('--algo', type=str, default='bc', 
                        help='训练策略使用的算法')
    parser.add_argument('--dataset', type=str, 
                        help='保存/加载数据集的文件路径')
    parser.add_argument('--model_file', type=str, 
                        help='加载模型文件的路径（用于 rollout）')
    parser.add_argument('--record_length', type=int, default=1200, 
                        help='每个 episode 录制的时长（秒）')
    parser.add_argument('--ctrl_device', type=str, default='vr', 
                        help='控制设备类型')
    parser.add_argument('--playback_mode', type=str, default='random', 
                        help='回放模式 (loop 或 random)')
    parser.add_argument('--rollout_times', type=int, default=10, 
                        help='rollout 测试次数')
    parser.add_argument('--augmented_sacle', type=float, default=0.01, 
                        help='数据增强的缩放比例')
    parser.add_argument('--augmented_rounds', type=int, default=3, 
                        help='数据增强的轮数')
    parser.add_argument('--teleoperation_rounds', type=int, default=20, 
                        help='遥操作的轮数')
    parser.add_argument('--sample_range', type=float, default=0.0, 
                        help='采样物体和目标位置的区域范围')
    parser.add_argument('--realtime_playback', type=bool, default=True, 
                        help='是否启用实时回放或 rollout')
    
    args = parser.parse_args()
    
    # 如果请求列出配置，则打印并退出
    if args.list_configs:
        print("可用的机器人配置：")
        configs = list_available_configs()
        for i, config in enumerate(configs, 1):
            print(f"  {i}. {config}")
        print("\n使用方法：")
        print("  单个配置: --robot_config openloong")
        print("  多个配置: --robot_configs 'robot1:openloong,robot2:d12'")
        sys.exit(0)
    
    # 解析机器人配置
    robot_configs = {}
    
    if args.robot_configs:
        # 为不同机器人指定不同配置
        robot_configs = parse_robot_configs(args.robot_configs)
        print(f"使用自定义机器人配置: {robot_configs}")
    elif args.robot_config:
        # 所有机器人使用相同配置
        agent_names = args.agent_name.split()
        robot_configs = {name: args.robot_config for name in agent_names}
        print(f"所有机器人使用配置: {args.robot_config}")
    else:
        print("未指定配置，将根据机器人名称自动推断配置")
    
    # 将配置添加到 args 中
    args.robot_configs = robot_configs
    
    # 运行主程序
    openloong_manipulation.run_openloong_sim(args, project_root, current_file_path)

