"""
Lite3 ONNX策略运行Demo
展示如何在OrcaGym-dev中使用迁移后的Lite3配置和ONNX策略运行仿真

使用方法:
    python run_lite3_onnx_demo.py --onnx_model_path /path/to/policy.onnx
"""

import argparse
import numpy as np
import sys
import os

# 添加项目根目录到Python路径
current_file_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from envs.legged_gym.utils.onnx_policy import load_onnx_policy
from envs.legged_gym.utils.lite3_obs_helper import compute_lite3_obs, get_dof_pos_default_policy
from envs.legged_gym.robot_config.Lite3_config import Lite3Config

from orca_gym.log.orca_log import get_orca_logger
_logger = get_orca_logger()



def main():
    parser = argparse.ArgumentParser(description="Lite3 ONNX策略运行Demo")
    parser.add_argument(
        "--onnx_model_path",
        type=str,
        default=None,
        help="Path to ONNX model file"
    )
    parser.add_argument(
        "--test_obs",
        action="store_true",
        help="Test observation computation"
    )
    args = parser.parse_args()
    
    _logger.info("=" * 60)
    _logger.info("Lite3 ONNX策略运行Demo")
    _logger.info("=" * 60)
    
    # ========== 1. 加载配置 ==========
    _logger.info("\n[1] 加载Lite3配置...")
    config = Lite3Config
    
    _logger.info(f"  - 关节数量: {len(config['leg_joint_names'])}")
    _logger.info(f"  - PD参数: kp={config['kps'][0]}, kd={config['kds'][0]}")
    print(f"  - 观测缩放: omega_scale={config.get('omega_scale', 'N/A')}, "
          f"dof_vel_scale={config.get('dof_vel_scale', 'N/A')}")
    
    # ========== 2. 加载ONNX策略 ==========
    if args.onnx_model_path:
        _logger.info(f"\n[2] 加载ONNX策略: {args.onnx_model_path}")
        try:
            policy = load_onnx_policy(args.onnx_model_path, device="cpu")
            _logger.info("  ✓ ONNX策略加载成功")
        except Exception as e:
            _logger.info(f"  ✗ ONNX策略加载失败: {e}")
            policy = None
    else:
        _logger.info("\n[2] 未指定ONNX模型路径，跳过策略加载")
        _logger.info("    提示: 使用 --onnx_model_path 参数指定模型路径")
        policy = None
    
    # ========== 3. 测试观测计算 ==========
    if args.test_obs:
        _logger.info("\n[3] 测试观测计算...")
        
        # 创建模拟数据
        base_ang_vel = np.random.randn(3)
        projected_gravity = np.random.randn(3)
        commands = np.random.randn(3)
        dof_pos = np.random.randn(12)
        dof_vel = np.random.randn(12)
        last_actions = np.zeros(12)
        
        # 获取配置参数
        omega_scale = config.get('omega_scale', 0.25)
        dof_vel_scale = config.get('dof_vel_scale', 0.05)
        max_cmd_vel = np.array(config.get('max_cmd_vel', [0.8, 0.8, 0.8]))
        dof_pos_default = get_dof_pos_default_policy()
        
        # 计算观测
        obs = compute_lite3_obs(
            base_ang_vel=base_ang_vel,
            projected_gravity=projected_gravity,
            commands=commands,
            dof_pos=dof_pos,
            dof_vel=dof_vel,
            last_actions=last_actions,
            omega_scale=omega_scale,
            dof_vel_scale=dof_vel_scale,
            max_cmd_vel=max_cmd_vel,
            dof_pos_default=dof_pos_default
        )
        
        _logger.info(f"  ✓ 观测计算完成，维度: {obs.shape}")
        assert obs.shape == (45,), f"观测维度应为(45,)，实际为{obs.shape}"
        
        # 测试批量观测
        num_envs = 10
        base_ang_vel_batch = np.random.randn(num_envs, 3)
        projected_gravity_batch = np.random.randn(num_envs, 3)
        commands_batch = np.random.randn(num_envs, 3)
        dof_pos_batch = np.random.randn(num_envs, 12)
        dof_vel_batch = np.random.randn(num_envs, 12)
        last_actions_batch = np.zeros((num_envs, 12))
        
        obs_batch = compute_lite3_obs(
            base_ang_vel=base_ang_vel_batch,
            projected_gravity=projected_gravity_batch,
            commands=commands_batch,
            dof_pos=dof_pos_batch,
            dof_vel=dof_vel_batch,
            last_actions=last_actions_batch,
            omega_scale=omega_scale,
            dof_vel_scale=dof_vel_scale,
            max_cmd_vel=max_cmd_vel,
            dof_pos_default=dof_pos_default
        )
        
        _logger.info(f"  ✓ 批量观测计算完成，维度: {obs_batch.shape}")
        assert obs_batch.shape == (num_envs, 45), f"批量观测维度应为({num_envs}, 45)，实际为{obs_batch.shape}"
        
        # 测试策略推理
        if policy is not None:
            try:
                actions = policy(obs)
                _logger.info(f"  ✓ 策略推理完成，动作维度: {actions.shape}")
                assert actions.shape == (12,), f"动作维度应为(12,)，实际为{actions.shape}"
                
                actions_batch = policy(obs_batch)
                _logger.info(f"  ✓ 批量策略推理完成，动作维度: {actions_batch.shape}")
                assert actions_batch.shape == (num_envs, 12), \
                    f"批量动作维度应为({num_envs}, 12)，实际为{actions_batch.shape}"
                
                _logger.info(f"\n  [示例输出]")
                _logger.info(f"  - 观测前3维 (base_ang_vel): {obs[:3]}")
                _logger.info(f"  - 动作前3维: {actions[:3]}")
            except Exception as e:
                _logger.info(f"  ✗ 策略推理失败: {e}")
                import traceback
                traceback.print_exc()
    
    # ========== 4. 使用说明 ==========
    _logger.info("\n" + "=" * 60)
    _logger.info("使用说明:")
    _logger.info("=" * 60)
    print("""
1. 在OrcaGym-dev环境中使用Lite3配置:
   - 配置文件: envs/legged_gym/robot_config/Lite3_config.py
   - 已添加迁移参数: omega_scale, dof_vel_scale, max_cmd_vel等

2. 使用ONNX策略:
   from envs.legged_gym.utils.onnx_policy import load_onnx_policy
   policy = load_onnx_policy("path/to/policy.onnx")

3. 计算Lite3格式观测:
   from envs.legged_gym.utils.lite3_obs_helper import compute_lite3_obs
   obs = compute_lite3_obs(...)

4. 运行策略:
   actions = policy(obs)
   
5. 在仿真环境中使用:
   - 参考 examples/legged_gym/run_legged_sim.py
   - 在环境循环中计算观测并运行策略
    """)
    
    _logger.info("=" * 60)
    _logger.info("Demo完成！")
    _logger.info("=" * 60)
    _logger.info("\n下一步:")
    _logger.info("1. 准备ONNX模型文件")
    _logger.info("2. 运行: python run_lite3_onnx_demo.py --onnx_model_path /path/to/policy.onnx --test_obs")
    _logger.info("3. 在仿真环境中集成ONNX策略")


if __name__ == "__main__":
    main()

