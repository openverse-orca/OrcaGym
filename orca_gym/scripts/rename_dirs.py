import os
import shutil
import sys

from orca_gym.log.orca_log import get_orca_logger
_logger = get_orca_logger()


def get_absolute_base_path():
    """获取绝对基础路径(OrcaGym目录)"""
    # 获取脚本绝对路径
    script_path = os.path.abspath(__file__)
    # 向上回溯3级目录到OrcaGym
    base_path = os.path.dirname(os.path.dirname(os.path.dirname(script_path)))
    return base_path

def main():
    # 检查参数
    if len(sys.argv) != 3:
        _logger.info("用法: python rename_dirs.py <相对路径> <目标目录名>")
        _logger.info("示例: python rename_dirs.py examples/openpi/records_tmp/ shop")
        sys.exit(1)
    
    relative_path = sys.argv[1].strip('/')  # 去除可能的末尾斜杠
    target_dir_name = sys.argv[2]
    
    # 计算基础路径
    base_path = get_absolute_base_path()
    _logger.info(f"[调试] 计算出的基础路径: {base_path}")
    
    # 构建完整目标路径
    target_path = os.path.join(base_path, relative_path, target_dir_name)
    _logger.info(f"[调试] 完整目标路径: {target_path}")
    
    # 验证路径是否存在
    if not os.path.exists(target_path):
        _logger.info(f"错误: 目标目录不存在: {target_path}")
        _logger.info("可能原因:")
        _logger.info(f"1. 相对路径 '{relative_path}' 不正确")
        _logger.info(f"2. 目标目录名 '{target_dir_name}' 不正确")
        _logger.info("请检查:")
        _logger.info(f"   - 脚本所在目录: {os.path.dirname(os.path.abspath(__file__))}")
        _logger.info(f"   - 计算出的基础路径: {base_path}")
        _logger.info(f"   - 尝试访问的路径: {target_path}")
        sys.exit(1)
    
    # 确认是目录
    if not os.path.isdir(target_path):
        _logger.info(f"错误: 目标路径不是目录: {target_path}")
        sys.exit(1)
    
    # 处理目录
    try:
        # 1. 获取子场景目录 (如 pick_catch)
        subscene_dirs = [d for d in os.listdir(target_path) if os.path.isdir(os.path.join(target_path, d))]
        if len(subscene_dirs) != 1:
            _logger.info(f"错误: {target_path} 下子场景目录数量不正确(应为1个，实际找到{len(subscene_dirs)}个)")
            _logger.info(f"找到的子目录: {subscene_dirs}")
            sys.exit(1)
        
        subscene_dir = os.path.join(target_path, subscene_dirs[0])
        _logger.info(f"[调试] 找到子场景目录: {subscene_dir}")
        
        # 2. 获取动作目录
        action_dirs = [d for d in os.listdir(subscene_dir) if os.path.isdir(os.path.join(subscene_dir, d))]
        if not action_dirs:
            _logger.info(f"错误: {subscene_dir} 下没有动作目录")
            sys.exit(1)
        _logger.info(f"[调试] 找到动作目录: {action_dirs}")
        
        # 3. 解析第一个动作目录名获取大小信息
        first_action = action_dirs[0]
        parts = first_action.split("_")
        if len(parts) < 4:
            _logger.info(f"错误: {first_action} 目录名格式不正确")
            _logger.info(f"目录名应包含至少3个下划线分隔的部分，如: 名称_53p21GB_2000counts_85p30h")
            sys.exit(1)
        
        size_part, files_part, duration_part = parts[-3], parts[-2], parts[-1]
        _logger.performance(f"[调试] 解析出的大小信息: {size_part}, {files_part}, {duration_part}")
        
        # 4. 创建新目录名
        new_target_name = f"{target_dir_name}-{size_part}_{files_part}_{duration_part}"
        new_subscene_name = f"{subscene_dirs[0]}-{size_part}_{files_part}_{duration_part}"
        _logger.info(f"[调试] 新目录名: {new_target_name}/{new_subscene_name}")
        
        # 5. 创建新目录结构
        parent_dir = os.path.dirname(target_path)
        new_target_dir = os.path.join(parent_dir, new_target_name)
        new_subscene_dir = os.path.join(new_target_dir, new_subscene_name)
        os.makedirs(new_subscene_dir, exist_ok=True)
        _logger.info(f"[调试] 创建新目录结构: {new_target_dir}/{new_subscene_dir}")
        
        # 6. 移动并重命名动作目录
        for action in action_dirs:
            new_action_name = f"{action}-{size_part}_{files_part}_{duration_part}"
            src_path = os.path.join(subscene_dir, action)
            dst_path = os.path.join(new_subscene_dir, new_action_name)
            _logger.info(f"移动: {src_path} -> {dst_path}")
            shutil.move(src_path, dst_path)
        
        # 7. 删除空目录
        os.rmdir(subscene_dir)
        os.rmdir(target_path)
        _logger.info(f"成功: {target_path} 已重命名为 {new_target_dir}/{new_subscene_dir}/...")
        
    except Exception as e:
        _logger.info(f"错误: 处理过程中发生异常: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()