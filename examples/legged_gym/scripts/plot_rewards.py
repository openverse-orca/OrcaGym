import re
import matplotlib
matplotlib.use('Agg')  # 无GUI环境必需
import matplotlib.pyplot as plt
from datetime import datetime
import math
import os

from orca_gym.log.orca_log import get_orca_logger
_logger = get_orca_logger()


def parse_rewards(file_path):
    """解析所有奖励项（支持不同长度数据）"""
    reward_data = {}
    # 使用您提供的正确正则表达式
    pattern = re.compile(
        r'^([\w\s]+)\s+reward_\d\.\d+e[+-]?\d+:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)$'
    )

    with open(file_path, 'r') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            match = pattern.match(line)
            if match:
                name = match.group(1).strip()
                value = float(match.group(2))
                
                if name not in reward_data:
                    reward_data[name] = []
                reward_data[name].append(value)
            else:
                _logger.error(f"Warning: Line {line_num} format invalid, skipped: {line}")
    
    if not reward_data:
        raise ValueError("No valid reward data found. Check file format.")
    
    return reward_data

def auto_subplot_layout(num_plots):
    """自动生成子图布局"""
    if num_plots <= 3:
        return (1, num_plots)
    elif num_plots <= 6:
        return (2, 3)
    elif num_plots <= 12:
        return (3, 4)
    else:
        rows = math.ceil(math.sqrt(num_plots))
        cols = math.ceil(num_plots / rows)
        return (rows, cols)

def plot_all_rewards(reward_data, file_path):
    """绘制所有奖励曲线（独立x轴）"""
    plt.figure(figsize=(24, 16), dpi=100)
    
    # 获取所有奖励名称和对应数据
    reward_items = list(reward_data.items())
    num_plots = len(reward_items)
    
    # 自动确定子图布局
    rows, cols = auto_subplot_layout(num_plots)
    
    # 设置全局字体
    plt.rcParams.update({'font.size': 8})
    
    for idx, (name, values) in enumerate(reward_items):
        ax = plt.subplot(rows, cols, idx+1)
        steps = range(len(values))  # 每个子图独立的x轴
        
        # 强制设置Y轴方向始终向上
        ax.yaxis.set_inverted(False)
        
        # 动态设置y轴范围
        ax.set_ylim(auto_y_range(values))
        
        # 绘制数据
        ax.plot(steps, values, linewidth=0.8, alpha=0.7)
        
        # 特殊处理总奖励
        if "Total reward" in name:
            ax.plot(steps, values, color='red', linewidth=1.5, label='Total')
            ax.legend(loc='upper right')
        
        # 设置标题和标签
        ax.set_title(name.replace('_', '\n'), pad=10, fontsize=8)
        ax.grid(alpha=0.3)
        ax.set_xlabel('Steps')
        ax.set_ylabel('Value')
        
        # 确保Y轴刻度标签方向正确
        ax.yaxis.set_tick_params(direction='out')
    
    # 调整整体布局
    plt.tight_layout(pad=1.5)
    plt.subplots_adjust(top=0.92)
    
    # 添加总标题
    plt.suptitle(f"All Rewards Progress (Total Steps: {len(next(iter(reward_data.values())))})",
                fontsize=16, fontweight='bold', y=0.98)
    
    # 保存图片
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_dir = os.path.dirname(file_path)
    file_prefix = os.path.basename(file_path).split('.')[0]
    filename = os.path.join(file_dir, f"{file_prefix}.png")
    plt.savefig(filename, bbox_inches='tight', dpi=300)
    plt.close()
    _logger.info(f"Saved all rewards plot to {filename}")

def auto_y_range(values):
    """自动确定y轴范围"""
    max_val = max(values)
    min_val = min(values)
    return [min_val - abs(min_val)*0.2, max_val + abs(max_val)*0.2]

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        _logger.info("Usage: python plot_all_rewards.py <reward_file>")
        sys.exit(1)
    
    file_path = sys.argv[1]
    try:
        reward_data= parse_rewards(file_path)
        _logger.info(f"Successfully parsed {len(reward_data)} reward types")
        plot_all_rewards(reward_data, file_path)
    except Exception as e:
        _logger.error(f"Error: {str(e)}")
        sys.exit(1)