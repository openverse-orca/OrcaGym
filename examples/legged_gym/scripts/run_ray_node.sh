#!/bin/bash

# Ray集群管理脚本
# 用于启动Ray集群的head节点和worker节点
# 支持从配置文件读取head节点IP地址

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 脚本目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

# 配置文件路径
CONFIG_FILE="$SCRIPT_DIR/../configs/rllib_appo_config.yaml"

# 默认端口
RAY_PORT=6379
RAY_DASHBOARD_PORT=8265

# 函数：打印带颜色的消息
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 函数：检查conda环境
check_conda_env() {
    if ! command -v conda &> /dev/null; then
        print_error "Conda未安装或不在PATH中"
        exit 1
    fi
    
    # 初始化conda（如果需要）
    if [[ -z "$CONDA_DEFAULT_ENV" ]]; then
        print_info "初始化conda环境..."
        # 尝试不同的conda初始化方法
        if [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
            source "$HOME/miniconda3/etc/profile.d/conda.sh"
        elif [[ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]]; then
            source "$HOME/anaconda3/etc/profile.d/conda.sh"
        elif [[ -f "/opt/conda/etc/profile.d/conda.sh" ]]; then
            source "/opt/conda/etc/profile.d/conda.sh"
        else
            print_warning "无法找到conda.sh，尝试使用conda命令..."
        fi
    fi
    
    if [[ "$CONDA_DEFAULT_ENV" != "orca" ]]; then
        print_warning "当前conda环境不是'orca'，正在激活..."
        conda activate orca
        if [[ "$CONDA_DEFAULT_ENV" != "orca" ]]; then
            print_error "无法激活'orca'环境"
            exit 1
        fi
    fi
    
    print_success "使用conda环境: $CONDA_DEFAULT_ENV"
}

# 函数：从配置文件读取head节点IP
get_head_node_ip() {
    if [[ ! -f "$CONFIG_FILE" ]]; then
        print_error "配置文件不存在: $CONFIG_FILE"
        exit 1
    fi
    
    # 使用grep和awk提取IP地址
    HEAD_IP=$(grep "orcagym_addresses:" "$CONFIG_FILE" | awk -F'"' '{print $2}' | awk -F':' '{print $1}')
    
    if [[ -z "$HEAD_IP" ]]; then
        print_error "无法从配置文件中提取head节点IP地址"
        exit 1
    fi
    
    echo "$HEAD_IP"
}

# 函数：检查端口是否被占用
check_port() {
    local port=$1
    if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null 2>&1; then
        return 0  # 端口被占用
    else
        return 1  # 端口空闲
    fi
}

# 函数：启动Ray head节点
start_head_node() {
    local head_ip=$1
    
    print_info "启动Ray head节点..."
    print_info "Head节点IP: $head_ip"
    print_info "Ray端口: $RAY_PORT"
    print_info "Dashboard端口: $RAY_DASHBOARD_PORT"
    
    # 检查端口是否被占用
    if check_port $RAY_PORT; then
        print_warning "Ray端口 $RAY_PORT 已被占用，尝试停止现有进程..."
        pkill -f "ray start.*head" || true
        sleep 2
    fi
    
    if check_port $RAY_DASHBOARD_PORT; then
        print_warning "Dashboard端口 $RAY_DASHBOARD_PORT 已被占用"
    fi
    
    # 启动Ray head节点
    ray start --head \
        --port=$RAY_PORT \
        --num-cpus=0 \
        --temp-dir=/tmp/ray \
        --block
    
    if [[ $? -eq 0 ]]; then
        print_success "Ray head节点启动成功"
        print_info "Ray地址: ray://$head_ip:$RAY_PORT"
        print_warning "注意：当前Ray安装为minimal版本，不支持Dashboard"
    else
        print_error "Ray head节点启动失败"
        exit 1
    fi
}

# 函数：启动Ray worker节点
start_worker_node() {
    local head_ip=$1
    
    print_info "启动Ray worker节点..."
    print_info "连接到head节点: $head_ip:$RAY_PORT"
    
    # 启动Ray worker节点
    ray start \
        --address=$head_ip:$RAY_PORT \
        --num-cpus=0 \
        --temp-dir=/tmp/ray
    
    if [[ $? -eq 0 ]]; then
        print_success "Ray worker节点启动成功"
        print_info "已连接到head节点: $head_ip:$RAY_PORT"
    else
        print_error "Ray worker节点启动失败"
        exit 1
    fi
}

# 函数：停止Ray节点
stop_ray() {
    print_info "停止Ray节点..."
    ray stop
    if [[ $? -eq 0 ]]; then
        print_success "Ray节点已停止"
    else
        print_warning "停止Ray节点时出现警告"
    fi
}

# 函数：显示Ray状态
show_ray_status() {
    print_info "Ray集群状态:"
    ray status
}

# 函数：显示帮助信息
show_help() {
    echo "Ray集群管理脚本"
    echo ""
    echo "用法: $0 [选项]"
    echo ""
    echo "选项:"
    echo "  head                启动Ray head节点"
    echo "  worker [HEAD_IP]    启动Ray worker节点（可选指定head节点IP）"
    echo "  stop                停止Ray节点"
    echo "  status              显示Ray集群状态"
    echo "  help                显示此帮助信息"
    echo ""
    echo "示例:"
    echo "  $0 head              # 启动head节点"
    echo "  $0 worker            # 启动worker节点（使用配置文件中的IP）"
    echo "  $0 worker 192.168.1.100  # 启动worker节点（指定IP）"
    echo "  $0 stop              # 停止Ray节点"
    echo "  $0 status            # 显示状态"
    echo ""
    echo "配置文件: $CONFIG_FILE"
}

# 主函数
main() {
    # 检查参数
    if [[ $# -eq 0 ]]; then
        show_help
        exit 1
    fi
    
    # 检查conda环境
    check_conda_env
    
    # 解析命令
    case "$1" in
        "head")
            head_ip=$(get_head_node_ip)
            start_head_node "$head_ip"
            ;;
        "worker")
            if [[ $# -eq 2 ]]; then
                # 使用指定的IP
                start_worker_node "$2"
            else
                # 从配置文件读取IP
                head_ip=$(get_head_node_ip)
                start_worker_node "$head_ip"
            fi
            ;;
        "stop")
            stop_ray
            ;;
        "status")
            show_ray_status
            ;;
        "help"|"-h"|"--help")
            show_help
            ;;
        *)
            print_error "未知命令: $1"
            show_help
            exit 1
            ;;
    esac
}

# 执行主函数
main "$@"
