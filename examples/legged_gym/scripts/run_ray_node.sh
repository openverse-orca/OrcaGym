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

# 注意：不再使用配置文件读取head节点IP，改为通过参数传入

# 默认端口
RAY_PORT=6379
RAY_DASHBOARD_PORT=8265

# NFS配置 - 使用动态路径
NFS_EXPORT_PATH="$(dirname "$SCRIPT_DIR")/trained_models_tmp"
NFS_MOUNT_PATH="/tmp/trained_models_tmp"

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
    
    print_success "使用conda环境: $CONDA_DEFAULT_ENV"
}

# 函数：验证IP地址格式
validate_ip() {
    local ip=$1
    if [[ $ip =~ ^[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}$ ]]; then
        return 0
    else
        return 1
    fi
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

# 函数：检查NFS服务是否运行
check_nfs_service() {
    if systemctl is-active --quiet nfs-kernel-server; then
        return 0  # NFS服务运行中
    else
        return 1  # NFS服务未运行
    fi
}

# 函数：启动NFS服务
start_nfs_service() {
    print_info "启动NFS服务..."
    sudo systemctl start nfs-kernel-server
    sudo systemctl enable nfs-kernel-server
    if [[ $? -eq 0 ]]; then
        print_success "NFS服务启动成功"
    else
        print_error "NFS服务启动失败"
        exit 1
    fi
}

# 函数：配置NFS共享目录（head节点）
setup_nfs_export() {
    local head_ip=$1
    
    print_info "配置NFS共享目录..."
    print_info "共享路径: $NFS_EXPORT_PATH"
    print_info "实际路径: $(realpath "$NFS_EXPORT_PATH")"
    
    # 检查NFS服务
    if ! check_nfs_service; then
        print_warning "NFS服务未运行，正在启动..."
        start_nfs_service
    fi
    
    # 确保共享目录存在
    if [[ ! -d "$NFS_EXPORT_PATH" ]]; then
        print_info "创建共享目录: $NFS_EXPORT_PATH"
        mkdir -p "$NFS_EXPORT_PATH"
    fi
    
    # 配置NFS导出
    local export_line="$NFS_EXPORT_PATH *(rw,sync,no_subtree_check,no_root_squash)"
    
    # 检查是否已经配置过
    if ! grep -q "$NFS_EXPORT_PATH" /etc/exports 2>/dev/null; then
        print_info "添加NFS导出配置..."
        echo "$export_line" | sudo tee -a /etc/exports
        sudo exportfs -ra
        print_success "NFS导出配置完成"
    else
        print_info "NFS导出配置已存在"
    fi
    
    # 显示导出状态
    print_info "当前NFS导出状态:"
    sudo exportfs -v
}

# 函数：挂载NFS共享目录（worker节点）
mount_nfs_share() {
    local head_ip=$1
    
    print_info "挂载NFS共享目录..."
    print_info "从 $head_ip:$NFS_EXPORT_PATH 挂载到 $NFS_MOUNT_PATH"
    print_info "实际路径: $(realpath "$NFS_EXPORT_PATH")"
    
    # 创建挂载点
    if [[ ! -d "$NFS_MOUNT_PATH" ]]; then
        print_info "创建挂载点: $NFS_MOUNT_PATH"
        sudo mkdir -p "$NFS_MOUNT_PATH"
    fi
    
    # 检查是否已经挂载
    if mountpoint -q "$NFS_MOUNT_PATH"; then
        print_warning "目录已挂载，先卸载..."
        sudo umount "$NFS_MOUNT_PATH"
    fi
    
    # 挂载NFS共享
    sudo mount -t nfs "$head_ip:$NFS_EXPORT_PATH" "$NFS_MOUNT_PATH"
    
    if [[ $? -eq 0 ]]; then
        print_success "NFS共享挂载成功"
        print_info "挂载点: $NFS_MOUNT_PATH"
        print_info "注意: 此挂载仅在当前会话中有效，重启后需要重新运行脚本"
    else
        print_error "NFS共享挂载失败"
        exit 1
    fi
}

# 函数：启动Ray head节点
start_head_node() {
    local head_ip=$1
    
    print_info "启动Ray head节点..."
    print_info "Head节点IP: $head_ip"
    print_info "Ray端口: $RAY_PORT"
    print_info "Dashboard端口: $RAY_DASHBOARD_PORT"
    
    # 配置NFS共享目录
    setup_nfs_export "$head_ip"
    
    # 检查端口是否被占用
    if check_port $RAY_PORT; then
        print_warning "Ray端口 $RAY_PORT 已被占用，尝试停止现有进程..."
        pkill -f "ray start.*head" || true
        sleep 2
    fi
    
    if check_port $RAY_DASHBOARD_PORT; then
        print_warning "Dashboard端口 $RAY_DASHBOARD_PORT 已被占用"
    fi
    
    # 检测可用的GPU数量
    local num_gpus=0
    if command -v nvidia-smi &> /dev/null; then
        num_gpus=$(nvidia-smi --list-gpus | wc -l)
        print_info "检测到 $num_gpus 个GPU"
    else
        print_warning "未检测到nvidia-smi，GPU数量设为0"
    fi
    
    # 检测可用的CPU数量并分配为8的倍数，且不超过50%
    local num_cpus=$(nproc)
    local max_cpus=$((num_cpus * 50 / 100))
    local allocated_cpus=$((max_cpus / 8 * 8 + 2))  # 1 for ray scheduler, 1 for leaner
    print_info "检测到 $num_cpus 个CPU核心，最大分配 $max_cpus 个核心，实际分配 $allocated_cpus 个核心给Ray"
    
    # 启动Ray head节点
    ray start --head \
        --port=$RAY_PORT \
        --num-cpus=$allocated_cpus \
        --num-gpus=$num_gpus \
        --temp-dir=/tmp/ray \
        --node-ip-address=$head_ip
    
    if [[ $? -eq 0 ]]; then
        print_success "Ray head节点启动成功"
        # 获取实际的本地IP地址
        local actual_ip=$(hostname -I | awk '{print $1}')
        print_info "Ray地址: ray://$actual_ip:$RAY_PORT"
        print_info "GPU资源: $num_gpus"
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
    
    # 挂载NFS共享目录
    mount_nfs_share "$head_ip"
    
    # 检测可用的GPU数量
    local num_gpus=0
    if command -v nvidia-smi &> /dev/null; then
        num_gpus=$(nvidia-smi --list-gpus | wc -l)
        print_info "检测到 $num_gpus 个GPU"
    else
        print_warning "未检测到nvidia-smi，GPU数量设为0"
    fi
    
    # 检测可用的CPU数量，并分配为8的倍数
    local num_cpus=$(nproc)
    local allocated_cpus=$((num_cpus / 8 * 8))
    print_info "检测到 $num_cpus 个CPU核心，最大分配 $max_cpus 个核心，实际分配 $allocated_cpus 个核心给Ray（8的倍数）"
    
    # 启动Ray worker节点
    ray start \
        --address=$head_ip:$RAY_PORT \
        --num-cpus=$allocated_cpus \
        --num-gpus=$num_gpus
    
    if [[ $? -eq 0 ]]; then
        print_success "Ray worker节点启动成功"
        print_info "已连接到head节点: $head_ip:$RAY_PORT"
        print_info "GPU资源: $num_gpus"
        # 获取实际的本地IP地址
        local actual_ip=$(hostname -I | awk '{print $1}')
        print_info "当前节点IP: $actual_ip"
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
    echo "  head <HEAD_IP>       启动Ray head节点（指定IP地址）"
    echo "  worker <HEAD_IP>     启动Ray worker节点（指定head节点IP）"
    echo "  stop                 停止Ray节点"
    echo "  status               显示Ray集群状态"
    echo "  help                 显示此帮助信息"
    echo ""
    echo "示例:"
    echo "  $0 head 192.168.1.100        # 在192.168.1.100启动head节点"
    echo "  $0 worker 192.168.1.100      # 连接到192.168.1.100的head节点"
    echo "  $0 stop                       # 停止Ray节点"
    echo "  $0 status                     # 显示状态"
    echo ""
    echo "NFS共享功能:"
    echo "  - head节点会自动配置NFS共享，将trained_models_tmp目录共享给worker节点"
    echo "  - worker节点会自动挂载head节点的trained_models_tmp目录到/tmp/trained_models_tmp"
    echo "  - 挂载仅在当前会话中有效，重启后需要重新运行脚本"
    echo "  - 需要sudo权限来配置NFS服务和挂载点"
    echo ""
    echo "注意: head节点IP地址必须作为参数传入，不再从配置文件读取"
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
            if [[ $# -ne 2 ]]; then
                print_error "head命令需要指定IP地址"
                echo "用法: $0 head <HEAD_IP>"
                exit 1
            fi
            if ! validate_ip "$2"; then
                print_error "无效的IP地址格式: $2"
                exit 1
            fi
            start_head_node "$2"
            ;;
        "worker")
            if [[ $# -ne 2 ]]; then
                print_error "worker命令需要指定head节点IP地址"
                echo "用法: $0 worker <HEAD_IP>"
                exit 1
            fi
            if ! validate_ip "$2"; then
                print_error "无效的IP地址格式: $2"
                exit 1
            fi
            start_worker_node "$2"
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
