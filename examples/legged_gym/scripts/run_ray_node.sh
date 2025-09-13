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

# NFS配置 - 使用统一的挂载方案
# Head节点：将./trained_models_tmp挂载到/mnt/nfs/下并共享
# Worker节点：挂载head节点的/mnt/nfs/trained_models_tmp并创建软链接
LOCAL_TRAINED_MODELS_PATH="$(dirname "$SCRIPT_DIR")/trained_models_tmp"
NFS_BASE_PATH="/mnt/nfs"
NFS_EXPORT_PATH="$NFS_BASE_PATH/trained_models_tmp"
NFS_MOUNT_PATH="/mnt/nfs/trained_models_tmp"

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

# 函数：验证路径和权限
validate_paths_and_permissions() {
    print_info "验证路径和权限..."
    
    # 检查脚本目录是否可写
    if [[ ! -w "$(dirname "$SCRIPT_DIR")" ]]; then
        print_error "脚本目录不可写: $(dirname "$SCRIPT_DIR")"
        exit 1
    fi
    
    # 检查是否有sudo权限
    if ! sudo -n true 2>/dev/null; then
        print_warning "需要sudo权限来配置NFS，请确保当前用户有sudo权限"
    fi
    
    # 检查NFS相关工具是否可用
    if ! command -v exportfs &> /dev/null; then
        print_error "exportfs命令不可用，请安装nfs-kernel-server"
        exit 1
    fi
    
    if ! command -v mount &> /dev/null; then
        print_error "mount命令不可用"
        exit 1
    fi
    
    print_success "路径和权限验证通过"
}

# 函数：验证worker节点的路径和权限（不包含NFS服务相关检查）
validate_worker_paths_and_permissions() {
    print_info "验证worker节点路径和权限..."
    
    # 检查脚本目录是否可写
    if [[ ! -w "$(dirname "$SCRIPT_DIR")" ]]; then
        print_error "脚本目录不可写: $(dirname "$SCRIPT_DIR")"
        exit 1
    fi
    
    # 检查是否有sudo权限（用于挂载NFS）
    if ! sudo -n true 2>/dev/null; then
        print_warning "需要sudo权限来挂载NFS，请确保当前用户有sudo权限"
    fi
    
    # 检查mount命令是否可用（用于挂载NFS）
    if ! command -v mount &> /dev/null; then
        print_error "mount命令不可用"
        exit 1
    fi
    
    print_success "worker节点路径和权限验证通过"
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
    print_info "本地路径: $LOCAL_TRAINED_MODELS_PATH"
    print_info "NFS导出路径: $NFS_EXPORT_PATH"
    print_info "实际本地路径: $(realpath "$LOCAL_TRAINED_MODELS_PATH")"
    
    # 检查NFS服务
    if ! check_nfs_service; then
        print_warning "NFS服务未运行，正在启动..."
        start_nfs_service
    fi
    
    # 创建NFS基础目录
    if [[ ! -d "$NFS_BASE_PATH" ]]; then
        print_info "创建NFS基础目录: $NFS_BASE_PATH"
        sudo mkdir -p "$NFS_BASE_PATH"
    fi
    
    # 确保本地trained_models_tmp目录存在
    if [[ ! -d "$LOCAL_TRAINED_MODELS_PATH" ]]; then
        print_info "创建本地trained_models_tmp目录: $LOCAL_TRAINED_MODELS_PATH"
        mkdir -p "$LOCAL_TRAINED_MODELS_PATH"
    fi
    
    # 创建NFS导出目录
    if [[ ! -d "$NFS_EXPORT_PATH" ]]; then
        print_info "创建NFS导出目录: $NFS_EXPORT_PATH"
        sudo mkdir -p "$NFS_EXPORT_PATH"
    fi
    
    # 检查是否已经存在软链接
    if [[ -L "$NFS_EXPORT_PATH" ]]; then
        print_info "NFS导出目录软链接已存在"
    else
        # 如果NFS导出目录是普通目录，先删除
        if [[ -d "$NFS_EXPORT_PATH" ]]; then
            print_info "删除现有的NFS导出目录: $NFS_EXPORT_PATH"
            sudo rm -rf "$NFS_EXPORT_PATH"
        fi
        
        # 创建软链接：将NFS导出目录链接到本地trained_models_tmp目录
        print_info "创建软链接: $NFS_EXPORT_PATH -> $LOCAL_TRAINED_MODELS_PATH"
        sudo ln -s "$LOCAL_TRAINED_MODELS_PATH" "$NFS_EXPORT_PATH"
        
        if [[ $? -eq 0 ]]; then
            print_success "软链接创建成功"
        else
            print_error "软链接创建失败"
            exit 1
        fi
    fi
    
    # 设置目录权限，确保所有用户都可以访问
    print_info "设置目录权限..."
    sudo chmod 755 "$NFS_BASE_PATH"
    sudo chmod 755 "$LOCAL_TRAINED_MODELS_PATH"
    
    # 设置NFS导出目录的权限（通过软链接）
    if [[ -L "$NFS_EXPORT_PATH" ]]; then
        # 软链接的权限由目标目录决定，这里确保目标目录权限正确
        sudo chmod 755 "$(readlink -f "$NFS_EXPORT_PATH")"
    fi
    
    # 配置NFS导出
    local export_line="$NFS_EXPORT_PATH *(rw,sync,no_subtree_check,no_root_squash,all_squash,anonuid=1000,anongid=1000)"
    
    # 检查是否已经配置过
    if ! grep -q "$NFS_EXPORT_PATH" /etc/exports 2>/dev/null; then
        print_info "添加NFS导出配置..."
        echo "$export_line" | sudo tee -a /etc/exports
        sudo exportfs -ra
        print_success "NFS导出配置完成"
    else
        print_info "NFS导出配置已存在"
    fi

    # 重启NFS服务
    print_info "重启NFS服务..."
    sudo systemctl restart nfs-kernel-server
    if [[ $? -eq 0 ]]; then
        print_success "NFS服务重启成功"
    else
        print_error "NFS服务重启失败"
        exit 1
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
    print_info "本地trained_models_tmp路径: $LOCAL_TRAINED_MODELS_PATH"
    
    # 创建NFS基础目录
    if [[ ! -d "$NFS_BASE_PATH" ]]; then
        print_info "创建NFS基础目录: $NFS_BASE_PATH"
        sudo mkdir -p "$NFS_BASE_PATH"
    fi
    
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
        
        # 设置挂载点的权限，确保当前用户可以访问
        print_info "设置挂载点权限..."
        sudo chmod 755 "$NFS_MOUNT_PATH"
        
        # 创建软链接：将本地的./trained_models_tmp指向挂载的NFS路径
        if [[ -L "$LOCAL_TRAINED_MODELS_PATH" ]]; then
            print_info "软链接已存在，先删除..."
            rm "$LOCAL_TRAINED_MODELS_PATH"
        elif [[ -d "$LOCAL_TRAINED_MODELS_PATH" ]]; then
            print_warning "本地目录已存在，先备份为trained_models_tmp_backup..."
            mv "$LOCAL_TRAINED_MODELS_PATH" "${LOCAL_TRAINED_MODELS_PATH}_backup"
        fi
        
        # 创建软链接
        print_info "创建软链接: $LOCAL_TRAINED_MODELS_PATH -> $NFS_MOUNT_PATH"
        ln -s "$NFS_MOUNT_PATH" "$LOCAL_TRAINED_MODELS_PATH"
        
        if [[ $? -eq 0 ]]; then
            print_success "软链接创建成功"
            print_info "本地路径 $LOCAL_TRAINED_MODELS_PATH 现在指向共享的NFS目录"
        else
            print_error "软链接创建失败"
            exit 1
        fi
        
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
    
    # 验证路径和权限
    validate_paths_and_permissions
    
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
    
    local num_cpus=$(nproc)
    local allocated_cpus=$((num_cpus - 8))  # 分配少一些worker, 给OrcaLab预留CPU和GPU资源
    print_info "检测到 $num_cpus 个CPU核心，实际分配 $allocated_cpus 个核心给Ray"
    
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
    
    # 验证路径和权限（worker节点不需要NFS服务相关检查）
    validate_worker_paths_and_permissions
    
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
    local allocated_cpus=$((num_cpus))
    print_info "检测到 $num_cpus 个CPU核心，实际分配 $allocated_cpus 个核心给Ray"
    
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

# 函数：清理NFS挂载和软链接
cleanup_nfs() {
    print_info "清理NFS挂载和软链接..."
    
    # 清理worker节点的软链接
    if [[ -L "$LOCAL_TRAINED_MODELS_PATH" ]]; then
        print_info "删除worker节点软链接: $LOCAL_TRAINED_MODELS_PATH"
        rm "$LOCAL_TRAINED_MODELS_PATH"
        
        # 如果存在备份目录，恢复它
        if [[ -d "${LOCAL_TRAINED_MODELS_PATH}_backup" ]]; then
            print_info "恢复备份目录: ${LOCAL_TRAINED_MODELS_PATH}_backup -> $LOCAL_TRAINED_MODELS_PATH"
            mv "${LOCAL_TRAINED_MODELS_PATH}_backup" "$LOCAL_TRAINED_MODELS_PATH"
        fi
    fi
    
    # 清理worker节点的NFS挂载
    if mountpoint -q "$NFS_MOUNT_PATH"; then
        print_info "卸载worker节点NFS挂载: $NFS_MOUNT_PATH"
        sudo umount "$NFS_MOUNT_PATH"
    fi
    
    # 清理head节点的NFS导出软链接
    if [[ -L "$NFS_EXPORT_PATH" ]]; then
        print_info "删除head节点NFS导出软链接: $NFS_EXPORT_PATH"
        sudo rm "$NFS_EXPORT_PATH"
    fi
    
    # 清理NFS导出配置（可选，注释掉以避免影响其他NFS共享）
    # if grep -q "$NFS_EXPORT_PATH" /etc/exports 2>/dev/null; then
    #     print_info "清理NFS导出配置..."
    #     sudo sed -i "/$NFS_EXPORT_PATH/d" /etc/exports
    #     sudo exportfs -ra
    # fi
    
    print_success "NFS清理完成"
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
    
    # 询问是否清理NFS
    echo -n "是否清理NFS挂载和软链接？(y/N): "
    read -r response
    if [[ "$response" =~ ^[Yy]$ ]]; then
        cleanup_nfs
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
    echo "  cleanup              清理NFS挂载和软链接"
    echo "  status               显示Ray集群状态"
    echo "  help                 显示此帮助信息"
    echo ""
    echo "示例:"
    echo "  $0 head 192.168.1.100        # 在192.168.1.100启动head节点"
    echo "  $0 worker 192.168.1.100      # 连接到192.168.1.100的head节点"
    echo "  $0 stop                       # 停止Ray节点"
    echo "  $0 cleanup                    # 清理NFS挂载和软链接"
    echo "  $0 status                     # 显示状态"
    echo ""
    echo "NFS共享功能:"
    echo "  - head节点会创建软链接：/mnt/nfs/trained_models_tmp -> ./trained_models_tmp"
    echo "  - head节点通过NFS共享/mnt/nfs/trained_models_tmp目录"
    echo "  - worker节点会挂载head节点的/mnt/nfs/trained_models_tmp到本地/mnt/nfs/trained_models_tmp"
    echo "  - worker节点会创建软链接：./trained_models_tmp -> /mnt/nfs/trained_models_tmp"
    echo "  - 这样所有节点都可以通过./trained_models_tmp访问共享的模型文件"
    echo "  - 支持不同用户名的worker节点访问共享目录"
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
        "cleanup")
            cleanup_nfs
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
