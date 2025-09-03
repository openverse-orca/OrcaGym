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
CONFIG_FILE="$SCRIPT_DIR/../configs/rllib_appo_cluster_config.yaml"

# 默认端口
RAY_PORT=6379
RAY_DASHBOARD_PORT=8265

# 共享存储配置
SHARED_STORAGE_PATH="/mnt/nfs/ray_results"
SHARED_STORAGE_OWNER="orca"  # 可以根据实际情况修改

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

# 函数：设置共享存储（head节点）
setup_shared_storage_head() {
    print_info "设置共享存储目录..."
    
    # 检查是否为root用户
    if [[ $EUID -eq 0 ]]; then
        print_info "以root用户运行，直接创建共享目录..."
        mkdir -p "$SHARED_STORAGE_PATH"
        chmod 777 "$SHARED_STORAGE_PATH"
        
        # 尝试设置所有者（如果用户存在）
        if id "$SHARED_STORAGE_OWNER" &>/dev/null; then
            chown "$SHARED_STORAGE_OWNER:$SHARED_STORAGE_OWNER" "$SHARED_STORAGE_PATH"
            print_info "设置目录所有者为: $SHARED_STORAGE_OWNER"
        fi
    else
        print_info "以普通用户运行，尝试创建共享目录..."
        
        # 尝试创建目录
        if mkdir -p "$SHARED_STORAGE_PATH" 2>/dev/null; then
            print_success "成功创建共享目录: $SHARED_STORAGE_PATH"
        else
            print_warning "无法创建共享目录，需要sudo权限"
            print_info "尝试使用sudo创建共享目录..."
            
            # 交互式提示用户输入sudo密码
            if sudo -n true 2>/dev/null; then
                # 用户有sudo权限且不需要密码
                print_info "用户有sudo权限，无需输入密码"
                if sudo mkdir -p "$SHARED_STORAGE_PATH" && sudo chmod 777 "$SHARED_STORAGE_PATH"; then
                    print_success "使用sudo成功创建共享目录: $SHARED_STORAGE_PATH"
                    
                    # 尝试设置所有者（如果用户存在）
                    if id "$SHARED_STORAGE_OWNER" &>/dev/null; then
                        sudo chown "$SHARED_STORAGE_OWNER:$SHARED_STORAGE_OWNER" "$SHARED_STORAGE_PATH"
                        print_info "设置目录所有者为: $SHARED_STORAGE_OWNER"
                    fi
                else
                    print_error "sudo创建目录失败"
                    return 1
                fi
            else
                # 需要输入sudo密码
                print_info "需要sudo权限来创建共享目录"
                print_info "请输入您的sudo密码:"
                
                # 使用sudo创建目录
                if sudo mkdir -p "$SHARED_STORAGE_PATH" && sudo chmod 777 "$SHARED_STORAGE_PATH"; then
                    print_success "使用sudo成功创建共享目录: $SHARED_STORAGE_PATH"
                    
                    # 尝试设置所有者（如果用户存在）
                    if id "$SHARED_STORAGE_OWNER" &>/dev/null; then
                        sudo chown "$SHARED_STORAGE_OWNER:$SHARED_STORAGE_OWNER" "$SHARED_STORAGE_PATH"
                        print_info "设置目录所有者为: $SHARED_STORAGE_OWNER"
                    fi
                else
                    print_error "sudo创建目录失败，请检查密码或权限"
                    return 1
                fi
            fi
        fi
    fi
    
    # 验证目录权限
    if [[ -d "$SHARED_STORAGE_PATH" ]] && [[ -w "$SHARED_STORAGE_PATH" ]]; then
        print_success "共享存储目录设置成功: $SHARED_STORAGE_PATH"
        
        # 测试写入权限
        local test_file="$SHARED_STORAGE_PATH/.test_write"
        if echo "test" > "$test_file" 2>/dev/null; then
            rm -f "$test_file"
            print_success "共享存储目录写入测试通过"
        else
            print_warning "共享存储目录写入测试失败，请检查权限"
        fi
    else
        print_error "共享存储目录设置失败"
        return 1
    fi
}

# 函数：挂载共享存储（worker节点）
mount_shared_storage_worker() {
    local head_ip=$1
    print_info "挂载共享存储目录..."
    
    # 检查目录是否已挂载
    if mount | grep -q "$SHARED_STORAGE_PATH"; then
        print_info "共享存储已挂载: $SHARED_STORAGE_PATH"
        return 0
    fi
    
    # 检查目录是否存在
    if [[ ! -d "$SHARED_STORAGE_PATH" ]]; then
        print_info "创建挂载点目录..."
        if ! mkdir -p "$SHARED_STORAGE_PATH" 2>/dev/null; then
            print_warning "无法创建挂载点，需要sudo权限"
            print_info "尝试使用sudo创建挂载点目录..."
            
            # 交互式提示用户输入sudo密码
            if sudo -n true 2>/dev/null; then
                # 用户有sudo权限且不需要密码
                print_info "用户有sudo权限，无需输入密码"
                sudo mkdir -p "$SHARED_STORAGE_PATH"
            else
                # 需要输入sudo密码
                print_info "需要sudo权限来创建挂载点目录"
                print_info "请输入您的sudo密码:"
                sudo mkdir -p "$SHARED_STORAGE_PATH"
            fi
        fi
    fi
    
    # 尝试挂载NFS
    print_info "尝试挂载NFS共享: $head_ip:$SHARED_STORAGE_PATH"
    
    if [[ $EUID -eq 0 ]]; then
        # 以root用户运行
        mount "$head_ip:$SHARED_STORAGE_PATH" "$SHARED_STORAGE_PATH" 2>/dev/null
    else
        # 以普通用户运行，尝试sudo
        print_info "需要sudo权限来挂载NFS共享"
        
        # 交互式提示用户输入sudo密码
        if sudo -n true 2>/dev/null; then
            # 用户有sudo权限且不需要密码
            print_info "用户有sudo权限，无需输入密码"
            sudo mount "$head_ip:$SHARED_STORAGE_PATH" "$SHARED_STORAGE_PATH" 2>/dev/null
        else
            # 需要输入sudo密码
            print_info "请输入您的sudo密码:"
            sudo mount "$head_ip:$SHARED_STORAGE_PATH" "$SHARED_STORAGE_PATH" 2>/dev/null
        fi
    fi
    
    # 检查挂载结果
    if mount | grep -q "$SHARED_STORAGE_PATH"; then
        print_success "共享存储挂载成功: $SHARED_STORAGE_PATH"
        
        # 测试读写权限
        local test_file="$SHARED_STORAGE_PATH/.test_mount"
        if echo "test" > "$test_file" 2>/dev/null; then
            rm -f "$test_file"
            print_success "挂载点写入测试通过"
        else
            print_warning "挂载点写入测试失败，请检查NFS配置"
        fi
    else
        print_warning "NFS挂载失败，尝试使用rsync同步..."
        
        # 如果NFS挂载失败，尝试使用rsync同步（作为备选方案）
        if command -v rsync &> /dev/null; then
            print_info "使用rsync同步共享目录..."
            if [[ $EUID -eq 0 ]]; then
                rsync -av --delete "$head_ip:$SHARED_STORAGE_PATH/" "$SHARED_STORAGE_PATH/" 2>/dev/null
            else
                print_info "需要sudo权限来执行rsync同步"
                
                # 交互式提示用户输入sudo密码
                if sudo -n true 2>/dev/null; then
                    # 用户有sudo权限且不需要密码
                    print_info "用户有sudo权限，无需输入密码"
                    sudo rsync -av --delete "$head_ip:$SHARED_STORAGE_PATH/" "$SHARED_STORAGE_PATH/" 2>/dev/null
                else
                    # 需要输入sudo密码
                    print_info "请输入您的sudo密码:"
                    sudo rsync -av --delete "$head_ip:$SHARED_STORAGE_PATH/" "$SHARED_STORAGE_PATH/" 2>/dev/null
                fi
            fi
            
            if [[ $? -eq 0 ]]; then
                print_success "rsync同步成功"
            else
                print_error "rsync同步失败，请检查网络连接和权限"
                return 1
            fi
        else
            print_error "无法挂载共享存储，且rsync不可用"
            return 1
        fi
    fi
}

# 函数：从配置文件读取head节点IP
get_head_node_ip() {
    if [[ ! -f "$CONFIG_FILE" ]]; then
        print_error "配置文件不存在: $CONFIG_FILE"
        exit 1
    fi
    
    # 使用grep和awk提取IP地址
    HEAD_IP=$(grep "ray_cluster_address:" "$CONFIG_FILE" | awk -F'"' '{print $2}' | awk -F'ray://' '{print $2}' | awk -F':' '{print $1}')
    
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
    
    # 设置共享存储
    if ! setup_shared_storage_head; then
        print_error "共享存储设置失败，无法启动Ray head节点"
        exit 1
    fi
    
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
    local allocated_cpus=$((max_cpus / 8 * 8 + 2))
    print_info "检测到 $num_cpus 个CPU核心，最大分配 $max_cpus 个核心，实际分配 $allocated_cpus 个核心给Ray（8的倍数）"
    
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
    
    # 挂载共享存储
    if ! mount_shared_storage_worker "$head_ip"; then
        print_error "共享存储挂载失败，无法启动Ray worker节点"
        exit 1
    fi
    
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
        --num-gpus=$num_gpus \
        --temp-dir=/tmp/ray
    
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

# 函数：测试共享存储
test_shared_storage() {
    print_info "测试共享存储访问..."
    
    if [[ ! -d "$SHARED_STORAGE_PATH" ]]; then
        print_error "共享存储目录不存在: $SHARED_STORAGE_PATH"
        return 1
    fi
    
    # 测试写入
    local test_file="$SHARED_STORAGE_PATH/.test_$(date +%s)"
    if echo "test_content" > "$test_file" 2>/dev/null; then
        print_success "写入测试通过"
        
        # 测试读取
        if [[ -f "$test_file" ]] && [[ "$(cat "$test_file")" == "test_content" ]]; then
            print_success "读取测试通过"
        else
            print_error "读取测试失败"
            rm -f "$test_file" 2>/dev/null
            return 1
        fi
        
        # 清理测试文件
        rm -f "$test_file"
        print_success "共享存储测试完成"
        return 0
    else
        print_error "写入测试失败，请检查权限"
        return 1
    fi
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
    echo "  storage setup       设置共享存储目录（head节点）"
    echo "  storage mount [IP]  挂载共享存储（worker节点）"
    echo "  storage test        测试共享存储访问"
    echo "  help                显示此帮助信息"
    echo ""
    echo "示例:"
    echo "  $0 head              # 启动head节点"
    echo "  $0 worker            # 启动worker节点（使用配置文件中的IP）"
    echo "  $0 worker 192.168.1.100  # 启动worker节点（指定IP）"
    echo "  $0 storage setup     # 设置共享存储"
    echo "  $0 storage mount     # 挂载共享存储"
    echo "  $0 storage test      # 测试共享存储"
    echo "  $0 stop              # 停止Ray节点"
    echo "  $0 status            # 显示状态"
    echo ""
    echo "配置文件: $CONFIG_FILE"
    echo "共享存储路径: $SHARED_STORAGE_PATH"
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
        "storage")
            case "${2:-}" in
                "setup")
                    setup_shared_storage_head
                    ;;
                "mount")
                    if [[ $# -eq 3 ]]; then
                        mount_shared_storage_worker "$3"
                    else
                        head_ip=$(get_head_node_ip)
                        mount_shared_storage_worker "$head_ip"
                    fi
                    ;;
                "test")
                    test_shared_storage
                    ;;
                *)
                    print_info "存储管理命令:"
                    print_info "  storage setup        - 设置共享存储目录（head节点）"
                    print_info "  storage mount [IP]   - 挂载共享存储（worker节点）"
                    print_info "  storage test         - 测试共享存储访问"
                    ;;
            esac
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
