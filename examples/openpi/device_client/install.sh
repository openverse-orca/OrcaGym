#!/bin/bash
# 设备客户端安装脚本

echo "🚀 开始安装设备客户端..."

# 检查Python环境
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 未安装，请先安装 Python3"
    exit 1
fi

# 安装依赖
echo "📦 安装Python依赖..."
pip3 install -r requirements.txt

# 设置权限
chmod +x client.py
chmod +x start.sh
chmod +x stop.sh

# 创建数据目录
echo "📁 创建数据目录..."
if [ ! -d "/data" ]; then
    sudo mkdir -p /data
    sudo chown $USER:$USER /data
fi

echo "✅ 设备客户端安装完成"
echo "📝 请编辑 client_config.json 配置文件"
echo "🚀 运行 ./start.sh 启动客户端"
