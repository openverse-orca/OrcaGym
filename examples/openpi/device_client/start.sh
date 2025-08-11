#!/bin/bash
# 启动设备客户端

echo "🚀 启动数据采集设备客户端..."

# 检查配置文件
if [ ! -f "client_config.json" ]; then
    echo "❌ 配置文件 client_config.json 不存在"
    echo "📝 请先复制并编辑配置文件"
    exit 1
fi

# 启动客户端
python3 client.py client_config.json
