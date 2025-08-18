# 数据采集设备客户端

## 📖 概述

这是运行在每个数据采集设备上的客户端程序，负责监控本地目录文件变化并上报数据到中央服务器。

## 🏗️ 架构说明

```
设备客户端 -> 中央服务器 -> 前端界面
     ↑           ↑            ↑
   监控文件    汇总数据    实时显示
```

## 📋 功能特性

- **实时文件监控**: 使用watchdog监控目录文件变化
- **定时数据上报**: 定期统计并上报文件数量、大小等信息
- **心跳检测**: 定时发送心跳保持与服务器连接
- **自动注册**: 启动时自动向服务器注册设备信息
- **错误恢复**: 网络断开时自动重试连接

## 🚀 安装部署

### 1. 系统要求
- Python 3.7+
- 网络连接到中央服务器
- 对监控目录的读权限

### 2. 安装步骤

```bash
# 1. 下载客户端代码到设备
scp -r device_client/ user@device:/opt/data_collector/

# 2. 登录到设备
ssh user@device

# 3. 进入客户端目录
cd /opt/data_collector/device_client

# 4. 运行安装脚本
chmod +x install.sh
./install.sh

# 5. 编辑配置文件
nano client_config.json
```

### 3. 配置文件说明

```json
{
  "device_id": "device-001",                    // 设备唯一标识
  "server_url": "http://192.168.1.100:8000",   // 中央服务器地址
  "monitor_directory": "/data/device001",       // 监控目录路径
  "report_interval": 30,                        // 数据上报间隔（秒）
  "heartbeat_interval": 60,                     // 心跳间隔（秒）
  "file_extensions": [".txt", ".csv", ".json"]  // 监控的文件类型
}
```

### 4. 启动客户端

```bash
# 前台启动（调试用）
#./start.sh
在openpi 目录下 执行
python ./device_client/client.py --configfile ./device_client/client_config.json

# 后台启动
nohup ./start.sh > client.log 2>&1 &s

# 使用systemd管理（推荐）
sudo cp data-collector-client.service /etc/systemd/system/
sudo systemctl enable data-collector-client
sudo systemctl start data-collector-client
```

## 🔧 配置说明

### 设备ID规则
- 格式: `device-xxx`
- 必须在服务器端先添加设备记录
- 建议使用有意义的编号，如 `device-workshop-001`

### 监控目录
- 支持监控多级子目录
- 需要确保目录存在且有读取权限
- 建议使用绝对路径

### 网络配置
- 确保能访问服务器的 HTTP API 端口（默认8000）
- 防火墙需要允许出站HTTP连接

## 📊 运行状态

### 日志输出
```
📱 设备客户端初始化完成
🆔 设备ID: device-001
🌐 服务器地址: http://192.168.1.100:8000
📁 监控目录: /data/device001
✅ 设备注册成功: 设备注册成功
👁️  开始监控目录: /data/device001
⏰ 开始定时上报，间隔 30 秒
💓 开始心跳发送，间隔 60 秒
📊 数据上报成功: 文件1245个, 新增23个, 大小156.78MB
```

### 状态检查
```bash
# 查看进程状态
ps aux | grep client.py

# 查看日志
tail -f client.log

# 查看网络连接
netstat -anp | grep 8000
```

## 🚨 故障排查

### 常见问题

1. **连接服务器失败**
   - 检查服务器地址配置
   - 确认网络连通性: `ping server_ip`
   - 检查防火墙设置

2. **设备注册失败**
   - 确认设备ID在服务器端已添加
   - 检查API接口是否正常工作

3. **目录监控异常**
   - 确认监控目录存在: `ls -la /data/device001`
   - 检查目录权限: `stat /data/device001`

4. **文件数据上报异常**
   - 查看客户端日志文件
   - 检查磁盘空间: `df -h`

### 调试模式
```bash
# 启用详细日志
export PYTHONUNBUFFERED=1
python3 client.py client_config.json
```

## 🔄 更新升级

```bash
# 停止客户端
pkill -f client.py

# 更新代码
git pull origin main

# 重启客户端
./start.sh
```

## 📞 技术支持

如遇到问题，请提供以下信息：
- 设备ID和配置文件
- 客户端日志文件
- 服务器端相关日志
- 网络配置信息
