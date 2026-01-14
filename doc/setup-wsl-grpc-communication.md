# 配置GRPC访问

需确保 Windows 主机上的 gRPC 服务可被 WSL2 中的 Python 客户端访问。配置步骤如下：

## 确保gRPC服务监听所有IP地址

在 gRPC 服务端代码中，应将服务绑定到 0.0.0.0，使其监听所有可用网络接口，而不仅是本地回环接口（127.0.0.1）。在 OrcaStudio 配置 GrpcServer Component 时也需注意。如图所示：<img src="images/MujocoGrpcServerConfig.png" alt="GrpcServer Component" width="300">

## 获取Windows主机的IP地址

需获取 Windows 主机在 WSL2 中的网络适配器 IP 地址。通常为类似 `172.x.x.x` 的地址。获取步骤如下：

* 打开Windows命令提示符（CMD）或PowerShell，运行ipconfig命令。
* 找到与WSL2通信的适配器的IPv4地址（通常是一个虚拟网络适配器）。

## 配置防火墙

需确保 Windows 防火墙允许 gRPC 服务端口（例如 50051）被外部访问。可按以下方式配置：

1. 打开Windows防火墙设置。
2. 创建一个新的入站规则，允许指定端口的TCP流量。
3. 选择“允许连接”。
4. 选择适当的网络类型（私人、公共、域），然后保存。

## 配置Python客户端

在 WSL2 中，Python 客户端需连接到 Windows 主机的 IP 地址，而非 localhost。例如：

```python
import grpc

# 这里替换成 Windows 主机的实际 IP 地址
channel = grpc.insecure_channel('172.x.x.x:50051')
stub = YourGrpcServiceStub(channel)
```

## 测试连接

最后运行 gRPC 服务与客户端，验证可成功通信。

## 可能遇到的问题

* **网络隔离：** 某些Windows配置可能会阻止WSL2和Windows主机之间的通信。如果遇到连接问题，检查防火墙、网络适配器配置。
* **端口占用：** 确保Windows主机上的端口没有被其他应用程序占用，否则可能导致gRPC服务无法启动或客户端无法连接。