# Development Guide

本指南面向首次接入 OrcaGym 开发的用户，指导完成本地环境配置与 GPU 旁路设置。

`AGENTS.md` 描述的是 **AI agent 在 TRAE sandbox 下运行 CUDA 任务时所依赖的旁路机制**——它是一份机制说明文档，不是配置教程。该机制依赖一项 **用户级配置（TRAE 命令白名单）**，此配置存储在用户 home 目录下，**不随仓库分发**。因此新用户 clone 仓库后，必须手动完成一次本指南所述的配置，AI agent 才能正常调用 GPU。

## 路径占位符

本指南使用以下占位符，请按本机实际路径替换：

| 占位符 | 含义 | 获取方式 |
|--------|------|---------|
| `<conda-base>` | conda 安装根目录 | `conda info --base`（如 `~/miniconda3`） |
| `<repo-root>` | OrcaGym 仓库根目录 | clone 后的路径（如 `~/repo/OrcaGym`） |

## 前置条件

- **CUDA 驱动**：已安装 NVIDIA 驱动，`nvidia-smi` 可正常输出
- **conda**：已安装 Anaconda 或 Miniconda
- **TRAE IDE**：已安装 TRAE CN 版（sandbox 旁路依赖此 IDE 的白名单功能）
- **Euler GPU 运行时**：`orca` 环境中已安装 Flow / Warp 等原生 CUDA 库（详见 README）

## 步骤 1：创建 orca conda 环境

OrcaGym 使用单一 `orca` conda 环境（README 推荐的环境名称）：

```bash
conda create -n orca python=3.12 -y
conda activate orca
```

按 README 指引安装本项目依赖。若需 GPU 加速（Euler 体系使用 MuJoCoFlow / Flow 在 GPU 上求解），还需确保 `orca` 环境内已安装 Flow / Warp 的原生 CUDA 库。

## 步骤 2：配置国内镜像源（国内网络必需）

国内网络环境下，pip 与 conda 默认从官方源拉取包，几乎无法使用。在创建环境前先配置国内镜像源（以清华 TUNA 为例，阿里云源亦可）。

### 配置 pip 镜像

```bash
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
```

该命令写入 `~/.config/pip/pip.conf`（或 `~/.pip/pip.conf`），对所有 conda 环境生效。若仅对单次安装临时使用，可加 `-i` 参数：

```bash
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple <package>
```

### 配置 conda 镜像

编辑 `~/.condarc`（不存在则新建），写入以下内容：

```yaml
channels:
  - defaults
show_channel_urls: true
default_channels:
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/r
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/msys2
custom_channels:
  conda-forge: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  nvidia: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
```

配置后执行 `conda clean -i` 刷新索引缓存，再进行后续 `conda create`。

> **注意**：镜像源 URL 偶有调整，若拉取失败请参考 [TUNA Anaconda 镜像帮助](https://mirrors.tuna.tsinghua.edu.cn/help/anaconda/) 与 [TUNA PyPI 镜像帮助](https://mirrors.tuna.tsinghua.edu.cn/help/pypi/) 获取最新配置。

## 步骤 3：配置 TRAE 命令白名单（关键步骤）

> **此步骤不可省略。** 白名单存储在用户级 IDE 设置中，不随仓库分发。未配置时，AI agent 执行的所有命令都会被 `trae-sandbox` 包裹，CUDA 因 capability 被剥离而报 `CUDA_ERROR_304`，无法使用 GPU。

### 背景：Sandbox 限制

AI agent 运行在 TRAE sandbox（基于 Bubblewrap + Linux user namespaces）内。该 sandbox 剥离了 **所有进程能力**（`CapEff: 0000000000000000`），导致 CUDA 驱动无法完成 `cuInit`——返回 `CUDA_ERROR_304`（"OS call failed"）。`/dev/nvidia*` 设备节点在 sandbox 内可见，但对其的 ioctl 操作被 namespace 拒绝（`read /dev/nvidia0` 返回 `EINVAL`，而非 `EACCES`）。这是 **能力/namespace 限制，不是文件系统权限问题**——配置 `sandbox.json` 授予 `/usr/local/cuda` 或 `/dev` 访问权限 **无效**。

| 症状 | 原因 |
|------|------|
| `cuInit` 返回 304 | 所有能力被剥离（`CapEff=0`）；驱动无法执行特权 ioctl |
| `head /dev/nvidia0` → `EINVAL` | 设备节点可见，但 namespace 阻止字符设备 ioctl |
| `nvidia-smi` 正常 | NVML 不需要 `cuInit`；它直接读取 sysfs/PCI 信息 |
| `sandbox.json` 文件系统授权无效 | 问题是能力，不是文件权限 |

### 唯一可靠的旁路方法：TRAE 白名单

TRAE IDE 支持 **命令白名单**，可完全跳过 sandbox 包裹。当命令前缀匹配白名单中的条目时，IDE **直接在宿主上执行**该命令（具有完整能力），而不是用 `trae-sandbox` 包裹。这是 **唯一可靠的旁路**——命令以宿主能力运行，`cuInit` 成功。

白名单存储在 TRAE 用户设置文件中：

**文件**：`~/.config/Trae CN/User/settings.json`

将 `orca` 环境的 Python 解释器路径加入 `allowList`（替换 `<conda-base>` 为本机实际路径）：

```json
{
  "AI.toolcall.v2.ide.command.mode": "whitelist",
  "AI.toolcall.v2.command.allowList": "[\"<conda-base>/envs/orca/bin/python\",\"uv run\",\"nvidia-smi\",\"nvcc\",\"cd\"]"
}
```

- `AI.toolcall.v2.ide.command.mode: "whitelist"` — 启用白名单模式（列表内命令直接在宿主执行，列表外命令被 sandbox 包裹）
- `AI.toolcall.v2.command.allowList` — JSON 编码的命令前缀数组

配置后 **重启 TRAE IDE** 以确保设置生效。

## 步骤 4：验证 GPU 旁路生效

在 TRAE IDE 的 AI agent 终端中执行（注意：不要用管道 `|`，会触发 sandbox 包裹）：

```bash
<conda-base>/envs/orca/bin/python -c "import orca.flow as flow; print(flow.get_devices())"
```

预期输出（旁路生效）：

```
devices: ['cuda:0']
```

若输出 `devices: []` 且日志出现 `CUDA driver not available`，说明白名单未生效，请回到步骤 3 检查。

## 常见问题

### Q: 白名单配置能否随仓库分发？

不能。`~/.config/Trae CN/User/settings.json` 是 TRAE IDE 的用户级设置，属于用户 home 目录，不归仓库管辖。每个用户必须在本机配置一次。本指南即为此而存在。

### Q: 配置完成后 AI agent 是否完全零干预？

基本是。AI agent 读取 `AGENTS.md` 后会理解占位符 `<conda-base>` 的含义，并可通过 `conda info --base`（CPU 命令，sandbox 内可执行）自动解析为实际路径。用户只需完成本指南的步骤 1–3 一次，之后 AI agent 即可自主工作。

### Q: 为什么 AI agent 运行的 GPU 命令仍然报 `CUDA_ERROR_304`？

最常见的原因是命令中包含了 shell 管道 `|` 或重定向 `2>&1 | ...`。这些构造会触发 IDE 用 `trae-sandbox` 包裹命令，重新引入能力限制。AI agent 应直接以白名单解释器路径开头运行命令，避免管道；如需捕获输出，应重定向到文件（`> /tmp/out.log 2>&1`），再单独读取文件。详见 `AGENTS.md` 中的 GPU 旁路规则。
