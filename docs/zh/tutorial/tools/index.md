# 🔧 工具与适配器

OrcaGym 提供了丰富的工具集和主流 RL/IL 框架的适配器。

## 工具集

| 工具 | 位置 | 说明 |
|------|------|------|
| 地形生成 | `tools/terrains/` | 生成高度图地形 |
| 资源处理 | `tools/assets/` | USDZ→XML、纹理处理 |
| HDF5 查看器 | `tools/hdf5_viewer/` | 数据集可视化 |
| 安装助手 | `tools/install_helpers/` | 项目下载与配置 |

## 适配器

| 适配器 | 位置 | 说明 |
|--------|------|------|
| RLlib | `adapters/rllib/` | PPO/APPO 训练 |
| Robomimic | `adapters/robomimic/` | 模仿学习数据集 |
| Robosuite | `adapters/robosuite/` | 操控任务环境 |
| 输入设备 | `devices/` | Xbox/键盘/手柄 |

## 章节导航

- [🗺️ 地形工具](terrain-tools.md)
- [🎨 资源处理](asset-processing.md)
- [📊 HDF5 查看器](hdf5-viewer.md)
- [🎮 输入设备](input-devices.md)
- [🤖 RLlib 适配器](rllib-adapter.md)
- [🧠 Robomimic 适配器](robomimic-adapter.md)
