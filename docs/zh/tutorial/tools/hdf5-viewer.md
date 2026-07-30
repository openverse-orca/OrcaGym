# 📊 HDF5 查看器

用于查看从 OrcaManipulation 采集的数据集。

## HDF5Viewer

```python
from orca_gym.tools.hdf5_viewer import hdf5Viewer
```

## VideoPlayer

```python
from orca_gym.tools.hdf5_viewer import videoPlayer
```

## 使用

```bash
# 启动 HDF5 数据集查看器
python -m orca_gym.tools.hdf5_viewer.hdf5Viewer
```

HDF5 查看器用于可视化：

- 机器人关节轨迹
- 相机图像序列
- 传感器数据曲线
- 末端执行器位姿轨迹
