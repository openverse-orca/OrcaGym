# 🎨 资源处理

资源处理工具用于管理 3D 资产。

## USDZ → XML 转换

```python
from orca_tools.assets import usdz_to_xml
```

将 Apple USDZ 格式的 3D 资产转换为 MuJoCo MJCF (XML) 格式。

## 纹理处理

```python
from orca_tools.assets import texture_processer
```

处理纹理文件，包括格式转换、分辨率调整等。

## 安装助手

```python
from orca_tools.install_helpers import down_projects
```

帮助下载和管理依赖项目。
