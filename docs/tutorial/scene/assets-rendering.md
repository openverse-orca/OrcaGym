# 🎨 资源与渲染

OrcaGym 通过 OrcaStudio/OrcaLab 实现场景的 3D 渲染。

## 支持的资源

| 资源类型 | 格式 | 说明 |
|----------|------|------|
| 网格 (Mesh) | OBJ, STL | 3D 几何形状 |
| 高度场 (HField) | PNG | 地形高度图 |
| 纹理 | PNG, JPG | 表面纹理 |
| 场景 | MJCF (XML) | MuJoCo 场景描述 |

## 资源处理工具

```python
# USDZ 转 XML
from orca_tools.assets import usdz_to_xml

# 纹理处理
from orca_tools.assets import texture_processer
```

## 资源缓存

资源文件缓存在 `~/.orcagym/tmp/`：

```python
# 缓存目录
# 资源缓存在 ~/.orcagym/tmp/

# 通过文件锁安全访问
# 支持多进程并发下载
```

## 渲染配置

渲染由 OrcaStudio/OrcaLab 服务端控制，包括：

- 光源位置和类型
- 相机视角
- 材质属性
- 阴影设置

Python 侧通过 `render()` 方法触发渲染帧的发送。
