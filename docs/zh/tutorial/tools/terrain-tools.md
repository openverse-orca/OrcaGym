# 🗺️ 地形工具

OrcaGym 提供地形生成工具用于创建复杂的训练环境。

## TerrainGenerator

```python
from orca_tools.terrains import terrain_generater
```

## HeightMapGenerator

```python
from orca_tools.terrains import height_map_generater
```

## 地形类型

| 类型 | 参数 | 示例用途 |
|------|------|----------|
| Flat | — | 基准测试 |
| Slope | angle | 爬坡训练 |
| Steps | step_height, step_count | 楼梯攀爬 |
| Rough | roughness | 崎岖地形适应 |
| Obstacles | obstacle_size, obstacle_count | 避障导航 |

## 工作流

1. 使用地形工具生成高度图
2. 将高度图嵌入 MuJoCo 场景 XML
3. OrcaGym 在模型加载时自动下载 hfield 资源
4. 创建环境进行训练
