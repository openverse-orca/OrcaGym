# 🏔️ 地形生成

OrcaGym 提供地形生成工具，用于创建复杂的地面环境。

## 地形生成器

```python
from orca_gym.tools.terrains import terrain_generater

# 高度图生成器
from orca_gym.tools.terrains import height_map_generater
```

## 地形类型

| 类型 | 说明 |
|------|------|
| 平面 | 平坦地面 |
| 斜坡 | 有坡度的地面 |
| 阶梯 | 离散高度变化 |
| 崎岖 | 随机起伏地面 |
| 障碍物 | 散落的障碍物 |

## 使用地形

生成的地形以 height field (hfield) 形式嵌入 MuJoCo 场景：

1. 通过工具生成高度图
2. 导出为 MuJoCo hfield 支持的格式
3. 在模型 XML 中引用
4. 加载时自动下载和缓存
