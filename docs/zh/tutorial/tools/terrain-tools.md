# 🗺️ 地形工具

OrcaGym 提供地形生成工具用于创建复杂的训练环境。

## terrain_generater

生成由 geom 体素（box/sphere/cylinder 等）组成的随机地形，输出 MuJoCo XML：

```python
from orca_gym.tools.terrains import terrain_generater

# 核心函数签名
terrain_generater.generate_geom_terrain(
    num_x, num_y,                    # 网格规模
    geom_type,                        # geom 原语类型：box/sphere/ellipsoid/cylinder/capsule
    geom_size,                        # geom 尺寸
    geom_size_cale_range,             # 尺寸缩放范围
    max_tilt,                         # 最大倾斜角度
    min_step, max_step,               # 相邻 geom 高度差范围
    max_total_height,                 # 最大地形高度
    min_spacing, max_spacing,         # 相邻 geom 间距范围
    rotation_z_min, rotation_z_max,   # Z 轴旋转范围
)
```

也可通过命令行调用：

```bash
python -m orca_gym.tools.terrains.terrain_generater \
    --num_x 10 --num_y 10 --geom_type box \
    --max_tilt 0.3 --min_step 0.2 --max_step 0.5
```

## height_map_generater

通过物理碰撞检测生成当前场景的高程图（`HeightMapGenerater` 类，继承 `OrcaGymLocalEnv`）：

```python
from orca_gym.tools.terrains import height_map_generater
# HeightMapGenerater 通过 env 接口运行仿真并采样高度
```

## 工作流

1. 使用 `terrain_generater` 生成随机地形 XML（或 `height_map_generater` 生成高程图）
2. 将地形 XML 嵌入场景模型
3. OrcaGym 在模型加载时自动处理 mesh/hfield 资源
4. 创建环境进行训练
