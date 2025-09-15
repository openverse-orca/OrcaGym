# 地形工具集 (Terrains Tools)

本目录包含用于生成和处理地形的工具，主要用于机器人仿真环境中的地形创建和高程图生成。

## 工具列表

### 1. 高程图生成器 (Height Map Generator)

**文件**: `height_map_generater.py`

通过物理碰撞检测生成当前环境的高程图，分辨率为0.1m。

#### 功能特点
- 使用射线投射技术检测地形高度
- 支持自定义高程图边界和高度范围
- 生成高精度和高分辨率的高程图
- 适用于legged_gym等机器人仿真任务

#### 使用方法

```bash
# 基本用法
python height_map_generater.py --orcagym_addresses localhost:50051

# 自定义参数
python height_map_generater.py \
    --orcagym_addresses localhost:50051 \
    --height_map_border -50 -50 50 50 \
    --height_range 0 20 \
    --render_mode none \
    --output_file my_height_map.npy
```

#### 参数说明
- `--orcagym_addresses`: OrcaGym服务器地址
- `--height_map_border`: 高程图边界坐标 (左-上-右-下)
- `--height_range`: 高度范围 (最小值-最大值)
- `--render_mode`: 渲染模式 (human/none)
- `--output_file`: 输出文件名

#### 使用步骤
1. 首先将关卡中的agents移除，避免高程图包含agents信息
2. 运行脚本生成高程图文件
3. 在legged_gym等任务中使用生成的高程图文件

### 2. 地形生成器 (Terrain Generator)

**文件**: `terrain_generater.py`

生成随机地形的MuJoCo XML文件，支持多种几何形状和参数化配置。

#### 功能特点
- 支持多种几何形状：box、sphere、ellipsoid、cylinder、capsule
- 随机生成地形高度、倾斜角度和旋转
- 可配置几何体大小、间距和颜色
- 输出标准MuJoCo XML格式

#### 使用方法

```bash
# 基本用法 - 生成16x16的box地形
python terrain_generater.py

# 自定义参数 - 生成椭球体地形
python terrain_generater.py \
    --num_x 20 \
    --num_y 20 \
    --geom_type ellipsoid \
    --geom_size "(0.8,0.8,0.15)" \
    --max_tilt 5 \
    --min_step 0.1 \
    --max_step 0.3 \
    --max_total_height 2.0 \
    --output rough_terrain.xml
```

#### 参数说明
- `--num_x/--num_y`: X/Y方向的几何体数量
- `--geom_type`: 几何体类型 (box/sphere/ellipsoid/cylinder/capsule)
- `--geom_size`: 几何体尺寸 "(x,y,z)"
- `--geom_size_cale_range`: 尺寸随机变化范围
- `--max_tilt`: 最大倾斜角度
- `--min_step/--max_step`: 相邻几何体高度差范围
- `--max_total_height`: 地形最大总高度
- `--min_spacing/--max_spacing`: 几何体间距范围
- `--rotation_z_min/--rotation_z_max`: Z轴旋转角度范围
- `--output`: 输出XML文件名

#### 示例配置

**粗糙地形**:
```bash
python terrain_generater.py \
    --geom_type box \
    --geom_size "(1,1,0.2)" \
    --max_tilt 10 \
    --min_step 0.2 \
    --max_step 0.8 \
    --max_total_height 3.0 \
    --output rough_terrain.xml
```

**平滑地形**:
```bash
python terrain_generater.py \
    --geom_type ellipsoid \
    --geom_size "(0.5,0.5,0.1)" \
    --max_tilt 2 \
    --min_step 0.05 \
    --max_step 0.15 \
    --max_total_height 1.0 \
    --output smooth_terrain.xml
```

## 环境要求

- Python 3.7+
- MuJoCo
- OrcaGym环境
- numpy
- gymnasium

## 注意事项

1. 使用高程图生成器前，确保OrcaGym服务器正在运行
2. 地形生成器输出的XML文件可直接用于MuJoCo仿真
3. 建议在生成高程图前移除环境中的动态对象
4. 高程图分辨率固定为0.1m，可根据需要调整边界范围

## 文件说明

- `terrain.xml`: 示例地形文件
- `terrain_brics_rough.xml`: 粗糙砖块地形
- `terrain_ellipsoid_small.xml`: 小椭球地形
- `__init__.py`: Python包初始化文件
