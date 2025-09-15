# Assets 工具集

本目录包含用于处理 3D 资产和纹理的 Python 工具。

## 工具列表

### 1. texture_processer.py - 纹理处理器

用于处理纹理图像的工具，支持通道分离等功能。

#### 功能特性
- **通道分离**：将 RGB 图像分离为单独的 R、G、B 通道图像
- **临时目录管理**：自动创建临时目录存储处理结果

#### 使用方法

```bash
# 激活 conda 环境
conda activate orca

# 分离图像通道
python texture_processer.py --image /path/to/your/image.png --task split_channel
```

#### 参数说明
- `--image`: 输入图像路径（必需）
- `--task`: 处理任务类型，目前支持 `split_channel`（默认）

#### 输出
- 在临时目录中生成三个文件：
  - `原文件名_r.png` - 红色通道
  - `原文件名_g.png` - 绿色通道  
  - `原文件名_b.png` - 蓝色通道

---

### 2. usdz_to_xml.py - USDZ 到 MJCF 转换器

将 USDZ 格式的 3D 模型转换为 MuJoCo 物理仿真所需的 MJCF XML 格式。

#### 功能特性
- **格式转换**：USDZ → USDC → OBJ → MJCF XML
- **网格分割**：使用 VHACD 算法将复杂网格分割为凸包
- **碰撞体生成**：支持包围盒和凸包两种碰撞体类型
- **批量处理**：支持通过配置文件批量处理多个文件
- **智能合并**：自动合并重叠的碰撞体以减少计算开销

#### 安装依赖
```bash
pip install usd-core trimesh
```

#### 使用方法

```bash
# 激活 conda 环境
conda activate orca

# 使用配置文件批量处理
python usdz_to_xml.py --config usdz_to_xml_config_example.yaml
```

#### 配置文件示例

```yaml
# 默认设置
default_settings:
  transform_options:
    scale: 1.0
  
  collision_options:
    collider_type: "bounding_box"  # "bounding_box" 或 "convex_hull"
    split_mesh: true
    max_split_mesh_number: 16
    merge_threshold: 0.5
    merge_iterations: 10
  
  physics_options:
    free_joint: true
    density: 1000.0
    mass: 0.0
  
  debug_options:
    visualize_obj: false

# 文件列表
files:
  - filename: "model1.usdz"
    transform_options:
      scale: 0.1
    physics_options:
      mass: 0.2
```

#### 配置参数说明

**transform_options（变换选项）**
- `scale`: 模型缩放比例

**collision_options（碰撞选项）**
- `collider_type`: 碰撞体类型（"bounding_box" 或 "convex_hull"）
- `split_mesh`: 是否分割网格
- `max_split_mesh_number`: 最大分割数量
- `merge_threshold`: 包围盒合并阈值（0-1）
- `merge_iterations`: 合并迭代次数

**physics_options（物理选项）**
- `free_joint`: 是否添加自由关节
- `density`: 材料密度
- `mass`: 物体质量

**debug_options（调试选项）**
- `visualize_obj`: 是否可视化 OBJ 网格

#### 输出文件
- 在 `converted_files/` 目录下生成：
  - `模型名.xml` - MuJoCo MJCF 文件
  - `模型名.obj` - 导出的网格文件
  - 分割后的网格文件（如果启用分割）

#### 依赖要求
- USD Python 绑定
- trimesh
- PIL
- PyYAML
- TestVHACD 可执行文件（用于网格分割）

---

## 使用建议

1. **纹理处理**：使用 `texture_processer.py` 处理需要分离通道的纹理图像
2. **3D 模型转换**：使用 `usdz_to_xml.py` 将 USDZ 模型转换为 MuJoCo 可用的格式
3. **批量处理**：通过配置文件批量处理多个模型文件
4. **性能优化**：根据模型复杂度调整 `max_split_mesh_number` 和 `merge_threshold` 参数

## 注意事项

- 确保在运行脚本前激活 `orca` conda 环境
- USDZ 文件路径应相对于配置文件位置
- 网格分割需要 TestVHACD 可执行文件支持
- 大型模型处理可能需要较长时间，建议先在小型模型上测试参数
