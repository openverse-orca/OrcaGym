# 🎬 Scene API

场景管理接口，用于管理仿真场景中的 Actor、光源、相机、材质等元素。

## 主要类

| 类 | 说明 |
|----|------|
| `OrcaGymScene` | 完整的场景管理器 |
| `OrcaGymSceneRuntime` | 运行时安全封装 |

## 架构说明

- **OrcaGymScene**：完整场景控制能力，包含添加 Actor、发布场景等操作。适合场景初始化阶段。
- **OrcaGymSceneRuntime**：运行时安全封装。仅允许安全操作（调整光源、切换相机、动画参数等）。

---

## OrcaGymScene

### 构造

```python
class OrcaGymScene:
    def __init__(self, grpc_addr: str)
```

### 场景发布

```python
def publish_scene()
```
发布当前场景配置。在所有 `add_actor()` 调用完成后调用。

> ⚠️ **注意**：调用后仿真场景会被重新加载，仿真对象会被重建。

### Actor 管理

```python
def add_actor(actor: Actor)
```
添加一个 Actor 到场景。需在 `publish_scene()` 前调用。

### 光源

```python
def set_light_info(actor_name: str, light_info: LightInfo)
```

### 相机传感器

```python
def set_camera_sensor_info(actor_name: str, camera_sensor_info: CameraSensorInfo)
def make_camera_viewport_active(actor_name: str, entity_name: str)
```

### 材质

```python
def set_material_info(actor_name: str, material_info: MaterialInfo)
```

### 动画参数

```python
def set_actor_anim_param_number(actor_name: str, param_name: str, value: float)
def set_actor_anim_param_bool(actor_name: str, param_name: str, value: bool)
def set_actor_anim_param_string(actor_name: str, param_name: str, value: str)
```

### UI 文本

```python
def set_ui_text(
    self,
    actor_name: int,          # 1-6，对应不同 UI 位置
    message: str = "",        # 显示的文本内容
    showtime: int = 0,        # 显示时长（int，单位由 Studio 解释）
    blinkfreq: int = 0,       # 闪烁频率
    color: str = "",          # 颜色（如 "0x00ff00"）
    size: int = 0,            # 字号
)
```

actor_name 映射表：

| actor_name | 对应 UI 元素 |
|------------|-------------|
| 1 | SimMessText |
| 2 | SimTipText |
| 3 | SimUpleftText |
| 4 | SimUprightText |
| 5 | SimBottomleft |
| 6 | SimBottomrightText |

### 图片显示控制

```python
def set_image_enabled(actor_name: int, enabled: bool)
```

### 生命周期

```python
def close()
```

---

## OrcaGymSceneRuntime

运行时安全封装，仅允许不影响仿真对象的操作。

```python
class OrcaGymSceneRuntime:
    def __init__(self, scene: OrcaGymScene)
    def set_light_info(actor_name: str, light_info: LightInfo)
    def make_camera_viewport_active(actor_name: str, entity_name: str)
    def set_actor_anim_param_number(actor_name: str, param_name: str, value: float)
    def set_actor_anim_param_bool(actor_name: str, param_name: str, value: bool)
    def set_actor_anim_param_string(actor_name: str, param_name: str, value: str)
```

---

## 场景元素类型

### Actor

```python
class Actor:
    def __init__(
        self,
        name: str,              # Actor 名称（场景内唯一）
        asset_path: str,        # Spawnable 资源路径
        position: np.ndarray,   # 初始位置 [x, y, z]
        rotation: np.ndarray,   # 初始旋转，四元数 [w, x, y, z]
        scale: float,           # 缩放系数
    )
```

### LightInfo

```python
class LightInfo:
    def __init__(
        self,
        color: np.ndarray,      # 光源颜色 [r, g, b]
        intensity: float,       # 光源强度
    )
```

### CameraSensorInfo

基础 4 参数为必填；扩展参数为 optional（None 表示不修改 server 现有值），
对应 proto3 optional 语义，兼容老客户端。

```python
class CameraSensorInfo:
    def __init__(
        self,
        capture_rgb: bool,      # 是否捕获 RGB 图像
        capture_depth: bool,    # 是否捕获深度图
        save_mp4_file: bool,    # 是否保存 MP4 视频文件
        use_dds: bool,          # 是否使用 DDS 纹理格式
        **kwargs,               # 扩展 optional 字段（None 表示不修改）
    )
```

`**kwargs` 支持的扩展字段（共 16 个，对应 proto optional 语义）：
`capture_normal`、`capture_object_color`、`is_recording`、`use_nvenc`、
`nvenc_gpu_index`、`random_object_color`、`width`、`height`、`vertical_fov`、
`near_clip`、`far_clip`、`gamma`、`color_port`、`depth_port`、`dds_topic`、
`dds_stream_id`。

### MaterialInfo

```python
class MaterialInfo:
    def __init__(
        self,
        base_color: np.ndarray,  # 基础颜色 [r, g, b, a]
    )
```

---

## 使用示例

### 场景初始化

```python
from orca_gym.scene.orca_gym_scene import OrcaGymScene, Actor, LightInfo, CameraSensorInfo, MaterialInfo
import numpy as np

# 1. 连接场景
scene = OrcaGymScene("localhost:50051")

# 2. 添加物体
table = Actor(
    name="table",
    asset_path="/props/table_1",
    position=np.array([0.5, 0.0, 0.0]),
    rotation=np.array([1.0, 0.0, 0.0, 0.0]),
    scale=1.0,
)
scene.add_actor(table)

# 3. 发布场景
scene.publish_scene()

# 4. 配置光源（发布后依然可以设置）
scene.set_light_info("main_light", LightInfo(
    color=np.array([1.0, 1.0, 1.0]),
    intensity=2.5,
))

# 5. 清理
scene.close()
```

### 运行时操作（通过 Runtime 封装）

```python
from orca_gym.scene.orca_gym_scene_runtime import OrcaGymSceneRuntime
from orca_gym.scene.orca_gym_scene import LightInfo

runtime = OrcaGymSceneRuntime(scene)

# 运行时调整光源
runtime.set_light_info("main_light", LightInfo(
    color=np.array([0.8, 0.8, 1.0]),
    intensity=1.5,
))

# 运行时切换相机视口
runtime.make_camera_viewport_active("camera_2", "viewport_1")
```
