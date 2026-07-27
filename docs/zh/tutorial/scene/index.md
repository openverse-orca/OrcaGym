# 🎬 场景管理

OrcaGym 的场景系统提供了对仿真场景的运行时控制。

> 场景加载在 Euler 环境中通过 `model_xml_path` 参数自动完成，详见 [🔧 MuJoCo 后端](../physics/mujoco-backend.md)。

## 场景对象

场景由以下元素构成：

- **Actor** — 场景中的角色/物体
- **Light** — 光源
- **Camera** — 相机（含 RGB-D 传感器信息）
- **Material** — 材质

```python
from orca_scene import OrcaGymScene, OrcaGymSceneRuntime
```

## 章节导航

- [🏞️ 场景加载](scene-loading.md) — XML 加载流程、资源缓存
- [🎭 物体操作](actor-manipulation.md) — Mocap 锚定、等式约束
- [🏔️ 地形生成](terrain-generation.md) — 高度图地形工具
- [🎨 资源与渲染](assets-rendering.md) — 3D 资源格式、渲染配置
