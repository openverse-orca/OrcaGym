# 🎬 Scene Management

OrcaGym's scene system provides runtime control over the simulation scene.

> Scene loading is handled automatically in the Euler environment via the `model_xml_path` parameter. See [🔧 MuJoCo Backend](../physics/mujoco-backend.md) for details.

## Scene Objects

A scene consists of the following elements:

- **Actor** — Characters/objects in the scene
- **Light** — Light sources
- **Camera** — Cameras (including RGB-D sensor data)
- **Material** — Materials

```python
from orca_gym.scene.orca_gym_scene import OrcaGymScene
from orca_gym.scene.orca_gym_scene_runtime import OrcaGymSceneRuntime
```

## Chapter Navigation

- [🏞️ Scene Loading](scene-loading.md) — XML loading process, resource caching
- [🎭 Actor Manipulation](actor-manipulation.md) — Mocap anchoring, equality constraints
- [🏔️ Terrain Generation](terrain-generation.md) — Height map terrain tools
- [🎨 Assets and Rendering](assets-rendering.md) — 3D asset formats, rendering configuration
