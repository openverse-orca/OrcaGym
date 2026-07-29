# 🎬 Scene API

Scene management interface for managing Actors, lights, cameras, materials, and other elements in the simulation scene.

## Main Classes

| Class | Description |
|----|------|
| `OrcaGymScene` | Complete scene manager |
| `OrcaGymSceneRuntime` | Runtime-safe wrapper |

## Architecture

- **OrcaGymScene**: Full scene control capability, including adding Actors, publishing scenes, etc. Suitable for the scene initialization phase.
- **OrcaGymSceneRuntime**: Runtime-safe wrapper. Only allows safe operations (light adjustment, camera switching, animation parameters, etc.).

---

## OrcaGymScene

### Constructor

```python
class OrcaGymScene:
    def __init__(self, grpc_addr: str)
```

### Scene Publishing

```python
def publish_scene()
```
Publishes the current scene configuration. Call after all `add_actor()` calls are complete.

> ⚠️ **Note**: After calling, the simulation scene will be reloaded and simulation objects will be reconstructed.

### Actor Management

```python
def add_actor(actor: Actor)
```
Adds an Actor to the scene. Must be called before `publish_scene()`.

### Lights

```python
def set_light_info(actor_name: str, light_info: LightInfo)
```

### Camera Sensors

```python
def set_camera_sensor_info(actor_name: str, camera_sensor_info: CameraSensorInfo)
def make_camera_viewport_active(actor_name: str, entity_name: str)
```

### Materials

```python
def set_material_info(actor_name: str, material_info: MaterialInfo)
```

### Animation Parameters

```python
def set_actor_anim_param_number(actor_name: str, param_name: str, value: float)
def set_actor_anim_param_bool(actor_name: str, param_name: str, value: bool)
def set_actor_anim_param_string(actor_name: str, param_name: str, value: str)
```

### UI Text

```python
def set_ui_text(
    self,
    actor_name: int,          # 1-6, corresponding to different UI positions
    message: str = "",        # Text content to display
    showtime: int = 0,        # Display duration (int, unit interpreted by Studio)
    blinkfreq: int = 0,       # Blink frequency
    color: str = "",          # Color (e.g. "0x00ff00")
    size: int = 0,            # Font size
)
```

actor_name mapping table:

| actor_name | Corresponding UI Element |
|------------|-------------|
| 1 | SimMessText |
| 2 | SimTipText |
| 3 | SimUpleftText |
| 4 | SimUprightText |
| 5 | SimBottomleft |
| 6 | SimBottomrightText |

### Image Display Control

```python
def set_image_enabled(actor_name: int, enabled: bool)
```

actor_name mapping table:

| actor_name | Corresponding UI Element |
|------------|-------------|
| 0 | Imagemidlebig |
| 1 | Imagetoplit |

### Lifecycle

```python
def close()
```

---

## OrcaGymSceneRuntime

Runtime-safe wrapper, only allows operations that do not affect simulation objects.

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

## Scene Element Types

### Actor

```python
class Actor:
    def __init__(
        self,
        name: str,              # Actor name (unique within the scene)
        asset_path: str,        # Spawnable asset path
        position: np.ndarray,   # Initial position [x, y, z]
        rotation: np.ndarray,   # Initial rotation, quaternion [w, x, y, z]
        scale: float,           # Scale factor
    )
```

### LightInfo

```python
class LightInfo:
    def __init__(
        self,
        color: np.ndarray,      # Light color [r, g, b]
        intensity: float,       # Light intensity
    )
```

### CameraSensorInfo

The first 4 parameters are required; extension parameters are optional (None means do not modify the server's existing value),
corresponding to proto3 optional semantics, compatible with old clients.

```python
class CameraSensorInfo:
    def __init__(
        self,
        capture_rgb: bool,      # Whether to capture RGB images
        capture_depth: bool,    # Whether to capture depth maps
        save_mp4_file: bool,    # Whether to save MP4 video files
        use_dds: bool,          # Whether to use DDS texture format
        **kwargs,               # Extension optional fields (None means do not modify)
    )
```

Extension fields supported by `**kwargs` (16 in total, corresponding to proto optional semantics):
`capture_normal`, `capture_object_color`, `is_recording`, `use_nvenc`,
`nvenc_gpu_index`, `random_object_color`, `width`, `height`, `vertical_fov`,
`near_clip`, `far_clip`, `gamma`, `color_port`, `depth_port`, `dds_topic`,
`dds_stream_id`.

### MaterialInfo

```python
class MaterialInfo:
    def __init__(
        self,
        base_color: np.ndarray,  # Base color [r, g, b, a]
    )
```

---

## Usage Examples

### Scene Initialization

```python
from orca_gym.scene.orca_gym_scene import OrcaGymScene, Actor, LightInfo, CameraSensorInfo, MaterialInfo
import numpy as np

# 1. Connect to the scene
scene = OrcaGymScene("localhost:50051")

# 2. Add objects
table = Actor(
    name="table",
    asset_path="/props/table_1",
    position=np.array([0.5, 0.0, 0.0]),
    rotation=np.array([1.0, 0.0, 0.0, 0.0]),
    scale=1.0,
)
scene.add_actor(table)

# 3. Publish the scene
scene.publish_scene()

# 4. Configure lights (can still be set after publishing)
scene.set_light_info("main_light", LightInfo(
    color=np.array([1.0, 1.0, 1.0]),
    intensity=2.5,
))

# 5. Cleanup
scene.close()
```

### Runtime Operations (via Runtime Wrapper)

```python
from orca_gym.scene.orca_gym_scene_runtime import OrcaGymSceneRuntime
from orca_gym.scene.orca_gym_scene import LightInfo

runtime = OrcaGymSceneRuntime(scene)

# Adjust lights at runtime
runtime.set_light_info("main_light", LightInfo(
    color=np.array([0.8, 0.8, 1.0]),
    intensity=1.5,
))

# Switch camera viewport at runtime
runtime.make_camera_viewport_active("camera_2", "viewport_1")
```
