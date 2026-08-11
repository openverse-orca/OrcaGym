# 🎬 Scene API

Scene management interface for managing Actors, lights, cameras, materials, and other elements in the simulation scene.

## Main Classes

| Class | Description |
|----|------|
| `OrcaGymScene` | Complete scene manager |
| `OrcaGymSceneRuntime` | Runtime-safe wrapper |

## Architecture

- **OrcaGymScene**: Full scene control capability, including adding Actors, publishing the scene, etc. Suitable for the scene initialization phase.
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
Publishes the current scene configuration. Call this after all `add_actor()` calls are complete.

> ⚠️ **Note**: After calling this, the simulation scene will be reloaded and simulation objects will be reconstructed.

### Actor Management

```python
def add_actor(actor: Actor)
```
Adds an Actor to the scene. Must be called before `publish_scene()`.

### Lights

```python
def set_light_info(actor_name: str, light_info: LightInfo)
```

### Camera Property Query/Set + Streaming State Machine

```python
def get_camera_names() -> list[str]
def get_camera_properties(camera_name: str) -> GetCameraPropertiesResponse
def set_camera_properties(camera_name: str, camera_property: CameraProperty)
def set_streaming_enabled(camera_name: str, enabled: bool)
def make_camera_viewport_active(actor_name: str, entity_name: str)
```

State machine constraints:
- `Idle --set_streaming_enabled(True)--> Streaming` (Studio side InitCameraSensor, ports start streaming)
- `Streaming --set_streaming_enabled(False)--> Idle` (Studio side UninitCameraSensor, stop streaming)
- `set_camera_properties` is only allowed in `Idle` state; in `Streaming` state, stop streaming first before setting properties
- `camera_name` can be enumerated via `get_camera_names()`
- MP4 recording is controlled by the environment-layer `save_streaming(camera_name, camera_type, file_path, start_simulate_index, end_simulate_index)` (client-side PyAV remux, non-blocking, returns a `Future`), orthogonal to this group of interfaces

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

`actor_name` mapping table:

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

`actor_name` mapping table:

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

### CameraProperty

Camera property batch update parameters. All fields are optional (None means do not modify the server's existing value),
corresponding to proto3 optional semantics. State machine constraint: property Set is only allowed in Idle state.

```python
class CameraProperty:
    def __init__(
        self,
        **kwargs,               # All fields optional (None means do not modify)
    )
```

Fields supported by `**kwargs` (18 in total, corresponding to proto `CameraProperty` optional semantics):
`capture_rgb`, `capture_depth`, `capture_normal`, `capture_object_color`,
`random_object_color`, `use_nvenc`, `nvenc_gpu_index`, `width`, `height`,
`vertical_fov`, `near_clip`, `far_clip`, `gamma`, `color_port`, `depth_port`,
`use_dds`, `dds_topic`, `dds_stream_id`.

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
from orca_gym.scene.orca_gym_scene import OrcaGymScene, Actor, LightInfo, CameraProperty, MaterialInfo
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

# 5. Clean up
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

# Switch the camera viewport at runtime
runtime.make_camera_viewport_active("camera_2", "viewport_1")
```
