# 🎬 Scene Setup — Putting Things in the Scene

In the previous section, we connected directly to an already-built scene. In this section, you will learn how to **build a scene yourself**: load robots, place objects, and set up lights.

---

## What Is a Scene?

In OrcaGym, a "scene" consists of the following elements:

```
Scene
├── Actor — robots, tables, cubes... any 3D object
├── Light — point lights, directional lights, etc.
├── Camera — rendering viewpoints
└── Material — object color/texture
```

**Key concept**: `OrcaGymScene` is responsible for scene **construction** (adding/removing objects), while `OrcaGymEulerEnv` is responsible for scene **simulation** (physics stepping).

---

## Step 1: Create an Empty Scene

```python
"""
setup_my_scene.py — Build a simple scene: table + robot arm + cube
"""

import numpy as np
from orca_gym.scene.orca_gym_scene import (
    OrcaGymScene,    # scene manager
    Actor,           # objects in the scene
    LightInfo,       # light parameters
    MaterialInfo,    # material parameters
)


def build_scene():
    """Build a scene and publish it to the simulation server."""

    # 1. Create scene manager — connect to simulation server
    print("Connecting to scene manager...")
    scene = OrcaGymScene(grpc_addr="localhost:50051")

    # 2. Publish an empty scene (clear previous content)
    scene.publish_scene()
    print("✅ Empty scene published")

    # ============================================================
    # 3. Add Actors (characters/objects)
    # ============================================================

    # --- 3a. Add a table ---
    # An Actor is defined by four elements: name, asset path, position, rotation, scale
    table = Actor(
        name="table_1",                        # unique name in the scene
        asset_path="assets/tables/table_80x80", # asset path on the server
        position=np.array([0.5, 0.0, 0.0]),     # [x, y, z] world coordinates (meters)
        rotation=np.array([1.0, 0.0, 0.0, 0.0]), # quaternion [w, x, y, z]
        scale=1.0,                               # scale factor
    )
    scene.add_actor(table)
    print(f"✅ Added: {table.name}")

    # --- 3b. Add a robot arm ---
    robot = Actor(
        name="robot_arm",
        asset_path="robots/franka_panda/panda_arm",  # Franka robot arm
        position=np.array([0.0, 0.0, 0.8]),           # place above the table
        rotation=np.array([1.0, 0.0, 0.0, 0.0]),      # unit quaternion = no rotation
        scale=1.0,
    )
    scene.add_actor(robot)
    print(f"✅ Added: {robot.name}")

    # --- 3c. Add a cube to manipulate ---
    cube = Actor(
        name="target_cube",
        asset_path="assets/blocks/red_cube_5cm",
        position=np.array([0.5, 0.2, 0.82]),  # place on the table
        rotation=np.array([1.0, 0.0, 0.0, 0.0]),
        scale=1.0,
    )
    scene.add_actor(cube)
    print(f"✅ Added: {cube.name}")

    # ============================================================
    # 4. Set up lights
    # ============================================================
    light = LightInfo(
        color=np.array([1.0, 1.0, 1.0]),   # RGB white light
        intensity=2.0,                       # brightness
    )
    scene.set_light_info("light_main", light)
    print("✅ Lights set up")

    # ============================================================
    # 5. Optional: set material color
    # ============================================================
    # Turn the cube blue
    blue_material = MaterialInfo(
        base_color=np.array([0.2, 0.4, 0.9, 1.0]),  # RGBA
    )
    scene.set_material_info("target_cube", blue_material)
    print("✅ Cube material changed to blue")

    # ============================================================
    # 6. Close scene manager
    # ============================================================
    scene.close()
    print("\n🎉 Scene setup complete! You can now load this scene with gym.make().")


if __name__ == "__main__":
    build_scene()
```

---

## Key API Details

### Actor — Every Object in the Scene

```python
Actor(
    name="unique name",                    # must be unique within the scene
    asset_path="asset path",               # corresponds to assets imported in OrcaStudio
    position=np.array([x, y, z]),          # world coordinate position (meters)
    rotation=np.array([w, x, y, z]),       # quaternion rotation
    scale=1.0,                              # scale (1.0 = original size)
)
```

### Quaternion Primer

A quaternion represents a 3D rotation using 4 numbers, in the format `[w, x, y, z]`:

```python
# Some common rotations
no_rotation = np.array([1.0, 0.0, 0.0, 0.0])       # no rotation
rotate_z_90  = np.array([0.707, 0.0, 0.0, 0.707])   # 90° around Z axis
rotate_y_180 = np.array([0.0, 0.0, 1.0, 0.0])       # 180° around Y axis

# Generate quaternion from Euler angles
from scipy.spatial.transform import Rotation as R
quat = R.from_euler('xyz', [0, 0, 1.57]).as_quat()        # [x, y, z, w]
quat_wxyz = np.array([quat[3], quat[0], quat[1], quat[2]]) # convert to [w, x, y, z]
```

### Operation Order

```
1. OrcaGymScene(grpc_addr) ← create scene manager
2. scene.publish_scene() ← publish empty scene (clear existing content)
3. scene.add_actor(...) ← add objects one by one (only enqueued, not spawned yet)
4. scene.append_scene() ← incremental publish: spawn newly added Actors, keep existing entities
   (first publish may also call publish_scene(), either one works)
5. scene.set_light_info(...) ← set lights (optional)
6. scene.set_material_info() ← modify materials (optional)
7. scene.close() ← close connection
```

> ⚠️ **`publish_scene()` clears the scene!**
> `publish_scene()` clears all entities in the current scene and republishes, suitable for initialization or a full scene reset.
> If you only want to append Actors at runtime without destroying existing entities, use `append_scene()` instead:
> it only spawns the Actors newly added via `add_actor()`, keeping previously spawned entities (such as a running robot),
> suitable for time-sequential spawning (gradually `add_actor` + `append_scene`).

---

## Getting Assets

Actors in a scene need an `asset_path`; these assets must first be fetched and imported into OrcaStudio / OrcaLab. The main workflow is as follows:

1. **Search assets**: go to the [SimAssets asset store](https://simassets.orca3d.cn/?tab=scene&sort=name) to search for the relevant assets you need.
2. **Enter the asset pack**: navigate to the corresponding asset pack page.
3. **Fetch the asset**:
   - **OrcaLab users**: subscribe directly on the asset pack page.
   - **OrcaStudio users**: download the raw assets and extract them into your project directory.
4. **Locate the asset in the editor**: open OrcaLab / OrcaStudio, find the asset in the asset search, and you can get its `asset_path`.

> For an example of dynamically adding Actors at runtime, see [OrcaPlayground - scene_building](https://github.com/openverse-orca/OrcaPlayground/tree/main/examples/scene_building).

---

## Common Asset Path Reference

Below are some example asset paths (actual paths depend on the resources imported in OrcaStudio):

| Category | Example Path |
|----------|--------------|
| Robot Arm | `robots/franka_panda/panda_arm` |
| Robot Hand | `robots/franka_panda/panda_hand` |
| Table | `assets/tables/table_80x80` |
| Cube | `assets/blocks/red_cube_5cm` |
| Sphere | `assets/balls/tennis_ball` |
| Floor | `assets/floors/checker_floor` |

> The available asset paths depend on the resources imported in your OrcaStudio. Contact your OrcaStudio administrator for a complete list.

---

## Hands-on Exercises

### Exercise 1: Build a Tabletop Scene

Place 3 cubes of different colors on a table at `(0.5, 0.0, 0.0)`, spaced 10 cm apart.

### Exercise 2: Adjust Object Pose

Place a cube rotated 45 degrees around the Z axis.

Hint:
```python
from scipy.spatial.transform import Rotation as R
quat = R.from_euler('z', 45, degrees=True).as_quat()
rotation = np.array([quat[3], quat[0], quat[1], quat[2]])
```

---

## Next Step

The scene is set up. Next, learn about the **MuJoCo backend** model loading, stepping control, and solver configuration: [🔧 MuJoCo Backend](mujoco-backend.md).
