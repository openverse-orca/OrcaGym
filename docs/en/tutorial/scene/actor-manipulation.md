# 🎭 Actor Manipulation

Manipulating scene objects in OrcaGym/OrcaStudio.

## Anchor System

OrcaGym uses a **Mocap anchor + equality constraint** system to manipulate objects:

```
User/code → set mocap pose →
 WELD/CONNECT constraint → object follows the anchor
```

## Python Code Manipulation

Under the Euler system, `OrcaGymEulerEnv` does not provide `anchor_actor` / `release_body_anchored`
high-level public methods (these exist only in the Local system). Programmatic operations must be
orchestrated with **public equality-constraint primitives**:
`equality_find_slot_by_body` + `equality_constraint` + `equality_update` +
`set_mocap_pos_and_quat`. For a complete example, see [🎭 Mocap Control](../robot_control/mocap-control.md).

A simplified snippet to move the anchor:

```python
# Move the anchor (mocap body name must exist in the model)
env.set_mocap_pos_and_quat({
    "ActorManipulator_Anchor": {
        "pos": np.array([0.5, 0.0, 0.8]),
        "quat": np.array([1.0, 0.0, 0.0, 0.0]),
    }
})

# Release must be done by restoring the constraint from the original snapshot via equality_update (see the Mocap Control doc)
```

## UI Drag Manipulation

When dragging objects in the OrcaStudio UI, under the Euler system you must query status via the
**async** methods of `env.studio_bridge()` (returns `OrcaStudioBridge`), driven by the event loop:

```python
# Detect UI manipulation (async methods, driven via env.loop)
body_name, anchor_type = env.loop.run_until_complete(
    env.studio_bridge().get_body_manipulation_anchored()
)
if body_name is not None:
    # Returns a dict with "delta_pos" and "delta_quat" keys (not a tuple)
    movement = env.loop.run_until_complete(
        env.studio_bridge().get_body_manipulation_movement()
    )
    delta_pos = movement["delta_pos"]
    delta_quat = movement["delta_quat"]
    print(f"Object {body_name} moved by {delta_pos}")
```

> Note: `OrcaGymEulerEnv` internally wraps the above flow in `_do_body_manipulation`,
> driven automatically by `render()`; users usually do not need to call it manually.

## Bounding Box Query

```python
# Returns np.ndarray(3,), the geom half-sizes (hx, hy, hz), not a dict
bbox = env.get_goal_bounding_box("target_object")
print(f"Half-sizes: {bbox}")  # e.g. [0.05, 0.05, 0.1]
```
