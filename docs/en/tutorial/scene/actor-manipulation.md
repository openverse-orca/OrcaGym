# 🎭 Actor Manipulation

Manipulating scene objects in OrcaGym/OrcaStudio.

## Anchor System

OrcaGym uses a **Mocap anchor + equality constraint** system to manipulate objects:

```
User/code → set mocap pose →
 WELD/CONNECT constraint → object follows the anchor
```

## Python Code Manipulation

```python
# Anchor an object
env.anchor_actor("target_object", AnchorType.WELD)

# Move the anchor (mocap body name must exist in the model)
env.set_mocap_pos_and_quat({
    "ActorManipulator_Anchor": {
        "pos": np.array([0.5, 0.0, 0.8]),
        "quat": np.array([1.0, 0.0, 0.0, 0.0]),
    }
})

# Release
env.release_body_anchored()
```

## UI Drag Manipulation

When dragging objects in the OrcaStudio UI:

```python
# Detect UI manipulation
body_name, anchor_type = env.get_body_manipulation_anchored()
if body_name is not None:
 delta_pos, delta_quat = env.get_body_manipulation_movement()
 print(f"Object {body_name} moved by {delta_pos}")
```

## Bounding Box Query

```python
# Compute the axis-aligned bounding box of an object
bbox = env.get_goal_bounding_box("target_object")
print(f"Bounding box: min={bbox['min']}, max={bbox['max']}, size={bbox['size']}")
```
