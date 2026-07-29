# 🎭 Mocap Control

Mocap (Motion Capture) bodies are special bodies in MuJoCo that can be controlled by directly setting their pose.

## What is a Mocap Body?

- A body in MuJoCo with `body_mocapid != -1`
- Its pose can be **set directly** without being affected by forces/dynamics
- Often used with equality constraints (WELD/CONNECT) to implement grasping/dragging
- Typical uses: anchor points, virtual hands, tool attachment points

## Finding Mocap Bodies

Under the Euler system, `OrcaGymEulerEnv` does not directly expose a public method to list mocap names.
You can verify whether a given mocap body exists via `env.data.mocap_pos(name)`
(it raises `KeyError` if the name does not exist), or traverse via `env.model.get_body_names()`
(mocap bodies usually carry suffixes like `Anchor`).

```python
# Read the pose of a known mocap body via env.data (raises if the name does not exist)
mocap_pos = env.data.mocap_pos("ActorManipulator_Anchor")  # (3,)
mocap_quat = env.data.mocap_quat("ActorManipulator_Anchor")  # (4,)
```

> Note: the UI-grasp dedicated mocap body name is `ORCA_MANIPULATOR_<uuid>_Anchor` in default levels,
> and `ActorManipulator_Anchor` in legacy levels.

## Setting Mocap Pose

```python
# Directly set the world-frame pose of a mocap body
env.set_mocap_pos_and_quat({
	"ActorManipulator_Anchor": {
		"pos": np.array([0.5, 0.0, 0.8], dtype=np.float64),
		"quat": np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
	}
})

# Must call mj_forward() to refresh derived quantities
env.mj_forward()
```

## Reading Mocap Pose

```python
# Read mocap pose via env.data
mocap_pos = env.data.mocap_pos("ActorManipulator_Anchor") # (3,)
mocap_quat = env.data.mocap_quat("ActorManipulator_Anchor") # (4,)
```

## Mocap + Equality Constraints = Object Manipulation

Under the Euler system, `OrcaGymEulerEnv` does not provide `anchor_actor` / `release_body_anchored`
high-level public methods (these exist only in the Local system). Programmatic operations must be
orchestrated with **public equality-constraint primitives**, consistent with the UI-grasp internal
methods `_anchor_actor` / `_release_body_anchored`:

- `equality_find_slot_by_body(body_name)` — find the equality constraint slot containing the specified body
- `equality_constraint(slot)` — read the full data of the slot (for snapshot/restore)
- `equality_update(slot, ...)` — atomically write slot fields (type/obj1/obj2/data/active...)
- `set_mocap_pos_and_quat(...)` — align the mocap pose to the target body

```python
import mujoco

# ── 1. Anchor an object ──
anchor_mocap_name = "ActorManipulator_Anchor"  # or ORCA_MANIPULATOR_<uuid>_Anchor
actor_name = "target_object"

# Find the equality constraint slot containing the anchor mocap
slot = env.equality_find_slot_by_body(anchor_mocap_name)
if slot == -1:
    raise ValueError(f"No equality slot containing {anchor_mocap_name} in the model")

# Save the original constraint snapshot (restore on release)
original_eq = env.equality_constraint(slot)

# Align the mocap pose to the actor's current pose (avoid yanking on the next frame)
actor_pose = env.get_body_xpos_xmat_xquat([actor_name])[actor_name]
env.set_mocap_pos_and_quat({
    anchor_mocap_name: {
        "pos": actor_pose["xpos"],
        "quat": actor_pose["xquat"],
    }
})

# Decide whether to change obj1 or obj2 (keep the mocap end, change the other end to actor)
mocap_id = env.model.body_name2id(anchor_mocap_name)
if original_eq["obj1_id"] == mocap_id:
    new_obj1_name = anchor_mocap_name
    new_obj2_name = actor_name
else:
    new_obj1_name = actor_name
    new_obj2_name = anchor_mocap_name

# Write the constraint (type/obj, internal mj_forward)
env.equality_update(
    slot,
    eq_type=mujoco.mjtEq.mjEQ_WELD,
    obj1_name=new_obj1_name,
    obj2_name=new_obj2_name,
)

# ── 2. Move the anchor → object follows ──
env.set_mocap_pos_and_quat({
    anchor_mocap_name: {
        "pos": new_target_pos,
        "quat": new_target_quat,
    }
})
env.mj_forward()

# ── 3. Release (restore the original constraint from the snapshot) ──
slot = env.equality_find_slot_by_body(actor_name)
if slot != -1:
    env.equality_update(
        slot,
        eq_type=original_eq["type"],
        obj1_name=env.model.body_id2name(original_eq["obj1_id"]),
        obj2_name=env.model.body_id2name(original_eq["obj2_id"]),
        data=original_eq["data"],
    )
```

> Note: `obj1_name` / `obj2_name` of `equality_update` must be **full body names including the agent prefix**
> (this primitive does not perform namespace resolution).

## Trajectory Tracking Example

```python
def follow_trajectory(env, trajectory: list[np.ndarray], duration: float):
 """Have the anchor follow a trajectory"""
 steps = int(duration / env.dt)
 
 for i in range(steps):
 t = i / steps
 idx = min(int(t * len(trajectory)), len(trajectory) - 1)
 target_pos = trajectory[idx]
 
 env.set_mocap_pos_and_quat({
 "ActorManipulator_Anchor": {
 "pos": target_pos,
 "quat": np.array([1, 0, 0, 0]),
 }
 })
 
 env.mj_forward()
 env.render()
```
