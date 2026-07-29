# 🔗 Equality Constraints

MuJoCo's equality constraints are the core mechanism in OrcaGym for implementing object grasping and manipulation.

> See [OrcaPlayground examples/euler/05_force_apply/](https://github.com/OrcaGym/OrcaPlayground) and [09_body_manipulation/](https://github.com/OrcaGym/OrcaPlayground) for complete runnable code.

---

## Complete Example: Overview First

Below is a complete grasp → move → release demo:

```python
"""Equality Constraints Complete Demo: Grasp → Move → Release"""
import numpy as np
from orca_gym.environment.euler.orca_gym_euler_env import OrcaGymEulerEnv


class GraspDemo(OrcaGymEulerEnv):
    """Demonstrates mocap + weld constraint object grasping"""

    def __init__(self, model_xml_path, **kwargs):
        super().__init__(
            frame_skip=kwargs.pop("frame_skip", 20),
            orcagym_addr=kwargs.pop("orcagym_addr", "localhost:50051"),
            agent_names=kwargs.pop("agent_names", ["g1"]),
            time_step=kwargs.pop("time_step", 0.001),
            model_xml_path=model_xml_path,
            **kwargs,
        )

    def demo_grasp_and_move(self):
        """Complete demo: grasp object → move to target → release"""
        agent = self.agent_name
        object_name = f"{agent}_manipulation_box"
        ctrl = np.zeros(self.model.nu)

        # ─── Step 1: Grasp ───
        print("Step 1: Grasping object...")
        self.anchor_actor(object_name, "weld")
        print(f"  ✅ {object_name} anchored (WELD constraint)")

        # ─── Step 2: Move ───
        target_pos = np.array([0.7, 0.0, 0.5])
        target_quat = np.array([1.0, 0.0, 0.0, 0.0])
        print(f"\nStep 2: Moving object to {target_pos}...")

        self.set_mocap_pos_and_quat({
            "ActorManipulator_Anchor": {
                "pos": target_pos,
                "quat": target_quat,
            }
        })
        self.mj_forward()

        # Step to let the constraint take effect
        for _ in range(10):
            self.do_simulation(ctrl, self.frame_skip)

        # Verify: object has followed to target
        box = self.get_body_xpos_xmat_xquat([object_name])
        box_pos = box[object_name]["xpos"]
        dist = np.linalg.norm(box_pos - target_pos)
        print(f"  Current object position: {box_pos}")
        print(f"  Distance to target: {dist:.4f}m")
        print(f"  {'✅ Object reached target' if dist < 0.05 else '⚠️ Not reached'}")

        # ─── Step 3: Release ───
        print(f"\nStep 3: Releasing object...")
        self.release_body_anchored()
        self.mj_forward()
        print("  ✅ Object released")

        # ─── Step 4: View constraint info ───
        print(f"\nCurrent equality constraints:")
        eq_list = self.model.get_eq_list()
        for eq in eq_list:
            print(f"  type={eq['eq_type']}, obj1={eq['obj1_id']}, "
                  f"obj2={eq['obj2_id']}, active={eq['active']}")

    def step(self, action):
        self.do_simulation(action, self.frame_skip)
        return self._get_obs(), 0.0, False, False, {}

    def reset_model(self):
        self.set_joint_qpos(self.init_qpos)
        self.set_joint_qvel(self.init_qvel)
        self.mj_forward()
        self._sync_view()
        return self._get_obs(), {}

    def _get_obs(self):
        return self.data.qpos.copy()


if __name__ == "__main__":
    env = GraspDemo(
        model_xml_path="/path/to/scene.xml",
        skip_grpc_load=False,
    )
    env.reset()
    env.demo_grasp_and_move()
    env.close()
```

---

## Section-by-Section Explanation

### What Are Equality Constraints

Equality constraints enforce a certain kinematic relationship between two bodies:

| Constraint Type | Effect | Degrees of Freedom |
|----------|------|--------|
| `mjEQ_WELD` | Fully fixed (position + orientation), like welded together | 0 DOF |
| `mjEQ_CONNECT` (BALL) | Fixed position, allows rotation, like a ball joint | 3 DOF (rotation) |

In OrcaGym, equality constraints are typically used together with **mocap bodies**:
```
User sets mocap pose → WELD constraint → anchored object follows the movement
```

### 1. Anchoring an Object — Public Primitive Orchestration

> ⚠️ **Euler path**: `OrcaGymEulerEnv` does **not** have `anchor_actor` / `release_body_anchored` public methods (these are only provided in the Local system `OrcaGymLocalEnv`). Programmatic operations should follow the UI-grasp internal method `_anchor_actor` / `_release_body_anchored` orchestration pattern, combining these public primitives:

```python
# Equivalent to Local system's env.anchor_actor("target_object", "weld")
import mujoco

mocap_name = "ActorManipulator_Anchor"   # mocap body in the scene
object_name = "target_object"

# 1. Find the equality constraint slot containing the mocap
slot = env.equality_find_slot_by_body(mocap_name)
# 2. Read the original constraint (restore on release)
original_eq = env.equality_constraint(slot)
# 3. Align mocap pose to the object's current pose (avoid yanking on the next frame)
obj_pose = env.get_body_xpos_xmat_xquat([object_name])[object_name]
env.set_mocap_pos_and_quat({
    mocap_name: {"pos": obj_pose["xpos"], "quat": obj_pose["xquat"]}
})
# 4. Write the WELD constraint
env.equality_update(
    slot,
    eq_type=mujoco.mjtEq.mjEQ_WELD,
    obj1_name=mocap_name,
    obj2_name=object_name,
)
```

This set of operations does three things:
1. Reads the object's current world pose
2. Moves the mocap body to that pose
3. Establishes a WELD equality constraint between the mocap and the object

Constraint type constants (from `mujoco.mjtEq`, no need to import `AnchorType`):
```python
import mujoco

mujoco.mjtEq.mjEQ_WELD      # Weld — fully fixed (position + orientation)
mujoco.mjtEq.mjEQ_CONNECT   # Ball joint — fixed position, allows rotation
mujoco.mjtEq.mjEQ_JOINT     # Joint coupling
```

> 📝 **Local system**: If using `OrcaGymLocalEnv`, you can directly call `env.anchor_actor(name, AnchorType.WELD)`, importing `AnchorType` from `orca_gym.core.orca_gym_local`. The Euler path does not provide this convenience wrapper; use the primitive orchestration above.

### 2. Moving an Object — Mocap Pose Setting

```python
env.set_mocap_pos_and_quat({
    "ActorManipulator_Anchor": {
        "pos": np.array([0.7, 0.0, 0.5]),          # target position [x, y, z]
        "quat": np.array([1.0, 0.0, 0.0, 0.0]),    # target quaternion [w, x, y, z]
    }
})
env.mj_forward()  # ← Required! Update derived quantities
```

**Mocap bodies** are special bodies in MuJoCo (`body_mocapid != -1`):
- Their pose can be **directly set**, unaffected by forces/dynamics
- They move like an "invisible hand"
- Combined with a WELD constraint, the anchored object automatically follows

**Read-back verification** (via `env.data` zero-copy view):
```python
read_pos = env.data.mocap_pos("mocap_name")    # (3,)
read_quat = env.data.mocap_quat("mocap_name")  # (4,) [w, x, y, z]
```

### 3. Releasing an Object — Restore Original Constraint

> ⚠️ **Euler path**: `OrcaGymEulerEnv` does not have a `release_body_anchored` public method. On release, use `equality_update` to restore the original constraint fields saved before grasping:

```python
# Equivalent to Local system's env.release_body_anchored()
slot = env.equality_find_slot_by_body(object_name)
if slot != -1:
    env.equality_update(
        slot,
        eq_type=original_eq["type"],
        obj1_name=env.model.body_id2name(original_eq["obj1_id"]),
        obj2_name=env.model.body_id2name(original_eq["obj2_id"]),
        data=original_eq["data"],
    )
env.mj_forward()
```

Removes the WELD constraint, returning the object to freedom (it will fall under gravity).

### 4. Equality Constraint Management

**Viewing constraints** (two equivalent paths):
```python
# Path A: read one by one via env.equality_constraint(slot) (key is type)
for slot in range(env._gym.n_equality()):
    eq = env.equality_constraint(slot)
    print(f"type={eq['type']}, obj1={eq['obj1_id']}, "
          f"obj2={eq['obj2_id']}, active={eq['active']}")

# Path B: read initial snapshot via env.model.get_eq_list() (key is eq_type)
eq_list = env.model.get_eq_list()
for eq in eq_list:
    print(f"type={eq['eq_type']}, obj1={eq['obj1_id']}, "
          f"obj2={eq['obj2_id']}, active={eq['active']}")
```

> ⚠️ **Key name difference**: The dict returned by `env.equality_constraint(slot)` uses the key `type`; the dict returned by `env.model.get_eq_list()` uses the key `eq_type`. Both correspond to MuJoCo's `eq_type` field, just named differently.

**Modifying constraint associated objects** (Euler path uses `equality_update`, auto-resolves ids by name):
```python
# Euler path: env.equality_update(slot, obj1_name=..., obj2_name=...)
env.equality_update(
    0,                                        # equality constraint slot index
    obj1_name="ActorManipulator_Anchor",      # new obj1 (auto-resolved to id)
    obj2_name="target_object",                # new obj2 (auto-resolved to id)
)
```

> 📝 **Local system**: `OrcaGymLocalEnv` provides `env.modify_equality_objects(eq_ids, obj1_ids, obj2_ids)` (parameters are id lists, gym-layer API). The Euler path supersedes this with `equality_update`, and passing by name is more intuitive.

**Deactivating constraints** (via `equality_update` with `active=False`):
```python
env.equality_update(0, active=False)   # deactivate the constraint in slot 0
env.equality_update(0, active=True)    # reactivate
```

> ⚠️ The Euler path has removed the `env.update_equality_constraints(eq_list)` public method (it remains in the gym layer `OrcaGymEuler` / SimCore as the underlying implementation of `equality_update`). The Env layer uniformly uses `equality_update(slot, ...)` to update per-slot.

### 5. Anchoring in UI Interaction

When dragging objects in the OrcaStudio UI, the system automatically handles anchoring:

```python
# render() internally calls do_body_manipulation()
# UI operations can be detected via studio_bridge()
bridge = env.studio_bridge()
body_name, anchor_type = bridge.get_body_manipulation_anchored()
if body_name is not None:
    delta_pos, delta_quat = bridge.get_body_manipulation_movement()
    print(f"User is dragging: {body_name}, displacement: {delta_pos}")
```

---

## Complete Workflow Summary

```
Grasp:  equality_find_slot_by_body(mocap) → equality_constraint(slot) (save snapshot)
        → set_mocap_pos_and_quat(...) (align) → equality_update(slot, WELD, obj1, obj2)
         ↓
Move:   set_mocap_pos_and_quat({mocap: {pos, quat}})
         ↓
        mj_forward()
         ↓
        do_simulation(ctrl, n_frames)  ← constraint takes effect, object follows
         ↓
Release: equality_find_slot_by_body(object) → equality_update(slot, restore from snapshot)
```

---

## Next Steps

Now that you understand equality constraints, learn how to **apply external forces and IK**: [🔄 Force Application and IK](../physics/force-apply.md).
