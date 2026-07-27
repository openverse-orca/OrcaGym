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

### 1. Anchoring an Object — `anchor_actor`

```python
env.anchor_actor("target_object", "weld")
```

This single line does three things:
1. Reads the object's current world pose
2. Moves the mocap body to that pose
3. Establishes a WELD equality constraint between the mocap and the object

Anchor types:
```python
from orca_core.orca_gym_local import AnchorType

AnchorType.WELD   # Weld — fully fixed (position + orientation)
AnchorType.BALL   # Ball joint — fixed position, allows rotation
AnchorType.NONE   # No anchoring
```

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

### 3. Releasing an Object — `release_body_anchored`

```python
env.release_body_anchored()
env.mj_forward()
```

Removes the WELD constraint, returning the object to freedom (it will fall under gravity).

### 4. Equality Constraint Management

**Viewing constraints**:
```python
eq_list = env.model.get_eq_list()
for eq in eq_list:
    print(f"type={eq['eq_type']}, obj1={eq['obj1_id']}, "
          f"obj2={eq['obj2_id']}, active={eq['active']}")
```

**Modifying constraint associated objects** (by name, auto-resolves ids):
```python
env.modify_equality_objects(
    eq_ids=[0],                              # equality constraint indices
    obj1_names=["ActorManipulator_Anchor"],  # new obj1
    obj2_names=["target_object"],            # new obj2
)
```

**Deactivating constraints**:
```python
env.update_equality_constraints([{
    "type": 0, "obj1_id": -1, "obj2_id": -1,
    "data": np.zeros(7),
}])
```

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
Grasp:  anchor_actor("object", "weld")
         ↓
Move:   set_mocap_pos_and_quat({mocap: {pos, quat}})
         ↓
        mj_forward()
         ↓
        do_simulation(ctrl, n_frames)  ← constraint takes effect, object follows
         ↓
Release: release_body_anchored()
```

---

## Next Steps

Now that you understand equality constraints, learn how to **apply external forces and IK**: [🔄 Force Application and IK](../physics/force-apply.md).
