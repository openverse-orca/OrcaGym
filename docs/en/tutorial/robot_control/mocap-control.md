# 🎭 Mocap Control

Mocap (Motion Capture) bodies are special bodies in MuJoCo that can be controlled by directly setting their pose.

## What is a Mocap Body

- A body in MuJoCo with `body_mocapid != -1`
- Its pose can be **set directly** without being affected by forces/dynamics
- Often used with equality constraints (WELD/CONNECT) to implement grasping/dragging
- Typical uses: anchor points, virtual hands, tool attachment points

## Finding Mocap Bodies

```python
# View all mocap bodies in the model
mocap_dict = env.model.get_mocap_dict()
for name, mocap_id in mocap_dict.items():
 print(f"Mocap: {name} (id={mocap_id})")

# Can also query via _gym
mocap_names = env._mocap_body_names()
```

## Setting Mocap Pose

```python
# Directly set the world-frame pose of a mocap body
env.set_mocap_pos_and_quat({
 "ActorManipulator_Anchor": {
 "pos": np.array([0.5, 0.0, 0.8], dtype=np.float64),
 "quat": np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
 }
})

# Must forward
env.mj_forward()
```

## Reading Mocap Pose

```python
# Read mocap pose via env.data
mocap_pos = env.data.mocap_pos("ActorManipulator_Anchor") # (3,)
mocap_quat = env.data.mocap_quat("ActorManipulator_Anchor") # (4,)
```

## Mocap + Equality Constraints = Object Manipulation

```python
# Use the high-level API
# 1. Anchor an object — auto-query pose + set mocap + create constraint
env.anchor_actor("target_object", "weld")

# 2. Move anchor → object follows
env.set_mocap_pos_and_quat({
 "ActorManipulator_Anchor": {
 "pos": new_target_pos,
 "quat": new_target_quat,
 }
})
env.mj_forward()

# 3. Release
env.release_body_anchored()
```

### Low-Level Control (When Needed)

```python
# Modify the associated object of an equality constraint
env.modify_equality_objects(
 eq_ids=[0],
 obj2_names=["target_object"], # Change obj2 from old body to target body
)

# Update constraints
env.update_equality_constraints(eq_list)
```

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
