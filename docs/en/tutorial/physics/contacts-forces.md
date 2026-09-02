# 💥 Contacts and Forces

OrcaGym provides comprehensive contact and force querying interfaces for reward computation, debugging, and analysis.

> See [OrcaPlayground examples/euler/06_force_apply/](https://github.com/openverse-orca/OrcaPlayground/tree/main/examples/euler/06_force_apply) (`force_apply.py` + `force_apply_env.py`) for complete runnable code.

---

## Real-World Example: G1 Standing Contact Force Verification

Below is the real verification flow from [Lesson 6](https://github.com/openverse-orca/OrcaPlayground/tree/main/examples/euler/06_force_apply): G1 humanoid stands, feet contact ground, query contact pairs and normal forces.

```python
"""Contact force query verification (from force_apply_env.py phase 1)"""

# ── step 0: G1 upright, feet on ground ──
contacts = env.query_contact_simple()
print(f"Contact pairs: {len(contacts)}")  # actually ~25 (multi-geom foot-ground)

# Verify: at least 1 contact (standing on ground)
assert len(contacts) >= 1, "G1 should have contacts when standing"

# Query contact forces (needs contact index list)
contact_ids = list(range(len(contacts)))
forces = env.query_contact_force(contact_ids)
# → {0: array([normal, shear1, shear2, torque1, torque2, torque3]), ...}

# First 3 components in contact frame; [0] = normal force
max_normal = max(abs(f[0]) for f in forces.values())
print(f"Max normal force: {max_normal:.1f}N")  # actually ~109931.9N (G1 weight ~343N)

# Verify: significant normal force (> 50N, proves real foot contact)
assert max_normal > 50.0, "Normal force should exceed 50N"
```

**Key Points**:

- `query_contact_simple()` returns all active contact pairs, each with `geom1`/`geom2` IDs
- `query_contact_force(contact_ids)` returns 6D force (contact frame: normal + shear + torque)
- Component [0] = normal force, the key metric for "real contact"

---

## Contact Force Visualization (Studio Auto-Rendering)

In Euler online mode, `env.render()` automatically builds contact snapshots and sends them to OrcaStudio for rendering:

```python
# Normal step + render
env.do_simulation(ctrl, env.frame_skip)
env.render(simulate_index=step_idx)
#                ↑ internally:
#   1. _build_contact_data() collects pos + world_force for all contacts
#   2. Sends to Studio, draws contact force vectors in 3D viewport
```

**Rendered Content**:

- Contact point position (3D arrow origin)
- Contact force vector (world frame, arrow direction and length reflect magnitude)
- Force converted from contact frame to world frame (`frame.T @ force`)

> No manual rendering call needed — `render()` handles it automatically. Offline mode has no rendering (no Studio connection).

---

## Applying External Force

Lesson 6 step 10 applies 500N upward force to G1 pelvis, verifying `apply_body_force`:

```python
"""Force application (from force_apply_env.py phase 2)"""

# ── step 10: record pelvis initial height, apply 500N upward ──
agent = env._agent_names[0]  # e.g. "g1_29dof_camera_usda"
pelvis_body = f"{agent}_pelvis"

pelvis = env.get_body_xpos_xmat_xquat([pelvis_body])
z_before = float(pelvis[pelvis_body]["xpos"][2])  # actually ~0.7864m
print(f"Pelvis height before force: {z_before:.4f}m")

env.apply_body_force(
    pelvis_body,
    force=np.array([0.0, 0.0, 500.0]),   # 500N up (> G1 weight ~343N)
    torque=np.array([0.0, 0.0, 0.0]),
)

# ── step 30: verify pelvis rise + xfrc record ──
for _ in range(20):  # 20 control cycles (0.4s) for force to take effect
    env.do_simulation(np.zeros(env.model.nu), env.frame_skip)

pelvis = env.get_body_xpos_xmat_xquat([pelvis_body])
z_after = float(pelvis[pelvis_body]["xpos"][2])  # actually ~1.1777m
print(f"Pelvis height after force: {z_after:.4f}m (Δ={z_after - z_before:.4f}m)")
# Verify: pelvis rose > 1cm
assert z_after > z_before + 0.01
```

**Why force on pelvis not torso**: G1 uses force-controlled motor actuators; with `ctrl=0`, joints have no torque, lumbar joints are slack under zero control. Force on torso cannot transfer through slack joints to pelvis. Directly forcing pelvis reliably verifies the API and produces visible whole-body lift.

### Verifying xfrc_applied Record

```python
# xfrc_applied is DataView read-only view, indexed by body_id
body_id = env.model.body_name2id(pelvis_body)
xfrc = env.data.xfrc_applied[body_id, :3]
print(f"xfrc record: {xfrc}")  # [0.0, 0.0, 500.0]
assert np.any(xfrc != 0), "xfrc should record applied force"
```

### Clearing External Force

```python
# ── end of step 30: clear force after verifying lift ──
env.clear_body_force(pelvis_body)

# ── step 35: verify xfrc is zeroed ──
xfrc = env.data.xfrc_applied[body_id, :3]
print(f"xfrc after clear: {xfrc}")  # [0.0, 0.0, 0.0]
assert np.all(xfrc == 0), "xfrc should be zero after clear"

# step 50: clear all (smoke test)
env.clear_all_forces()
assert np.all(env.data.xfrc_applied == 0), "All xfrc should be zero after clear_all"
```

---

## Friction Coefficient Setting

Lesson 6 step 50 sets friction on a G1 geom:

```python
"""Friction setting (from force_apply_env.py phase 3)"""

# Get geom name dynamically from model.get_geom_dict() (with agent prefix + GUID suffix)
geom_dict = env.model.get_geom_dict()
g1_geom = next(name for name in geom_dict if name.startswith(f"{agent}_"))

env.set_geom_friction(
    {g1_geom: np.array([0.8, 0.005, 0.0001])}
    # [slide, torsion, rolling friction]
)
print(f"Set friction for {g1_geom}")
```

---

## Body External Constraint Force

Query external constraint forces on all bodies (contact reaction, equality constraint forces, etc.):

```python
cfrc_ext = env.get_cfrc_ext()  # shape: (nbody, 6)
# Each row: [mx, my, mz, fx, fy, fz] — MuJoCo spatial vector layout (torque first, force last)

# Find body with max force (by linear force magnitude)
max_idx = np.argmax(np.linalg.norm(cfrc_ext[:, 3:], axis=1))
print(f"Max force body ID: {max_idx}, force: {cfrc_ext[max_idx, 3:]}")
```

---

## Collision Detection

Check if two bodies collide:

```python
def detect_collision(env, body_a, body_b):
    """Check if two bodies collide"""
    contacts = env.query_contact_simple()
    id_a = env.model.body_name2id(body_a)
    id_b = env.model.body_name2id(body_b)

    for c in contacts:
        g1 = env.model.get_geom_body_id(c["geom1"])
        g2 = env.model.get_geom_body_id(c["geom2"])
        if (g1 == id_a and g2 == id_b) or (g1 == id_b and g2 == id_a):
            return True
    return False

# Check if G1 left and right feet collide
left_foot = f"{agent}_left_ankle_roll_link"
right_foot = f"{agent}_right_ankle_roll_link"
print(f"Left↔Right foot collision: {detect_collision(env, left_foot, right_foot)}")
```

---

## Contacts in Reward Functions

```python
def contact_reward(env):
    """Reward moderate contact forces"""
    contacts = env.query_contact_simple()
    if not contacts:
        return -1.0          # no contact = penalty

    contact_ids = list(range(len(contacts)))
    forces = env.query_contact_force(contact_ids)
    total_force = sum(np.linalg.norm(f[:3]) for f in forces.values())

    if total_force < 100:    return 0.5   # light contact
    elif total_force < 500:  return 1.0   # ideal contact
    else:                    return -0.5  # excessive force
```

---

## Contact Frame Explanation

`query_contact_force` returns forces in the **contact frame**:

| Component | Meaning |
|-----------|---------|
| `[0]` | Normal force (perpendicular to contact surface) |
| `[1:3]` | Shear force (friction) |
| `[3:6]` | Torque components |

> **Note**: `query_contact_simple()` returns dict keys in **lowercase** `"geom1"` / `"geom2"`.

---

## API Quick Reference

| Operation | API | Description |
|-----------|-----|-------------|
| Get contacts | `env.query_contact_simple()` | Returns `list[dict]`, lowercase keys |
| Get contact force | `env.query_contact_force(ids)` | 6D force, contact frame |
| Get constraint force | `env.get_cfrc_ext()` | (nbody, 6), world frame |
| Apply external force | `env.apply_body_force(name, f, τ)` | World frame |
| Clear body force | `env.clear_body_force(name)` | Clear specific body |
| Clear all forces | `env.clear_all_forces()` | Clear all external forces |
| Set friction | `env.set_geom_friction({name: arr})` | [slide, torsion, rolling] |
| Render contacts | `env.render()` | Auto-builds contact snapshot for Studio |

---

## Next Step

Now that you understand contacts and forces, learn how to **grab objects with equality constraints**: [🔗 Equality Constraints](equality-constraints.md).
