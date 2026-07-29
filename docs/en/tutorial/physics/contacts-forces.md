# 💥 Contacts and Forces

OrcaGym provides comprehensive contact and force querying interfaces for reward computation, debugging, and analysis.

> See [OrcaPlayground examples/euler/05_force_apply/](https://github.com/OrcaGym/OrcaPlayground) for complete runnable code.

---

## Complete Example: Overview First

Below is a complete contact and force analysis demo, showing contact detection, contact force querying, external force application, collision detection, and standing detection:

```python
"""Complete Contacts and Forces Demo"""
import numpy as np
from orca_gym.environment.euler.orca_gym_euler_env import OrcaGymEulerEnv


class ContactForceDemo(OrcaGymEulerEnv):
    """Demonstrates contact detection, force querying, and external force application"""

    def __init__(self, model_xml_path, **kwargs):
        super().__init__(
            frame_skip=kwargs.pop("frame_skip", 20),
            orcagym_addr=kwargs.pop("orcagym_addr", "localhost:50051"),
            agent_names=kwargs.pop("agent_names", ["g1"]),
            time_step=kwargs.pop("time_step", 0.001),
            model_xml_path=model_xml_path,
            **kwargs,
        )

    # ─── Utility Functions ───

    def analyze_contacts(self):
        """Analyze all current contacts"""
        contacts = self.query_contact_simple()
        if not contacts:
            print("  No active contacts")
            return

        print(f"  Active contacts: {len(contacts)}")
        contact_ids = list(range(len(contacts)))
        forces = self.query_contact_force(contact_ids)

        # Show first 5 contacts
        for i, c in enumerate(contacts[:5]):
            f = forces[i]
            f_linear = f[:3]
            f_magnitude = np.linalg.norm(f_linear)
            print(f"    Contact {i}: geom{c['geom1']}↔geom{c['geom2']}, "
                  f"force={f_magnitude:.2f}N, direction={f_linear}")

        # Max normal force
        max_normal = max(abs(f[0]) for f in forces.values())
        print(f"  Max normal force: {max_normal:.1f}N")

    def detect_collision(self, body_a, body_b):
        """Check if two bodies are colliding"""
        contacts = self.query_contact_simple()
        id_a = self.model.body_name2id(body_a)
        id_b = self.model.body_name2id(body_b)

        for c in contacts:
            g1 = self.model.get_geom_body_id(c["geom1"])
            g2 = self.model.get_geom_body_id(c["geom2"])
            if (g1 == id_a and g2 == id_b) or (g1 == id_b and g2 == id_a):
                return True
        return False

    def is_standing(self, min_force=50.0):
        """Check if the robot is standing (sufficient foot contact force)"""
        contacts = self.query_contact_simple()
        if not contacts:
            return False
        contact_ids = list(range(len(contacts)))
        forces = self.query_contact_force(contact_ids)
        max_normal = max(abs(f[0]) for f in forces.values())
        return max_normal > min_force

    # ─── Demo Flow ───

    def demo(self):
        self.reset()
        agent = self.agent_name
        ctrl = np.zeros(self.model.nu)

        # Step a few frames to let the robot settle on the ground
        for _ in range(5):
            self.do_simulation(ctrl, self.frame_skip)

        # ─── 1. Contact Detection ───
        print("=" * 50)
        print("1. Contact Detection (G1 standing on ground)")
        print("=" * 50)
        contacts = self.query_contact_simple()
        print(f"  Contact pairs: {len(contacts)}")
        print(f"  Standing: {'✅ Standing' if self.is_standing() else '⚠️ Not standing'}")
        self.analyze_contacts()

        # ─── 2. Body External Constraint Forces ───
        print("\n" + "=" * 50)
        print("2. Body External Constraint Forces")
        print("=" * 50)
        cfrc_ext = self.get_cfrc_ext()
        max_idx = np.argmax(np.linalg.norm(cfrc_ext[:, :3], axis=1))
        print(f"  Body with max force ID: {max_idx}, force: {cfrc_ext[max_idx, :3]}")

        # ─── 3. Apply External Force to Lift Pelvis ───
        print("\n" + "=" * 50)
        print("3. Apply External Force to Lift the Robot")
        print("=" * 50)

        pelvis_body = f"{agent}_pelvis"
        pelvis = self.get_body_xpos_xmat_xquat([pelvis_body])
        z_before = float(pelvis[pelvis_body]["xpos"][2])
        print(f"  Pelvis height before force: {z_before:.3f}m")

        # Apply 500N upward force
        self.apply_body_force(
            pelvis_body,
            force=np.array([0.0, 0.0, 500.0]),
            torque=np.array([0.0, 0.0, 0.0]),
        )

        # Step to let the force take effect
        for _ in range(20):
            self.do_simulation(ctrl, self.frame_skip)

        pelvis = self.get_body_xpos_xmat_xquat([pelvis_body])
        z_after = float(pelvis[pelvis_body]["xpos"][2])
        print(f"  Pelvis height after force: {z_after:.3f}m (Δ={z_after - z_before:.3f}m)")

        # Verify xfrc_applied
        body_id = self.model.body_name2id(pelvis_body)
        xfrc = self.data.xfrc_applied[body_id, :3]
        print(f"  Force recorded in xfrc_applied: {xfrc}")

        # ─── 4. Clear External Force ───
        print("\n" + "=" * 50)
        print("4. Clear External Force")
        print("=" * 50)

        self.clear_body_force(pelvis_body)
        xfrc = self.data.xfrc_applied[body_id, :3]
        print(f"  xfrc after clearing: {xfrc}")
        assert np.all(xfrc == 0), "xfrc should be zero after clearing force"
        print("  ✅ clear_body_force succeeded")

        self.clear_all_forces()  # Clear all (smoke test)
        print("  ✅ clear_all_forces succeeded")

        # ─── 5. Collision Detection Test ───
        print("\n" + "=" * 50)
        print("5. Collision Detection")
        print("=" * 50)
        left_foot = f"{agent}_left_ankle_roll_link"
        right_foot = f"{agent}_right_ankle_roll_link"
        # Ground body name depends on XML definition
        print(f"  Left↔Right foot collision: {self.detect_collision(left_foot, right_foot)}")

        print("\n✅ All contact and force demos completed")

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
    env = ContactForceDemo(
        model_xml_path="/path/to/scene.xml",
        skip_grpc_load=False,
    )
    env.demo()
    env.close()
```

---

## Section-by-Section Explanation

### 1. Contact Detection

```python
contacts = env.query_contact_simple()
# → [{"geom1": 12, "geom2": 34, "dist": ..., "pos": ..., "frame": ...}, ...]
```

`query_contact_simple()` returns all currently active contact pairs. Each contact is a dictionary
containing the IDs of the two colliding geoms, penetration distance, contact point position, and contact frame.

> **Note**: Dictionary keys are **lowercase** `"geom1"` / `"geom2"`, not uppercase.

**Getting Contact Forces** (requires two steps):

```python
# Step 1: Build a list of contact IDs (by contact list index)
contact_ids = list(range(len(contacts)))

# Step 2: Query contact forces
forces = env.query_contact_force(contact_ids)
# → {0: array([normal, shear1, shear2, torque1, torque2, torque3]), ...}

# Contact frame: component 0 = normal force
max_normal = max(abs(f[0]) for f in forces.values())
```

**Contact Frame**: Contact forces are expressed in the **contact frame**.
- Component 0: normal force (perpendicular to contact surface)
- Components 1-2: tangential forces (friction)
- Components 3-5: torque components

### 2. Body External Constraint Forces

```python
cfrc_ext = env.get_cfrc_ext()  # shape: (nbody, 6)
# Each row: [mx, my, mz, fx, fy, fz] — external constraint force acting on each body
# (layout is [torque(3), force(3)], linear force is in [:, 3:])

# Find the body with the largest force
max_idx = np.argmax(np.linalg.norm(cfrc_ext[:, 3:], axis=1))
print(f"Max force: body {max_idx}, force={cfrc_ext[max_idx, 3:]}")
```

### 3. Applying External Forces

```python
env.apply_body_force(
    "g1_pelvis",                          # body name
    force=np.array([0.0, 0.0, 500.0]),    # force (N), world frame
    torque=np.array([0.0, 0.0, 0.0]),     # torque (N·m)
)
```

**Principle**: Directly writes force/torque into MuJoCo's `xfrc_applied` array.
The force acts at the body's center of mass, and the torque acts about the body's center of mass. These forces participate in the dynamics computation on the next `mj_step()`.

**Verification**: `env.data.xfrc_applied[body_id, :3]` — the first 3 components are the force.

**Clearing**:
```python
env.clear_body_force("body_name")   # Clear a single body
env.clear_all_forces()              # Clear all external forces
```

### 4. Physics of External Forces

When applying a force at a site point, the equivalent force and torque at the body center are:

- **Force unchanged**: F_body = F
- **Additional torque**: τ = r × F (r = site_pos - body_pos)
- **Total torque**: τ_total = r × F + τ_user

This means applying a force at a site produces an additional torque, equivalent to applying the same force at the body center plus the moment arm torque.

### 5. Contacts in Reward Functions

```python
def contact_reward(env):
    """Reward moderate contact forces"""
    contacts = env.query_contact_simple()
    if not contacts:
        return -1.0          # No contact = penalty

    contact_ids = list(range(len(contacts)))
    forces = env.query_contact_force(contact_ids)
    total_force = sum(np.linalg.norm(f[:3]) for f in forces.values())

    if total_force < 100:    return 0.5   # Light contact
    elif total_force < 500:  return 1.0   # Ideal contact
    else:                    return -0.5  # Excessive force
```

---

## API Quick Reference

| Operation | API | Description |
|------|-----|------|
| Get contact list | `env.query_contact_simple()` | Returns `list[dict]`, keys are lowercase |
| Get contact forces | `env.query_contact_force(ids)` | 6D force, contact frame |
| Get constraint forces | `env.get_cfrc_ext()` | (nbody, 6), world frame |
| Apply external force | `env.apply_body_force(name, f, τ)` | World frame |
| Clear body force | `env.clear_body_force(name)` | Clear a specific body |
| Clear all | `env.clear_all_forces()` | Clear all external forces |

---

## Next Steps

Now that you understand contacts and forces, learn how to **grasp objects with equality constraints**: [🔗 Equality Constraints](equality-constraints.md).
