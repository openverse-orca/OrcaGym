# 🦾 Making the Robot Move — Controlling a Single Joint

So far we have only been "looking". In this section, we start **making the robot move**. We begin with the simplest case: understanding `qpos`/`qvel` and manually controlling joints.

> See [OrcaPlayground examples/euler/01_hello_euler/](https://github.com/OrcaGym/OrcaPlayground) for complete runnable code.

---

## Complete Example: See the Big Picture First

Below is a **runnable** complete example demonstrating three ways to control and query joint state.
We suggest reading through it once, then reviewing the section-by-section explanation that follows.

```python
"""Joint control complete demo: torque drive -> direct position set -> set by name."""
import time
import numpy as np
from gymnasium import spaces
from orca_gym.environment.euler.orca_gym_euler_env import OrcaGymEulerEnv


class JointControlDemo(OrcaGymEulerEnv):
    """Joint control demonstration environment."""

    def __init__(self, model_xml_path, **kwargs):
        super().__init__(
            frame_skip=kwargs.pop("frame_skip", 5),
            orcagym_addr=kwargs.pop("orcagym_addr", "localhost:50051"),
            agent_names=kwargs.pop("agent_names", ["agent0"]),
            time_step=kwargs.pop("time_step", 0.002),
            model_xml_path=model_xml_path,
            **kwargs,
        )

    # --- Method 1: Torque drive (through physics) ⭐ Recommended ---
    def demo_torque_drive(self, joint_index=0, steps=200):
        """Drive a joint with constant torque, observing its natural motion under gravity + inertia.

        This is the "through physics" approach: torque -> acceleration -> velocity -> position
        """
        ctrlrange = self.model.get_actuator_ctrlrange()
        max_torque = ctrlrange[joint_index, 1]
        print(f"Joint {joint_index} torque range: "
              f"[{ctrlrange[joint_index, 0]:.1f}, {max_torque:.1f}] N*m")

        for i in range(steps):
            ctrl = np.zeros(self.model.nu, dtype=np.float64)

            # First half: positive torque, second half: reverse -> observe reciprocating motion
            if i < steps // 2:
                ctrl[joint_index] = 0.3 * max_torque   # 30% positive
            else:
                ctrl[joint_index] = -0.3 * max_torque  # 30% reverse

            self.do_simulation(ctrl, self.frame_skip)

            if i % 20 == 0:
                pos = self.data.qpos[joint_index]
                vel = self.data.qvel[joint_index]
                print(f"  Step {i:3d}: pos={pos:+.4f} rad, "
                      f"vel={vel:+.4f} rad/s, torque={ctrl[joint_index]:+.2f}")

    # --- Method 2: Direct position set (sine wave, suitable for reset) ---
    def demo_wiggle(self, joint_index=0, amplitude=0.5, steps=200):
        """Make the joint swing sinusoidally. Direct qpos setting, without going through physics."""
        print(f"Joint {joint_index} initial position: {self.data.qpos[joint_index]:.3f} rad")

        for i in range(steps):
            target_angle = amplitude * np.sin(i * 0.1)

            # Canonical write: copy -> modify -> set -> forward
            new_qpos = self.data.qpos.copy()
            new_qpos[joint_index] = target_angle
            self.set_joint_qpos(new_qpos)
            self.set_joint_qvel(np.zeros(self.model.nv))
            self.mj_forward()
            self._sync_view()

            if i % 20 == 0:
                actual = self.data.qpos[joint_index]
                print(f"  Step {i:3d}: target={target_angle:+.3f}, "
                      f"actual={actual:.3f}")

    # --- Method 3: Set position by name ---
    def demo_set_named_joint(self, joint_name, target_angle):
        """Set position by joint name (rather than index)."""
        qpos = self.data.qpos.copy()
        qpos_addr = self.jnt_qposadr(joint_name)
        qpos[qpos_addr] = target_angle

        self.set_joint_qpos(qpos)
        self.mj_forward()
        self._sync_view()

        # Verify
        actual = self.query_joint_qpos([joint_name])[joint_name]
        print(f"{joint_name}: target={target_angle:.3f}, actual={actual[0]:.3f}")

    # --- Utility: print qpos layout ---
    def print_qpos_layout(self):
        """Print the qpos layout to understand how many elements each joint occupies."""
        offset = 0
        for i in range(len(self.model.get_joint_dict())):
            name = self.model.joint_id2name(i)
            info = self.model.get_joint_byname(name)
            nq = info.get("NQ", 1)
            print(f"  qpos[{offset:2d}:{offset+nq:2d}]  {name}  (nq={nq})")
            offset += nq

    # --- Gymnasium interface ---
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
    env = JointControlDemo(
        model_xml_path="/path/to/scene.xml",
        skip_grpc_load=True,  # offline mode
    )
    env.reset()

    print("=" * 50)
    print("1. Torque drive (through physics)")
    print("=" * 50)
    env.demo_torque_drive(joint_index=0, steps=100)

    print("\n" + "=" * 50)
    print("2. Direct position set (sine wave)")
    print("=" * 50)
    env.demo_wiggle(joint_index=0, amplitude=0.5, steps=100)

    print("\n" + "=" * 50)
    print("3. qpos layout")
    print("=" * 50)
    env.print_qpos_layout()

    env.close()
```

---

## Section-by-Section Explanation

### Core Concept: qpos and qvel

MuJoCo uses two arrays to describe the entire simulation world:

```
qpos = [joint0 angle, joint1 angle, ..., free body pose(xyz+qwxyz)]
       length = model.nq (number of generalized coordinates)

qvel = [joint0 angular velocity, joint1 angular velocity, ..., free body velocity(v+omega)]
       length = model.nv (number of degrees of freedom)
```

Different joint types occupy different numbers of elements in qpos:

| Joint Type | qpos Elements | Meaning |
|------------|---------------|---------|
| `hinge` (revolute) | 1 | Rotation angle (radians) |
| `slide` (prismatic) | 1 | Sliding distance (meters) |
| `ball` (spherical) | 4 | Quaternion [w, x, y, z] |
| `free` | 7 | [x, y, z, qw, qx, qy, qz] |

### Method 1: Torque Drive (Recommended) ⭐

```python
ctrl = np.zeros(env.model.nu)
ctrl[joint_index] = 0.3 * max_torque   # apply 30% of max torque
env.do_simulation(ctrl, env.frame_skip)
```

**Principle**: torque -> acceleration -> velocity -> position. This is the "through physics" approach — the joint moves naturally under gravity, inertia, friction, and other physical effects, rather than teleporting to a target position.

**Use case**: Normal simulation control, RL training. This is the **recommended standard approach**.

### Method 2: Direct Position Set (Suitable for Reset)

```python
qpos = env.data.qpos.copy()        # 1. Copy
qpos[joint_index] = target_angle   # 2. Modify the copy
env.set_joint_qpos(qpos)           # 3. Canonical write
env.mj_forward()                   # 4. Required! Update derived quantities
env._sync_view()                   # 5. Sync to DataView
```

> ⚠️ **This method does NOT go through physics!** The joint teleports to the target angle without experiencing acceleration/deceleration.
> Use case: **Resetting the environment** (quickly setting initial pose), debugging.

### Method 3: Set Position by Name

```python
qpos = env.data.qpos.copy()
addr = env.jnt_qposadr("robot_0_joint1")  # look up address by name
qpos[addr] = target_angle
env.set_joint_qpos(qpos)
env.mj_forward()
```

Use this when you know the joint **name** (rather than its index). `jnt_qposadr(name)` returns the starting address of that joint in the qpos array.

### The Golden Rule of State Writing

```
1. copy()                            <- copy current qpos (data.qpos is a read-only zero-copy view)
2. Modify the copy
3. set_joint_qpos(qpos_copy)         <- canonical write
4. mj_forward()                      <- required! Update derived quantities
5. _sync_view()                      <- sync to DataView
```

Skipping step 4 -> body poses and sensor readings will still hold the old values.

### Safety Tips

- Setting excessively large joint angles may cause **self-collision**
- Setting excessively large torques may cause simulation **instability** (numerical explosion)
- Test with small amplitudes first (within +/-0.5 rad)
- There are no consequences for breaking things in simulation — feel free to experiment!

---

## Next Step

Now you can control joints. Next, learn how to **write a PD controller**: [🎮 Simple Controller](simple-controller.md).
