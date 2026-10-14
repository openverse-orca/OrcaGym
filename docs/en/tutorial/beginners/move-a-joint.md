# 🦾 Making the Robot Move — Controlling a Single Joint

So far we have only been "looking". In this section, we start **making the robot move**. We begin with the simplest case: understanding `qpos`/`qvel` and manually controlling joints.

> This section is explained using the scene and examples from [OrcaPlayground examples/euler/01_hello_euler/](https://github.com/openverse-orca/OrcaPlayground/tree/main/examples/euler/01_hello_euler),
> whose scene XML is [assets/scenes/simple_pendulum.xml](https://github.com/openverse-orca/OrcaPlayground/tree/main/examples/euler/assets/scenes/simple_pendulum.xml).
> The sample code (`simple_env.py`) uses direct index access; this section rewrites it to demonstrate **querying and setting by name**.

The examples in this section **reuse that XML**; the key name mappings are listed below (these names are defined in the XML, and all subsequent reads/writes use them):

| Element | `name` in XML | Meaning |
|---------|---------------|---------|
| body | `pendulum` | pendulum body |
| joint | `hinge` | hinge joint rotating around the Y axis |
| geom | `arm` | pendulum geometry |
| site | `tip` | pendulum tip site |
| actuator | `hinge_motor` | motor actuator (`joint="hinge"`) |

---

## Complete Example: See the Big Picture First

Below is a **directly runnable** complete example demonstrating three ways to control/query joint state.
It **consistently uses names** (rather than indices) to query and set state, making it easier to map the code to the XML.

```python
"""Complete joint control demo: torque drive → set position by name → query by name"""
import numpy as np
from orca_gym.environment.euler.orca_gym_euler_env import OrcaGymEulerEnv

# Names from simple_pendulum.xml
JOINT_NAME   = "hinge"        # <joint name="hinge">
ACTUATOR_NAME = "hinge_motor"  # <motor name="hinge_motor">
BODY_NAME    = "pendulum"     # <body name="pendulum">
SITE_NAME    = "tip"          # <site name="tip">


class JointControlDemo(OrcaGymEulerEnv):
    """Joint control demonstration environment (offline mode, no Studio needed)."""

    def __init__(self, model_xml_path, **kwargs):
        super().__init__(
            frame_skip=kwargs.pop("frame_skip", 5),
            orcagym_addr=kwargs.pop("orcagym_addr", "localhost:50051"),
            agent_names=kwargs.pop("agent_names", ["agent0"]),
            time_step=kwargs.pop("time_step", 0.002),
            model_xml_path=model_xml_path,
            skip_grpc_load=kwargs.pop("skip_grpc_load", True),
            **kwargs,
        )

    # ─── Method 1: Torque drive (through physics) ⭐ Recommended ───
    def demo_torque_drive(self, actuator_name, steps=200):
        """Drive the joint with constant torque, observing its natural motion under gravity + inertia.

        This is the "through physics" approach: torque → acceleration → velocity → position
        """
        # Query the actuator torque range (by name, no need to remember indices)
        ctrlrange = self.model.get_actuator_ctrlrange()
        act_id = self.model.actuator_name2id(actuator_name)
        max_torque = ctrlrange[act_id, 1]
        print(f"Actuator {actuator_name} torque range: "
              f"[{ctrlrange[act_id, 0]:.1f}, {max_torque:.1f}] N·m")

        for i in range(steps):
            ctrl = np.zeros(self.model.nu, dtype=np.float64)

            # First half: positive torque, second half: reverse → observe reciprocating motion
            if i < steps // 2:
                ctrl[act_id] = 0.3 * max_torque   # 30% forward
            else:
                ctrl[act_id] = -0.3 * max_torque  # 30% reverse

            self.do_simulation(ctrl, self.frame_skip)

            if i % 20 == 0:
                # Query joint state by name (returns dict, intuitive mapping)
                qpos = self.query_joint_qpos([JOINT_NAME])
                qvel = self.query_joint_qvel([JOINT_NAME])
                pos = float(qpos[JOINT_NAME][0])
                vel = float(qvel[JOINT_NAME][0])
                print(f"  Step {i:3d}: pos={pos:+.4f} rad, "
                      f"vel={vel:+.4f} rad/s, torque={ctrl[act_id]:+.2f}")

    # ─── Method 2: Set position by name (sine wave, suitable for reset) ───
    def demo_wiggle(self, joint_name, amplitude=0.5, steps=200):
        """Make the joint swing sinusoidally. Direct qpos setting, without going through physics."""
        # First query the initial position by name
        init_pos = float(self.query_joint_qpos([joint_name])[joint_name][0])
        print(f"Joint {joint_name} initial position: {init_pos:.3f} rad")

        for i in range(steps):
            target_angle = amplitude * np.sin(i * 0.1)

            # Canonical write: copy → locate by name → modify → set → forward
            new_qpos = self.data.qpos.copy()
            qpos_addr = self.jnt_qposadr(joint_name)  # look up address by name
            new_qpos[qpos_addr] = target_angle

            self.set_joint_qpos(new_qpos)             # full write
            self.set_joint_qvel(np.zeros(self.model.nv))
            self.mj_forward()                         # update derived quantities
            self._sync_view()                          # sync DataView

            if i % 20 == 0:
                # Verify by querying by name again
                actual = float(self.query_joint_qpos([joint_name])[joint_name][0])
                print(f"  Step {i:3d}: target={target_angle:+.3f}, "
                      f"actual={actual:+.3f}")

    # ─── Method 3: Query body / site pose by name ───
    def demo_query_body_site(self, body_name, site_name):
        """Demonstrate querying body / site by name (names also come from XML)."""
        # Body pose (returns dict, key is the body name from XML)
        body_pose = self.get_body_xpos_xmat_xquat([body_name])
        bp = body_pose[body_name]
        print(f"Body '{body_name}':")
        print(f"  position: {bp['xpos']}")
        print(f"  quaternion: {bp['xquat']}")

        # Site pose
        site_pose = self.query_site_pos_and_mat([site_name])
        sp = site_pose[site_name]
        print(f"Site '{site_name}':")
        print(f"  position: {sp['xpos']}")

    # ─── Utility: print qpos layout ───
    def print_qpos_layout(self):
        """Print the qpos layout to understand how many elements each joint occupies."""
        offset = 0
        for name in self.model.get_joint_dict().keys():
            # Look up each joint's starting address in qpos by name
            qpos_addr = self.jnt_qposadr(name)
            # Different joint types have different qpos lengths (hinge/slide=1, ball=4, free=7)
            info = self.model.get_joint_byname(name)
            nq = 1  # default hinge/slide
            print(f"  qpos[{qpos_addr:2d}:{qpos_addr+nq:2d}]  {name}  (nq={nq})")
            offset = qpos_addr + nq

    # ─── Gymnasium interface ───
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
        # Read by name and assemble the observation vector
        qpos = self.query_joint_qpos([JOINT_NAME])[JOINT_NAME]
        qvel = self.query_joint_qvel([JOINT_NAME])[JOINT_NAME]
        return np.concatenate([qpos, qvel]).astype(np.float32)


if __name__ == "__main__":
    env = JointControlDemo(
        model_xml_path="tests/orca_gym/environment/euler/fixtures/simple_pendulum.xml",
        skip_grpc_load=True,  # offline mode
    )
    env.reset()

    print("=" * 50)
    print("1. Torque drive (through physics)")
    print("=" * 50)
    env.demo_torque_drive(actuator_name=ACTUATOR_NAME, steps=100)

    print("\n" + "=" * 50)
    print("2. Direct position set (sine wave)")
    print("=" * 50)
    env.demo_wiggle(joint_name=JOINT_NAME, amplitude=0.5, steps=100)

    print("\n" + "=" * 50)
    print("3. Body / Site query")
    print("=" * 50)
    env.demo_query_body_site(body_name=BODY_NAME, site_name=SITE_NAME)

    print("\n" + "=" * 50)
    print("4. qpos layout")
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

> **Names vs indices**: every `<joint>`, `<body>`, `<site>`, `<actuator>`, and `<sensor>` in the XML has a `name` attribute.
> All of OrcaGym's query APIs support name-based access (`query_joint_qpos(names)`, `get_body_xpos_xmat_xquat(names)`, etc.),
> so you don't have to remember indices that may change. In "State Writing" below, first call `jnt_qposadr(name)` to get the address, then modify the qpos copy.

### Method 1: Torque Drive (Recommended) ⭐

```python
ctrl = np.zeros(env.model.nu)
ctrl[act_id] = 0.3 * max_torque   # apply 30% of max torque
env.do_simulation(ctrl, env.frame_skip)
```

**Principle**: torque → acceleration → velocity → position. This is the "through physics" approach — the joint moves
naturally under gravity, inertia, friction, and other physical effects, rather than teleporting to a target position.

**Use case**: normal simulation control and RL training. This is the **recommended standard approach**.

### Method 2: Set Position by Name (Suitable for Reset)

```python
qpos = env.data.qpos.copy()            # 1. Copy the current qpos
addr = env.jnt_qposadr("hinge")        # 2. Look up the starting address by name
qpos[addr] = target_angle              # 3. Modify the copy
env.set_joint_qpos(qpos)               # 4. Full write (canonical)
env.mj_forward()                       # 5. Required! Update derived quantities
env._sync_view()                        # 6. Sync to DataView
```

> ⚠️ **This method does NOT go through physics!** The joint teleports to the target angle without experiencing acceleration/deceleration.
> Use case: **resetting the environment** (quickly setting initial pose), debugging.
>
> ⚠️ On the Euler path, `set_joint_qpos(qpos)` accepts the **full qpos array** (not a name-keyed dict).
> To change only one joint, still copy the full qpos first, locate the change via `jnt_qposadr(name)`, then write the whole array back.

### Method 3: Query body / site by Name

```python
# Body pose (name from <body name="pendulum"> in the XML)
body_pose = env.get_body_xpos_xmat_xquat(["pendulum"])
# → {"pendulum": {"xpos": ..., "xmat": ..., "xquat": ...}}

# Site pose (name from <site name="tip"> in the XML)
site_pose = env.query_site_pos_and_mat(["tip"])
# → {"tip": {"xpos": ..., "xmat": ...}}
```

Body, Site, Sensor, and Actuator names all come from the `name` attribute in the XML;
use names directly as keys in queries and the code reads clearly at a glance.

### The Golden Rule of State Writing

```
1. copy()                   ← copy the current qpos (data.qpos is a read-only zero-copy view)
2. jnt_qposadr(name)         ← look up the starting address by name
3. Modify the corresponding slice in the copy
4. set_joint_qpos(qpos_copy) ← full canonical write
5. mj_forward()              ← required! Update derived quantities
6. _sync_view()              ← sync to DataView
```

Skipping step 5 → body poses and sensor readings will still hold the old values.

### Safety Tips

- Setting excessively large joint angles may cause **self-collision**
- Setting excessively large torques may cause simulation **instability** (numerical explosion)
- Test with small amplitudes first (within ±0.5 rad)
- There are no consequences for breaking things in simulation — feel free to experiment!

---

## Next Step

Now you can control joints. Next, learn how to **write a PD controller**: [🎮 Simple Controller](simple-controller.md).
