# 🧬 Core API

Core simulation interface, encapsulating the MuJoCo physics engine and model management functionality.

## Main Classes

| Class | Description |
|----|------|
| `OrcaGymModel` | Static model information (geometry, joints, actuators, etc.) |
| `OrcaGymDataView` | Read-only view of simulation state (qpos, qvel, etc.) |
| `SimConfig` | Solver parameter configuration |

---

## OrcaGymModel — Static Model Information

Contains all information that does not change during simulation, accessible via `env.model`.

### Dimension Properties

```python
model.nq: int        # qpos length
model.nv: int        # qvel/qacc length (degrees of freedom)
model.nu: int        # Number of actuators
model.ngeom: int     # Number of geometries
model.neq: int       # Number of equality constraints
model.nmocap: int    # Number of mocap bodies
```

### Entity Type Terminology

| Entity | Description |
|------|------|
| **Body** | Rigid body, the basic unit of physics simulation. Has mass, inertia, position, and orientation. |
| **Joint** | Joint, a constraint connecting bodies. Defines relative motion (rotation/sliding/free). |
| **Actuator** | Actuator, the element that drives the robot (motors, etc.). Corresponds to the action space dimension. |
| **Geom** | Geometry, a geometric shape used for collision detection. |
| **Site** | Marker point, does not participate in physics simulation. Used to mark key positions. |
| **Sensor** | Sensor, a virtual device that measures physical quantities. |
| **Equality** | Equality constraint, enforces a specific relationship between two bodies. Commonly used for grasping. |
| **Mocap Body** | Virtual body, freely movable without physics constraints. |

### Name-ID Mappings

| Entity | name→id | id→name | Get All Info |
|------|---------|---------|------------|
| Body | `body_name2id(n)` | `body_id2name(i)` | `get_body_dict()` |
| Joint | `joint_name2id(n)` | `joint_id2name(i)` | `get_joint_dict()` |
| Actuator | `actuator_name2id(n)` | `actuator_id2name(i)` | `get_actuator_dict()` |
| Geom | `geom_name2id(n)` | `geom_id2name(i)` | `get_geom_dict()` |
| Site | `site_name2id(n)` | `site_id2name(i)` | `get_site_dict()` |
| Sensor | `sensor_name2id(n)` | `sensor_id2name(i)` | `gen_sensor_dict()` |
| Mesh | `mesh_name2id(n)` | `mesh_id2name(i)` | `get_mesh_dict()` |

### Other Queries

```python
def get_body_names()
def get_actuator_ctrlrange() -> np.ndarray    # (nu, 2) control range
def get_joint_qposrange(joint_names) -> np.ndarray
def get_eq_list() -> list
def get_geom_body_name(geom_id: int) -> str
def get_geom_body_id(geom_id: int) -> int

# mocap: initialized via init_mocap_dict(mocap_dict),
# count obtained via the nmocap attribute (there is no get_mocap_dict method)
```

### Get Entity by ID / by Name

```python
def get_body_byid(id: int) -> dict
def get_body_byname(name: str) -> dict
def get_joint_byid(id: int) -> dict
def get_joint_byname(name: str) -> dict
def get_actuator_byid(id: int) -> dict
def get_actuator_byname(name: str) -> dict
def get_geom_byid(id: int) -> dict
def get_geom_byname(name: str) -> dict
def get_sensor(name_or_id: Union[str, int]) -> dict | None
def get_site(name_or_id: Union[str, int]) -> dict | None
def get_mesh_byid(id: int) -> dict
def get_mesh_byname(name: str) -> dict | None
```

### Equality Constraint Type Constants

```python
model.mjEQ_CONNECT = 0      # Two bodies connected at a point (ball joint)
model.mjEQ_WELD = 1         # Weld constraint (fixed relative position and orientation)
model.mjEQ_JOINT = 2        # Joint coupling
model.mjEQ_TENDON = 3       # Tendon coupling
model.mjEQ_FLEX = 4         # Flex edge length constraint
model.mjEQ_DISTANCE = 5     # Distance constraint (not recommended)
```

Used for the `eq_type` parameter of `equality_update()`.

### Flex Body Queries

```python
def get_flex_info_by_body_id(body_id: int) -> Optional[Tuple[int, int]]   # (flex_id, local_vertex_index)
def parse_flex_vertex_name(body_name: str) -> Optional[Tuple[str, int]]   # (flex_name, global_vertex_index)
def resolve_flex_body_name(body_name: str) -> Optional[FlexBodyInfo]      # Unified parsing of all flex types
```

---

## OrcaGymDataView — Read-Only View of Simulation State

Accessed via `env.data`. Automatically updated to the latest state after `do_simulation()`.

### Basic State Fields

```python
qpos: np.ndarray       # (nq,)  Generalized coordinates
qvel: np.ndarray       # (nv,)  Generalized velocity
qacc: np.ndarray       # (nv,)  Generalized acceleration
qfrc_bias: np.ndarray  # (nv,)  Bias forces (gravity + Coriolis + centrifugal)
time: float            # Simulation time (seconds)
```

### Extended Fields

```python
xfrc_applied: np.ndarray    # External forces (read-only, use apply_body_force to write)
actuator_force: np.ndarray  # Actuator forces
contact: list               # Contact list
cfrc_ext: np.ndarray        # External constraint forces (nbody, 6)
```

### Body Queries (by name, no ID needed)

```python
def body_xpos(body_name: str) -> np.ndarray       # World-coordinate position (3,)
def body_xquat(body_name: str) -> np.ndarray      # Quaternion [w,x,y,z] (4,)
def body_xmat(body_name: str) -> np.ndarray       # Rotation matrix in flat storage (9,)
def body_cvel(body_name: str) -> np.ndarray       # Spatial velocity [ang(3), lin(3)] (6,)
def body_subtree_mass(body_name: str) -> float    # Total subtree mass
```

### Site Queries

```python
def site_xpos(site_name: str) -> np.ndarray       # World-coordinate position (3,)
def site_xmat(site_name: str) -> np.ndarray       # Rotation matrix in flat storage (9,)
```

### Geom Queries

```python
def geom_xpos(geom_name: str) -> np.ndarray       # World-coordinate position (3,)
def geom_xmat(geom_name: str) -> np.ndarray       # Rotation matrix in flat storage (9,)
def geom_size(geom_name: str) -> np.ndarray       # Size (3,)
```

### Mocap Queries

```python
def mocap_pos(body_name: str) -> np.ndarray       # Mocap position (3,)
def mocap_quat(body_name: str) -> np.ndarray      # Mocap quaternion [w,x,y,z] (4,)
```

---

## SimConfig — Solver Configuration

Accessed via `env.sim_config`. Changes take effect at the next simulation step.

### Properties

| Property | Type | Description |
|------|------|------|
| `timestep` | `float` | Physics time step |
| `integrator` | `int` | Integrator type (0=Euler, 1=RK4, 2=IMPLICIT, 3=IMPLICITFAST) |
| `iterations` | `int` | Solver iteration count |
| `gravity` | `np.ndarray` | Gravity vector, shape (3,) |

### Methods

```python
def load_from_dict(config: dict) -> None    # Set parameters in batch
def to_dict() -> dict                       # Export the configuration as a dict
```

### Usage Example

```python
env.sim_config.timestep = 0.002
env.sim_config.iterations = 100
env.sim_config.load_from_dict({"integrator": 0, "iterations": 100})
```

---

## Auxiliary Enums and Functions

### AnchorType

Module path: in the Euler system, this is located at `orca_gym.core.euler.orca_studio_bridge`;
in the Local/Warp system, it is located at `orca_gym.core.orca_gym_local`.

```python
class AnchorType:
    NONE = 0   # No anchoring
    WELD = 1   # Weld anchoring (fully fixed position and orientation)
    BALL = 2   # Ball joint anchoring (fixed position, allows rotation)
```

### CaptureMode

Module path: only defined in the Local/Warp system (`orca_gym.core.orca_gym_local`) and
`orca_gym.core.orca_gym_warp`; the Euler system does not export this enum.

```python
class CaptureMode:
    ASYNC = 0  # Asynchronous video capture
    SYNC = 1   # Synchronous video capture
```

### Utility Functions

Module path: `orca_gym.core.orca_gym_local` (Local/Warp system).
The Euler system does not export these two functions.

```python
def get_qpos_size(joint_type: int) -> int  # Number of elements in qpos for a joint
def get_dof_size(joint_type: int) -> int   # Degrees of freedom for a joint
```
