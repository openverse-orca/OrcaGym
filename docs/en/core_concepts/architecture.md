# ⚙️ OrcaGym Euler Architecture

> 📌 **Prerequisite Reading**: This document covers overall layering, API boundaries, and component design details. If this is your first time encountering the OrcaGym architecture, we recommend first reading [Model / Data / Config](model-data-opt.md) and [Gymnasium Interface](gym-interface.md) to build a basic understanding, then return to this document for architecture design.

## 1. Why a New Architecture Is Needed

### 1.1 Problems with the Existing System

`OrcaGymLocalEnv` + `OrcaGymLocal`, as the current primary path, suffers from three categories of structural defects:

| Defect Type | Manifestation | Consequence |
|---------|------|------|
| **God Class** | `OrcaGymLocal` alone shoulders simulation core, Studio integration, model registration, solver configuration, object manipulation, and all other responsibilities | Hard to maintain, hard to extend, coupled responsibilities |
| **Incomplete API** | `OrcaGymData` only caches 5 fields (qpos/qvel/qacc/qfrc_bias/time), missing xfrc_applied/cvel/contact, etc. | You are forced to bypass through `gym._mjData` for direct access |
| **Encapsulation Leak** | `_mjModel`/`_mjData` exposed as public attributes, `self.gym` is both an internal component and an external library handle | 83 direct accesses, encapsulation exists in name only |

### 1.2 Current State of User Code

Analysis of the OrcaPlayground and OrcaManipulation repositories reveals:

- **17 direct subclasses** + **4 indirect subclasses** inheriting from `OrcaGymLocalEnv`
- **83 instances** of direct access to `gym._mjData` / `gym._mjModel`
- Typical bypass scenarios: external force injection (`xfrc_applied`), solver configuration (`opt.*`), body property queries (`body_subtreemass`/`cvel`), equality constraint structure access (`eq_data`)

You bypass not because you do not understand encapsulation, but because **the encapsulation does not cover your needs**.

### 1.3 What the New Architecture Does

`OrcaGymEulerEnv` + `OrcaGymEuler` adopts a **Facade + responsibility-cohesive decomposition** design, replacing the god-class pattern of `OrcaGymLocalEnv` + `OrcaGymLocal`:

- **Complete public API**: Covers all legitimate MuJoCo operation needs, so you no longer need to bypass to internal data
- **Encapsulation isolation**: `_mjModel`/`_mjData` are not exposed externally, mechanically guiding you and AI down the correct path
- **Smooth migration**: ~70% zero-change API, 25% mechanical replacement, 5% design adjustment

---

## 2. Core Design Principles

### 2.1 Six Principles

| Principle | Meaning | Compared to Old System |
|------|------|-----------|
| **P1 Completeness** | Public API covers all legitimate MuJoCo operation needs | Many gaps force bypassing |
| **P2 No Engine Internals Exposed** | `_mjModel`/`_mjData` not exposed as public attributes | Directly exposed |
| **P3 State Consistency** | After any write operation, `self.data` is guaranteed consistent; all reads go through `self.data` or explicit queries | `self.data` and `_mjData` dual-track system |
| **P4 Traceable Force Application** | External force injection via explicit methods; future Euler coupler can perceive it | `xfrc_applied` written directly, no perception |
| **P5 Responsibility Cohesion** | Modules divided by cohesive responsibilities; a group of methods change for the same reason and share the same data | God class |
| **P6 Framework Stateless, Business Self-Orchestrated** | Framework only provides stateless primitives (single atomic reads/writes); multi-step orchestration flows are composed by your business code and you manage your own state | Framework mixes primitives and orchestration, prone to misuse |

#### P6 Supplementary Explanation

All framework public APIs are **stateless primitives** -- a single call completes a single data read/write, does not depend on previous or subsequent call order, and does not hold snapshots or binding markers across calls. For example:

- `equality_find_slot_by_body` — find slot by body
- `equality_constraint(slot)` — read a single slot
- `equality_update(slot, **fields)` — write a single slot
- `set_mocap_pos_and_quat` — write mocap pose

If your business requires multi-step orchestration flows such as "bind/release/grasp", compose these primitives yourself and manage your own business state. This is easier to review and less error-prone than framework-managed state.

### 2.2 Design Patterns

| Pattern | Applied At | Problem Solved |
|------|---------|-----------|
| **Facade** | `OrcaGymEulerEnv` / `OrcaGymEuler` | Combine multiple subcomponents, provide unified API, avoid god class |
| **Composition over Inheritance** | Env holds Gym, Gym holds subcomponents | Avoid inheritance chain rot, responsibilities can evolve independently |
| **Strategy Pattern** | `OrcaGymEuler._euler` field (placeholder) | Currently always None; switching between with/without Euler strategies is encapsulated via `has_euler()` / `step_with_coupling()` (Euler orchestrator design TBD) |
| **Dependency Inversion** | `OrcaStudioBridge` does not hold mjData, achieves decoupling by receiving data parameters | Studio integration decoupled from simulation core |
| **Read-Only View** | `OrcaGymDataView` | Provide complete state reads, prohibit writes |

---

## 3. Architecture Overview

### 3.1 Overall Structure

```
gym.Env
  └── OrcaGymEulerEnv (new)                    (Facade + contract executor, directly inherits gym.Env)
        │   ↑ OrcaGymEnvMixin (namespace, action/observation space generation, reset orchestration)
        │
        │   Composition (not inheritance)
        ├── _gym: OrcaGymEuler           (Simulation core Facade)
        │     ├── _sim: MuJoCoSimCore    # Holds _mjModel/_mjData (not exposed externally)
        │     ├── _studio: OrcaStudioBridge  # gRPC integration
        │     ├── _registry: ModelRegistry  # Model information
        │     ├── _opt: SimConfig        # Solver configuration (typed)
        │     └── _euler: None  # Euler coupling placeholder (currently unimplemented, design TBD)
        │
        │   Public API (the interface you face)
        ├── .data → OrcaGymDataView      # Complete state view
        ├── .model → OrcaGymModel        # Model structure (reused as-is)
        ├── .sim_config → SimConfig      # Solver configuration
        ├── .ctrl → np.ndarray           # Control array
        │
        ├── Simulation control
        ├── State query
        ├── State setting
        ├── Namespace
        └── Studio interaction
```

### 3.2 Comparison with the Old System

| Dimension | OrcaGymLocalEnv + OrcaGymLocal | OrcaGymEulerEnv + OrcaGymEuler |
|------|-------------------------------|-------------------------------|
| Class Structure | God class, single class bears all responsibilities | Facade + responsibility-cohesive decomposition |
| `_mjModel`/`_mjData` | Public attributes, 83 direct accesses | Internal components, multi-layer isolation |
| `OrcaGymData` | 5-field cache, incomplete | `OrcaGymDataView` complete read-only view |
| Solver Configuration | No interface, bypass through `opt.*` | `SimConfig` typed configuration |
| External Force Injection | Directly write `xfrc_applied` | `apply_body_force()` explicit method |
| Inheritance System | Inherits from corrupted `OrcaGymBase` chain | Directly inherits `gym.Env` + `OrcaGymEnvMixin` |

### 3.3 Coexistence with the Old System

The existing `OrcaGymBase` → Remote / Local / Warp inheritance system remains untouched; existing systems continue to run unaffected. `OrcaGymEulerEnv` is an independent new class that coexists with `OrcaGymLocal` long-term. On the migration path, `OrcaGymLocal` will eventually be deprecated, at which point the old system can simply be deleted.

---

## 4. Component Descriptions

### 4.1 OrcaGymEulerEnv — Environment Facade

Implements the Gymnasium `Env`, composing the `OrcaGymEuler` simulation core, exposing a unified API to you. Inheritance structure: `OrcaGymEulerEnv(OrcaGymEnvMixin, gym.Env)`.

```python
class OrcaGymEulerEnv(OrcaGymEnvMixin, gym.Env):
    """OrcaGym Euler dual-engine environment.

    Usage contract:
        Read state:    env.data.qpos / env.data.body_xpos(name) / env.query_*()
        Write state:   env.set_joint_qpos() / env.apply_body_force()
        Simulation:    env.do_simulation(ctrl, n_frames)
        Solver config: env.sim_config.timestep = 0.002

    Prohibited:
        Do not access env._gym._sim._mjData or any internal MuJoCo objects.
        env.gym/env.stub/env.channel do not exist; directly inheriting gym.Env does not create these attributes.
        If functionality is missing, extend this class's public methods.
    """
```

**Key Attributes**:

| Attribute | Type | Description |
|------|------|------|
| `data` | `OrcaGymDataView` | Complete state read-only view, replaces `_mjData` reads |
| `model` | `OrcaGymModel` | Model structure information (reused as-is) |
| `sim_config` | `SimConfig` | Solver parameter configuration, replaces direct `opt.*` access |
| `ctrl` | `np.ndarray` | Control array |
| `frame_skip` | `int` | Number of physics steps per `step()` |
| `dt` | `float` | Environment timestep = timestep × frame_skip (`timestep` is the single physics step time) |

### 4.2 OrcaGymEuler — Simulation Core Facade

Composes simulation subcomponents, providing simulation operation interfaces to `OrcaGymEulerEnv`. Holds `MuJoCoSimCore`, `OrcaStudioBridge`, `ModelRegistry`, `SimConfig`, and the `_euler` placeholder field (currently `None`, Euler coupling orchestrator design TBD). **Does not expose** `_mjModel`/`_mjData`, relies on `_` prefix convention + ruff SLF001 static checking, controls IDE autocompletion via `__dir__` to show only public API.

```python
class OrcaGymEuler:
    """Dual-engine orchestration core.

    ┌─────────────────────────────────────────────────────────────┐
    │  API Contract: You should not directly access _mjData /     │
    │  _mjModel.                                                  │
    │  Read MuJoCo state → use env.data (OrcaGymDataView)         │
    │  Write external forces → use env.apply_body_force()          │
    │  Configure solver → use env.sim_config                      │
    │  Missing functionality → extend OrcaGymEulerEnv public      │
    │  methods                                                    │
    └─────────────────────────────────────────────────────────────┘
    """
```

### 4.3 MuJoCoSimCore — Simulation Core

Holds `_mjModel`/`_mjData`, executes pure MuJoCo operations such as `mj_step`/`mj_forward`/`set_ctrl`. `_mjModel`/`_mjData` exist only inside this class and are not exposed externally.

```python
class MuJoCoSimCore:
    def __init__(self):
        self._mjModel: mujoco.MjModel | None = None
        self._mjData: mujoco.MjData | None = None

    def init_simulation(self, model_xml_path: str) -> None: ...
    def step(self, nstep: int) -> None: ...
    def forward(self) -> None: ...
    def set_ctrl(self, ctrl: np.ndarray) -> None: ...
    def sync_to_view(self, view: OrcaGymDataView) -> None: ...
    def apply_body_force(self, body_id: int, force: np.ndarray, torque: np.ndarray) -> None: ...
```

### 4.4 OrcaStudioBridge — Studio Integration

Handles gRPC interaction with OrcaStudio, including rendering, video saving, object manipulation, etc. **Dependency inversion** design: does not hold `_mjData`, achieves decoupling by receiving data parameters; does not touch `mj_step`, only handles communication and scene synchronization.

```python
class OrcaStudioBridge:
    def __init__(self, stub=None) -> None: ...
    async def render(self, qpos: np.ndarray, sim_time: float) -> None: ...
    async def load_model_xml(self) -> str: ...
    async def begin_save_video(self, file_path: str, capture_mode) -> None: ...
    async def stop_save_video(self) -> None: ...
    async def get_current_frame(self) -> int: ...
    async def get_body_manipulation_anchored(self) -> tuple: ...
    async def get_body_manipulation_movement(self) -> dict: ...
```

### 4.5 ModelRegistry — Model Registration

Builds `OrcaGymModel`/`OrcaGymData`, provides model information queries such as `query_all_*`. `OrcaGymModel` is a successful abstraction, reused as-is, and extended with missing model structure queries.

```python
class ModelRegistry:
    def __init__(self, mj_model=None) -> None: ...   # When None, must call _bind(mj_model) before use
    def build_orca_gym_model(self) -> OrcaGymModel: ...
    def build_orca_gym_data(self): ...   # Raises NotImplementedError: Euler system uses OrcaGymDataView instead
    def body_subtree_mass(self, body_name: str) -> float: ...
    def equality_data_width(self) -> int: ...
    def equality_object_ids(self, eq_idx: int) -> tuple[int, int]: ...
```

### 4.6 SimConfig — Solver Configuration

Provides a typed read/write interface for MuJoCo solver parameters, replacing direct `_mjModel.opt.*` access. Covers all user-accessible `opt` fields; changes take effect on the next `mj_step`.

```python
class SimConfig:
    """MuJoCo solver parameter configuration. Replaces direct _mjModel.opt.* access.
    Changes take effect on the next mj_step."""

    @property
    def timestep(self) -> float: ...
    @timestep.setter
    def timestep(self, value: float): ...

    @property
    def integrator(self) -> int: ...
    @integrator.setter
    def integrator(self, value: int): ...

    @property
    def iterations(self) -> int: ...
    @iterations.setter
    def iterations(self, value: int): ...

    @property
    def gravity(self) -> np.ndarray: ...
    @gravity.setter
    def gravity(self, value: np.ndarray): ...

    # ... covers all user-accessible opt fields ...

    def load_from_dict(self, config: dict) -> None: ...
```

**Migration Mapping**:

| Old Code | New Code |
|--------|--------|
| `gym._mjModel.opt.timestep = 0.002` | `env.sim_config.timestep = 0.002` |
| `gym._mjModel.opt.iterations = 100` | `env.sim_config.iterations = 100` |
| `gym._mjModel.opt.integrator = 0` | `env.sim_config.integrator = 0` |
| 30 lines of `opt.*` settings | `env.sim_config.load_from_dict({...})` |

### 4.7 OrcaGymDataView — Complete State View

Provides a complete read-only view of the MuJoCo state, replacing direct `_mjData` access. Covers all fields you need to read (not just the 5 from the original `OrcaGymData`). Provides body/site/geom property queries via methods; you access by name without needing to know IDs.

```python
class OrcaGymDataView:
    """Complete read-only view of MuJoCo state.

    Replaces direct _mjData access. All fields are guaranteed consistent after update_data().
    You never need to access _mjData.

    If fields not provided by this view are needed, extend OrcaGymDataView,
    do not bypass through env._gym._sim._mjData.
    """

    # --- Basic State ---
    qpos: np.ndarray
    qvel: np.ndarray
    qacc: np.ndarray
    qfrc_bias: np.ndarray
    time: float

    # --- Extended Fields ---
    xfrc_applied: np.ndarray       # Read-only view (write via apply_body_force)
    actuator_force: np.ndarray     # Actuator forces
    contact: list                  # Contact list

    def body_xpos(self, body_name: str) -> np.ndarray: ...
    def body_xquat(self, body_name: str) -> np.ndarray: ...
    def body_xmat(self, body_name: str) -> np.ndarray: ...
    def body_cvel(self, body_name: str) -> np.ndarray: ...
    def body_subtree_mass(self, body_name: str) -> float: ...

    def site_xpos(self, site_name: str) -> np.ndarray: ...
    def site_xmat(self, site_name: str) -> np.ndarray: ...
```

**Migration Mapping**:

| Old Code | New API |
|--------|--------|
| `gym._mjData.qpos` | `env.data.qpos` |
| `gym._mjData.body(id).xpos` | `env.data.body_xpos(name)` |
| `gym._mjData.cvel[id]` | `env.data.body_cvel(name)` |
| `gym._mjData.xpos[body_id, 2]` | `env.data.body_xpos(name)[2]` |
| `gym._mjData.time` | `env.data.time` |

### 4.8 Euler Coupling — Placeholder (Currently Unimplemented)

Orchestrates the coupled stepping of Euler non-rigid-body solver and MuJoCo rigid-body solver. **Currently a design placeholder, no `EulerOrchestrator` class exists in the code**: `OrcaGymEuler._euler` field is always `None`, `OrcaGymEulerEnv` behaves as a pure MuJoCo environment. Euler coupling toggling is queried via `OrcaGymEuler.has_euler()` and stepping is encapsulated via `OrcaGymEuler.step_with_coupling(ctrl, n_frames, dt)` (when `has_euler()` is `False`, equivalent to pure MuJoCo stepping). Detailed orchestrator design will be discussed in a separate document.

```python
# Actual encapsulation in current code (OrcaGymEuler public API)
def has_euler(self) -> bool:
    """Query whether an Euler coupling orchestrator exists. Returns False during skeleton phase."""
    return self._euler is not None

def step_with_coupling(self, ctrl: np.ndarray, n_frames: int, dt: float) -> None:
    """Stepping with Euler coupling. When has_euler()=False, equivalent to set_ctrl + step."""
    # Skeleton phase: set_ctrl + step (no Euler coupling)
    self._sim.set_ctrl(ctrl)
    self._sim.step(n_frames)
```

### 4.9 OrcaGymEnvMixin — Environment Public Method Mixin

Extracts public methods from `OrcaGymLocalEnv`/`OrcaGymBaseEnv` that are independent of the simulation engine, for use by `OrcaGymEulerEnv`. Currently `OrcaGymLocalEnv` still directly inherits from `OrcaGymBaseEnv` and does not use `OrcaGymEnvMixin` (optional refactor, not mandatory).

```python
class OrcaGymEnvMixin:
    """OrcaGym environment public method Mixin.

    Provides namespace resolution, action/observation space generation,
    reset orchestration and other methods.
    Does not define __init__, does not hold state; subclasses initialize
    _agent_names and other fields themselves.
    """

    # --- Namespace Resolution (automatically adds agent prefix) ---
    def body(self, name: str, agent_id: int = None) -> str: ...
    def joint(self, name: str, agent_id: int = None) -> str: ...
    def actuator(self, name: str, agent_id: int = None) -> str: ...
    def site(self, name: str, agent_id: int = None) -> str: ...
    def mocap(self, name: str, agent_id: int = None) -> str: ...
    def sensor(self, name: str, agent_id: int = None) -> str: ...

    # --- Space Generation ---
    def generate_action_space(self, bounds: np.ndarray) -> Space: ...
    def generate_observation_space(self, obs: Union[Dict, np.ndarray]) -> Space: ...

    # --- Reset Orchestration ---
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None): ...
    def set_seed_value(self, seed: int = None) -> list: ...
    def _get_reset_info(self) -> Dict[str, float]: ...

    # --- Helpers ---
    def _name_with_agent0(self, name: str) -> str: ...
    def _name_with_agent(self, agent_id: int, name: str) -> str: ...
    @property
    def agent_num(self) -> int: ...
```

**Usage**:

```python
# Euler system (currently in use)
class OrcaGymEulerEnv(OrcaGymEnvMixin, gym.Env):
    def __init__(self, ...):
        self._agent_names = agent_names
        self._gym = OrcaGymEuler(...)
        # Mixin methods can be used directly
        ...

# Local system (currently actually inherits OrcaGymBaseEnv, does not use Mixin)
class OrcaGymLocalEnv(OrcaGymBaseEnv):
    ...
```

---

## 5. API Usage Contract

### 5.1 Contract Levels

| Level | Meaning | Violation Consequence |
|------|------|---------|
| **L1 Public API** | Methods and attributes exposed via `__dir__`, intended for your use | Works normally |
| **L2 Internal Components** | `_gym`/`_sim`/`_studio`, etc., you should not access | ruff SLF001 warning |
| **L3 Engine Internals** | `_mjModel`/`_mjData`, you must never access | ruff SLF001 warning |

### 5.2 State Reading

**Rule**: All state reads go through `env.data` (`OrcaGymDataView`) or `env.query_*()` methods. `env.data` is guaranteed consistent after `do_simulation()` returns and after `mj_forward()` returns. `env.data` is a read-only view; write operations must go through explicit methods.

```python
# ✅ Correct
qpos = env.data.qpos
body_pos = env.data.body_xpos("link1")

# ❌ Wrong
qpos = env._gym._sim._mjData.qpos  # ruff SLF001 warning
```

### 5.3 State Writing

**Rule**: All state writes go through explicit methods; do not directly manipulate MuJoCo data structures. External force injection goes through `apply_body_force()`, do not directly write `xfrc_applied`. If you need to immediately read consistent state after a write operation, you must call `mj_forward()`.

```python
# ✅ Correct
env.set_joint_qpos({"joint1": np.array([0.5])})
env.apply_body_force("link1", force, torque)
env.mj_forward()

# ❌ Wrong
env._gym._sim._mjData.xfrc_applied[body_id, :3] = force  # ruff SLF001 warning
```

### 5.4 Simulation Stepping

| Method | Responsibility | Euler Coupling | Applicable Scenario |
|------|------|-----------|---------|
| `do_simulation(ctrl, n)` | Standard stepping | Yes (future) | Most Env `step()` calls |
| `mj_step(n)` | Pure MuJoCo stepping | No | Advanced users needing fine-grained timing control |
| `mj_forward()` | Forward computation | No | Update derived quantities after state setting |

Two usage modes:

```python
# Mode A (recommended, includes Euler coupling)
env.do_simulation(ctrl, self.frame_skip)
# After do_simulation returns, env.data is automatically synchronized (sync_to_view called internally)

# Mode B (pure MuJoCo, no coupling)
for _ in range(self.frame_skip):
    env.set_ctrl(torques)
    env.mj_step(1)
# After the loop, call env.mj_forward() to refresh derived quantities, then read via env.data
```

> Mode B currently behaves identically to OrcaGymLocalEnv. If Euler coupling is needed in the future, Mode B users must switch to Mode A.

### 5.5 Solver Configuration

All `opt.*` parameters are read/written through `env.sim_config`; configuration changes take effect on the next `mj_step`.

```python
# ✅ Correct
env.sim_config.timestep = 0.002
env.sim_config.iterations = 100
env.sim_config.load_from_dict({"integrator": 0, "iterations": 100})

# ❌ Wrong
env._gym._sim._mjModel.opt.timestep = 0.002  # ruff SLF001 warning
```

### 5.6 Namespace

All names are resolved through `env.joint()`/`env.body()`/`env.site()`/`env.actuator()`/`env.sensor()`, automatically adding agent prefixes. This part of the API is completely identical to OrcaGymLocalEnv, with zero-change migration.

```python
joint_name = env.joint("joint1")  # → "agent_name/joint1"
body_name = env.body("object")
```

### 5.7 Complete Public API Reference

| Category | API |
|------|-----|
| **State Reading** | `data` (OrcaGymDataView), `model` (OrcaGymModel), `ctrl`, `frame_skip`, `dt`, `realtime_step` |
| **Simulation Control** | `do_simulation(ctrl, n)`, `mj_step(n)`, `mj_forward()` |
| **State Query** | `query_joint_qpos/qvel/qacc/offsets/lengths()`, `query_site_pos_and_mat()`, `query_site_pos_and_quat_B()`, `query_site_xvalp_xvalr()`, `query_site_xvalp_xvalr_B()`, `query_actuator_torques()`, `query_sensor_data()`, `query_contact_simple()`, `get_body_xpos_xmat_xquat()` |
| **State Setting** | `set_joint_qpos/qvel()`, `set_mocap_pos_and_quat()`, `set_geom_friction()`, `apply_body_force()`, `clear_body_force()`, `clear_all_forces()` |
| **Equality Constraint Primitives (Stateless, L1)** | `equality_find_slot_by_body(body_name)`, `equality_constraint(slot)`, `equality_update(slot, **fields, forward=True)` |
| **Solver Configuration** | `sim_config` (SimConfig) |
| **Namespace** | `joint()`, `body()`, `site()`, `actuator()`, `sensor()` |
| **Studio Interaction** | `render()`, `begin_save_video()`, `stop_save_video()`, `get_current_frame()`, `get_frame_png()` |
| **Lifecycle** | `initialize_simulation()`, `initialize_grpc()`, `pause_simulation()`, `close()` |

> **Studio UI grasping is an internal API**: The original `anchor_actor()` / `release_body_anchored()` / `do_body_manipulation()` have been changed to `_`-prefixed internal methods per the P6 principle, driven internally by `render()`, and do not enter the public API. Programmatic body manipulation should use equality constraint stateless primitives implemented by yourself.

---

## 6. Encapsulation and Isolation

### 6.1 Mechanism Overview

This architecture uses multiple layers of guidance to make the "correct way" the path of least resistance. `OrcaGymEulerEnv` directly inherits `gym.Env`, does not create `gym`/`stub`/`channel` attributes -- Python natively rejects access; other internal objects rely on `_` prefix convention + ruff SLF001 static checking + AGENTS.md AI behavior constraints:

| Mechanism | Implementation | Effect |
|------|------|------|
| **Python Native Attribute Absence** | `OrcaGymEulerEnv` directly inherits `gym.Env`; all internal components assigned in `__init__` are prefixed with underscore (e.g. `_gym`/`_stub`/`_channel`/`_studio_bridge`, etc.), no unprefixed `gym`/`stub`/`channel` attributes are created | `env.gym`/`env.stub`/`env.channel` raise `AttributeError` |
| **ruff SLF001 Static Check** | Configure `ruff check --select SLF001` to scan code accessing `_`-prefixed attributes externally | Detect wall-bypassing access at pre-commit / CI stage |
| **AGENTS.md AI Constraints** | Each in-house repo configures `AGENTS.md` at root, explicitly prohibiting AI from using `_`-prefixed attributes | Constrain AI code generation behavior from the input side |
| **`__dir__` Control** | Env/Gym/DataView implement `__dir__`, only exposing public API | IDE autocompletion guides correct path |
| **DataView Fallback** | `OrcaGymDataView.__getattr__` guides extension when fields are missing | Guides extension rather than bypassing when functionality is missing |
| **Type Annotations** | Public methods return typed objects, never `mujoco.MjData` | AI code generation follows correct path |
| **Docstring Contracts** | Class documentation explicitly lists correct usage and prohibitions | Know the contract just by reading the API |
| **Path Depth** | `_mjData` is three layers deep at `env._gym._sim._mjData` | Natural barrier |

### 6.2 Isolation Effect Comparison

| Scenario | Trigger Mechanism | What You/AI See |
|------|---------|-------------|
| AI generates `env._mjData.qpos` | ruff SLF001 | Pre-commit warning: use `env.data.qpos` |
| AI generates `env._gym._mjData` | ruff SLF001 | Pre-commit warning: use `env.data` |
| AI generates `env._mjModel.opt.iterations` | ruff SLF001 | Pre-commit warning: use `env.sim_config.iterations` |
| AI autocompletes `env.` in IDE | `__dir__` control | Only sees public API |
| AI reads class docstring | Type annotations + contract | Knows correct usage and prohibitions |
| AI commits without running ruff | AGENTS.md constraint + CI gate | CI rejects merge |

### 6.3 Isolation Strength Comparison with Old System

| System | Bypass Path | Depth | Internal Components Visible | Static Check |
|------|---------|------|----------------|---------|
| OrcaGymLocalEnv | `env.gym._mjData` | 2 | `gym` is a public attribute | None |
| OrcaGymEulerEnv | `env._gym._sim._mjData` | 3 | `__dir__` does not list | ruff SLF001 warning |

---

## 7. Step Orchestration

### 7.1 `do_simulation` Internal Flow

```python
def do_simulation(self, ctrl: np.ndarray, n_frames: int):
    """Standard simulation stepping (includes Euler coupling).

    Contract:
    - Set control input → step n_frames times → synchronize state
    - Euler coupling is encapsulated via step_with_coupling (do not write if self._gym._euler is not None)
    - After stepping completes, self.data is guaranteed consistent
    """
    # K8 compliance: do not write if self._gym._euler is not None, encapsulated via step_with_coupling
    self._gym.step_with_coupling(ctrl, n_frames, self.dt)
    self._gym.sync_to_view()
```

> See actual implementation in `orca_gym/environment/euler/orca_gym_euler_env.py` `do_simulation`. `step_with_coupling` is equivalent to `set_ctrl + step` when `has_euler()=False` (current skeleton phase), and will be extended when Euler coupling is implemented.

### 7.2 Two Usage Modes

**Mode A (Delegated, Recommended)**:

```python
def step(self, action):
    torque = self._compute_torque(action)
    self.do_simulation(torque, self.frame_skip)
    obs = self._get_obs()
    return obs, reward, terminated, truncated, info
```

**Mode B (Manual Loop)**:

```python
def step(self, action):
    for _ in range(self.frame_skip):
        torque = self._compute_torque(action)
        self.set_ctrl(torque)
        self.mj_step(nstep=1)
    # Refresh derived quantities after loop, then read env.data
    self.mj_forward()
    obs = self._get_obs()
    return obs, reward, terminated, truncated, info
```

**Contract**: Mode B currently behaves identically to OrcaGymLocalEnv (pure MuJoCo). If Euler coupling is needed in the future, Mode B users must switch to Mode A.

---

## 8. Migration Guide

### 8.1 Migration Cost

| API Category | Migration Difficulty | Notes |
|---------|---------|------|
| Lifecycle and Attributes | Low | `model`/`data`/`ctrl`/`frame_skip`, etc. provided as-is |
| Simulation Stepping (Mode A) | Low | `do_simulation` delegates internally, same signature |
| Simulation Stepping (Mode B) | Medium | `mj_step(1)` behavior note: no Euler coupling |
| State Query | Low | `query_*` methods copied as-is |
| State Setting | Low | `set_*` methods copied as-is + new `apply_body_force` |
| Namespace Resolution | Low | `joint()`/`body()`/`site()`, etc. provided as-is |
| `_mjData`/`_mjModel` Direct Access | **Low** | Formal API replacements available, mechanical substitution |
| Studio Interaction | Low | gRPC logic copied as-is |

**Overall**: ~70% zero-change, 25% mechanical replacement, 5% design adjustment.

### 8.2 Alternatives for the 83 Direct Accesses

#### Read Type (→ OrcaGymDataView)

| Old Code | New API |
|--------|--------|
| `gym._mjData.qpos` | `env.data.qpos` |
| `gym._mjData.qvel` | `env.data.qvel` |
| `gym._mjData.body(id).xpos` | `env.data.body_xpos(name)` |
| `gym._mjData.cvel[id]` | `env.data.body_cvel(name)` |
| `gym._mjData.xpos[body_id, 2]` | `env.data.body_xpos(name)[2]` |
| `gym._mjData.time` | `env.data.time` |

#### Write Type (→ Explicit Methods)

| Old Code | New API |
|--------|--------|
| `gym._mjData.xfrc_applied[id, :3] = f` | `env.apply_body_force(name, f, tau)` |
| `gym._mjData.xfrc_applied[id].fill(0)` | `env.clear_body_force(name)` |
| `gym._mjData.eq_active[gi] = bool` | `env.equality_update(slot, active=bool)` |

#### Configuration Type (→ SimConfig)

| Old Code | New API |
|--------|--------|
| `gym._mjModel.opt.timestep = 0.002` | `env.sim_config.timestep = 0.002` |
| `gym._mjModel.opt.iterations = 100` | `env.sim_config.iterations = 100` |
| `gym._mjModel.opt.integrator = 0` | `env.sim_config.integrator = 0` |
| `gym._mjModel.opt.gravity = ...` | `env.sim_config.gravity = ...` |
| 30 lines of `opt.*` settings | `env.sim_config.load_from_dict({...})` |

#### Model Structure Type (→ env public methods / OrcaGymModel)

| Old Code | New API |
|--------|--------|
| `gym._mjModel.body_subtreemass[id]` | `env.body_subtree_mass(name)` or `env.data.body_subtree_mass(name)` |
| `gym._mjModel.eq_data.shape[1]` | Currently `env.model` has no such public API; use `env._gym.equality_data_width()` or extend `env.model` public methods |
| `gym._mjModel.eq_obj1id[gi]` | Currently `env.model` has no such public API; use `env._gym.equality_object_ids(idx)` or extend `env.model` public methods |
| `gym._mjModel.joint(i).name` | `env.model.joint_id2name(i)` |
| `gym._mjModel.njnt` | `len(env.model.get_joint_dict())` |

### 8.3 Migration Category Examples

**Category 1: Zero-Change (Business Logic Preserved)**

Joint queries, state setting, namespace, rendering, stepping -- API signatures are completely identical.

**Category 2: Mechanical Replacement (`_mjData` → Formal API)**

```python
# Before migration
self.gym._mjData.xfrc_applied[body_id, :3] = force

# After migration
self.apply_body_force(body_name, force, torque)
```

**Category 3: Design Adjustment (Few Cases)**

```python
# Before migration (Local system): manual loop stepping
for _ in range(self.frame_skip):
    self.set_ctrl(torques)
    self.mj_step(nstep=1)
    self.gym.update_data()   # Old API of Local system

# After migration (Euler system): if Euler coupling is needed, switch to do_simulation
# do_simulation internally encapsulates step_with_coupling + sync_to_view, data is synchronized automatically
self.do_simulation(torques, self.frame_skip)
```

---

## 9. Design Decisions

### 9.1 Why Directly Inherit gym.Env + OrcaGymEnvMixin

`OrcaGymEulerEnv` directly inherits `gym.Env`, shares common methods via `OrcaGymEnvMixin`, and does not inherit `OrcaGymBaseEnv` or `OrcaGymLocalEnv`.

**Rationale**:
- `OrcaGymLocalEnv` is a god class; inheriting it would inherit all its coupled responsibilities
- `OrcaGymLocal`'s `_mjModel`/`_mjData` exposure design conflicts with the P2 principle
- `OrcaGymBaseEnv`'s `self.gym`/`self.model`/`self.data` assignments directly conflict with the Euler system, requiring workaround mechanisms like `__setattr__` shielding
- After directly inheriting `gym.Env`, `env.gym`/`env.stub`/`env.channel` naturally do not exist (Python native AttributeError), requiring no interception mechanism
- `OrcaGymEnvMixin` extracts namespace, space generation, reset orchestration, and other engine-independent methods, avoiding code duplication

### 9.2 Why MuJoCoAdapter Was Abandoned

`MuJoCoAdapter` (a controlled MuJoCo handle adapter) is not provided.

**Rationale**: The original need came from external libraries like robosuite controllers requiring direct MuJoCo object manipulation. After deciding not to support robosuite components, this need disappeared. The design is simpler without it -- no "escape hatch"; all needs are met by extending Env/Gym public methods.

### 9.3 Why `_mjData`/`_mjModel` Are Placed in MuJoCoSimCore

`_mjModel`/`_mjData` exist only inside `MuJoCoSimCore`; `OrcaGymEuler` and `OrcaGymEulerEnv` do not hold references.

**Rationale**:
- Increases bypass path depth (`env._gym._sim._mjData` -- three layers)
- Responsibility cohesion: native MuJoCo operations are concentrated in `MuJoCoSimCore`
- `OrcaGymEuler` as a Facade only coordinates, does not directly manipulate engine data

### 9.4 Why Retain the `self.gym` Concept but Rename to `_gym`

`OrcaGymEulerEnv` holds `_gym: OrcaGymEuler`, but does not expose it as a public attribute.

**Rationale**:
- Retaining the layered structure (Env → Gym → SimCore) facilitates responsibility separation
- `_gym` is not listed in `__dir__`, making it hard for AI to discover
- You use Gym indirectly through Env's public methods, without direct contact

---

## 10. Summary

Core points of this document:

1. **Facade + responsibility-cohesive decomposition** replaces the god class; components are divided by responsibility into `MuJoCoSimCore`/`OrcaStudioBridge`/`ModelRegistry`/`SimConfig` (the `_euler` field is a placeholder, currently unimplemented)
2. **Directly inherit `gym.Env` + `OrcaGymEnvMixin`**: does not inherit `OrcaGymBaseEnv`; common methods shared via Mixin
3. **Complete public API contract** covers all legitimate MuJoCo operation needs, eliminating reasons to bypass
4. **Multi-layer encapsulation isolation** (ruff SLF001 + AGENTS.md + Python native attribute absence + `__dir__` + DataView fallback + type annotations + docstring) guides you and AI down the correct path
5. **Step orchestration contract** clearly distinguishes the semantics of `do_simulation` (with coupling) and `mj_step` (pure MuJoCo)
6. **Migration strategy**: ~70% zero-change, 25% mechanical replacement, 5% design adjustment
