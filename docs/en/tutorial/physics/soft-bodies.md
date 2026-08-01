# 🧤 Soft Bodies and Flexibles

OrcaGym supports flexible body simulation through MuJoCo's Flex system.

## MuJoCo Flex

MuJoCo 3.0+ introduced Flex (flexible body) support, and OrcaGym wraps the relevant interfaces.

## Model Querying

```python
# Get flex information via env.model.model_info (public attribute of OrcaGymModel)
model_info = env.model.model_info

# Flex-related information
nflex = model_info['nflex']           # number of flexible bodies
nflexvert = model_info['nflexvert']   # total number of flex vertices
flex_vertbodyid = model_info['flex_vertbodyid']  # body each vertex belongs to
flex_names = model_info['flex_names'] # flexible body names

if nflex > 0:
    print(f"Model contains {nflex} flexible bodies: {flex_names}")
```

> ⚠️ **Euler path**: `OrcaGymEulerEnv` does not have a `_query_model_info()` method (that method is only in the Local/Warp system). The Euler path accesses model dimension info via the `env.model.model_info` public attribute (assigned in `OrcaGymModel.init_model_info`). `env.model` also directly exposes common dimension fields like `nq`/`nv`/`nu`/`ngeom`.

## Flexible Body Anchoring

> ⚠️ **Euler path**: `OrcaGymEulerEnv` does not have an `anchor_actor` public method, nor an `_is_flex_vertex_anchored` internal flag field (these exist only in the Local system `OrcaGymLocalEnv`). Under the Euler path, operating on flexible bodies should use equality-constraint primitive orchestration, consistent with rigid body operations:

```python
# Euler path: anchor a flex vertex via equality-constraint primitive orchestration
import mujoco

mocap_name = "ActorManipulator_Anchor"
flex_body_name = "flex_body_name"

slot = env.equality_find_slot_by_body(mocap_name)
original_eq = env.equality_constraint(slot)
# Align mocap to the flex vertex pose (query via body_xpos)
flex_pose = env.get_body_xpos_xmat_xquat([flex_body_name])[flex_body_name]
env.set_mocap_pos_and_quat({
    mocap_name: {"pos": flex_pose["xpos"], "quat": flex_pose["xquat"]}
})
env.equality_update(
    slot,
    eq_type=mujoco.mjtEq.mjEQ_WELD,
    obj1_name=mocap_name,
    obj2_name=flex_body_name,
)
```

> 📝 **Local system**: `OrcaGymLocalEnv` provides the `env.anchor_actor(name, AnchorType.WELD)` convenience wrapper, which internally detects whether the target is a flex vertex and sets the `_is_flex_vertex_anchored` flag. The Euler path does not provide this wrapper; anchoring of flex vertices and rigid bodies is unified through equality-constraint primitives under the Euler path.

## Flexible Body State

```python
# Flexible body state is still accessed via qpos/qvel
# Each flex body has corresponding dimensions in qpos
```

## Limitations

- Flex support in the current version is experimental
- Flexible body operations depend on MuJoCo 3.0+ flex features
- Test in OrcaStudio before using for training
