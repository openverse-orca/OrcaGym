# 🧤 Soft Bodies and Flexibles

OrcaGym supports flexible body simulation through MuJoCo's Flex system.

## MuJoCo Flex

MuJoCo 3.0+ introduced Flex (flexible body) support, and OrcaGym wraps the relevant interfaces.

## Model Querying

```python
# Get flex information via model_info
model_info = env._query_model_info()

# Flex-related information
nflex = model_info['nflex'] # number of flexible bodies
nflexvert = model_info['nflexvert'] # total number of flex vertices
flex_vertbodyid = model_info['flex_vertbodyid'] # body each vertex belongs to
flex_names = model_info['flex_names'] # flexible body names

if nflex > 0:
 print(f"Model contains {nflex} flexible bodies: {flex_names}")
```

## Flexible Body Anchoring

```python
# Anchor flexible body vertices (similar to rigid body operations)
env.anchor_actor("flex_body_name", AnchorType.WELD)

# Internally detects whether the anchored target is a flex vertex
# env._is_flex_vertex_anchored = True
```

## Flexible Body State

```python
# Flexible body state is still accessed via qpos/qvel
# Each flex body has corresponding dimensions in qpos
```

## Limitations

- Flex support in the current version is experimental
- Flexible body operations depend on MuJoCo 3.0+ flex features
- It is recommended to test in OrcaStudio before using for training
