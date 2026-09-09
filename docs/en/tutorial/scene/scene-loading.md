# 🏞️ Scene Loading

Scene loading involves creating a MuJoCo model from an XML file and initializing all scene elements.

## Scene Loading Process

```
1. gRPC: LoadLocalEnv → retrieve model XML
2. Local: parse XML → download mesh/hfield dependencies
3. Local: create MuJoCo model and data structures
4. Local: query and populate Model / Opt / Data
5. Local: initialize all dictionaries (body, joint, actuator, ...)
```

## OrcaGymScene Utilities

```python
from orca_gym.scene.orca_gym_scene import OrcaGymScene

# Connect to a scene
scene = OrcaGymScene("localhost:50051")

# Retrieve runtime data (parameter names are scriptname / stepname)
scene.get_rundata(scriptname="my_script", stepname="beginscene")

# Display UI text
scene.set_ui_text(
	actor_name=1,
	message="Simulation started!",
	showtime=5,
	color="0xff0000",
	size=32,
)

scene.close()
```

## OrcaGymSceneRuntime

```python
from orca_gym.scene.orca_gym_scene_runtime import OrcaGymSceneRuntime

# OrcaGymSceneRuntime wraps OrcaGymScene's runtime operations (lighting, camera viewport, etc.)
scene_runtime = OrcaGymSceneRuntime(scene)

# Note: currently neither OrcaGymEulerEnv nor OrcaGymLocalEnv defines a set_scene_runtime method.
# run_sim_loop.py detects it via hasattr(env, "set_scene_runtime"),
# so if an Env subclass extends that method and accepts OrcaGymSceneRuntime, the script injects it automatically.
# To hold scene_runtime in an Env, extend it in your subclass.
```

## Model XML Assets

Mesh and hfield files referenced by the model XML are downloaded on demand via gRPC:

```
1. Retrieve XML file content
2. Parse XML, locate <mesh> and <hfield> nodes
3. Check local cache (~/.orcagym/tmp/)
4. If not present, download via gRPC
5. Atomic write + file lock to avoid conflicts
```
