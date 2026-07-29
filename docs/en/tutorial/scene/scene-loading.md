# 🏞️ Scene Loading

Scene loading involves creating a MuJoCo model from an XML file and initializing all scene elements.

## Scene Loading Process

```
1. gRPC: LoadLocalEnv → retrieve model XML
2. Local: parse XML → download mesh/hfield dependencies
3. Local: mujoco.MjModel.from_xml_path()
4. Local: mujoco.MjData(model)
5. Local: query and populate Model / Opt / Data
6. Local: initialize all dictionaries (body, joint, actuator, ...)
```

## OrcaGymScene Utilities

```python
from orca_gym.scene.orca_gym_scene import OrcaGymScene

# Connect to a scene
scene = OrcaGymScene("localhost:50051")

# Retrieve runtime data
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

# Inject the scene runtime into the environment
scene_runtime = OrcaGymSceneRuntime(...)
env.set_scene_runtime(scene_runtime)
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
