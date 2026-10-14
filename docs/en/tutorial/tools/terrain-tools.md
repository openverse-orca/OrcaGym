# 🗺️ Terrain Tools

OrcaGym provides terrain generation tools for creating complex training environments.

## terrain_generater

Generates random terrain composed of geom voxels (box/sphere/cylinder, etc.) and outputs MuJoCo XML:

```python
from orca_gym.tools.terrains import terrain_generater

# Core function signature
terrain_generater.generate_geom_terrain(
    num_x, num_y,                    # grid size
    geom_type,                        # geom primitive type: box/sphere/ellipsoid/cylinder/capsule
    geom_size,                        # geom size
    geom_size_cale_range,             # size scaling range
    max_tilt,                         # maximum tilt angle
    min_step, max_step,               # adjacent geom height-difference range
    max_total_height,                 # maximum terrain height
    min_spacing, max_spacing,         # adjacent geom spacing range
    rotation_z_min, rotation_z_max,   # Z-axis rotation range
)
```

It can also be invoked from the command line:

```bash
python -m orca_gym.tools.terrains.terrain_generater \
    --num_x 10 --num_y 10 --geom_type box \
    --max_tilt 0.3 --min_step 0.2 --max_step 0.5
```

## height_map_generater

Generates the current scene's height map via physical collision detection (the `HeightMapGenerater` class, inheriting `OrcaGymLocalEnv`):

```python
from orca_gym.tools.terrains import height_map_generater
# HeightMapGenerater runs the simulation and samples heights via the env interface
```

## Workflow

1. Use `terrain_generater` to generate random terrain XML (or `height_map_generater` to generate a height map)
2. Embed the terrain XML into the scene model
3. OrcaGym automatically processes mesh/hfield assets during model loading
4. Create an environment for training
