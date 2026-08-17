# 🗺️ Terrain Tools

OrcaGym provides terrain generation tools for creating complex training environments.

## TerrainGenerator

```python
from orca_gym.tools.terrains import terrain_generater
```

## HeightMapGenerator

```python
from orca_gym.tools.terrains import height_map_generater
```

## Terrain Types

| Type | Parameters | Example Use Case |
|------|------------|------------------|
| Flat | — | Baseline testing |
| Slope | angle | Slope climbing training |
| Steps | step_height, step_count | Stair climbing |
| Rough | roughness | Rough terrain adaptation |
| Obstacles | obstacle_size, obstacle_count | Obstacle avoidance navigation |

## Workflow

1. Use terrain tools to generate heightmaps
2. Embed the heightmap into the MuJoCo scene XML
3. OrcaGym automatically downloads hfield assets during model loading
4. Create environments for training
