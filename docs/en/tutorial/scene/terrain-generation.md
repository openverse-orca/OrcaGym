# 🏔️ Terrain Generation

OrcaGym provides terrain generation tools for creating complex ground environments.

## Terrain Generator

```python
from orca_tools.terrains import terrain_generater

# Height map generator
from orca_tools.terrains import height_map_generater
```

## Terrain Types

| Type | Description |
|------|-------------|
| Flat | Flat ground |
| Slope | Sloped ground |
| Steps | Discrete height variations |
| Rough | Randomly undulating ground |
| Obstacles | Scattered obstacles |

## Using Terrain

Generated terrains are embedded into MuJoCo scenes as height fields (hfield):

1. Generate a height map using the tools
2. Export to a MuJoCo hfield-supported format
3. Reference it in the model XML
4. Automatically downloaded and cached at load time
