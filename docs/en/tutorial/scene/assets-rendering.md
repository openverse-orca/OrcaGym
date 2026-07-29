# 🎨 Assets and Rendering

OrcaGym implements 3D rendering of scenes through OrcaStudio/OrcaLab.

## Supported Assets

| Asset Type | Format | Description |
|------------|--------|-------------|
| Mesh | OBJ, STL | 3D geometry |
| Height Field (HField) | PNG | Terrain height map |
| Texture | PNG, JPG | Surface textures |
| Scene | MJCF (XML) | MuJoCo scene description |

## Asset Processing Tools

```python
# USDZ to XML conversion
from orca_gym.tools.assets import usdz_to_xml

# Texture processing
from orca_gym.tools.assets import texture_processer
```

## Asset Cache

Asset files are cached at `~/.orcagym/tmp/`:

```python
# Cache directory
# Assets are cached at ~/.orcagym/tmp/

# Safe access via file locks
# Supports concurrent multi-process downloads
```

## Rendering Configuration

Rendering is controlled by the OrcaStudio/OrcaLab server side, including:

- Light position and type
- Camera viewpoint
- Material properties
- Shadow settings

On the Python side, rendering frame transmission is triggered via the `render()` method.
