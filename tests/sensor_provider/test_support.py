"""Small preassembled scenes owned by the Euler integration tests."""

from pathlib import Path
import xml.etree.ElementTree as ET

from orca_gym.sensor.providers.custom_scene import (
    read_custom_sensor_instances, validate_main_xml_sensor_declarations,
)

HAND = Path(__file__).with_name("fixtures") / "ten_sites.xml"
GRID = "com.orca.examples.contact_grid"
RANGE = "com.orca.examples.rangefinder"


def append_custom_sensor_instances(root, instances):
    """Create test declarations; this helper is not a model assembly API."""
    block = ET.SubElement(root, "custom")
    for spec in instances:
        prefix = "orca.sensor.v1/" + spec["instance_id"] + "/"
        ET.SubElement(block, "text", name=prefix + "plugin", data=spec["type_id"])
        reference = ET.SubElement(block, "tuple", name=prefix + "site")
        ET.SubElement(reference, "element", objtype="site", objname=spec["site"])
        for key, value in spec.get("global_parameters", {}).items():
            ET.SubElement(block, "numeric", name=prefix + "config/" + key, data=str(value))


def read_sensor_instances(path):
    root = ET.parse(path).getroot()
    validate_main_xml_sensor_declarations(path, root)
    return read_custom_sensor_instances(root) or ()


def scene_root():
    return ET.fromstring('''<mujoco>
      <option timestep="0.001" integrator="implicit"/>
      <worldbody>
        <body name="pad">
          <joint type="hinge" axis="0 0 1" damping="1"/>
          <geom name="pad_shape" type="box" pos="0 0 0.01" size="0.03 0.03 0.01" mass="1"/>
          <site name="original_tip_site" size="0.001"/>
        </body>
        <body name="payload" pos="0.004 0.003 0.06">
          <freejoint/>
          <geom name="ball" type="sphere" size="0.02" mass="0.2"/>
        </body>
      </worldbody>
    </mujoco>''')



def specs():
    return [
        {"instance_id": "hand.tip", "type_id": GRID, "site": "original_tip_site"},
        {"instance_id": "second", "type_id": GRID, "site": "original_tip_site",
         "global_parameters": {"gain": 2}},
        {"instance_id": "range", "type_id": RANGE, "site": "original_tip_site"},
        {"instance_id": "biased", "type_id": RANGE, "site": "original_tip_site",
         "global_parameters": {"bias_m": 0.01}},
    ]



def write_scene(tmp_path, root):
    path = tmp_path / "scene.xml"
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)
    return path



def declarations_root():
    root = scene_root()
    append_custom_sensor_instances(root, specs())
    return root



def packages(sdk_build):
    build = sdk_build[0].parent
    return [build / "providers/contact_grid/provider.json", build / "providers/rangefinder/provider.json"]
