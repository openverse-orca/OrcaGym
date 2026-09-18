"""Read standard custom declarations from an already assembled MJCF."""

from copy import deepcopy
import xml.etree.ElementTree as ET

import mujoco
import pytest

from orca_gym.sensor.providers.custom_scene import CUSTOM_SENSOR_PREFIX
from test_support import declarations_root, read_sensor_instances, write_scene


def test_custom_is_typed_native_xml_not_embedded_payload(tmp_path):
    root = declarations_root()
    path = write_scene(tmp_path, root)
    parsed = read_sensor_instances(path)
    assert [item["instance_id"] for item in parsed] == ["hand.tip", "second", "range", "biased"]
    assert all(item["site"] == "original_tip_site" for item in parsed)
    assert parsed[1]["global_parameters"] == {"gain": 2.0}
    assert parsed[2]["global_parameters"] == {}
    assert "&lt;" not in path.read_text()
    assert root.find("sensor") is None
    model = mujoco.MjModel.from_xml_path(str(path))
    assert model.nsensor == 0 and model.nplugin == 0
    assert model.ntuple == 4 and model.ntext == 4 and model.nnumeric == 2



def test_field_order_unrelated_custom_and_native_save_roundtrip(tmp_path):
    root = declarations_root()
    custom = root.find("custom")
    custom[:] = list(reversed(list(custom)))
    ET.SubElement(custom, "text", name="application.note", data="not a sensor")
    path = write_scene(tmp_path, root)
    original = {item["instance_id"]: item for item in read_sensor_instances(path)}
    model = mujoco.MjModel.from_xml_path(str(path))
    saved = tmp_path / "saved.xml"
    mujoco.mj_saveLastXML(str(saved), model)
    assert {item["instance_id"]: item for item in read_sensor_instances(saved)} == original



@pytest.mark.parametrize("mutate", [
    lambda block: block.append(deepcopy(block[0])),
    lambda block: block.remove(block[0]),
    lambda block: block.remove(block[1]),
    lambda block: block[0].set("name", "orca.sensor.v2/hand.tip/plugin"),
    lambda block: block[0].set("name", CUSTOM_SENSOR_PREFIX + "bad/name/plugin"),
    lambda block: block[0].set("name", CUSTOM_SENSOR_PREFIX + "hand.tip/library"),
    lambda block: block[0].set("data", "vendor.dll" + "/"),
    lambda block: block[0].set("unexpected", "true"),
    lambda block: block[1][0].set("objtype", "body"),
    lambda block: block[1][0].set("objname", ""),
    lambda block: block[1][0].set("prm", "1"),
    lambda block: block[1][0].set("prm", "nan"),
    lambda block: block[1].append(deepcopy(block[1][0])),
])
def test_custom_rejects_malformed_or_ambiguous_metadata(tmp_path, mutate):
    root = declarations_root()
    mutate(root.find("custom"))
    with pytest.raises(ValueError):
        read_sensor_instances(write_scene(tmp_path, root))



@pytest.mark.parametrize("data", ["", "1 2", "nan", "inf", "1e999", "abc", "true"])
def test_numeric_config_requires_one_finite_number(tmp_path, data):
    root = declarations_root()
    root.find("custom/numeric").set("data", data)
    with pytest.raises(ValueError, match="finite scalar"):
        read_sensor_instances(write_scene(tmp_path, root))



def test_config_declared_size_and_unknown_fields_rejected(tmp_path):
    root = declarations_root()
    root.find("custom/numeric").set("size", "2")
    with pytest.raises(ValueError, match="scalar numeric"):
        read_sensor_instances(write_scene(tmp_path, root))
    root = declarations_root()
    ET.SubElement(root.find("custom"), "text", name=CUSTOM_SENSOR_PREFIX + "hand.tip/root", data="pad")
    with pytest.raises(ValueError, match="Unknown custom sensor field"):
        read_sensor_instances(write_scene(tmp_path, root))

def object_root():
    root = ET.fromstring('''<mujoco><worldbody><body name="pad">
      <geom name="surface" type="box" size=".03 .03 .01"/>
      <site name="frame"/></body></worldbody><custom>
      <text name="orca.sensor.v1/touch/plugin" data="com.orca.examples.touch_grid"/>
      <tuple name="orca.sensor.v1/touch/object/force_f1"><element objtype="body" objname="pad"/></tuple>
      <tuple name="orca.sensor.v1/touch/object/surface_frame"><element objtype="site" objname="frame"/></tuple>
      <tuple name="orca.sensor.v1/touch/geoms"><element objtype="geom" objname="surface"/></tuple>
      <text name="orca.sensor.v1/touch/seed" data="18446744073709551615"/>
    </custom></mujoco>''')
    return root


def test_object_bindings_are_typed_references_and_roundtrip_without_geometry_changes(tmp_path):
    root = object_root()
    geometry = ET.tostring(root.find("worldbody"))
    path = write_scene(tmp_path, root)
    (spec,) = read_sensor_instances(path)
    assert spec["objects"] == {
        "force_f1": {"kind": "body", "name": "pad"},
        "surface_frame": {"kind": "site", "name": "frame"},
    }
    assert spec["instance_geoms"] == ("surface",)
    assert spec["seed"] == 2**64 - 1
    assert "site" not in spec
    model = mujoco.MjModel.from_xml_path(str(path))
    assert model.nbody == 2 and model.ngeom == 1 and model.nsite == 1
    assert ET.tostring(root.find("worldbody")) == geometry
    saved = tmp_path / "saved.xml"
    mujoco.mj_saveLastXML(str(saved), model)
    assert read_sensor_instances(saved) == (spec,)


@pytest.mark.parametrize("value", ["", "-1", "1.0", "nan", "true", "18446744073709551616", "1 2"])
def test_declared_seed_is_exact_uint64_text(tmp_path, value):
    root = object_root()
    root.find("custom/text[@name='orca.sensor.v1/touch/seed']").set("data", value)
    with pytest.raises(ValueError, match="uint64"):
        read_sensor_instances(write_scene(tmp_path, root))


@pytest.mark.parametrize("failure", ["mixed", "joint", "duplicate_geom", "wrong_geom_kind", "missing_plugin"])
def test_object_bindings_reject_ambiguous_or_untyped_declarations(tmp_path, failure):
    root = object_root()
    block = root.find("custom")
    if failure == "mixed":
        reference = ET.SubElement(block, "tuple", name="orca.sensor.v1/touch/site")
        ET.SubElement(reference, "element", objtype="site", objname="frame")
    elif failure == "joint":
        block.find("tuple/element").set("objtype", "joint")
    elif failure == "duplicate_geom":
        node = block.find("tuple[@name='orca.sensor.v1/touch/geoms']")
        node.append(deepcopy(node[0]))
    elif failure == "wrong_geom_kind":
        block.find("tuple[@name='orca.sensor.v1/touch/geoms']/element").set("objtype", "body")
    else:
        block.remove(block[0])
    with pytest.raises(ValueError):
        read_sensor_instances(write_scene(tmp_path, root))
