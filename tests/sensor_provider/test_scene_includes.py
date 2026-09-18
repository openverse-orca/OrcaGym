"""Main-file declarations cannot silently disappear through MJCF includes."""

import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import pytest

from orca_gym.sensor.providers import SensorHost
from test_support import append_custom_sensor_instances, read_sensor_instances
from test_support import GRID, scene_root, specs, write_scene
from test_euler_provider_env import ProviderTaskEnv, configured_env


def write_xml(path, root):
    path.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)
    return path


def included_declarations(tmp_path, *, layout="top", kind="custom", main=False):
    root = scene_root()
    if main:
        append_custom_sensor_instances(root, [
            {"instance_id": "main", "type_id": GRID, "site": "original_tip_site"},
        ])
    fragment = ET.Element("mujoco")
    if kind == "custom":
        append_custom_sensor_instances(fragment, specs())
    else:
        custom = ET.SubElement(fragment, "custom")
        name = {"legacy_v1": "orca.sensor.instances.v1",
                "legacy_v2": "orca.sensor.instances.v2",
                "future": "orca.sensor.v99/hidden/plugin"}[kind]
        ET.SubElement(custom, "text", name=name, data="[]")
    if layout == "custom":
        # Includes can splice fields directly into a main-file custom block.
        fragment = fragment.find("custom")
        parent = ET.SubElement(root, "custom")
    else:
        parent = root
    if layout == "nested":
        outer = ET.Element("mujocoinclude")
        ET.SubElement(outer, "include", file="declarations.xml")
        write_xml(tmp_path / "parts/outer.xml", outer)
        ET.SubElement(parent, "include", file="parts/outer.xml")
    else:
        ET.SubElement(parent, "include", file="declarations.xml")
    write_xml(tmp_path / "declarations.xml", fragment)
    return write_scene(tmp_path, root)


@pytest.mark.parametrize("kind", ["custom", "legacy_v1", "legacy_v2", "future"])
@pytest.mark.parametrize("layout", ["top", "custom", "nested"])
@pytest.mark.parametrize("main", [False, True])
def test_reader_rejects_included_declarations_with_source(tmp_path, kind, layout, main):
    path = included_declarations(tmp_path, layout=layout, kind=kind, main=main)
    # These files are valid native MJCF; rejection is Orca's explicit boundary.
    assert mujoco.MjModel.from_xml_path(str(path)).ntext > 0
    with pytest.raises(ValueError, match="must be in the main XML.*declarations.xml"):
        read_sensor_instances(path)


@pytest.mark.parametrize("kind", ["custom", "legacy_v1", "legacy_v2"])
@pytest.mark.parametrize("main", [False, True])
def test_loaders_reject_includes_before_vendor_registration(
        sdk_build, tmp_path, monkeypatch, kind, main):
    path = included_declarations(tmp_path, layout="nested", kind=kind, main=main)

    def unexpected_registration(*args, **kwargs):
        pytest.fail("Included declarations must fail before loading a vendor library")

    monkeypatch.setattr(SensorHost, "register_provider", unexpected_registration)
    with pytest.raises(ValueError, match="must be in the main XML.*declarations.xml"):
        with configured_env(sdk_build, path):
            pytest.fail("EulerEnv must not silently disable included sensors")


@pytest.mark.parametrize("lookup", ["main", "relative", "main_precedence"])
def test_geometry_includes_still_compute_for_main_declarations(sdk_build, tmp_path, lookup):
    root = scene_root()
    body = root.find("worldbody")
    root.remove(body)
    geometry = ET.Element("mujocoinclude")
    geometry.append(body)
    # Unrelated custom data is legal, even when its value mentions Orca names.
    note = ET.SubElement(geometry, "custom")
    ET.SubElement(note, "text", name="application.note", data="orca.sensor.v1/example/plugin")
    outer = ET.Element("mujocoinclude")
    ET.SubElement(outer, "include", file="geometry.xml")
    write_xml(tmp_path / "parts/outer.xml", outer)
    leaf = tmp_path / ("parts/geometry.xml" if lookup == "relative" else "geometry.xml")
    write_xml(leaf, geometry)
    if lookup == "main_precedence":
        # Native MuJoCo chooses the main-directory file; do not inspect a
        # shadowed file that would not become part of the loaded scene.
        shadow = ET.Element("mujocoinclude")
        custom = ET.SubElement(shadow, "custom")
        ET.SubElement(custom, "text", name="orca.sensor.instances.v2", data="[]")
        write_xml(tmp_path / "parts/geometry.xml", shadow)
    ET.SubElement(root, "include", file="parts/outer.xml")
    append_custom_sensor_instances(root, specs())
    path = write_scene(tmp_path, root)
    assert len(read_sensor_instances(path)) == len(specs())
    assert mujoco.MjModel.from_xml_path(str(path)).nsite == 1
    with configured_env(sdk_build, path, frame_skip=1) as env:
        outputs = env.step(np.zeros(0))[0]["sensors"]
    assert set(outputs) == {spec["instance_id"] for spec in specs()}
    assert outputs["hand.tip"].shape == (4, 4)
    assert outputs["range"].shape == (1,)


def test_unconfigured_euler_keeps_native_include_loading(tmp_path):
    path = included_declarations(tmp_path)
    env = ProviderTaskEnv(path)
    try:
        assert env.step(np.zeros(0))[0]["time"] == pytest.approx(.004)
    finally:
        env.close()


def test_fragment_config_alone_cannot_be_silently_dropped(tmp_path):
    root = scene_root()
    append_custom_sensor_instances(root, specs()[:1])
    ET.SubElement(root.find("custom"), "include", file="config.xml")
    fragment = ET.Element("mujocoinclude")
    ET.SubElement(fragment, "numeric", name="orca.sensor.v1/hand.tip/config/gain", data="2")
    write_xml(tmp_path / "config.xml", fragment)
    path = write_scene(tmp_path, root)
    assert mujoco.MjModel.from_xml_path(str(path)).nnumeric == 1
    with pytest.raises(ValueError, match="must be in the main XML.*config.xml"):
        read_sensor_instances(path)


@pytest.mark.parametrize("failure", ["missing", "directory", "cycle"])
def test_uninspectable_includes_fail_explicitly(tmp_path, failure):
    root = scene_root()
    filename = {"missing": "missing.xml", "directory": ".", "cycle": "scene.xml"}[failure]
    ET.SubElement(root, "include", file=filename)
    path = write_scene(tmp_path, root)
    with pytest.raises(ValueError, match="Cannot inspect XML include|Cyclic XML include"):
        read_sensor_instances(path)
