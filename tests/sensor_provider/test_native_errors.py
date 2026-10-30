"""Native registration must reject incompatible packages without partial state."""

import json
import shutil

import numpy as np
import pytest

from orca_gym.sensor.providers import SensorError, SensorHost, StepInput

TYPE_ID = "com.orca.examples.contact_grid"


def sample(index=0):
    values = np.zeros((4, 4, 3), dtype=np.float64)
    values[0, 0, 2] = -3.0
    return StepInput(index * .001, .001, index,
                     {"sample_time": index * .001, "force_grid": values})


@pytest.mark.parametrize("mutation", ["fingerprint", "output", "version", "parameters", "duplicate", "escape"])
def test_registration_mismatch_is_atomic(sdk_build, tmp_path, mutation):
    package = tmp_path / "changed-provider"
    shutil.copytree(sdk_build[1].parent, package)
    manifest = package / "provider.json"
    document = json.loads(manifest.read_text())
    contract_path = package / document["types"][0]["contract_schema"]
    contract = json.loads(contract_path.read_text())
    if mutation == "fingerprint":
        contract["inputs"][0]["name"] = "different_time_field"
    elif mutation == "output":
        contract["output"]["shape"] = [2, 8]
        contract["output"]["channels"][0]["shape"] = [2, 8]
    elif mutation == "version":
        document["version"] = "999.0.0"
    elif mutation == "parameters":
        document["types"][0]["global_parameters"].clear()
    elif mutation == "duplicate":
        document["types"].append(document["types"][0].copy())
    else:
        document["types"][0]["contract_schema"] = str(sdk_build[1].parent / "contract.json")
    contract_path.write_text(json.dumps(contract))
    manifest.write_text(json.dumps(document))
    with SensorHost(sdk_build[0]) as host:
        with pytest.raises((ValueError, SensorError)):
            host.register_provider(manifest)
        assert host.register_provider(sdk_build[1]) == [TYPE_ID]
        with pytest.raises(ValueError, match="already registered"):
            host.register_provider(sdk_build[1])
        host.create_sensor("left", TYPE_ID)
        assert host.process_step({"left": sample()})["left"].sum() == 3.0


def test_external_model_metadata_is_not_opened(sdk_build, tmp_path):
    package = tmp_path / "preassembled-provider"
    shutil.copytree(sdk_build[1].parent, package)
    manifest = package / "provider.json"
    document = json.loads(manifest.read_text())
    document["types"][0].update(model={"asset": "not-installed.xml"},
                                  model_schema="not-installed.json",
                                  presentation_schema="not-installed-presentation.json")
    manifest.write_text(json.dumps(document))
    with SensorHost(sdk_build[0]) as host:
        host.register_provider(manifest)
        host.create_sensor("left", TYPE_ID)
        assert host.process_step({"left": sample()})["left"].sum() == 3.0


def test_incomplete_object_contract_is_rejected_explicitly(sdk_build, tmp_path):
    package = tmp_path / "model-object-provider"
    shutil.copytree(sdk_build[1].parent, package)
    manifest = package / "provider.json"
    document = json.loads(manifest.read_text())
    path = package / document["types"][0]["contract_schema"]
    contract = json.loads(path.read_text())
    contract["inputs"][1] = {"name": "force_grid", "capability": "orca.contact.v1", "objects": ["pad"]}
    path.write_text(json.dumps(contract))
    with SensorHost(sdk_build[0]) as host:
        with pytest.raises(ValueError, match="required properties"):
            host.register_provider(manifest)


def test_invalid_typed_input_faults_results_until_reset(sdk_build):
    with SensorHost(sdk_build[0]) as host:
        host.register_provider(sdk_build[1])
        host.create_sensor("left", TYPE_ID)
        host.process_step({"left": sample()})
        bad = sample(1)
        bad.fields["force_grid"] = np.zeros((4, 4, 3), dtype=np.float32)
        with pytest.raises((ValueError, SensorError)):
            host.process_step({"left": bad})
        with pytest.raises((ValueError, SensorError, RuntimeError)):
            host.read("left")
        host.reset(9)
        assert host.process_step({"left": sample()})["left"].sum() == 3.0
