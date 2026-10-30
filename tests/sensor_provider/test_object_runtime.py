"""Object contracts retain native identity, atomic inputs and reproducible seeds."""

import json
import shutil

import numpy as np
import pytest

from orca_gym.sensor.providers import SensorHost, StepInput
from orca_gym.sensor.providers.queries import SampleStamp
from orca_gym.sensor.providers.sampled_runtime import SampledSensorRuntime


def package_path(sdk_build, name):
    return sdk_build[0].parent / "providers" / name / "provider.json"


def touch_binding(instance_id):
    return {
        "instance_id": instance_id,
        "type_id": "com.orca.examples.touch_grid",
        "objects": {
            "force_f1": {"kind": "body", "name": f"robot_pad_{instance_id}"},
            "surface_frame": {"kind": "site", "name": f"measurement_{instance_id}"},
        },
        "global_parameters": {},
    }


def touch_samples(queries):
    values = {}
    for query in queries:
        if query.kind == "orca.sample.time.v1":
            values[query] = {"value": np.asarray(0.0)}
        else:
            assert query.kind == "orca.contact.v1"
            values[query] = {
                "surface_index": np.asarray([0], dtype=np.uint32),
                "counterpart_rigid_body_id": np.asarray([2], dtype=np.uint64),
                "position": np.asarray([[0.0, 0.0, 0.0]]),
                "normal_force": np.asarray([[0.0, 0.0, 3.0]]),
                "tangential_force": np.zeros((1, 3)),
            }
    return values


@pytest.mark.parametrize("name", ["touch_grid", "seven_pad"])
def test_real_object_providers_reject_contract_changes_before_registration(sdk_build, tmp_path, name):
    source = package_path(sdk_build, name)
    changed = tmp_path / name
    shutil.copytree(source.parent, changed)
    contract_path = changed / "contract.json"
    contract = json.loads(contract_path.read_text())
    contact = next(item for item in contract["inputs"] if item["capability"] == "orca.contact.v1")
    contact["capacity"] += 1  # Valid metadata, but no longer the layout compiled into the DLL.
    contract_path.write_text(json.dumps(contract))
    with SensorHost(sdk_build[0]) as host:
        with pytest.raises(ValueError, match="Input contract/ABI mismatch"):
            host.register_provider(changed / "provider.json")
        assert host.register_provider(source) == [f"com.orca.examples.{name}"]
        assert host.contract_for(f"com.orca.examples.{name}").objects


@pytest.mark.parametrize("mistake", ["missing", "renamed", "wrong_kind"])
def test_object_bindings_require_exact_local_aliases_and_compatible_types(sdk_build, monkeypatch, mistake):
    spec = touch_binding("one")
    if mistake == "missing":
        del spec["objects"]["surface_frame"]
    elif mistake == "renamed":
        spec["objects"]["pad_guess"] = spec["objects"].pop("force_f1")
    else:
        spec["objects"]["force_f1"]["kind"] = "site"
    created = []
    create = SensorHost.create_sensor

    def record_create(host, *args, **kwargs):
        created.append(args)
        return create(host, *args, **kwargs)

    monkeypatch.setattr(SensorHost, "create_sensor", record_create)
    with pytest.raises(ValueError, match="exactly match|semantic type"):
        SampledSensorRuntime(sdk_build[0], [package_path(sdk_build, "touch_grid")], [spec])
    assert created == []


@pytest.mark.parametrize("mistake", ["capacity", "rows", "dtype", "shape"])
def test_late_bad_contact_columns_fail_before_any_provider_compute(sdk_build, monkeypatch, mistake):
    specs = [touch_binding("first"), touch_binding("second")]
    runtime = SampledSensorRuntime(sdk_build[0], [package_path(sdk_build, "touch_grid")], specs)
    computed = []
    compute = SensorHost.compute

    def record_compute(host, instance_id, *args, **kwargs):
        computed.append(instance_id)
        return compute(host, instance_id, *args, **kwargs)

    monkeypatch.setattr(SensorHost, "compute", record_compute)
    try:
        queries = runtime.sampling_queries
        contacts = [query for query in queries if query.kind == "orca.contact.v1"]
        assert contacts[0].payload["objects"] == specs[0]["objects"]
        values = touch_samples(queries)
        bad = values[contacts[-1]]
        if mistake == "capacity":
            for column, value in bad.items():
                bad[column] = np.repeat(value, 65, axis=0)
        elif mistake == "rows":
            bad["normal_force"] = np.zeros((2, 3))
        elif mistake == "dtype":
            bad["normal_force"] = bad["normal_force"].astype(np.float32)
        else:
            bad["normal_force"] = np.zeros((1, 2))
        with pytest.raises(ValueError, match="capacity|row counts|dtype|row shape"):
            runtime.compute(SampleStamp(0.0, 0.001, 0), values)
        assert computed == []
        with pytest.raises(RuntimeError, match="faulted"):
            runtime.publish()

        runtime.reset(17)
        runtime.compute(SampleStamp(0.0, 0.001, 0), touch_samples(queries))
        runtime.publish()
        assert computed == ["first", "second"]
        for output in runtime.read().values():
            assert output.sum() == pytest.approx(3.0)
    finally:
        runtime.close()


def seven_pad_sequence(host):
    outputs = []
    for index in range(3):
        sample = StepInput(index * 0.001, 0.001, index, {
            "proximity": 0.05,
            "contacts.pad_index": np.asarray([0], dtype=np.uint32),
            "contacts.normal": np.asarray([[0.0, 0.0, 3.0]]),
            "contacts.tangent": np.asarray([[0.25, 0.0, 0.0]]),
        })
        outputs.append(host.process_step({"sensor": sample})["sensor"])
    return np.asarray(outputs)


def test_declared_seed_survives_global_reset_and_replays_native_noise(sdk_build):
    manifest = package_path(sdk_build, "seven_pad")
    sequences = []
    for declared_seed in (19, 20):
        with SensorHost(sdk_build[0]) as host:
            host.register_provider(manifest)
            host.create_sensor("sensor", "com.orca.examples.seven_pad", seed=declared_seed)
            host.reset(37)
            first = seven_pad_sequence(host)
            host.reset(37)
            np.testing.assert_array_equal(first, seven_pad_sequence(host))
            sequences.append(first)
    assert not np.array_equal(sequences[0][:, 1:4], sequences[1][:, 1:4])


def test_zero_declared_seed_preserves_the_previous_native_noise_sequence(sdk_build):
    with SensorHost(sdk_build[0]) as host:
        host.register_provider(package_path(sdk_build, "seven_pad"))
        sensor = host.create_sensor("sensor", "com.orca.examples.seven_pad", seed=0)
        host.reset(37)
        actual = seven_pad_sequence(host)
        # Frozen ABI-2 mapping for global seed 37 and the exact instance name "sensor".
        sensor.reset(7028369333356403401)
        np.testing.assert_array_equal(actual, seven_pad_sequence(host))
