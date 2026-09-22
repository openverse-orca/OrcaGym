"""Public transport and C-layout reconstruction, independent of compiler code."""

import ctypes
import json
import subprocess
from types import SimpleNamespace

import pytest
import numpy as np

from orca_gym.sensor.providers.contracts import _rpc, contract


@pytest.fixture
def transport(tmp_path, monkeypatch):
    executable = tmp_path / "trusted-tool"
    executable.write_bytes(b"test tool")
    monkeypatch.setattr(_rpc, "tool_path", lambda: executable)
    _rpc._exchange.cache_clear()
    calls = []

    def run(command, **kwargs):
        request = json.loads(kwargs["input"])
        calls.append((command, request, kwargs))
        return SimpleNamespace(returncode=0, stdout=json.dumps({
            "protocol": 1, "id": 1, "ok": True, "result": {"items": [1, 2]},
        }))

    monkeypatch.setattr(_rpc.subprocess, "run", run)
    return executable, calls


def test_transport_cache_uses_content_and_returns_independent_values(transport):
    executable, calls = transport
    first = _rpc.request("example", {"payload": {"a": 1, "b": 2}})
    first["items"].append(99)
    second = _rpc.request("example", {"payload": {"b": 2, "a": 1}})
    assert second == {"items": [1, 2]}
    assert len(calls) == 1
    assert calls[0][0] == [str(executable), "rpc"]
    assert calls[0][2]["timeout"] == 30
    assert calls[0][2]["encoding"] == "utf-8"
    _rpc.request("example", {"payload": {"a": 3, "b": 2}})
    assert len(calls) == 2


def test_contract_file_cache_uses_current_text_not_just_its_path(transport, tmp_path):
    path = tmp_path / "contract.json"
    path.write_text('{"a":1}')
    _rpc.request("example", {"source": _rpc.source_value(path)})
    path.write_text('{"a":2}')
    _rpc.request("example", {"source": _rpc.source_value(path)})
    assert len(transport[1]) == 2
    assert transport[1][0][1]["params"]["source"]["text"] == '{"a":1}'


@pytest.mark.parametrize("value", [(1, 2), {1: "number key"}, {"x": float("nan")}, {"x": object()}])
def test_transport_rejects_lossy_or_nonfinite_json_without_spawning(transport, value):
    with pytest.raises(contract.ContractError, match="finite JSON"):
        _rpc.request("compile_contract", {"source": value}, error_type=contract.ContractError)
    assert transport[1] == []


@pytest.mark.parametrize("kind,error_type", [
    ("ContractError", _rpc.ContractError), ("XmlModelError", _rpc.XmlModelError),
    ("SDKCompatibilityError", _rpc.SDKCompatibilityError), ("ValueError", ValueError),
    ("ToolProtocolError", _rpc.SensorToolError),
])
def test_remote_errors_preserve_public_exception_types(transport, monkeypatch, kind, error_type):
    reply = {"protocol": 1, "id": 1, "ok": False, "error": {"kind": kind, "message": "test error"}}
    monkeypatch.setattr(_rpc.subprocess, "run", lambda *args, **kwargs:
                        SimpleNamespace(returncode=0, stdout=json.dumps(reply)))
    with pytest.raises(error_type, match="test error"):
        _rpc.request("example")


@pytest.mark.parametrize("reply", [
    "not JSON", '{"protocol":1,"protocol":1,"id":1,"ok":true,"result":1}',
    '{"protocol":1,"id":1,"ok":true,"result":NaN}',
    '{"protocol":2,"id":1,"ok":true,"result":1}',
    '{"protocol":true,"id":1,"ok":true,"result":1}',
    '{"protocol":1,"id":2,"ok":true,"result":1}',
    '{"protocol":1,"id":1,"ok":true}',
    '{"protocol":1,"id":1,"ok":false,"error":{"kind":[],"message":"bad"}}',
])
def test_malformed_tool_replies_are_rejected(transport, monkeypatch, reply):
    monkeypatch.setattr(_rpc.subprocess, "run", lambda *args, **kwargs:
                        SimpleNamespace(returncode=0, stdout=reply))
    with pytest.raises(_rpc.SensorToolError):
        _rpc.request("example")


def test_tool_crash_and_timeout_are_reported(transport, monkeypatch):
    monkeypatch.setattr(_rpc.subprocess, "run", lambda *args, **kwargs:
                        SimpleNamespace(returncode=7, stdout=""))
    with pytest.raises(_rpc.SensorToolError, match="status 7"):
        _rpc.request("example")

    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(args[0], kwargs["timeout"])

    monkeypatch.setattr(_rpc.subprocess, "run", timeout)
    with pytest.raises(_rpc.SensorToolError, match="could not complete"):
        _rpc.request("example")


def compiled_dto():
    document = {"schema_version": 2, "contract_id": "example.time.v1", "c_struct": "ExampleInput",
                "fields": [{"name": "time", "dtype": "f64"}]}
    return {
        "document": document, "fingerprint": "a" * 64, "sdk_version": "1.0.0",
        "input_size": 8, "input_alignment": 8, "resolved_fields": document["fields"],
        "requirements": [], "resolved_objects": {},
        "layout_fields": [{"field_id": 1, "dtype": 1, "offset": 0, "stride": 0,
                           "capacity": 1, "components": 1, "count_offset": 0xFFFFFFFF}],
        "value_fields": [{"field_id": 1, "path": "time", "dtype": "f64", "shape": [],
                          "capacity": 1, "variable": False, "query_source": None}],
        "ctype_schema": {"kind": "struct", "name": "ExampleInput", "size": 8, "alignment": 8,
                         "fields": [{"name": "time", "offset": 0,
                                     "type": {"kind": "primitive", "dtype": "f64"}}]},
    }


def test_resolved_layout_reconstructs_plain_ctypes_without_compiling_semantics(monkeypatch):
    monkeypatch.setattr(contract, "request", lambda *args, **kwargs: compiled_dto())
    result = contract.compile_contract({})
    record = result.ctypes_type()
    record.time = 1.25
    assert result.size == ctypes.sizeof(record) == 8
    assert record.time == 1.25
    assert result.value_fields[0].shape == ()
    assert result.layout_fields[0].offset == 0


@pytest.mark.parametrize("bad_type", [
    {"kind": "pointer", "dtype": "f64"}, {"kind": "primitive", "dtype": "py_object"},
    {"kind": "array", "length": -1, "item": {"kind": "primitive", "dtype": "u8"}},
    {"kind": "array", "length": 2**40, "item": {"kind": "primitive", "dtype": "u8"}},
])
def test_layout_whitelist_rejects_pointers_python_objects_and_invalid_lengths(monkeypatch, bad_type):
    dto = compiled_dto()
    dto["ctype_schema"]["fields"][0]["type"] = bad_type
    monkeypatch.setattr(contract, "request", lambda *args, **kwargs: dto)
    with pytest.raises(_rpc.SensorToolError):
        contract.compile_contract({})


@pytest.mark.parametrize("field,value", [("size", 16), ("alignment", 4)])
def test_layout_rejects_native_abi_mismatch(monkeypatch, field, value):
    dto = compiled_dto()
    dto["ctype_schema"][field] = value
    monkeypatch.setattr(contract, "request", lambda *args, **kwargs: dto)
    with pytest.raises(_rpc.SensorToolError, match="platform ABI"):
        contract.compile_contract({})










def test_development_override_must_be_absolute(monkeypatch):
    monkeypatch.setenv("ORCA_SENSOR_TOOL", "relative-tool")
    with pytest.raises(_rpc.SensorToolError, match="absolute path"):
        _rpc.tool_path()


def test_prepared_sensor_steps_publish_and_reset_without_tool_calls(sdk_build, monkeypatch):
    from orca_gym.sensor.providers.queries import SampleStamp
    from orca_gym.sensor.providers.sampled_runtime import SampledSensorRuntime

    manifest = sdk_build[0].parent / "providers/contact_grid/provider.json"
    runtime = SampledSensorRuntime(sdk_build[0], [manifest], [{
        "instance_id": "left", "type_id": "com.orca.examples.contact_grid",
        "site": "left_tip", "global_parameters": {},
    }])

    def no_tool():
        raise AssertionError("A prepared physics step must not call the sensor tool")

    monkeypatch.setattr(_rpc, "tool_path", no_tool)
    try:
        for index in range(3):
            stamp = SampleStamp(index * 0.002, 0.002, index)
            results = {}
            for query in runtime.sampling_queries:
                shape = query.fields["value"].shape
                results[query] = {"value": np.full(shape, stamp.time if not shape else 1.0, dtype=np.float64)}
            runtime.compute(stamp, results)
            runtime.publish()
            assert runtime.read()["left"].shape == (4, 4)
        runtime.reset(42)
    finally:
        runtime.close()
