"""Value-only projection validates every site before mutating any algorithm."""

from types import SimpleNamespace

import numpy as np
import pytest

from orca_gym.sensor.providers.queries import CompiledQuery, QueryField, SampleStamp, SensorRuntime


def contract(*, source="clock.value", dtype="f64", shape=(), variable=False):
    return SimpleNamespace(schema_version=4, objects={},
        requirements=[{"name": "clock", "capability": "orca.sample.time.v1"}],
        value_fields=[SimpleNamespace(path="private_clock_name", query_source=source,
                                      dtype=dtype, shape=shape, variable=variable)])


class FakeSensor:
    def __init__(self):
        self.samples, self.closed = [], False

    def compute(self, sample):
        self.samples.append(sample)

    def read(self):
        return np.asarray([self.samples[-1].time])

    def close(self):
        self.closed = True


class FakeHost:
    def __init__(self, spec):
        self.spec, self.created = spec, []
        self.invalidated, self.published, self.reset_seed = False, 0, None

    def contract_for(self, type_id):
        return self.spec

    def create_sensor(self, name, type_id, *, global_parameters, seed):
        if type_id == "fail.create":
            raise ValueError("synthetic create failure")
        sensor = FakeSensor()
        self.created.append(sensor)
        return sensor

    def invalidate(self):
        self.invalidated = True

    def publish(self):
        self.published += 1

    def reset(self, seed):
        self.invalidated, self.reset_seed = False, seed


class Mailbox:
    def __init__(self):
        self.queries, self.active, self.results = (), False, None
        self.stamp = SampleStamp(0.0, .002, 0)

    def compile_site_capability(self, requirement, site):
        return CompiledQuery({"value": QueryField("f64")}, requirement["capability"], site)

    def prepare_queries(self, queries):
        assert not self.active
        self.queries, self.active = tuple(queries), True
        return object()

    def release_queries(self, token):
        self.active = False

    def capture_queries(self):
        return self.stamp, self.results if self.results is not None else {
            query: {"value": np.asarray(self.stamp.time)} for query in self.queries}

    def reset(self):
        self.stamp, self.results = SampleStamp(0.0, .002, 0), None


def make_runtime(spec=None):
    host, mailbox = FakeHost(spec or contract()), Mailbox()
    return SensorRuntime(host, mailbox), host, mailbox


def test_field_projection_uses_shared_stamp_and_publishes_once():
    runtime, host, mailbox = make_runtime()
    first = runtime.attach("first", "vendor.one", site="shared_tip")
    runtime.attach("second", "vendor.two", site="shared_tip")
    with pytest.raises(RuntimeError, match="prepare"):
        first.read()
    runtime.prepare()
    for index in range(3):
        mailbox.stamp = SampleStamp(index * .002, .002, index)
        runtime.compute()
    assert host.published == 0
    runtime.publish()
    assert host.published == 1
    for sensor in host.created:
        assert [sample.step_index for sample in sensor.samples] == [0, 1, 2]
        assert set(sensor.samples[-1].fields) == {"private_clock_name"}
    np.testing.assert_array_equal(runtime.read_all()["first"], [.004])
    runtime.reset(27)
    runtime.compute()
    assert host.reset_seed == 27 and host.created[0].samples[-1].time == 0
    runtime.close()
    assert not mailbox.active and all(sensor.closed for sensor in host.created)


@pytest.mark.parametrize("changes", [{"source": "clock.missing"}, {"source": None},
                                      {"dtype": "f32"}, {"shape": (1,)}, {"variable": True}])
def test_incompatible_projection_fails_before_native_allocation(changes):
    runtime, host, _ = make_runtime(contract(**changes))
    runtime.attach("one", "vendor.type", site="tip")
    with pytest.raises(ValueError):
        runtime.prepare()
    assert not host.created


def test_prepare_rolls_back_created_instances_and_releases_mailbox():
    runtime, host, mailbox = make_runtime()
    first = runtime.attach("first", "vendor.type", site="tip")
    runtime.attach("second", "fail.create", site="tip")
    with pytest.raises(ValueError, match="synthetic create"):
        runtime.prepare()
    assert first.instance is None and host.created[0].closed and not mailbox.active
    replacement = SensorRuntime(host, mailbox)
    replacement.attach("replacement", "vendor.type", site="tip")
    replacement.prepare()
    replacement.compute()
    replacement.close()


@pytest.mark.parametrize("bad_value", [np.asarray(np.nan), np.asarray(0, dtype=np.float32), np.zeros(2)])
def test_all_site_inputs_are_checked_before_first_compute(bad_value):
    runtime, host, mailbox = make_runtime()
    runtime.attach("first", "vendor.one", site="left_tip")
    runtime.attach("second", "vendor.two", site="right_tip")
    runtime.prepare()
    mailbox.results = {query: {"value": np.asarray(0.) if index == 0 else bad_value}
                       for index, query in enumerate(mailbox.queries)}
    with pytest.raises(ValueError):
        runtime.compute()
    assert host.invalidated and all(not sensor.samples for sensor in host.created)
    with pytest.raises(RuntimeError, match="faulted"):
        runtime.publish()
    runtime.reset()
    runtime.compute()
    assert all(len(sensor.samples) == 1 for sensor in host.created)
    runtime.close()


def test_duplicate_substep_is_rejected_and_reset_allows_replay():
    runtime, host, _ = make_runtime()
    runtime.attach("one", "vendor.type", site="tip")
    runtime.prepare()
    runtime.compute()
    with pytest.raises(ValueError, match="already computed"):
        runtime.compute()
    assert host.invalidated and len(host.created[0].samples) == 1
    runtime.reset()
    runtime.compute()
    assert len(host.created[0].samples) == 2
    runtime.close()
