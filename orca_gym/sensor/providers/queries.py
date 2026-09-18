"""Route SimCore's samples from existing scene objects into provider contracts.

This module owns no physics state. Its input mailbox contains only typed
samples collected by SimCore from the existing model at one coherent substep.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from copy import deepcopy
import math
from typing import Any, Mapping

import numpy as np

from .runtime import SensorHost, SensorInstance, StepInput, PARAMETERS_UNSET, resolve_global_parameters


DTYPES = {"f64": np.dtype("float64"), "f32": np.dtype("float32"),
          "u64": np.dtype("uint64"), "u32": np.dtype("uint32"), "u8": np.dtype("uint8")}
SITE_CAPABILITIES = frozenset({"orca.contact_grid.v1", "orca.site_raycast.v1",
                               "orca.sample.time.v1", "orca.sample.dt.v1", "orca.sample.index.v1"})
OBJECT_CAPABILITIES = frozenset({"orca.contact.v1", "orca.raycast.v1",
                                 "orca.sample.time.v1", "orca.sample.dt.v1", "orca.sample.index.v1"})


@dataclass(frozen=True)
class QueryField:
    """The scalar type, trailing shape and row status of one sampled column."""

    dtype: str
    shape: tuple[int, ...] = ()
    variable: bool = False


@dataclass(frozen=True)
class SampleStamp:
    time: float
    dt: float
    index: int


@dataclass(eq=False)
class CompiledQuery:
    """Plain query metadata; never an evaluator or physics-object reference."""

    fields: Mapping[str, QueryField]
    kind: str
    payload: Any = None


@dataclass
class AttachedSensor:
    """Exact scene names and an independent provider instance; no model template."""

    instance_id: str
    type_id: str
    site: str | None
    global_parameters: Mapping[str, float]
    seed: int
    objects: Mapping[str, Any] | None = None
    instance_geoms: tuple[str, ...] | None = None
    instance: SensorInstance | None = field(default=None, repr=False)

    def read(self) -> np.ndarray:
        if self.instance is None:
            raise RuntimeError("SensorRuntime.prepare() has not created this instance")
        return self.instance.read()


class SensorRuntime:
    """Project scene samples and publish all provider results together.

    SimCore owns physics stepping. The mailbox receives only sampled values;
    this runtime validates every input before computing any provider instance.
    The Host is dedicated to this runtime because publish/reset are Host-wide.
    """

    def __init__(self, host: SensorHost, backend):
        self._host = host
        self._backend = backend
        self._attached: dict[str, AttachedSensor] = {}
        self._plans = {}
        self._prepared = False
        self._faulted = False
        self._closed = False
        self._backend_token = None
        self._last_stamp = None

    def attach(self, instance_id: str, type_id: str, *, site: str | None = None,
               objects: Mapping[str, Any] | None = None, instance_geoms=None,
               global_parameters=PARAMETERS_UNSET, seed: int = 0,
               parameters=PARAMETERS_UNSET) -> AttachedSensor:
        if self._prepared or self._closed:
            raise RuntimeError("Bindings cannot change after prepare or close; create a new runtime")
        if instance_id in self._attached:
            raise ValueError(f"Duplicate sensor instance: {instance_id}")
        if site is not None:
            if not isinstance(site, str) or not site or "\0" in site:
                raise ValueError("site must be an exact nonempty site name")
            if objects is not None or instance_geoms is not None:
                raise ValueError("site cannot be combined with objects or instance_geoms")
        elif not isinstance(objects, Mapping):
            raise ValueError("Provide one exact site or an explicit objects mapping")
        copied = None
        if objects is not None:
            copied = deepcopy(dict(objects))
            for local_name, reference in copied.items():
                if (not isinstance(local_name, str) or not local_name or "\0" in local_name
                        or not isinstance(reference, dict) or reference.keys() != {"kind", "name"}
                        or not isinstance(reference["kind"], str)
                        or reference["kind"] not in {"body", "geom", "site"}
                        or not isinstance(reference["name"], str) or not reference["name"]
                        or "\0" in reference["name"]):
                    raise ValueError("Object bindings require local names and exact body/geom/site kind/name references")
        owned = None
        if instance_geoms is not None:
            if (not isinstance(instance_geoms, (tuple, list)) or not instance_geoms
                    or any(not isinstance(name, str) or not name or "\0" in name for name in instance_geoms)
                    or len(set(instance_geoms)) != len(instance_geoms)):
                raise ValueError("instance_geoms must be distinct, nonempty exact geom names")
            owned = tuple(instance_geoms)
        if type(seed) is not int or not 0 <= seed < 2**64:
            raise ValueError("sensor seed must be a uint64 integer")
        sensor = AttachedSensor(instance_id, type_id, site,
                                resolve_global_parameters(global_parameters, parameters=parameters), seed,
                                copied, owned)
        self._attached[instance_id] = sensor
        return sensor

    def prepare(self) -> None:
        if self._prepared or self._closed:
            raise RuntimeError("Runtime is already prepared or closed")
        if not self._attached:
            raise ValueError("Attach at least one sensor before prepare")
        plans, all_queries = {}, []
        for name, sensor in self._attached.items():
            contract = self._host.contract_for(sensor.type_id)
            if sensor.site is not None:
                # Read-only sites can be shared; algorithms still own separate handles.
                if (contract.schema_version not in {3, 4} or contract.objects
                        or any(spec.get("capability") not in SITE_CAPABILITIES for spec in contract.requirements)):
                    raise ValueError("site binding requires an object-free contact_grid/site_raycast/sample contract")
                queries = {spec["name"]: self._backend.compile_site_capability(spec, sensor.site)
                           for spec in contract.requirements}
            else:
                if (contract.schema_version not in {3, 4}
                        or any(spec.get("capability") not in OBJECT_CAPABILITIES for spec in contract.requirements)):
                    raise ValueError("object binding requires a contact/raycast/sample contract")
                if sensor.objects.keys() != contract.objects.keys():
                    raise ValueError("Object bindings must exactly match the contract's local objects")
                allowed = {"surface": {"body", "geom"}, "frame": {"body", "geom", "site"}}
                for local_name, declaration in contract.objects.items():
                    if sensor.objects[local_name]["kind"] not in allowed.get(declaration["kind"], set()):
                        raise ValueError(f"Object binding kind does not match semantic type: {local_name}")
                queries = {spec["name"]: self._backend.compile_object_capability(
                                spec, sensor.objects, sensor.instance_geoms)
                           for spec in contract.requirements}
            projections = []
            for value in contract.value_fields:
                if value.query_source is None:
                    raise ValueError(f"{name}.{value.path}: automatic routing requires a 'from' query")
                alias, column = value.query_source.split(".", 1)
                query = queries.get(alias)
                if query is None or column not in query.fields:
                    raise ValueError(f"Unsupported query field: {value.query_source}")
                source = query.fields[column]
                if (source.dtype != value.dtype or source.shape != value.shape
                        or source.variable != value.variable):
                    raise ValueError(f"Query field dtype/shape mismatch: {value.path} <- {value.query_source}")
                projections.append((value, query, column))
            plans[name] = projections
            all_queries.extend(queries.values())

        token = self._backend.prepare_queries(all_queries)
        created = []
        try:
            for sensor in self._attached.values():
                sensor.instance = self._host.create_sensor(sensor.instance_id, sensor.type_id,
                                                          global_parameters=sensor.global_parameters, seed=sensor.seed)
                created.append(sensor)
        except BaseException:
            for sensor in reversed(created):
                try:
                    sensor.instance.close()
                except BaseException:
                    self._host.invalidate()
                finally:
                    sensor.instance = None
            self._backend.release_queries(token)
            raise
        self._plans = plans
        self._backend_token = token
        self._prepared = True

    def _check_ready(self):
        if self._closed or not self._prepared:
            raise RuntimeError("Runtime must be prepared and open")
        if self._faulted:
            raise RuntimeError("SensorRuntime is faulted; reset before continuing")

    def compute(self) -> None:
        """Stage every instance from the same current SimCore sample."""
        self._check_ready()
        try:
            stamp, results = self._backend.capture_queries()
            if (type(stamp.index) is not int or not 0 <= stamp.index < 2**64 or
                    isinstance(stamp.time, bool) or not math.isfinite(stamp.time) or
                    isinstance(stamp.dt, bool) or not math.isfinite(stamp.dt) or stamp.dt <= 0):
                raise ValueError("Query snapshot has an invalid source stamp")
            source_stamp = (stamp.index, stamp.time, stamp.dt)
            if source_stamp == self._last_stamp:
                raise ValueError("This physics substep was already computed; advance the simulation first")
            samples = {}
            for name, projections in self._plans.items():
                values, counts = {}, {}
                for descriptor, query, column in projections:
                    value = np.asarray(results[query][column])
                    if descriptor.variable:
                        if value.ndim != len(descriptor.shape) + 1 or value.shape[1:] != descriptor.shape:
                            raise ValueError(f"Query changed row shape: {name}.{descriptor.path}")
                        if value.shape[0] > descriptor.capacity:
                            raise ValueError(f"Input capacity exceeded: {name}.{descriptor.path}; no truncation")
                        group = descriptor.path.split(".", 1)[0]
                        if group in counts and counts[group] != value.shape[0]:
                            raise ValueError(f"Query columns have inconsistent row counts: {name}.{group}")
                        counts[group] = value.shape[0]
                    elif value.shape != descriptor.shape:
                        raise ValueError(f"Query changed shape: {name}.{descriptor.path}")
                    if value.dtype != DTYPES[descriptor.dtype]:
                        raise ValueError(f"Query changed dtype: {name}.{descriptor.path}")
                    if value.dtype.kind == "f" and not np.isfinite(value).all():
                        raise ValueError(f"Query returned nonfinite data: {name}.{descriptor.path}")
                    values[descriptor.path] = value
                samples[name] = StepInput(stamp.time, stamp.dt, stamp.index, fields=values)
            # Assemble every projection before mutating any provider state.
            for name, sample in samples.items():
                self._attached[name].instance.compute(sample)
            self._last_stamp = source_stamp
        except BaseException:
            self._faulted = True
            self._host.invalidate()
            raise

    def publish(self) -> None:
        self._check_ready()
        try:
            self._host.publish()
        except BaseException:
            self._faulted = True
            self._host.invalidate()
            raise

    def read_all(self) -> dict[str, np.ndarray]:
        self._check_ready()
        return {name: sensor.read() for name, sensor in self._attached.items()}

    def reset(self, seed: int = 0) -> None:
        if self._closed or not self._prepared:
            raise RuntimeError("Runtime must be prepared and open")
        self._faulted = True
        self._host.invalidate()
        self._backend.reset()
        self._host.reset(seed)
        self._last_stamp = None
        self._faulted = False

    def close(self) -> None:
        """Destroy this runtime's instances; the caller still owns the Host."""
        errors = []
        for sensor in self._attached.values():
            if sensor.instance is not None:
                try:
                    sensor.instance.close()
                except Exception as error:
                    errors.append(error)
                finally:
                    sensor.instance = None
        if self._backend_token is not None:
            try:
                self._backend.release_queries(self._backend_token)
            except Exception as error:
                errors.append(error)
            finally:
                self._backend_token = None
        self._prepared = False
        self._closed = True
        if errors:
            raise errors[0]
