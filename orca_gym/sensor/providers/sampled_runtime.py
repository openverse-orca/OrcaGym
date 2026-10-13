"""Internal data-only bridge from SimCore samples to the existing provider runtime.

No physics objects or callbacks enter this module. Native model/data ownership
and physical query evaluation remain entirely inside MuJoCoSimCore.
"""

from __future__ import annotations

from copy import deepcopy

from .queries import CompiledQuery, QueryField, SampleStamp, SensorRuntime
from .runtime import SensorHost
from .contracts.model import validate_bound_site_request, validate_raycast_request


_STAMPS = {"orca.sample.time.v1": "time", "orca.sample.dt.v1": "dt",
           "orca.sample.index.v1": "index"}


class _SampleBackend:
    """A typed mailbox, not a simulator or a native-object adapter."""

    def __init__(self):
        self._queries = ()
        self._token = None
        self._sample = None

    @property
    def queries(self):
        return self._queries

    def compile_site_capability(self, requirement, site):
        request = deepcopy(dict(requirement))
        capability = request.get("capability")
        if capability in _STAMPS:
            if set(request) != {"name", "capability"}:
                raise ValueError("Sample capabilities accept only name and capability")
            shape = ()
            dtype = "u64" if capability == "orca.sample.index.v1" else "f64"
        else:
            shape = tuple(validate_bound_site_request(request))
            dtype = "f64"
        return CompiledQuery({"value": QueryField(dtype, shape)}, capability,
                             {"site": site, "requirement": request})

    def compile_object_capability(self, requirement, objects, instance_geoms):
        request = deepcopy(dict(requirement))
        capability = request.get("capability")
        if capability in _STAMPS:
            if set(request) != {"name", "capability"}:
                raise ValueError("Sample capabilities accept only name and capability")
            fields = {"value": QueryField("u64" if capability == "orca.sample.index.v1" else "f64")}
        elif capability == "orca.contact.v1":
            # The contract compiler maps vendor aliases to these standard columns.
            fields = {
                "surface_index": QueryField("u32", variable=True),
                "counterpart_rigid_body_id": QueryField("u64", variable=True),
                "position": QueryField("f64", (3,), True),
                "normal_force": QueryField("f64", (3,), True),
                "tangential_force": QueryField("f64", (3,), True),
            }
        elif capability == "orca.raycast.v1":
            validate_raycast_request(request)
            fields = {"value": QueryField("f64")}
        else:
            raise ValueError(f"Unsupported object-bound capability: {capability}")
        return CompiledQuery(fields, capability, {
            "requirement": request, "objects": deepcopy(dict(objects)),
            "instance_geoms": None if instance_geoms is None else tuple(instance_geoms),
        })

    def prepare_queries(self, queries):
        if self._token is not None:
            raise RuntimeError("Sample mailbox already has an active plan")
        self._queries = tuple(queries)
        self._token = object()
        return self._token

    def release_queries(self, token):
        if token is None or token is not self._token:
            raise ValueError("Sample mailbox ownership token does not match")
        self._queries, self._token, self._sample = (), None, None

    def stage(self, stamp, results):
        self._sample = (stamp, results)

    def capture_queries(self):
        if self._sample is None:
            raise RuntimeError("No SimCore sample is available")
        return self._sample

    def reset(self):
        self._sample = None


class SampledSensorRuntime:
    """Own a Host and route already-collected, standard-valued samples."""

    def __init__(self, host_path, provider_manifests, instances, *, seed=0):
        self._host = SensorHost(host_path)
        self._mailbox = _SampleBackend()
        self._runtime = SensorRuntime(self._host, self._mailbox)
        self._names = tuple(spec["instance_id"] for spec in instances)
        self._closed = False
        self._faulted = False
        try:
            for manifest in provider_manifests:
                self._host.register_provider(manifest)
            for spec in instances:
                self._runtime.attach(spec["instance_id"], spec["type_id"],
                                     site=spec.get("site"), objects=spec.get("objects"),
                                     instance_geoms=spec.get("instance_geoms"),
                                     global_parameters=spec["global_parameters"], seed=spec.get("seed", 0))
            self._runtime.prepare()
            # Use name-derived seeds on the first run as well as every reset.
            self._runtime.reset(seed)
        except BaseException:
            try:
                self.close()
            except Exception:
                pass
            raise

    @property
    def sampling_queries(self) -> tuple[CompiledQuery, ...]:
        """Plain query metadata; it contains no native objects or evaluators."""
        return self._mailbox.queries

    def _check_ready(self):
        if self._closed:
            raise RuntimeError("Provider sensor runtime is closed")
        if self._faulted:
            raise RuntimeError("Provider sensor runtime is faulted; reset before continuing")

    def compute(self, stamp: SampleStamp, results) -> None:
        self._check_ready()
        try:
            self._mailbox.stage(stamp, results)
            self._runtime.compute()
        except BaseException:
            self.invalidate()
            raise
        finally:
            self._mailbox.reset()

    def publish(self) -> None:
        self._check_ready()
        try:
            self._runtime.publish()
        except BaseException:
            self.invalidate()
            raise

    def read(self, instance_ids=None):
        self._check_ready()
        if instance_ids is None:
            selected = self._names
        else:
            if (not isinstance(instance_ids, (list, tuple))
                    or any(not isinstance(name, str) for name in instance_ids)):
                raise ValueError("instance_ids must be a list or tuple of exact sensor names")
            selected = tuple(instance_ids)
            if len(set(selected)) != len(selected):
                raise ValueError("instance_ids must not contain duplicates")
            unknown = set(selected) - set(self._names)
            if unknown:
                raise ValueError(f"Unknown provider sensor instance: {sorted(unknown)}")
        # SensorHost.read copies native output into fresh NumPy-owned storage.
        return {name: self._host.read(name) for name in selected}

    def invalidate(self):
        if not self._closed:
            self._faulted = True
            self._host.invalidate()

    def reset(self, seed=0):
        if self._closed:
            raise RuntimeError("Provider sensor runtime is closed")
        self._faulted = True
        self._runtime.reset(seed)
        self._faulted = False

    def close(self):
        if self._closed:
            return
        try:
            self._runtime.close()
        finally:
            try:
                self._host.close()
            finally:
                self._closed = True
