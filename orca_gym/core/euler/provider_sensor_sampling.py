"""SimCore-owned provider sampling; native objects never leave the core.

This mixin is part of MuJoCoSimCore itself, not a separate backend object.
The only values sent to SampledSensorRuntime are query metadata and typed
NumPy samples. No second model/data or post-step forward pass is needed.
"""

from __future__ import annotations

import math
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from orca_gym.sensor.providers.custom_scene import (
    read_custom_sensor_instances, validate_main_xml_sensor_declarations,
)
from orca_gym.sensor.providers.native import default_host_path
from orca_gym.sensor.providers.queries import SampleStamp
from orca_gym.sensor.providers.sampled_runtime import SampledSensorRuntime


_RAY_PRIMITIVES = frozenset(int(value) for value in (
    mujoco.mjtGeom.mjGEOM_PLANE, mujoco.mjtGeom.mjGEOM_SPHERE,
    mujoco.mjtGeom.mjGEOM_CAPSULE, mujoco.mjtGeom.mjGEOM_ELLIPSOID,
    mujoco.mjtGeom.mjGEOM_CYLINDER, mujoco.mjtGeom.mjGEOM_BOX))
_STAMPS = {"orca.sample.time.v1": "time", "orca.sample.dt.v1": "dt",
           "orca.sample.index.v1": "index"}
_OBJECT_KINDS = {"body": mujoco.mjtObj.mjOBJ_BODY, "geom": mujoco.mjtObj.mjOBJ_GEOM,
                 "site": mujoco.mjtObj.mjOBJ_SITE}


def _add_grid_force(grid, resolution, fov, position, force):
    """Apply the SDK's spherical FOV projection to one site-local contact."""
    if not np.isfinite(position).all() or not np.isfinite(force).all():
        raise ValueError("Contact grid source position and force must be finite")
    x, y, z = position
    if z <= 0:
        return
    azimuth = math.degrees(math.atan2(x, z))
    elevation = math.degrees(math.atan2(y, math.hypot(x, z)))
    horizontal, vertical = fov
    if abs(azimuth) * 2 > horizontal or abs(elevation) * 2 > vertical:
        return
    rows, columns = resolution
    column = min(columns - 1, int((azimuth / horizontal + 0.5) * columns))
    row = min(rows - 1, int((elevation / vertical + 0.5) * rows))
    grid[row, column] += force


class ProviderSensorSamplingMixin:
    """Private sampling implementation plus SimCore's approved value-only API."""

    def _initialize_provider_sensor_state(self):
        self._provider_configuration = None
        self._provider_runtime = None
        self._provider_status = "inactive"
        self._provider_plans = {}
        self._provider_routes = {}
        self._provider_seed = 0
        self._provider_index = 0

    @staticmethod
    def _validate_provider_seed(seed):
        if type(seed) is not int or not 0 <= seed < 2**64:
            raise ValueError("sensor seed must be a uint64 integer")
        return seed

    def configure_provider_sensors(self, host_path, provider_manifests):
        """Save explicit trusted package paths before first model loading.

        This does not execute a Host or vendor library. A configured scene with
        no provider declarations also remains on the original bulk-step path.
        Reconfigure by constructing a new core, not by hot-swapping algorithms.
        """
        if self._mjModel is not None:
            raise RuntimeError("Configure provider sensors before model initialization; no hot reconfiguration")
        if host_path is not None and (not isinstance(host_path, (str, Path)) or not str(host_path)):
            raise ValueError("sensor_host_path must be a nonempty file path")
        if (not isinstance(provider_manifests, (list, tuple)) or not provider_manifests
                or any(not isinstance(path, (str, Path)) or not str(path) for path in provider_manifests)):
            raise ValueError("sensor_provider_manifests must be a nonempty list or tuple of file paths")
        host = default_host_path() if host_path is None else Path(host_path).expanduser().resolve()
        manifests = tuple(Path(path).expanduser().resolve() for path in provider_manifests)
        if not host.is_file() or any(not path.is_file() for path in manifests):
            raise ValueError("Provider Host and manifest paths must be existing files")
        if len(set(manifests)) != len(manifests):
            raise ValueError("Provider manifest paths must be distinct")
        self._provider_configuration = (host, manifests)
        self._provider_status = "uninitialized"

    def _bind_provider_sensors(self, model_xml_path):
        self._provider_index = 0
        self._provider_plans, self._provider_routes = {}, {}
        if self._provider_configuration is None:
            self._provider_status = "inactive"
            return
        self._provider_status = "faulted"
        root = ET.parse(model_xml_path).getroot()
        validate_main_xml_sensor_declarations(model_xml_path, root)
        if any(node.get("name", "").startswith("orca.sensor.instances")
               for custom in root.findall("custom") for node in custom):
            raise ValueError("Euler provider integration supports standard custom declarations, not legacy JSON scenes")
        instances = read_custom_sensor_instances(root)
        if not instances:
            self._provider_status = "inactive"
            return
        self._check_provider_integrator()
        runtime = None
        try:
            host_path, manifests = self._provider_configuration
            runtime = SampledSensorRuntime(host_path, manifests, instances, seed=self._provider_seed)
            self._compile_provider_sampling(runtime.sampling_queries)
        except BaseException:
            if runtime is not None:
                try:
                    runtime.close()
                except Exception:
                    pass
            raise
        self._provider_runtime = runtime
        self._provider_status = "ready"

    def _check_provider_integrator(self):
        allowed = {mujoco.mjtIntegrator.mjINT_EULER, mujoco.mjtIntegrator.mjINT_IMPLICIT,
                   mujoco.mjtIntegrator.mjINT_IMPLICITFAST}
        if self._mjModel.opt.integrator not in allowed:
            raise ValueError("Provider source-state sampling supports Euler/implicit/implicitfast, not RK4")

    def _resolve_provider_object(self, reference):
        if (not isinstance(reference, dict) or set(reference) != {"kind", "name"}
                or not isinstance(reference["kind"], str) or reference["kind"] not in _OBJECT_KINDS
                or not isinstance(reference["name"], str) or not reference["name"] or "\0" in reference["name"]):
            raise ValueError("Provider object requires an exact body, geom or site name")
        kind, name = reference["kind"], reference["name"]
        object_id = mujoco.mj_name2id(self._mjModel, _OBJECT_KINDS[kind], name)
        if object_id < 0:
            raise ValueError(f"Unknown provider sensor {kind}: {name}")
        return kind, int(object_id)

    def _compile_provider_raycast(self, plan, excluded):
        request = plan["request"]
        vector = request["direction"]
        if (not isinstance(vector, (list, tuple)) or len(vector) != 3
                or any(type(value) not in {int, float} for value in vector)):
            raise ValueError("Raycast direction must contain three finite numeric components")
        try:
            direction = np.asarray(vector, dtype=np.float64)
            maximum = float(request["max_distance"])
        except (OverflowError, TypeError, ValueError) as error:
            raise ValueError("Raycast direction and maximum distance must be finite") from error
        if (not np.isfinite(direction).all() or not np.any(direction)
                or type(request["max_distance"]) not in {int, float}
                or not math.isfinite(maximum) or maximum <= 0):
            raise ValueError("Raycast requires a finite nonzero direction and positive maximum distance")
        direction /= np.max(np.abs(direction))
        plan["direction"] = direction / np.linalg.norm(direction)
        plan["maximum"] = maximum
        targets = tuple(int(geom) for geom in np.flatnonzero(
            (self._mjModel.geom_contype != 0) | (self._mjModel.geom_conaffinity != 0))
                        if int(geom) not in excluded)
        supported = _RAY_PRIMITIVES | {int(mujoco.mjtGeom.mjGEOM_MESH),
                                       int(mujoco.mjtGeom.mjGEOM_HFIELD)}
        if any(int(self._mjModel.geom_type[geom]) not in supported for geom in targets):
            raise ValueError("Raycast supports primitive, mesh and heightfield geoms, not SDF/other geometry")
        if self._mjModel.nflex and np.any(
                (self._mjModel.flex_contype != 0) | (self._mjModel.flex_conaffinity != 0)):
            raise ValueError("Raycast does not support collidable flex geometry")
        plan["targets"] = targets

    def _compile_provider_objects(self, query):
        request = query.payload["requirement"]
        references = query.payload["objects"]
        if not isinstance(references, dict):
            raise ValueError("Provider object bindings must map aliases to exact scene objects")
        objects = {name: self._resolve_provider_object(reference) for name, reference in references.items()}
        owned_names = query.payload.get("instance_geoms")
        owned = None
        if owned_names is not None:
            if (not isinstance(owned_names, (list, tuple))
                    or any(not isinstance(name, str) or not name for name in owned_names)
                    or len(set(owned_names)) != len(owned_names)):
                raise ValueError("instance_geoms must contain distinct exact geom names")
            owned = frozenset(self._resolve_provider_object({"kind": "geom", "name": name})[1]
                              for name in owned_names)
        else:
            # Exact body/geom bindings establish only their directly named
            # geometry. Never expand a site, weld group, subtree or name prefix.
            surfaces = [reference for reference in objects.values() if reference[0] in {"body", "geom"}]
            if surfaces:
                owned = frozenset(int(geom) for kind, object_id in surfaces
                                  for geom in (np.flatnonzero(self._mjModel.geom_bodyid == object_id)
                                               if kind == "body" else (object_id,)))
        frame_name = request.get("frame", "world")
        if frame_name != "world" and frame_name not in objects:
            raise ValueError(f"Unknown provider frame alias: {frame_name}")
        plan = {"request": request, "frame": None if frame_name == "world" else objects[frame_name]}
        if query.kind in _STAMPS:
            return plan
        if query.kind == "orca.raycast.v1":
            if plan["frame"] is None:
                raise ValueError("Object raycast requires a bound body, geom or site frame")
            exclude = request.get("exclude_self", True)
            if type(exclude) is not bool:
                raise ValueError("Raycast exclude_self must be boolean")
            if exclude and owned is None:
                raise ValueError("Raycast exclude_self requires body/geom bindings or explicit instance_geoms")
            self._compile_provider_raycast(plan, owned if exclude else ())
            return plan
        if query.kind != "orca.contact.v1":
            raise ValueError(f"Unsupported object-bound sampling capability: {query.kind}")
        names = request.get("objects")
        if (not isinstance(names, (list, tuple)) or not names
                or any(not isinstance(name, str) or name not in objects for name in names)
                or len(set(names)) != len(names)):
            raise ValueError("Contact capability requires ordered, distinct bound surface aliases")
        indices = {}
        for index, name in enumerate(names):
            kind, object_id = objects[name]
            if kind not in {"body", "geom"}:
                raise ValueError("Contact surface must bind a body or geom")
            geoms = (np.flatnonzero(self._mjModel.geom_bodyid == object_id)
                     if kind == "body" else (object_id,))
            if len(geoms) == 0:
                raise ValueError("Contact surface body must directly own at least one geom")
            for geom in geoms:
                geom = int(geom)
                if geom in indices:
                    raise ValueError("Contact surfaces have overlapping geometry coverage")
                if (plan["frame"] is not None
                        and self._provider_weld_group(("geom", geom)) != self._provider_weld_group(plan["frame"])):
                    raise ValueError("Contact surface and frame must belong to the same rigid weld group")
                if owned is not None and geom not in owned:
                    raise ValueError("instance_geoms must include every bound surface geom")
                indices[geom] = index
        capacity = request.get("capacity")
        if type(capacity) is not int or capacity <= 0:
            raise ValueError("Contact capacity must be a positive integer")
        exclude = request.get("exclude_internal", True)
        if type(exclude) is not bool:
            raise ValueError("Contact exclude_internal must be boolean")
        plan.update(indices=indices, geoms=frozenset(indices), capacity=capacity, exclude=exclude)
        return plan

    def _compile_provider_sampling(self, queries):
        plans, routes = {}, {}
        for query in queries:
            if "objects" in query.payload:
                plan = self._compile_provider_objects(query)
                if query.kind == "orca.contact.v1":
                    for geom, index in plan["indices"].items():
                        routes.setdefault(geom, []).append((query, index))
            else:
                frame = self._resolve_provider_object({"kind": "site", "name": query.payload["site"]})
                plan = {"frame": frame, "request": query.payload["requirement"]}
                if query.kind not in _STAMPS:
                    body = int(self._mjModel.site_bodyid[frame[1]])
                    weld = int(self._mjModel.body_weldid[body])
                    geoms = frozenset(int(geom) for geom in np.flatnonzero(
                        self._mjModel.body_weldid[self._mjModel.geom_bodyid] == weld))
                    plan.update(geoms=geoms, exclude=True)
                    if query.kind == "orca.contact_grid.v1":
                        for geom in geoms:
                            routes.setdefault(geom, []).append((query, 0))
                    elif query.kind == "orca.site_raycast.v1":
                        self._compile_provider_raycast(plan, geoms)
                    else:
                        raise ValueError(f"Unsupported provider sampling capability: {query.kind}")
            plans[query] = plan
        self._provider_plans, self._provider_routes = plans, routes

    def _provider_weld_group(self, reference):
        kind, object_id = reference
        if kind == "body":
            body = object_id
        elif kind == "geom":
            body = self._mjModel.geom_bodyid[object_id]
        else:
            body = self._mjModel.site_bodyid[object_id]
        return int(self._mjModel.body_weldid[body])

    def _provider_frame_pose(self, frame):
        kind, object_id = frame
        if kind == "body":
            origin, axes = self._mjData.xpos[object_id], self._mjData.xmat[object_id]
        elif kind == "geom":
            origin, axes = self._mjData.geom_xpos[object_id], self._mjData.geom_xmat[object_id]
        else:
            origin, axes = self._mjData.site_xpos[object_id], self._mjData.site_xmat[object_id]
        return origin, axes.reshape(3, 3)

    def _sample_provider_raycast(self, plan):
        origin, axes = self._provider_frame_pose(plan["frame"])
        direction = axes @ plan["direction"]
        if not np.isfinite(origin).all() or not np.isfinite(direction).all():
            raise ValueError("Raycast source frame must be finite")
        closest = math.inf
        for geom in plan["targets"]:
            kind = int(self._mjModel.geom_type[geom])
            if kind == mujoco.mjtGeom.mjGEOM_MESH:
                distance = mujoco.mj_rayMesh(self._mjModel, self._mjData, geom, origin, direction)
            elif kind == mujoco.mjtGeom.mjGEOM_HFIELD:
                distance = mujoco.mj_rayHfield(self._mjModel, self._mjData, geom, origin, direction)
            else:
                distance = mujoco.mju_rayGeom(self._mjData.geom_xpos[geom], self._mjData.geom_xmat[geom],
                                            self._mjModel.geom_size[geom], origin, direction, kind)
            if not math.isfinite(distance):
                raise ValueError("Raycast returned a nonfinite intersection distance")
            if 0 <= distance < closest:
                closest = distance
        return np.asarray(closest if closest <= plan["maximum"] else -1.0, dtype=np.float64)

    def _capture_provider_sample(self, stamp):
        results = {}
        grids = {query: np.zeros(query.fields["value"].shape, dtype=np.float64)
                 for query in self._provider_plans if query.kind == "orca.contact_grid.v1"}
        rows = {query: [] for query in self._provider_plans if query.kind == "orca.contact.v1"}
        if grids or rows:
            force = np.empty(6, dtype=np.float64)
            for index in range(self._mjData.ncon):
                contact = self._mjData.contact[index]
                if contact.efc_address < 0:
                    continue
                first, second = int(contact.geom1), int(contact.geom2)
                first_routes = self._provider_routes.get(first, ())
                second_routes = self._provider_routes.get(second, ())
                if not first_routes and not second_routes:
                    continue
                if first < 0 or second < 0:
                    raise ValueError("Bound flex contacts require a flex-aware sampling capability")
                # One native force read feeds every interested sensor instance.
                mujoco.mj_contactForce(self._mjModel, self._mjData, index, force)
                axes = contact.frame.reshape(3, 3)
                world_normal = axes[0] * force[0]
                world_tangent = axes[1:].T @ force[1:3]
                for recipients, other, sign in ((first_routes, second, -1.0),
                                                (second_routes, first, 1.0)):
                    for query, surface_index in recipients:
                        plan = self._provider_plans[query]
                        if plan["exclude"] and other in plan["geoms"]:
                            continue
                        if plan["frame"] is None:
                            position = np.array(contact.pos, copy=True)
                            local_normal, local_tangent = sign * world_normal, sign * world_tangent
                        else:
                            origin, axes = self._provider_frame_pose(plan["frame"])
                            rotation = axes.T
                            position = rotation @ (contact.pos - origin)
                            local_normal = rotation @ (sign * world_normal)
                            local_tangent = rotation @ (sign * world_tangent)
                        if query in grids:
                            request = plan["request"]
                            _add_grid_force(grids[query], request["resolution"], request["fov_degrees"],
                                            position, local_normal + local_tangent)
                        else:
                            if len(rows[query]) >= plan["capacity"]:
                                raise ValueError("Contact input capacity exceeded; no truncation")
                            if not all(np.isfinite(value).all() for value in (position, local_normal, local_tangent)):
                                raise ValueError("Contact source position and force must be finite")
                            counterpart = int(self._mjModel.body_weldid[self._mjModel.geom_bodyid[other]])
                            rows[query].append((surface_index, counterpart, position, local_normal, local_tangent))
        for query, records in rows.items():
            columns = {
                "surface_index": np.asarray([row[0] for row in records], dtype=np.uint32),
                "counterpart_rigid_body_id": np.asarray([row[1] for row in records], dtype=np.uint64),
            }
            for offset, name in enumerate(("position", "normal_force", "tangential_force"), start=2):
                columns[name] = np.asarray([row[offset] for row in records], dtype=np.float64).reshape(-1, 3)
            results[query] = columns
        for query, plan in self._provider_plans.items():
            if query in results:
                continue
            if query in grids:
                value = grids[query]
            elif query.kind in {"orca.site_raycast.v1", "orca.raycast.v1"}:
                value = self._sample_provider_raycast(plan)
            else:
                column = _STAMPS[query.kind]
                value = np.asarray(getattr(stamp, column), dtype=np.uint64 if column == "index" else np.float64)
            results[query] = {"value": value}
        return results

    def _invalidate_provider_sensors(self):
        self._provider_status = "faulted"
        if self._provider_runtime is not None:
            self._provider_runtime.invalidate()

    def _step_provider_sensors(self, nstep):
        if self._provider_status != "ready":
            raise RuntimeError("Provider sensors are not ready or are faulted/closed; reset or reload first")
        if type(nstep) is not int or nstep <= 0:
            raise ValueError("nstep must be a positive integer")
        try:
            for _ in range(nstep):
                self._check_provider_integrator()
                source_time = float(self._mjData.time)
                dt = float(self._mjModel.opt.timestep)
                if not math.isfinite(source_time) or not math.isfinite(dt) or dt <= 0:
                    raise ValueError("Provider sample time/dt must be finite and dt positive")
                if self._provider_index >= 2**64:
                    raise ValueError("Provider sample index exhausted; reset before continuing")
                mujoco.mj_step(self._mjModel, self._mjData, 1)
                if not math.isclose(float(self._mjData.time), source_time + dt,
                                    rel_tol=1e-12, abs_tol=1e-12):
                    raise RuntimeError("MuJoCo reset or changed time during stepping; reset the runtime")
                # Retained contact/site/geom transforms belong to source_time;
                # do not forward here or combine them with integrated qpos.
                stamp = SampleStamp(source_time, dt, self._provider_index)
                values = self._capture_provider_sample(stamp)
                self._provider_runtime.compute(stamp, values)
                self._provider_index += 1
            self._provider_runtime.publish()
        except BaseException:
            self._invalidate_provider_sensors()
            raise

    def query_provider_sensor_data(self, instance_ids=None):
        """Copy published DLL outputs using exact XML instance identifiers."""
        if self._provider_configuration is None or self._provider_status == "inactive":
            if instance_ids is not None and (
                    not isinstance(instance_ids, (list, tuple)) or len(instance_ids) != 0):
                raise ValueError("Unknown provider sensor instance; this scene has none enabled")
            return {}
        if self._provider_status != "ready":
            raise RuntimeError("Provider sensors are uninitialized, faulted or closed")
        return self._provider_runtime.read(instance_ids)

    def reset_provider_sensors(self, seed=0):
        """Reset algorithm/RNG/sample state only, without advancing physics."""
        seed = self._validate_provider_seed(seed)
        if self._provider_configuration is not None and self._provider_status != "inactive":
            if self._provider_runtime is None or self._provider_status == "closed":
                raise RuntimeError("Provider sensors require model reload before reset")
            self._provider_status = "faulted"
            self._provider_runtime.reset(seed)
            self._provider_status = "ready"
        self._provider_seed = seed
        self._provider_index = 0

    def close_provider_sensors(self):
        """Release native algorithm instances; repeated close is harmless."""
        runtime = self._provider_runtime
        self._provider_runtime = None
        self._provider_plans, self._provider_routes = {}, {}
        if self._provider_configuration is not None:
            self._provider_status = "closed"
        if runtime is not None:
            runtime.close()
