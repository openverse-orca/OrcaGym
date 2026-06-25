"""ModelRegistry — 模型信息注册，构建 OrcaGymModel，提供扩展查询。

替代直接访问 _mjModel 的模型结构查询，构建 OrcaGymModel/OrcaGymData，
提供 body_subtree_mass、equality 等扩展查询。

属于 OrcaGymEuler 体系的 P2 状态视图与配置组件。
参见 docs/design/architecture/orca_gym_euler_architecture.md 第 5.5 节。
"""

from __future__ import annotations

import os
import xml.etree.ElementTree as ET

import mujoco

from orca_gym.core.orca_gym_model import OrcaGymModel
from orca_gym.core.orca_gym_data import OrcaGymData


class ModelRegistry:
    """模型信息注册，构建 OrcaGymModel，提供扩展查询。

    设计契约:
        - 从 _mjModel 构建完整的 OrcaGymModel（与 OrcaGymLocal 一致）。
        - 提供替代 _mjModel 直接访问的扩展查询方法。
        - 不暴露 _mjModel，内部持有用于查询。

    使用示例:
        ```python
        registry = ModelRegistry(mj_model, xml_path="scene.xml")
        model = registry.build_orca_gym_model()
        mass = registry.body_subtree_mass("link1")
        ```
    """

    def __init__(
        self, mj_model: mujoco.MjModel, xml_path: str | None = None
    ) -> None:
        self._mj_model = mj_model
        self._xml_path = xml_path

    # --- 构建 OrcaGymModel / OrcaGymData ---

    def build_orca_gym_model(self) -> OrcaGymModel:
        """构建完整的 OrcaGymModel，与 OrcaGymLocal 构建逻辑一致。"""
        model = OrcaGymModel(self._query_model_info())
        model.init_eq_list(self._query_all_equality_constraints())
        model.init_mocap_dict(self._query_all_mocap_bodies())
        model.init_actuator_dict(self._query_all_actuators())
        model.init_body_dict(self._query_all_bodies())
        model.init_joint_dict(self._query_all_joints())
        model.init_geom_dict(self._query_all_geoms())
        model.init_site_dict(self._query_all_sites())
        model.init_sensor_dict(self._query_all_sensors())
        model.init_mesh_dict(self._query_all_meshes())
        return model

    def build_orca_gym_data(self) -> OrcaGymData:
        """构建 OrcaGymData。"""
        return OrcaGymData(self.build_orca_gym_model())

    # --- 扩展查询（替代 _mjModel 直接访问）---

    def body_subtree_mass(self, body_name: str) -> float:
        """查询 body 子树总质量。"""
        body_id = mujoco.mj_name2id(
            self._mj_model, mujoco.mjtObj.mjOBJ_BODY, body_name
        )
        return float(self._mj_model.body_subtreemass[body_id])

    def equality_data_width(self) -> int:
        """等式约束数据宽度（eq_data 的列数）。"""
        if self._mj_model.neq == 0:
            return 0
        return int(self._mj_model.eq_data.shape[1])

    def equality_object_ids(self, eq_idx: int) -> tuple[int, int]:
        """查询第 eq_idx 个等式约束的两个对象 ID。"""
        return (
            int(self._mj_model.eq_obj1id[eq_idx]),
            int(self._mj_model.eq_obj2id[eq_idx]),
        )

    def joint_name_by_id(self, joint_id: int) -> str:
        """根据关节 ID 查询关节名称。"""
        name = mujoco.mj_id2name(
            self._mj_model, mujoco.mjtObj.mjOBJ_JOINT, joint_id
        )
        return name if name is not None else ""

    # --- 模型信息查询（从 OrcaGymLocal 迁移，逻辑一致）---

    def _query_model_info(self) -> dict:
        m = self._mj_model
        return {
            "nq": m.nq, "nv": m.nv, "nu": m.nu,
            "nbody": m.nbody, "njnt": m.njnt, "ngeom": m.ngeom,
            "nsite": m.nsite, "nmesh": m.nmesh, "ncam": m.ncam,
            "nlight": m.nlight,
            "nuser_body": m.nuser_body, "nuser_jnt": m.nuser_jnt,
            "nuser_geom": m.nuser_geom, "nuser_site": m.nuser_site,
            "nuser_tendon": m.nuser_tendon, "nuser_actuator": m.nuser_actuator,
            "nuser_sensor": m.nuser_sensor, "nconmax": m.nconmax,
            "nflex": m.nflex, "nflexvert": m.nflexvert,
            "flex_vertbodyid": list(m.flex_vertbodyid),
            "flex_vertadr": list(m.flex_vertadr) if m.nflex > 0 else [],
            "flex_vertnum": list(m.flex_vertnum) if m.nflex > 0 else [],
            "flex_names": (
                [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_FLEX, i) for i in range(m.nflex)]
                if m.nflex > 0 else []
            ),
        }

    def _query_all_equality_constraints(self) -> list:
        m = self._mj_model
        eq_list = []
        for i in range(m.neq):
            eq_list.append({
                "eq_type": m.eq_type[i],
                "obj1_id": m.eq_obj1id[i],
                "obj2_id": m.eq_obj2id[i],
                "active": m.eq_active0[i],
                "eq_solref": m.eq_solref[i],
                "eq_solimp": m.eq_solimp[i],
                "eq_data": m.eq_data[i],
            })
        return eq_list

    def _query_all_mocap_bodies(self) -> dict:
        m = self._mj_model
        mocap_dict = {}
        for i in range(m.nbody):
            if m.body_mocapid[i] != -1:
                mocap_dict[m.body(i).name] = m.body_mocapid[i]
        return mocap_dict

    def _query_all_actuators(self) -> dict:
        m = self._mj_model
        actuator_dict = {}
        idx = 0
        for i in range(m.nu):
            actuator = m.actuator(i)
            name = actuator.name if actuator.name else "actuator"
            if name in actuator_dict:
                name = f"{name}_{idx}"
                idx += 1
            if actuator.trntype == mujoco.mjtTrn.mjTRN_JOINT:
                joint_name = m.joint(actuator.trnid[0]).name
            elif actuator.trntype == mujoco.mjtTrn.mjTRN_TENDON:
                joint_name = m.tendon(actuator.trnid[0]).name
            elif actuator.trntype == mujoco.mjtTrn.mjTRN_SITE:
                joint_name = m.site(actuator.trnid[0]).name
            else:
                joint_name = "unknown"
            actuator_dict[name] = {
                "JointName": joint_name,
                "GearRatio": actuator.gear,
                "TrnId": actuator.trnid[0],
                "CtrlLimited": bool(actuator.ctrllimited[0]),
                "ForceLimited": bool(actuator.forcelimited[0]),
                "ActLimited": bool(actuator.actlimited[0]),
                "CtrlRange": actuator.ctrlrange,
                "ForceRange": actuator.forcerange,
                "ActRange": actuator.actrange,
                "TrnType": actuator.trntype[0],
                "DynType": actuator.dyntype[0],
                "GainType": actuator.gaintype[0],
                "BiasType": actuator.biastype[0],
                "ActAdr": actuator.actadr[0],
                "ActNum": actuator.actnum[0],
                "Group": actuator.group[0],
                "DynPrm": actuator.dynprm,
                "GainPrm": actuator.gainprm,
                "BiasPrm": actuator.biasprm,
                "ActEarly": bool(m.actuator_actearly[i]),
                "Gear": actuator.gear,
                "CrankLength": actuator.cranklength[0],
                "Acc0": actuator.acc0[0],
                "Length0": actuator.length0[0],
                "LengthRange": actuator.lengthrange,
            }
        return actuator_dict

    def _query_all_bodies(self) -> dict:
        m = self._mj_model
        body_dict = {}
        for i in range(m.nbody):
            body = m.body(i)
            body_dict[body.name] = {
                "ID": body.id,
                "ParentID": body.parentid[0],
                "RootID": body.rootid[0],
                "WeldID": body.weldid[0],
                "MocapID": body.mocapid[0],
                "JntNum": body.jntnum[0],
                "JntAdr": body.jntadr[0],
                "DofNum": body.dofnum[0],
                "DofAdr": body.dofadr[0],
                "TreeID": m.body_treeid[i],
                "GeomNum": body.geomnum[0],
                "GeomAdr": body.geomadr[0],
                "Simple": body.simple[0],
                "SameFrame": body.sameframe[0],
                "Pos": body.pos,
                "Quat": body.quat,
                "IPos": body.ipos,
                "IQuat": body.iquat,
                "Mass": body.mass[0],
                "SubtreeMass": body.subtreemass[0],
                "Inertia": body.inertia,
                "InvWeight": body.invweight0,
                "GravComp": m.body_gravcomp[i],
                "Margin": m.body_margin[i],
            }
        return body_dict

    def _query_all_joints(self) -> dict:
        m = self._mj_model
        joint_dict = {}
        for i in range(m.njnt):
            joint = m.joint(i)
            joint_dict[joint.name] = {
                "ID": joint.id,
                "BodyID": joint.bodyid[0],
                "Type": joint.type[0],
                "Range": joint.range,
                "QposIdxStart": joint.qposadr[0],
                "QvelIdxStart": joint.dofadr[0],
                "Group": joint.group[0],
                "Limited": bool(joint.limited[0]),
                "ActfrcLimited": bool(m.jnt_actfrclimited[i]),
                "Solref": joint.solref[0],
                "Solimp": joint.solimp[0],
                "Pos": joint.pos,
                "Axis": joint.axis,
                "Stiffness": joint.stiffness[0],
                "ActfrcRange": m.jnt_actfrcrange[i],
                "Margin": joint.margin[0],
                "Frictionloss": joint.frictionloss[0],
                "Damping": joint.damping[0],
            }
        return joint_dict

    def _query_all_geoms(self) -> dict:
        m = self._mj_model
        geom_dict = {}
        for i in range(m.ngeom):
            geom = m.geom(i)
            bodyname = m.body(geom.bodyid[0]).name
            geom_dict[geom.name] = {
                "BodyName": bodyname,
                "Type": geom.type[0],
                "Contype": geom.contype[0],
                "Conaffinity": geom.conaffinity[0],
                "Condim": geom.condim[0],
                "Solmix": geom.solmix[0],
                "Solref": geom.solref,
                "Solimp": geom.solimp,
                "Size": geom.size,
                "Friction": geom.friction,
                "DataID": geom.dataid[0],
                "MatID": geom.matid[0],
                "Group": geom.group[0],
                "Priority": geom.priority[0],
                "Plugin": -1,
                "SameFrame": geom.sameframe[0],
                "Pos": geom.pos,
                "Quat": geom.quat,
                "Margin": geom.margin[0],
                "Gap": geom.gap[0],
            }
        return geom_dict

    def _query_all_sites(self) -> dict:
        m = self._mj_model
        site_dict = {}
        for i in range(m.nsite):
            site = m.site(i)
            user_data = list(m.site_user[i]) if m.nuser_site > 0 else []
            site_dict[site.name] = {
                "ID": site.id,
                "BodyID": site.bodyid[0],
                "Type": site.type[0],
                "Pos": site.pos,
                "Mat": site.matid[0],
                "LocalPos": site.pos,
                "LocalQuat": site.quat,
                "Size": site.size,
                "User": user_data,
            }
        return site_dict

    def _query_all_sensors(self) -> dict:
        m = self._mj_model
        sensor_dict = {}
        for i in range(m.nsensor):
            sensor = m.sensor(i)
            sensor_dict[sensor.name] = {
                "ID": sensor.id,
                "Type": sensor.type[0],
                "ObjID": sensor.objid[0],
                "Dim": sensor.dim[0],
                "Adr": sensor.adr[0],
                "Noise": sensor.noise[0],
            }
        return sensor_dict

    def _query_all_meshes(self) -> dict:
        m = self._mj_model
        mesh_files: dict[str, str] = {}
        mesh_scales: dict[str, list] = {}

        if self._xml_path and os.path.exists(self._xml_path):
            try:
                tree = ET.parse(self._xml_path)
                root = tree.getroot()
                xml_dir = os.path.dirname(os.path.abspath(self._xml_path))
                meshdir = ""
                compiler = root.find("compiler")
                if compiler is not None:
                    meshdir = compiler.get("meshdir", "")
                for mesh_elem in root.findall(".//asset/mesh"):
                    name = mesh_elem.get("name")
                    file = mesh_elem.get("file", "")
                    scale_str = mesh_elem.get("scale", "")
                    if name and file:
                        full_path = os.path.join(xml_dir, meshdir, file) if meshdir else os.path.join(xml_dir, file)
                        mesh_files[name] = os.path.abspath(full_path)
                        if scale_str:
                            try:
                                vals = [float(x) for x in scale_str.split()]
                                mesh_scales[name] = vals * 3 if len(vals) == 1 else vals
                            except ValueError:
                                pass
            except Exception:
                pass

        mesh_dict = {}
        for i in range(m.nmesh):
            mesh = m.mesh(i)
            mesh_dict[mesh.name] = {
                "ID": mesh.id,
                "File": mesh_files.get(mesh.name, ""),
                "Scale": mesh_scales.get(mesh.name, [1.0, 1.0, 1.0]),
            }
        return mesh_dict
