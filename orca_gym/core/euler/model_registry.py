"""ModelRegistry — 模型注册与结构查询（阶段二填充）。

本模块属于 OrcaGym Euler 体系阶段二（P4-Step3），负责从 `_mj_model`
构建 `OrcaGymModel`（原样复用老体系 `query_all_*` 逻辑，但直接从
`_mj_model` 读取，不走 gRPC）。

阶段二 Step 3 实现 `build_orca_gym_model`，填充 body/joint/actuator/
site/sensor/geom/mesh/equality/mocap 字典。`build_orca_gym_data` 保留
`NotImplementedError`（阶段二使用 `OrcaGymDataView` 替代 `OrcaGymData`）。

注意:
    `build_orca_gym_data` 返回 `OrcaGymData`（原体系的状态容器），
    而非 `OrcaGymDataView`。`OrcaGymDataView` 由 `MuJoCoSimCore`
    的 `sync_to_view` 填充，不由此处构建。
"""

from typing import Tuple

import mujoco
import numpy as np

from orca_gym.core.orca_gym_model import OrcaGymModel


class ModelRegistry:
    """模型注册与结构查询。

    构建 OrcaGymModel/OrcaGymData，提供 body/equality 等模型信息查询。

    使用契约:
        绑定模型:   registry._bind(mj_model)  # 供 OrcaGymEuler.init_simulation 调用
        构建模型:   model = registry.build_orca_gym_model()
        查询结构:   mass = registry.body_subtree_mass("link1")

    禁止:
        不要通过本类直接访问 _mjModel.opt.*（用 SimConfig）。
    """

    def __init__(self, mj_model=None) -> None:
        """初始化模型注册器。

        Args:
            mj_model: MuJoCo 模型对象。None 时 build_orca_gym_model 抛 RuntimeError，
                待 _bind(mj_model) 绑定后可用。
        """
        self._mj_model = mj_model

    # --- 绑定方法（供 OrcaGymEuler.init_simulation 后调用）---

    def _bind(self, mj_model) -> None:
        """绑定真实 mjModel。

        供 OrcaGymEuler.init_simulation 在加载 mjModel 后调用。

        Args:
            mj_model: MuJoCo MjModel 对象。
        """
        self._mj_model = mj_model

    # --- 构建方法 ---

    def build_orca_gym_model(self) -> OrcaGymModel:
        """构建 OrcaGymModel 实例（模型结构抽象，原样复用）。

        从 `_mj_model` 读取维度参数和各结构字典，复用老体系 `query_all_*`
        的逻辑，但直接从 `_mj_model` 读取（不走 gRPC）。

        Returns:
            OrcaGymModel 实例，已填充 body/joint/actuator/site/sensor/geom/
            mesh/equality/mocap 字典。

        Raises:
            RuntimeError: 未绑定 mjModel 时抛错。
        """
        if self._mj_model is None:
            raise RuntimeError("ModelRegistry not bound to mjModel")

        m = self._mj_model
        model_info = {
            'nq': m.nq, 'nv': m.nv, 'nu': m.nu,
            'nbody': m.nbody, 'njnt': m.njnt, 'ngeom': m.ngeom,
            'nsite': m.nsite, 'nmesh': m.nmesh, 'ncam': m.ncam,
            'nlight': m.nlight, 'nuser_body': m.nuser_body,
            'nuser_jnt': m.nuser_jnt, 'nuser_geom': m.nuser_geom,
            'nuser_site': m.nuser_site, 'nuser_tendon': m.nuser_tendon,
            'nuser_actuator': m.nuser_actuator, 'nuser_sensor': m.nuser_sensor,
            'nconmax': m.nconmax, 'nflex': m.nflex, 'nflexvert': m.nflexvert,
            'flex_vertbodyid': list(m.flex_vertbodyid),
            'flex_vertadr': list(m.flex_vertadr) if m.nflex > 0 else [],
            'flex_vertnum': list(m.flex_vertnum) if m.nflex > 0 else [],
            'flex_names': [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_FLEX, i)
                           for i in range(m.nflex)] if m.nflex > 0 else [],
        }
        model = OrcaGymModel(model_info)

        # 填充各结构字典（复用老体系 query_all_* 逻辑）
        model.init_body_dict(self._query_all_bodies())
        model.init_joint_dict(self._query_all_joints())
        model.init_actuator_dict(self._query_all_actuators())
        model.init_site_dict(self._query_all_sites())
        model.init_sensor_dict(self._query_all_sensors())
        model.init_geom_dict(self._query_all_geoms())
        model.init_mesh_dict(self._query_all_meshes())
        model.init_eq_list(self._query_all_equality_constraints())
        model.init_mocap_dict(self._query_all_mocap_bodies())

        return model

    def build_orca_gym_data(self):
        """构建 OrcaGymData 实例（原体系的状态容器，非 DataView）。

        注意: 此处返回 OrcaGymData（原体系），不是 OrcaGymDataView。
        OrcaGymDataView 由 MuJoCoSimCore.sync_to_view 填充。

        Returns:
            OrcaGymData 实例。

        Raises:
            NotImplementedError: 阶段二使用 OrcaGymDataView 替代 OrcaGymData，
                此方法留待完整 P4 实现。
        """
        raise NotImplementedError("build_orca_gym_data 阶段二不实现（使用 OrcaGymDataView）")

    # --- 内部查询方法（复用老体系 query_all_* 逻辑，直接从 _mj_model 读取）---

    def _query_all_bodies(self) -> dict:
        """查询所有 body 信息（复用老体系 query_all_bodies 逻辑）。"""
        model = self._mj_model
        body_dict = {}
        for i in range(model.nbody):
            body = model.body(i)
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
                "TreeID": model.body_treeid[i],
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
                "GravComp": model.body_gravcomp[i],
                "Margin": model.body_margin[i],
            }
        return body_dict

    def _query_all_joints(self) -> dict:
        """查询所有关节信息（复用老体系 query_all_joints 逻辑）。"""
        model = self._mj_model
        joint_dict = {}
        for i in range(model.njnt):
            joint = model.joint(i)
            joint_dict[joint.name] = {
                "ID": joint.id,
                "BodyID": joint.bodyid[0],
                "Type": joint.type[0],
                "Range": joint.range,
                "QposIdxStart": joint.qposadr[0],
                "QvelIdxStart": joint.dofadr[0],
                "Group": joint.group[0],
                "Limited": bool(joint.limited[0]),
                "ActfrcLimited": bool(model.jnt_actfrclimited[i]),
                "Solref": joint.solref[0],
                "Solimp": joint.solimp[0],
                "Pos": joint.pos,
                "Axis": joint.axis,
                "Stiffness": joint.stiffness[0],
                "ActfrcRange": model.jnt_actfrcrange[i],
                "Margin": joint.margin[0],
                "Frictionloss": joint.frictionloss[0],
                "Damping": joint.damping[0],
            }
        return joint_dict

    def _query_all_actuators(self) -> dict:
        """查询所有执行器信息（复用老体系 query_all_actuators 逻辑）。"""
        model = self._mj_model
        actuator_dict = {}
        idx = 0
        for i in range(model.nu):
            actuator = model.actuator(i)

            actuator_name = actuator.name
            if actuator_name == "":
                actuator_name = "actuator"

            if actuator_name in actuator_dict:
                actuator_name = actuator_name + f"_{idx}"
                idx += 1

            if actuator.trntype == mujoco.mjtTrn.mjTRN_JOINT:
                joint_name = model.joint(actuator.trnid[0]).name
            elif actuator.trntype == mujoco.mjtTrn.mjTRN_TENDON:
                joint_name = model.tendon(actuator.trnid[0]).name
            elif actuator.trntype == mujoco.mjtTrn.mjTRN_SITE:
                joint_name = model.site(actuator.trnid[0]).name
            else:
                joint_name = "unknown"

            actuator_dict[actuator_name] = {
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
                "ActEarly": bool(model.actuator_actearly[i]),
                "Gear": actuator.gear,
                "CrankLength": actuator.cranklength[0],
                "Acc0": actuator.acc0[0],
                "Length0": actuator.length0[0],
                "LengthRange": actuator.lengthrange,
            }
        return actuator_dict

    def _query_all_sites(self) -> dict:
        """查询所有 site 信息（复用老体系 query_all_sites 逻辑）。"""
        model = self._mj_model
        site_dict = {}
        for i in range(model.nsite):
            site = model.site(i)
            user_data = list(model.site_user[i]) if model.nuser_site > 0 else []
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
        """查询所有传感器信息（复用老体系 query_all_sensors 逻辑）。"""
        model = self._mj_model
        sensor_dict = {}
        for i in range(model.nsensor):
            sensor = model.sensor(i)
            sensor_dict[sensor.name] = {
                "ID": sensor.id,
                "Type": sensor.type[0],
                "ObjID": sensor.objid[0],
                "Dim": sensor.dim[0],
                "Adr": sensor.adr[0],
                "Noise": sensor.noise[0]
            }
        return sensor_dict

    def _query_all_geoms(self) -> dict:
        """查询所有几何体信息（复用老体系 query_all_geoms 逻辑）。"""
        model = self._mj_model
        geom_dict = {}
        for i in range(model.ngeom):
            geom = model.geom(i)
            bodyname = model.body(geom.bodyid[0]).name
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

    def _query_all_meshes(self) -> dict:
        """查询所有 mesh 信息。

        注意：MuJoCo Python 绑定的 mesh 对象不包含原始文件路径，
        仅填充 mesh 名称与基本结构信息。
        """
        model = self._mj_model
        mesh_dict = {}
        for i in range(model.nmesh):
            mesh_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MESH, i)
            mesh_dict[mesh_name] = {
                "ID": i,
                "Name": mesh_name,
            }
        return mesh_dict

    def _query_all_equality_constraints(self) -> list:
        """查询所有等式约束（复用老体系 query_all_equality_constraints 逻辑）。"""
        model = self._mj_model
        equality_constraints = []
        for i in range(model.neq):
            eq_info = {
                "eq_type": model.eq_type[i],
                "obj1_id": model.eq_obj1id[i],
                "obj2_id": model.eq_obj2id[i],
                "active": model.eq_active0[i],
                "eq_solref": model.eq_solref[i],
                "eq_solimp": model.eq_solimp[i],
                "eq_data": model.eq_data[i],
            }
            equality_constraints.append(eq_info)
        return equality_constraints

    def _query_all_mocap_bodies(self) -> dict:
        """查询所有 mocap body（复用老体系 query_all_mocap_bodies 逻辑）。"""
        model = self._mj_model
        mocap_body_dict = {}
        for i in range(model.nbody):
            if model.body_mocapid[i] != -1:
                mocap_body_dict[model.body(i).name] = model.body_mocapid[i]
        return mocap_body_dict

    # --- 扩展查询方法（架构 §5.5，覆盖用户绕道访问的模型结构）---
    # 阶段三 3.1.5 填充：替换 NotImplementedError 为真实实现。

    def body_subtree_mass(self, body_name: str) -> float:
        """查询 body 子树总质量。

        替代直接访问 _mjModel.body_subtreemass[id]。

        Args:
            body_name: body 名称（已含 agent 前缀）。

        Returns:
            body 子树总质量（Python float 标量，非 numpy 泄漏）。
        """
        body_id = mujoco.mj_name2id(
            self._mj_model, mujoco.mjtObj.mjOBJ_BODY, body_name
        )
        mass = float(self._mj_model.body_subtreemass[body_id])
        return mass

    def geom_friction(self, geom_name: str) -> np.ndarray:
        """查询 geom 摩擦系数 (3,) [sliding, torsion, rolling]（只读视图）。

        替代直接访问 _mjModel.geom_friction[id]。

        Args:
            geom_name: geom 名称。

        Returns:
            geom 摩擦系数数组，形状 (3,)。
        """
        geom_id = mujoco.mj_name2id(
            self._mj_model, mujoco.mjtObj.mjOBJ_GEOM, geom_name
        )
        return np.asarray(self._mj_model.geom_friction[geom_id]).copy()

    def equality_data_width(self) -> int:
        """查询等式约束数据宽度（eq_data 每行元素数）。

        替代直接访问 _mjModel.eq_data.shape[1]。

        Returns:
            等式约束数据宽度（Python int）。无等式约束时返回 0。
        """
        if self._mj_model.neq == 0:
            return 0
        width = int(self._mj_model.eq_data.shape[1])
        return width

    def equality_object_ids(self, eq_idx: int) -> Tuple[int, int]:
        """查询等式约束关联的两个对象 id。

        替代直接访问 _mjModel.eq_obj1id[eq_idx] / eq_obj2id[eq_idx]。

        Args:
            eq_idx: 等式约束索引。

        Returns:
            (obj1_id, obj2_id) 元组，均为 Python int。
        """
        obj1 = int(self._mj_model.eq_obj1id[eq_idx])
        obj2 = int(self._mj_model.eq_obj2id[eq_idx])
        return (obj1, obj2)

    def n_equality(self) -> int:
        """查询等式约束数量。

        替代直接访问 _mjModel.neq。

        Returns:
            等式约束数量（Python int）。
        """
        return int(self._mj_model.neq)

    def mocap_body_names(self) -> list[str]:
        """查询所有 mocap body 名称。

        替代直接访问 _mjModel.body_mocapid 遍历。

        Returns:
            mocap body 名称列表（list[str]）。
        """
        names: list[str] = []
        for i in range(self._mj_model.nbody):
            if self._mj_model.body_mocapid[i] != -1:
                name = mujoco.mj_id2name(
                    self._mj_model, mujoco.mjtObj.mjOBJ_BODY, i
                )
                if name:
                    names.append(name)
        return names
