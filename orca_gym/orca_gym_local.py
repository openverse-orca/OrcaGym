import sys
import os
import grpc

current_dir = os.path.dirname(os.path.abspath(__file__))
proto_path = os.path.abspath(os.path.join(current_dir, "protos"))
sys.path.append(proto_path)
import mjc_message_pb2
import mjc_message_pb2_grpc

import numpy as np
import scipy.linalg
from datetime import datetime

from orca_gym.orca_gym_model import OrcaGymModel
from orca_gym.orca_gym_data import OrcaGymData
from orca_gym.orca_gym_opt_config import OrcaGymOptConfig
from orca_gym.orca_gym import OrcaGymBase

import mujoco

class OrcaGymLocal(OrcaGymBase):
    """
    OrcaGymLocal class
    """
    def __init__(self, stub):
        super().__init__(stub = stub)

        self._timestep = 0.001
        self._mjModel = None
        self._mjData = None


    async def init_simulation(self):

        model_xml_path = "/home/superfhwl/repo/OrcaStudio_2409/build/bin/profile/out.xml"

        self._mjModel = mujoco.MjModel.from_xml_path(model_xml_path)
        self._mjData = mujoco.MjData(self._mjModel)

        # Update the timestep setting form the env parameter.
        self.set_opt_timestep(self._timestep)

        opt_config = self.query_opt_config()
        self.opt = OrcaGymOptConfig(opt_config)
        self.print_opt_config()

        model_info = self.query_model_info()
        self.model = OrcaGymModel(model_info)      # 对应 Mujoco Model
        self.print_model_info(model_info)

        eq_list = self.query_all_equality_constraints()
        self.model.init_eq_list(eq_list)
        mocap_dict = self.query_all_mocap_bodies()
        self.model.init_mocap_dict(mocap_dict)
        actuator_dict = self.query_all_actuators()
        self.model.init_actuator_dict(actuator_dict)
        body_dict = self.query_all_bodies()
        self.model.init_body_dict(body_dict)
        joint_dict = self.query_all_joints()
        self.model.init_joint_dict(joint_dict)
        geom_dict = self.query_all_geoms()
        self.model.init_geom_dict(geom_dict)
        site_dict = self.query_all_sites()
        self.model.init_site_dict(site_dict)

        self.data = OrcaGymData(self.model)
        self.update_data()

    def set_time_step(self, time_step):
        self._timestep = time_step
        self.set_opt_timestep(time_step)

    def set_opt_timestep(self, timestep):
        if self._mjModel is not None:
            self._mjModel.opt.timestep = timestep

    def query_opt_config(self):
        opt_config = {
            "timestep": self._mjModel.opt.timestep,
            "apirate": self._mjModel.opt.apirate,
            "impratio": self._mjModel.opt.impratio,
            "tolerance": self._mjModel.opt.tolerance,
            "ls_tolerance": self._mjModel.opt.ls_tolerance,
            "noslip_tolerance": self._mjModel.opt.noslip_tolerance,
            "mpr_tolerance": self._mjModel.opt.mpr_tolerance,
            "gravity": list(self._mjModel.opt.gravity),
            "wind": list(self._mjModel.opt.wind),
            "magnetic": list(self._mjModel.opt.magnetic),
            "density": self._mjModel.opt.density,
            "viscosity": self._mjModel.opt.viscosity,
            "o_margin": self._mjModel.opt.o_margin,
            "o_solref": list(self._mjModel.opt.o_solref),
            "o_solimp": list(self._mjModel.opt.o_solimp),
            "o_friction": list(self._mjModel.opt.o_friction),
            "integrator": self._mjModel.opt.integrator,
            "cone": self._mjModel.opt.cone,
            "jacobian": self._mjModel.opt.jacobian,
            "solver": self._mjModel.opt.solver,
            "iterations": self._mjModel.opt.iterations,
            "ls_iterations": self._mjModel.opt.ls_iterations,
            "noslip_iterations": self._mjModel.opt.noslip_iterations,
            "mpr_iterations": self._mjModel.opt.mpr_iterations,
            "disableflags": self._mjModel.opt.disableflags,
            "enableflags": self._mjModel.opt.enableflags,
            "disableactuator": self._mjModel.opt.disableactuator,
            "sdf_initpoints": self._mjModel.opt.sdf_initpoints,
            "sdf_iterations": self._mjModel.opt.sdf_iterations
        }
        return opt_config

    def query_model_info(self):
        model_info = {
            'nq': self._mjModel.nq,
            'nv': self._mjModel.nv,
            'nu': self._mjModel.nu,
            'nbody': self._mjModel.nbody,
            'njnt': self._mjModel.njnt,
            'ngeom': self._mjModel.ngeom,
            'nsite': self._mjModel.nsite,
            'nmesh': self._mjModel.nmesh,
            'ncam': self._mjModel.ncam,
            'nlight': self._mjModel.nlight,
            'nuser_body': self._mjModel.nuser_body,
            'nuser_jnt': self._mjModel.nuser_jnt,
            'nuser_geom': self._mjModel.nuser_geom,
            'nuser_site': self._mjModel.nuser_site,
            'nuser_tendon': self._mjModel.nuser_tendon,
            'nuser_actuator': self._mjModel.nuser_actuator,
            'nuser_sensor': self._mjModel.nuser_sensor,
        }
        return model_info
    
    def query_all_equality_constraints(self):
        # 遍历模型中的所有等式约束
        model = self._mjModel

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

    def query_all_mocap_bodies(self):
        model = self._mjModel
        mocap_body_dict = {}
        for i in range(model.nbody):
            if model.body_mocapid[i] != -1:
                mocap_body_dict[model.body(i).name] = model.body_mocapid[i]

        return mocap_body_dict

    def query_all_actuators(self):
        model = self._mjModel
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

    def query_all_bodies(self):
        model = self._mjModel
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

    def query_all_joints(self):
        model = self._mjModel
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
            }
            
        return joint_dict

    def query_all_geoms(self):
        model = self._mjModel
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
    
    def query_all_sites(self):
        model = self._mjModel
        site_dict = {}
        for i in range(model.nsite):
            site = model.site(i)
            bodyname = model.body(site.bodyid[0]).name
            site_dict[site.name] = {
                "ID": site.id,
                "BodyID": site.bodyid[0],
                "Type": site.type[0],
                "Pos": site.pos,
                "Mat": site.matid[0],
                "LocalPos": site.pos,
                "LocalQuat": site.quat,
            }

        return site_dict 
    
    def update_data(self):
        qpos, qvel, qacc = self.query_all_qpos_qvel_qacc()
        qfrc_bias = self.query_qfrc_bias()
        
        self.data.update_qpos_qvel_qacc(qpos, qvel, qacc)        
        self.data.update_qfrc_bias(qfrc_bias)

        sim_time = datetime.now() - self.model._init_time
        self.data.update_time(sim_time.total_seconds())

    def query_all_qpos_qvel_qacc(self):
        data = self._mjData
        qpos = data.qpos.copy()
        qvel = data.qvel.copy()
        qacc = data.qacc.copy()

        return qpos, qvel, qacc
    
    def query_qfrc_bias(self):
        data = self._mjData
        qfrc_bias = data.qfrc_bias.copy()
        return qfrc_bias
    
    def load_initial_frame(self):
        mujoco.mj_resetData(self._mjModel, self._mjData)