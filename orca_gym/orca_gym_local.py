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



def get_qpos_size(joint_type):
    if joint_type == mujoco.mjtJoint.mjJNT_FREE:
        return 7
    elif joint_type == mujoco.mjtJoint.mjJNT_BALL:
        return 4
    elif joint_type == mujoco.mjtJoint.mjJNT_SLIDE or joint_type == mujoco.mjtJoint.mjJNT_HINGE:
        return 1
    else:
        return 0

def get_qvel_size(joint_type):
    if joint_type == mujoco.mjtJoint.mjJNT_FREE:
        return 6
    elif joint_type == mujoco.mjtJoint.mjJNT_BALL:
        return 3
    elif joint_type == mujoco.mjtJoint.mjJNT_SLIDE or joint_type == mujoco.mjtJoint.mjJNT_HINGE:
        return 1
    else:
        return 0

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

        model_xml_path = await self.load_local_env()
        print("Model XML Path: ", model_xml_path)

        self._mjModel = mujoco.MjModel.from_xml_path(model_xml_path)
        self._mjData = mujoco.MjData(self._mjModel)

        size_model = mujoco.mj_sizeModel(self._mjModel)
        print("size_model: ", size_model)


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

    async def render(self):
        await self.update_local_env(self.data.qpos, self._mjData.time)

    async def update_local_env(self, qpos, time):
        request = mjc_message_pb2.UpdateLocalEnvRequest(qpos=qpos, time=time)
        response = await self.stub.UpdateLocalEnv(request)
        return response
    
    async def load_local_env(self):
        request = mjc_message_pb2.LoadLocalEnvRequest()
        response = await self.stub.LoadLocalEnv(request)
        return response.xml_path

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

    def query_all_qpos_qvel_qacc(self):
        qpos = self._mjData.qpos
        qvel = self._mjData.qvel
        qacc = self._mjData.qacc

        return qpos, qvel, qacc
    
    def query_qfrc_bias(self):
        qfrc_bias = self._mjData.qfrc_bias
        return qfrc_bias
    
    def load_initial_frame(self):
        mujoco.mj_resetData(self._mjModel, self._mjData)

    def query_joint_offsets(self, joint_names):
        # 按顺序构建 offset 数组
        qpos_offsets = []
        qvel_offsets = []
        qacc_offsets = []

        # 将响应中每个关节的 offset 按顺序添加到数组中
        for joint_name in joint_names:
            joint_id = self._mjModel.joint(joint_name).id
            qpos_offsets.append(self._mjModel.jnt_qposadr[joint_id])
            qvel_offsets.append(self._mjModel.jnt_dofadr[joint_id])
            qacc_offsets.append(self._mjModel.jnt_dofadr[joint_id] + self._mjModel.njnt)

        return qpos_offsets, qvel_offsets, qacc_offsets    
    
    def query_body_xpos_xmat_xquat(self, body_name_list):
        body_pos_mat_quat_list = {}
        for body_name in body_name_list:
            body_id = self._mjModel.body(body_name).id
            body_pos_mat_quat = {
                "Pos": self._mjData.xpos[body_id],
                "Mat": self._mjData.xmat[body_id],
                "Quat": self._mjData.xquat[body_id],
            }
            body_pos_mat_quat_list[body_name] = body_pos_mat_quat
            
        return body_pos_mat_quat_list
    
    def query_sensor_data(self, sensor_names):
        sensor_data_dict = {}
        for sensor_name in sensor_names:
            sensor_id = self._mjModel.sensor(sensor_name).id
            sensor_dim = self._mjModel.sensor_dim[sensor_id]
            sensor_type = self._mjModel.sensor_type[sensor_id]

            if sensor_type == mujoco.mjtSensor.mjSENS_ACCELEROMETER:
                sensor_type_str = "accelerometer"
            elif sensor_type == mujoco.mjtSensor.mjSENS_GYRO:
                sensor_type_str = "gyro"
            elif sensor_type == mujoco.mjtSensor.mjSENS_TOUCH:
                sensor_type_str = "touch"
            elif sensor_type == mujoco.mjtSensor.mjSENS_VELOCIMETER:
                sensor_type_str = "velocimeter"
            elif sensor_type == mujoco.mjtSensor.mjSENS_FRAMEQUAT:
                sensor_type_str = "framequat"
            else:
                sensor_type_str = "unknown"

            sensor_values = np.copy(self._mjData.sensordata[sensor_id:sensor_id + sensor_dim])

            sensor_data_dict[sensor_name] = {
                "type": sensor_type_str,
                "values": sensor_values,
            }

        # print("Sensor Data Dict: ", sensor_data_dict)

        return sensor_data_dict    
    
    def set_ctrl(self, ctrl):
        self._mjData.ctrl = ctrl.copy()

    def mj_step(self, nstep):
        mujoco.mj_step(self._mjModel, self._mjData, nstep)

    def mj_forward(self):
        mujoco.mj_forward(self._mjModel, self._mjData)

    def mj_inverse(self):
        mujoco.mj_inverse(self._mjModel, self._mjData)

    def query_joint_qpos(self, joint_names):
        joint_qpos_dict = {}
        for joint_name in joint_names:
            joint_id = self._mjModel.joint(joint_name).id
            joint_type = self._mjModel.jnt_type[joint_id]
            joint_qpos = self._mjData.qpos[self._mjModel.jnt_qposadr[joint_id]:self._mjModel.jnt_qposadr[joint_id] + get_qpos_size(joint_type)]
            joint_qpos_dict[joint_name] = joint_qpos
        return joint_qpos_dict
    
    def query_joint_qvel(self, joint_names):
        joint_qvel_dict = {}
        for joint_name in joint_names:
            joint_id = self._mjModel.joint(joint_name).id
            joint_qvel_dict[joint_name] = self._mjData.qvel[self._mjModel.jnt_dofadr[joint_id]]
        return joint_qvel_dict
    
    def jnt_qposadr(self, joint_name):
        joint_id = self._mjModel.joint(joint_name).id
        return self._mjModel.jnt_qposadr[joint_id]
    
    def jnt_dofadr(self, joint_name):
        joint_id = self._mjModel.joint(joint_name).id
        return self._mjModel.jnt_dofadr[joint_id]
    
    def query_site_pos_and_mat(self, site_names: list[str]):
        site_pos_and_mat = {}
        for site_name in site_names:
            site_id = self._mjModel.site(site_name).id
            site = self._mjData.site_xpos[site_id]
            site_mat = self._mjData.site_xmat[site_id]
            site_pos_and_mat[site_name] = {"xpos": site, "xmat": site_mat}
        return site_pos_and_mat
    
    def set_joint_qpos(self, joint_qpos):
        for joint_name, qpos in joint_qpos.items():
            joint_id = self._mjModel.joint(joint_name).id
            qpos_size = get_qpos_size(self._mjModel.jnt_type[joint_id])
            self._mjData.qpos[self._mjModel.jnt_qposadr[joint_id]:self._mjModel.jnt_qposadr[joint_id] + qpos_size] = qpos.copy()

    def mj_jac_site(self, site_names: list[str]):
        site_jacs_dict = {}
        for site_name in site_names:
            site_id = self._mjModel.site(site_name).id
            jacp = np.zeros((3, self.model.nv))
            jacr = np.zeros((3, self.model.nv))
            mujoco.mj_jacSite(self._mjModel, self._mjData, jacp, jacr, site_id)
            site_jacs_dict[site_name] = {"jacp": jacp, "jacr": jacr}
        return site_jacs_dict            
    
    def update_equality_constraints(self, constraint_list):
        for constraint in constraint_list:
            obj1_id = constraint['obj1_id']
            obj2_id = constraint['obj2_id']
            eq_data = constraint['eq_data']
            for i in range(self.model.neq):
                if self._mjModel.eq_obj1id[i] == obj1_id and self._mjModel.eq_obj2id[i] == obj2_id:
                    self._mjModel.eq_data[i] = eq_data.copy()
                    break

            # print("eq_data: ", eq_data)
            # self._mjModel.eq_data[obj1_id:obj1_id + len(eq_data)] = eq_data.copy()
            # print("model.eq_data: ", self._mjModel.eq_data)


    async def _remote_set_mocap_pos_and_quat(self, mocap_data):
        request = mjc_message_pb2.SetMocapPosAndQuatRequest()
        for name, data in mocap_data.items():
            mocap_body_info = request.mocap_body_info.add()
            mocap_body_info.mocap_body_name = name
            mocap_body_info.pos.extend(data['pos'])
            mocap_body_info.quat.extend(data['quat'])

        response = await self.stub.SetMocapPosAndQuat(request)
        return response.success

    async def set_mocap_pos_and_quat(self, mocap_data, send_remote = False):
        for name, data in mocap_data.items():
            body_id = self._mjModel.body(name).id
            mocap_id = self._mjModel.body_mocapid[body_id]
            if mocap_id != -1:
                # print("mocap_pos: ", self._mjData.mocap_pos[mocap_id])
                # print("mocap_quat: ", self._mjData.mocap_quat[mocap_id])
                self._mjData.mocap_pos[mocap_id] = data['pos'].copy()
                self._mjData.mocap_quat[mocap_id] = data['quat'].copy()
        
        if send_remote:
            await self._remote_set_mocap_pos_and_quat(mocap_data)