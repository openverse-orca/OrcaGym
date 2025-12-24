import sys
import os
import grpc

proj_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
proto_path = os.path.abspath(os.path.join(proj_dir, "protos"))
sys.path.append(proto_path)
import mjc_message_pb2
import mjc_message_pb2_grpc

import numpy as np
import scipy.linalg
from datetime import datetime

from orca_gym.log.orca_log import get_orca_logger
_logger = get_orca_logger()

from orca_gym.core.orca_gym_model import OrcaGymModel
from orca_gym.core.orca_gym_data import OrcaGymData
from orca_gym.core.orca_gym_opt_config import OrcaGymOptConfig
from orca_gym.core.orca_gym import OrcaGymBase

class OrcaGymRemote(OrcaGymBase):
    """
    OrcaGymRemote class
    """
    def __init__(self, stub):
        super().__init__(stub = stub)

    # async def query_agents(self):
    #     body_info_list = await self.query_all_bodies()
    #     self.model.update_body_info(body_info_list)
    #     joint_info_list = await self.query_all_joints()
    #     self.model.update_joint_info(joint_info_list)
    #     actuator_info_list = await self.query_all_actuators()
    #     self.model.update_actuator_info(actuator_info_list)

    async def init_simulation(self):
        opt_config = await self.query_opt_config()
        self.opt = OrcaGymOptConfig(opt_config)
        self.print_opt_config()

        model_info = await self.query_model_info()
        self.model = OrcaGymModel(model_info)      # 对应 Mujoco Model
        self.print_model_info(model_info)

        eq_list = await self.query_all_equality_constraints()
        self.model.init_eq_list(eq_list)
        mocap_dict = await self.query_all_mocap_bodies()
        self.model.init_mocap_dict(mocap_dict)
        actuator_dict = await self.query_all_actuators()
        self.model.init_actuator_dict(actuator_dict)
        body_dict = await self.query_all_bodies()
        self.model.init_body_dict(body_dict)
        joint_dict = await self.query_all_joints()
        self.model.init_joint_dict(joint_dict)
        geom_dict = await self.query_all_geoms()
        self.model.init_geom_dict(geom_dict)
        site_dict = await self.query_all_sites()
        self.model.init_site_dict(site_dict)

        self.data = OrcaGymData(self.model)
        await self.update_data()

    async def update_data(self):
        qpos, qvel, qacc = await self.query_all_qpos_qvel_qacc()
        qfrc_bias = await self.query_qfrc_bias()
        
        self.data.update_qpos_qvel_qacc(qpos, qvel, qacc)        
        self.data.update_qfrc_bias(qfrc_bias)

    async def query_all_actuators(self):
        request = mjc_message_pb2.QueryAllActuatorsRequest()
        response = await self.stub.QueryAllActuators(request)
        actuator_dict = {}
        idx = 0
        for actuator in response.ActuatorDataList:
            actuator_name = actuator.ActuatorName
            if actuator_name == "":
                actuator_name = "actuator"

            if actuator_name in actuator_dict:
                actuator_name = actuator_name + f"_{idx}"
                idx += 1

            actuator_dict[actuator_name] = {
                "JointName": actuator.JointName,
                "GearRatio": actuator.GearRatio,
                "TrnId": list(actuator.actuator_trnid),
                "CtrlLimited": actuator.actuator_ctrllimited,
                "ForceLimited": actuator.actuator_forcelimited,
                "ActLimited": actuator.actuator_actlimited,
                "CtrlRange": list(actuator.actuator_ctrlrange),
                "ForceRange": list(actuator.actuator_forcerange),
                "ActRange": list(actuator.actuator_actrange),
                "TrnType": actuator.actuator_trntype,
                "DynType": actuator.actuator_dyntype,
                "GainType": actuator.actuator_gaintype,
                "BiasType": actuator.actuator_biastype,
                "ActAdr": actuator.actuator_actadr,
                "ActNum": actuator.actuator_actnum,
                "Group": actuator.actuator_group,
                "DynPrm": list(actuator.actuator_dynprm),
                "GainPrm": list(actuator.actuator_gainprm),
                "BiasPrm": list(actuator.actuator_biasprm),
                "ActEarly": actuator.actuator_actearly,
                "Gear": list(actuator.actuator_gear),
                "CrankLength": actuator.actuator_cranklength,
                "Acc0": actuator.actuator_acc0,
                "Length0": actuator.actuator_length0,
                "LengthRange": list(actuator.actuator_lengthrange),
            }
        return actuator_dict


    async def query_joint_qpos(self, joint_names):
        request = mjc_message_pb2.QueryJointQposRequest(JointNameList=joint_names)
        response = await self.stub.QueryJointQpos(request)
        joint_qpos_dict = {joint.JointName: np.array(list(joint.Qpos)) for joint in response.JointQposList}
        return joint_qpos_dict

    async def query_joint_qvel(self, joint_names):
        request = mjc_message_pb2.QueryJointQvelRequest(JointNameList=joint_names)
        response = await self.stub.QueryJointQvel(request)
        joint_qvel_dict = {joint.JointName: np.array(list(joint.Qvel)) for joint in response.JointQvelList}
        return joint_qvel_dict

    async def get_agent_state(self, joint_names):
        qpos_request = mjc_message_pb2.QueryJointQposRequest(JointNameList=joint_names)
        qvel_request = mjc_message_pb2.QueryJointQvelRequest(JointNameList=joint_names)

        qpos_response = await self.stub.QueryJointQpos(qpos_request)
        qvel_response = await self.stub.QueryJointQvel(qvel_request)

        qpos = []
        qvel = []
        for joint in qpos_response.JointQposList:
            qpos.extend(joint.Qpos)

        for joint in qvel_response.JointQvelList:
            qvel.extend(joint.Qvel)

        return np.array(qpos), np.array(qvel)

    async def set_control_input(self, control_input):
        controls = [mjc_message_pb2.SetControlInputRequest.ActuatorControl(ActuatorName=name, ControlInput=value) for name, value in control_input.items()]
        request = mjc_message_pb2.SetControlInputRequest(Controls=controls)
        response = await self.stub.SetControlInput(request)
        return response

    async def load_initial_frame(self):
        request = mjc_message_pb2.LoadInitialFrameRequest()
        response = await self.stub.LoadInitialFrame(request)
        # print("Initial frame loaded")
        return response

    async def query_model_info(self):
        request = mjc_message_pb2.QueryModelInfoRequest()
        response = await self.stub.QueryModelInfo(request)
        model_info = {
            'nq': response.nq,
            'nv': response.nv,
            'nu': response.nu,
            'nbody': response.nbody,
            'njnt': response.njnt,
            'ngeom': response.ngeom,
            'nsite': response.nsite,
            'nmesh': response.nmesh,
            'ncam': response.ncam,
            'nlight': response.nlight,
            'nuser_body': response.nuser_body,
            'nuser_jnt': response.nuser_jnt,
            'nuser_geom': response.nuser_geom,
            'nuser_site': response.nuser_site,
            'nuser_tendon': response.nuser_tendon,
            'nuser_actuator': response.nuser_actuator,
            'nuser_sensor': response.nuser_sensor,
        }
        return model_info

    async def query_opt_config(self):
        request = mjc_message_pb2.QueryOptConfigRequest()
        response = await self.stub.QueryOptConfig(request)
        opt_config = {
            "timestep": response.timestep,
            "apirate": response.apirate,
            "impratio": response.impratio,
            "tolerance": response.tolerance,
            "ls_tolerance": response.ls_tolerance,
            "noslip_tolerance": response.noslip_tolerance,
            "ccd_tolerance": response.ccd_tolerance,
            "gravity": list(response.gravity),
            "wind": list(response.wind),
            "magnetic": list(response.magnetic),
            "density": response.density,
            "viscosity": response.viscosity,
            "o_margin": response.o_margin,
            "o_solref": list(response.o_solref),
            "o_solimp": list(response.o_solimp),
            "o_friction": list(response.o_friction),
            "integrator": response.integrator,
            "cone": response.cone,
            "jacobian": response.jacobian,
            "solver": response.solver,
            "iterations": response.iterations,
            "ls_iterations": response.ls_iterations,
            "noslip_iterations": response.noslip_iterations,
            "ccd_iterations": response.ccd_iterations,
            "disableflags": response.disableflags,
            "enableflags": response.enableflags,
            "disableactuator": response.disableactuator,
            "sdf_initpoints": response.sdf_initpoints,
            "sdf_iterations": response.sdf_iterations
        }
        return opt_config
    
    async def set_opt_config(self, opt_config):
        request = mjc_message_pb2.SetOptConfigRequest(
            timestep=opt_config["timestep"],
            apirate=opt_config["apirate"],
            impratio=opt_config["impratio"],
            tolerance=opt_config["tolerance"],
            ls_tolerance=opt_config["ls_tolerance"],
            noslip_tolerance=opt_config["noslip_tolerance"],
            ccd_tolerance=opt_config["ccd_tolerance"],
            gravity=opt_config["gravity"],
            wind=opt_config["wind"],
            magnetic=opt_config["magnetic"],
            density=opt_config["density"],
            viscosity=opt_config["viscosity"],
            o_margin=opt_config["o_margin"],
            o_solref=opt_config["o_solref"],
            o_solimp=opt_config["o_solimp"],
            o_friction=opt_config["o_friction"],
            integrator=opt_config["integrator"],
            cone=opt_config["cone"],
            jacobian=opt_config["jacobian"],
            solver=opt_config["solver"],
            iterations=opt_config["iterations"],
            ls_iterations=opt_config["ls_iterations"],
            noslip_iterations=opt_config["noslip_iterations"],
            ccd_iterations=opt_config["ccd_iterations"],
            disableflags=opt_config["disableflags"],
            enableflags=opt_config["enableflags"],
            disableactuator=opt_config["disableactuator"],
            sdf_initpoints=opt_config["sdf_initpoints"],
            sdf_iterations=opt_config["sdf_iterations"]
        )
        response = await self.stub.SetOptConfig(request)
        return response

    async def mj_differentiate_pos(self, initial_qpos, qpos):
        request = mjc_message_pb2.DifferentiatePosRequest(
            InitialQpos=initial_qpos,
            Qpos=qpos
        )
        response = await self.stub.MJ_DifferentiatePos(request)
        dq = np.array(response.Dq)
        return dq

    async def mjd_transition_fd(self, epsilon, flg_centered):
        request = mjc_message_pb2.TransitionFDRequest(Epsilon=epsilon, FlgCentered=flg_centered)
        response = await self.stub.MJD_TransitionFD(request)
        nv = response.nv
        nu = response.nu
        A = np.array(response.A).reshape((2 * nv, 2 * nv))
        B = np.array(response.B).reshape((2 * nv, nu))
        return A, B, nv, nu

    async def mj_jac_subtree_com(self, body_name):
        request = mjc_message_pb2.JacSubtreeComRequest(BodyName=body_name)
        response = await self.stub.MJ_JacSubtreeCom(request)
        jac_com = np.array(response.JacCom).reshape((3, -1))
        return jac_com

    async def mj_jac_body_com(self, body_name):
        request = mjc_message_pb2.JacBodyComRequest(BodyName=body_name)
        response = await self.stub.MJ_JacBodyCom(request)
        jac_p = np.array(response.JacP).reshape((3, -1))
        jac_r = np.array(response.JacR).reshape((3, -1))
        return jac_p, jac_r

    async def query_joint_names(self):
        request = mjc_message_pb2.QueryJointNamesRequest()
        response = await self.stub.QueryJointNames(request)
        return response.JointNames

    async def query_joint_dofadr(self, joint_names):
        request = mjc_message_pb2.QueryJointDofadrRequest(JointNames=joint_names)
        response = await self.stub.QueryJointDofadr(request)
        return response.JointDofadrs

    async def query_all_qpos_qvel_qacc(self):
        request = mjc_message_pb2.QueryAllQposQvelQaccRequest()
        response = await self.stub.QueryAllQposQvelQacc(request)
        qpos = np.array(response.qpos)
        qvel = np.array(response.qvel)
        qacc = np.array(response.qacc)
        return qpos, qvel, qacc

    async def load_keyframe(self, keyframe_name):
        request = mjc_message_pb2.LoadKeyFrameRequest(KeyFrameName=keyframe_name)
        response = await self.stub.LoadKeyFrame(request)
        return response.success
    
    async def resume_simulation(self):
        request = mjc_message_pb2.SetSimulationStateRequest(state=mjc_message_pb2.RUNNING)
        response = await self.stub.SetSimulationState(request)
        return response
    
    async def query_actuator_moment(self):
        request = mjc_message_pb2.QueryActuatorMomentRequest()
        response = await self.stub.QueryActuatorMoment(request)
        actuator_moment_flat = response.actuator_moment
        nv = response.nv
        nu = response.nu
        
        # 将一维数组转换为二维矩阵
        actuator_moment = np.array(actuator_moment_flat).reshape((nu, nv))
        
        return actuator_moment

    async def query_qfrc_inverse(self):
        request = mjc_message_pb2.QueryQfrcInverseRequest()
        response = await self.stub.QueryQfrcInverse(request)
        qfrc_inverse = response.qfrc_inverse
        return np.array(qfrc_inverse)

    async def query_qfrc_actuator(self):
        request = mjc_message_pb2.QueryQfrcActuatorRequest()
        response = await self.stub.QueryQfrcActuator(request)
        return np.array(response.qfrc_actuator)

    async def query_body_subtreemass_by_name(self, body_name):
        request = mjc_message_pb2.QueryBodySubtreeMassByNameRequest(body_name=body_name)
        response = await self.stub.QueryBodySubtreeMassByName(request)
        return response.body_subtreemass

    async def set_qacc(self, qacc):
        request = mjc_message_pb2.SetQaccRequest(qacc=qacc)
        response = await self.stub.SetQacc(request)
        return response

    async def set_opt_timestep(self, timestep):
        request = mjc_message_pb2.SetOptTimestepRequest(timestep=timestep)
        response = await self.stub.SetOptTimestep(request)
        return response

    async def set_ctrl(self, ctrl_values):
        request = mjc_message_pb2.SetCtrlRequest(ctrl=ctrl_values)
        response = await self.stub.SetCtrl(request)
        return response
    
    async def query_joint_type_by_id(self, joint_id):
        request = mjc_message_pb2.QueryJointTypeByIdRequest(joint_id=joint_id)
        response = await self.stub.QueryJointTypeById(request)
        return response.joint_type    
    
    async def query_all_joints(self):
        request = mjc_message_pb2.QueryAllJointsRequest()
        response = await self.stub.QueryAllJoints(request)
        joint_dict = {
            joint.name: {
                "ID": joint.id,
                "BodyID": joint.body_id,
                "Type": joint.type,
                "Range": list(joint.range),
                "QposIdxStart": joint.qpos_idx_start,
                "QvelIdxStart": joint.qvel_idx_start,
                "Group": joint.group,
                "Limited": joint.limited,
                "ActfrcLimited": joint.actfrclimited,
                "Solref": list(joint.solref),
                "Solimp": list(joint.solimp),
                "Pos": list(joint.pos),
                "Axis": list(joint.axis),
                "Stiffness": joint.stiffness,
                "ActfrcRange": list(joint.actfrcrange),
                "Margin": joint.margin,
            }
            for joint in response.joint_info
        }
        return joint_dict

    async def query_all_bodies(self):
        request = mjc_message_pb2.QueryAllBodiesRequest()
        response = await self.stub.QueryAllBodies(request)
        body_dict = {
            body.name: {
                "ID": body.id,
                "ParentID": body.parent_id,
                "RootID": body.root_id,
                "WeldID": body.weld_id,
                "MocapID": body.mocap_id,
                "JntNum": body.jnt_num,
                "JntAdr": body.jnt_adr,
                "DofNum": body.dof_num,
                "DofAdr": body.dof_adr,
                "TreeID": body.tree_id,
                "GeomNum": body.geom_num,
                "GeomAdr": body.geom_adr,
                "Simple": body.simple,
                "SameFrame": body.same_frame,
                "Pos": list(body.pos),
                "Quat": list(body.quat),
                "IPos": list(body.ipos),
                "IQuat": list(body.iquat),
                "Mass": body.mass,
                "SubtreeMass": body.subtree_mass,
                "Inertia": list(body.inertia),
                "InvWeight": list(body.inv_weight),
                "GravComp": body.grav_comp,
                "Margin": body.margin,
            }
            for body in response.body_info
        }
        return body_dict

    
        
    async def query_cfrc_ext(self, body_names):
        request = mjc_message_pb2.QueryCfrcExtRequest(body_names=body_names)
        response = await self.stub.QueryCfrcExt(request)
        if response.success:
            return {body_cfrc_ext.body_name: list(body_cfrc_ext.cfrc_ext) for body_cfrc_ext in response.body_cfrc_exts}
        else:
            raise Exception("Failed to query cfrc_ext")        
        
    async def set_joint_qpos(self, joint_qpos):
        request = mjc_message_pb2.SetJointQposRequest()
        for joint_name, qpos in joint_qpos.items():
            joint_qpos = request.joint_qpos_list.add()
            joint_qpos.JointName = joint_name
            joint_qpos.Qpos.extend(qpos)        

        response = await self.stub.SetJointQpos(request)
        return response.success
    
    async def query_actuator_force(self):
        request = mjc_message_pb2.QueryActuatorForceRequest()
        response = await self.stub.QueryActuatorForce(request)
        actuator_force = response.actuator_force
        return np.array(actuator_force)
    
    async def query_joint_limits(self, joint_names):
        request = mjc_message_pb2.QueryJointLimitsRequest(joint_names=joint_names)
        response = await self.stub.QueryJointLimits(request)
        joint_limits = [{"joint_name": limit.joint_name, "has_limit": limit.has_limit, "range_min": limit.range_min, "range_max": limit.range_max} for limit in response.joint_limits]
        return joint_limits    
    
    async def query_body_velocities(self, body_names):
        request = mjc_message_pb2.QueryBodyVelocitiesRequest(body_names=body_names)
        response = await self.stub.QueryBodyVelocities(request)

        velocities_dict = {}
        for velocity in response.body_velocities:
            velocities_dict[velocity.body_name] = {
                "linear_velocity": list(velocity.linear_velocity),
                "angular_velocity": list(velocity.angular_velocity),
            }

        return velocities_dict    
    
    async def query_actuator_gain_prm(self, actuator_names):
        request = mjc_message_pb2.QueryActuatorGainPrmRequest(actuator_names=actuator_names)
        response = await self.stub.QueryActuatorGainPrm(request)
        gain_prm_dict = {item.actuator_name: item.gain_prm for item in response.gain_prm_list}
        return gain_prm_dict

    async def set_actuator_gain_prm(self, gain_prm_set_list):
        gain_prm_set_list_proto = [
            mjc_message_pb2.SetActuatorGainPrmRequest.ActuatorGainPrmSet(
                actuator_name=item["actuator_name"],
                gain_prm=item["gain_prm"]
            ) for item in gain_prm_set_list
        ]
        request = mjc_message_pb2.SetActuatorGainPrmRequest(gain_prm_set_list=gain_prm_set_list_proto)
        response = await self.stub.SetActuatorGainPrm(request)
        return response.success

    async def query_actuator_bias_prm(self, actuator_names):
        request = mjc_message_pb2.QueryActuatorBiasPrmRequest(actuator_names=actuator_names)
        response = await self.stub.QueryActuatorBiasPrm(request)
        bias_prm_dict = {item.actuator_name: item.bias_prm for item in response.bias_prm_list}
        return bias_prm_dict

    async def set_actuator_bias_prm(self, bias_prm_set_list):
        bias_prm_set_list_proto = [
            mjc_message_pb2.SetActuatorBiasPrmRequest.ActuatorBiasPrmSet(
                actuator_name=item["actuator_name"],
                bias_prm=item["bias_prm"]
            ) for item in bias_prm_set_list
        ]
        request = mjc_message_pb2.SetActuatorBiasPrmRequest(bias_prm_set_list=bias_prm_set_list_proto)
        response = await self.stub.SetActuatorBiasPrm(request)
        return response.success    
    
    async def query_all_mocap_bodies(self):
        request = mjc_message_pb2.QueryAllMocapBodiesRequest()
        response = await self.stub.QueryAllMocapBodies(request)
        mocap_body_dict = {item.mocap_body_name: item.mocap_body_id for item in response.mocap_bodies}
        return mocap_body_dict

    async def query_mocap_pos_and_quat(self, mocap_body_names):
        request = mjc_message_pb2.QueryMocapPosAndQuatRequest(mocap_body_names=mocap_body_names)
        response = await self.stub.QueryMocapPosAndQuat(request)
        mocap_info = {
            info.mocap_body_name: {'pos': info.pos, 'quat': info.quat}
            for info in response.mocap_body_info
        }
        return mocap_info

    async def set_mocap_pos_and_quat(self, mocap_data):
        request = mjc_message_pb2.SetMocapPosAndQuatRequest()
        for name, data in mocap_data.items():
            mocap_body_info = request.mocap_body_info.add()
            mocap_body_info.mocap_body_name = name
            mocap_body_info.pos.extend(data['pos'])
            mocap_body_info.quat.extend(data['quat'])

        response = await self.stub.SetMocapPosAndQuat(request)
        return response.success
    
    async def query_all_equality_constraints(self):
        request = mjc_message_pb2.QueryAllEqualityConstraintsRequest()
        response = await self.stub.QueryAllEqualityConstraints(request)

        equality_constraints = []
        for eq in response.equality_constraints:
            eq_info = {
                "eq_type": eq.eq_type,
                "obj1_id": eq.obj1_id,
                "obj2_id": eq.obj2_id,
                "active": eq.active,
                "eq_solref": np.array(list(eq.solref)),
                "eq_solimp": np.array(list(eq.solimp)),
                "eq_data": np.array(list(eq.data))
            }
            equality_constraints.append(eq_info)

        return equality_constraints
    
    async def query_site_pos_and_mat(self, site_names: list[str]):
        request = mjc_message_pb2.QuerySitePosAndMatRequest(site_names=site_names)
        response = await self.stub.QuerySitePosAndMat(request)
        site_pos_and_mat = {site.site_name: {"xpos": np.array(list(site.xpos)), "xmat": np.array(list(site.xmat))} for site in response.site_pos_and_mat}
        return site_pos_and_mat
    
    
    async def mj_jac_site(self, site_names: list[str]):
        request = mjc_message_pb2.QuerySiteJacRequest(site_names=site_names)
        response = await self.stub.QuerySiteJac(request)
        site_jacs_dict = {}
        for site_jac in response.site_jacs:
            site_jacs_dict[site_jac.site_name] = {
                "jacp": np.array(list(site_jac.jacp)),
                "jacr": np.array(list(site_jac.jacr))
            }
        return site_jacs_dict
    
    async def update_equality_constraints(self, constraint_list):
        request = mjc_message_pb2.UpdateEqualityConstraintsRequest()
        for constraint in constraint_list:
            info = request.equality_constraints.add()
            info.obj1_id = constraint['obj1_id']
            info.obj2_id = constraint['obj2_id']
            info.data.extend(constraint['eq_data'])

        response = await self.stub.UpdateEqualityConstraints(request)
        return response.success
    
    async def query_all_geoms(self):
        request = mjc_message_pb2.QueryAllGeomsRequest()
        response = await self.stub.QueryAllGeoms(request)
        geom_dict = {
            geom.geom_name: {
                "BodyName": geom.body_name,
                "Type": geom.geom_type,
                "Contype": geom.geom_contype,
                "Conaffinity": geom.geom_conaffinity,
                "Condim": geom.geom_condim,
                "Solmix": geom.geom_solmix,
                "Solref": list(geom.geom_solref),
                "Solimp": list(geom.geom_solimp),
                "Size": list(geom.geom_size),
                "Friction": list(geom.geom_friction),
                "DataID": geom.geom_dataid,
                "MatID": geom.geom_matid,
                "Group": geom.geom_group,
                "Priority": geom.geom_priority,
                "Plugin": geom.geom_plugin,
                "SameFrame": geom.geom_sameframe,
                "Pos": list(geom.geom_pos),
                "Quat": list(geom.geom_quat),
                "Margin": geom.geom_margin,
                "Gap": geom.geom_gap,
            }
            for geom in response.geom_data_list
        }
        return geom_dict

    async def query_contact(self):
        request = mjc_message_pb2.QueryContactRequest()
        response = await self.stub.QueryContact(request)
        
        contacts = []
        for i, contact in enumerate(response.contacts):
            contact_info = {
                "ID": i,
                "Dist": contact.dist,
                "Pos": list(contact.pos),
                "Frame": list(contact.frame),
                "IncludeMargin": contact.includemargin,
                "Friction": list(contact.friction),
                "Solref": list(contact.solref),
                "SolrefFriction": list(contact.solreffriction),
                "Solimp": list(contact.solimp),
                "Mu": contact.mu,
                "H": list(contact.H),
                "Dim": contact.dim,
                "Geom1": contact.geom1,
                "Geom2": contact.geom2,
                "Geom": list(contact.geom),
                "Flex": list(contact.flex),
                "Elem": list(contact.elem),
                "Vert": list(contact.vert),
                "Exclude": contact.exclude,
                "EfcAddress": contact.efc_address,
            }
            contacts.append(contact_info)
        
        return contacts

    async def query_contact_simple(self):
        request = mjc_message_pb2.QueryContactSimpleRequest()
        response = await self.stub.QueryContactSimple(request)
        
        contacts = []
        for i, contact in enumerate(response.contacts):
            contact_info = {
                "ID": i,
                "Dim": contact.dim,
                "Geom1": contact.geom1,
                "Geom2": contact.geom2,
            }
            contacts.append(contact_info)
        
        return contacts
    
    async def query_body_com_xpos_xmat(self, body_name_list):
        request = mjc_message_pb2.QueryBodyComPosMatRequest(body_name_list=body_name_list)
        response = await self.stub.QueryBodyComPosMat(request)
        body_com_pos_mat_dict = {
            body.body_name: {
                "Pos": list(body.com_pos),
                "Mat": list(body.com_mat),
            }
            for body in response.body_com_pos_mat_list
        }
        return body_com_pos_mat_dict

    async def query_body_xpos_xmat_xquat(self, body_name_list):
        request = mjc_message_pb2.QueryBodyPosMatQuatRequest(body_name_list=body_name_list)
        response = await self.stub.QueryBodyPosMatQuat(request)
        body_pos_mat_quat_dict = {
            body.body_name: {
                "Pos": list(body.pos),
                "Mat": list(body.mat),
                "Quat": list(body.quat),
            }
            for body in response.body_pos_mat_quat_list
        }
        return body_pos_mat_quat_dict

    async def query_geom_xpos_xmat(self, geom_name_list):
        request = mjc_message_pb2.QueryGeomPosMatRequest(geom_name_list=geom_name_list)
        response = await self.stub.QueryGeomPosMat(request)
        geom_pos_mat_dict = {
            geom.geom_name: {
                "Pos": list(geom.pos),
                "Mat": list(geom.mat),
            }
            for geom in response.geom_pos_mat_list
        }
        return geom_pos_mat_dict

    async def query_contact_force(self, contact_ids):
        request = mjc_message_pb2.QueryContactForceRequest(contact_ids=contact_ids)
        response = await self.stub.QueryContactForce(request)
        forces_dict = {}
        for contact_force in response.contact_forces:
            forces_dict[contact_force.id] = np.array(contact_force.forces)
        return forces_dict
    
    async def mj_jac(self, body_point_list, compute_jacp=True, compute_jacr=True):
        request = mjc_message_pb2.MJ_JacRequest(
            compute_jacp=compute_jacp,
            compute_jacr=compute_jacr
        )
        for body, point in body_point_list:
            body_point_proto = request.body_point_list.add()
            body_point_proto.body = body
            body_point_proto.point.extend(point)

        response = await self.stub.MJ_Jac(request)
        jacp_list = []
        jacr_list = []
        
        for jac_result in response.jac_results:
            if compute_jacp:
                jacp_list.append(np.array(jac_result.jacp).reshape((3, -1)))
            if compute_jacr:
                jacr_list.append(np.array(jac_result.jacr).reshape((3, -1)))

        return jacp_list if compute_jacp else None, jacr_list if compute_jacr else None
    
    async def calc_full_mass_matrix(self):
        request = mjc_message_pb2.CalcFullMassMatrixRequest()
        response = await self.stub.CalcFullMassMatrix(request)
        full_mass_matrix = np.array(response.full_mass_matrix).reshape((self.model.nv, self.model.nv))
        return full_mass_matrix
    
    async def query_qfrc_bias(self):
        request = mjc_message_pb2.QueryQfrcBiasRequest()
        response = await self.stub.QueryQfrcBias(request)
        qfrc_bias = np.array(response.qfrc_bias)
        return qfrc_bias

    async def query_subtree_com(self, body_name_list):
        request = mjc_message_pb2.QuerySubtreeComRequest(
            body_name_list=body_name_list
        )
        response = await self.stub.QuerySubtreeCom(request)
        subtree_com_dict = {}
        subtree_com_data = np.array(response.subtree_com).reshape((-1, 3))
        
        for body_name, subtree_com in zip(response.body_name_list, subtree_com_data):
            subtree_com_dict[body_name] = subtree_com

        return subtree_com_dict
    
    async def set_geom_friction(self, geom_name_list, friction_list):
        request = mjc_message_pb2.SetGeomFrictionRequest()
        request.geom_name_list.extend(geom_name_list)

        for friction in friction_list:
            friction_proto = request.friction_list.add()
            friction_proto.values.extend(friction)

        response = await self.stub.SetGeomFriction(request)
        return response.success_list    
    
    async def query_sensor_data(self, sensor_names):
        request = mjc_message_pb2.QuerySensorDataRequest(sensor_names=sensor_names)
        response = await self.stub.QuerySensorData(request)

        sensor_data_dict = {}
        for sensor_data in response.sensor_data_list:
            sensor_data_dict[sensor_data.sensor_name] = {
                "type": sensor_data.sensor_type,
                "values": np.array(sensor_data.sensor_value)
            }

        # print("sensor_data_dict", sensor_data_dict)

        return sensor_data_dict    
    
    async def query_joint_offsets(self, joint_names):
        request = mjc_message_pb2.QueryJointOffsetsRequest(joint_names=joint_names)
        response = await self.stub.QueryJointOffsets(request)

        # 按顺序构建 offset 数组
        qpos_offsets = []
        qvel_offsets = []
        qacc_offsets = []

        # 记录实际返回的关节名称，以便检查
        returned_joint_names = [joint_offset.joint_name for joint_offset in response.joint_offsets]

        # 检查是否所有请求的关节都返回了偏移量
        missing_joints = [joint_name for joint_name in joint_names if joint_name not in returned_joint_names]
        if missing_joints:
            raise Exception(f"Joints not found: {', '.join(missing_joints)}")

        # 将响应中每个关节的 offset 按顺序添加到数组中
        for joint_name in joint_names:
            for joint_offset in response.joint_offsets:
                if joint_offset.joint_name == joint_name:
                    qpos_offsets.append(joint_offset.qpos_offset)
                    qvel_offsets.append(joint_offset.qvel_offset)
                    qacc_offsets.append(joint_offset.qacc_offset)
                    break

        return qpos_offsets, qvel_offsets, qacc_offsets    
    
    async def query_all_sites(self):
        request = mjc_message_pb2.QueryAllSitesRequest()
        response = await self.stub.QueryAllSites(request)
        site_dict = {
            site.name: {
                "ID": site.id,
                "BodyID": site.site_bodyid,
                "Type": site.site_type,
                "Pos": np.array(site.pos),
                "Mat": np.array(site.mat),
                "LocalPos": np.array(site.local_pos),
                "LocalQuat": np.array(site.local_quat)
            }
            for site in response.site_info
        }
        return site_dict    
    
    async def begin_save_video(self, file_path):
        request = mjc_message_pb2.BeginSaveMp4FileRequest(file_path=file_path)
        response = await self.stub.BeginSaveMp4File(request)
        if response.success:
            _logger.info(f"Video saving started at {file_path}")
        else:
            _logger.error(f"Failed to start video saving: {response.error_message}")

    async def stop_save_video(self):
        request = mjc_message_pb2.StopSaveMp4FileRequest()
        await self.stub.StopSaveMp4File(request)
        
    async def get_current_frame(self):
        request = mjc_message_pb2.GetCurrentFrameIndexRequest()
        response = await self.stub.GetCurrentFrameIndex(request)
        return response.current_frame