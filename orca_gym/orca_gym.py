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

class OrcaGym:
    def __init__(self, stub):
        self.stub = stub
        self.model = None
        self.opt_config = None
    

    # async def query_agents(self):
    #     body_info_list = await self.query_all_bodies()
    #     self.model.update_body_info(body_info_list)
    #     joint_info_list = await self.query_all_joints()
    #     self.model.update_joint_info(joint_info_list)
    #     actuator_info_list = await self.query_all_actuators()
    #     self.model.update_actuator_info(actuator_info_list)

    async def init_simulation(self):
        self.opt_config = await self.query_opt_config()
        print("Opt config: ", self.opt_config)

        model_info = await self.query_model_info()
        self.model = OrcaGymModel(model_info)      # 对应 Mujoco Model

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
        actuator_ctrlrange = await self.query_actuator_ctrl_range()
        self.model.init_actuator_ctrlrange(actuator_ctrlrange)

    async def query_all_actuators(self):
        request = mjc_message_pb2.QueryAllActuatorsRequest()
        response = await self.stub.QueryAllActuators(request)
        actuator_dict = {actuator.ActuatorName : {"JointName": actuator.JointName, "GearRatio": actuator.GearRatio} for actuator in response.ActuatorDataList}
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
            "mpr_tolerance": response.mpr_tolerance,
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
            "mpr_iterations": response.mpr_iterations,
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
            mpr_tolerance=opt_config["mpr_tolerance"],
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
            mpr_iterations=opt_config["mpr_iterations"],
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

    async def query_all_qpos_and_qvel(self):
        request = mjc_message_pb2.QueryAllQposAndQvelRequest()
        response = await self.stub.QueryAllQposAndQvel(request)
        qpos = np.array(response.Qpos)
        qvel = np.array(response.Qvel)
        return qpos, qvel

    async def load_keyframe(self, keyframe_name):
        request = mjc_message_pb2.LoadKeyFrameRequest(KeyFrameName=keyframe_name)
        response = await self.stub.LoadKeyFrame(request)
        return response.success

    async def pause_simulation(self):
        request = mjc_message_pb2.SetSimulationStateRequest(state=mjc_message_pb2.PAUSED)
        response = await self.stub.SetSimulationState(request)
        return response

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

    async def set_qpos(self, qpos):
        request = mjc_message_pb2.SetQposRequest(qpos=qpos)
        response = await self.stub.SetQpos(request)
        return response

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
        joint_dict = {joint.name: {"joint_id": joint.id, "joint_body_id": joint.body_id, "joint_type": joint.type} for joint in response.joint_info}
        return joint_dict
    
    async def query_all_bodies(self):
        request = mjc_message_pb2.QueryAllBodiesRequest()
        response = await self.stub.QueryAllBodies(request)
        body_dict = {body.name: body.id for body in response.body_info}
        return body_dict
    
    async def mj_forward(self):
        request = mjc_message_pb2.MJ_ForwardRequest()
        response = await self.stub.MJ_Forward(request)
        return response

    async def mj_inverse(self):
        request = mjc_message_pb2.MJ_InverseRequest()
        response = await self.stub.MJ_Inverse(request)
        return response
    
    async def mj_step(self, nstep):
        request = mjc_message_pb2.MJ_StepRequest(nstep=nstep)
        response = await self.stub.MJ_Step(request)
        return response    
    
    async def set_qvel(self, qvel):
        request = mjc_message_pb2.SetQvelRequest(qvel=qvel)
        response = await self.stub.SetQvel(request)
        return response    
    
    async def query_body_com(self, body_names):
        request = mjc_message_pb2.QueryBodyComRequest(body_names=body_names)
        response = await self.stub.QueryBodyCom(request)
        if response.success:
            return {body_com.body_name: list(body_com.com) for body_com in response.body_coms}
        else:
            raise Exception("Failed to query body COM")    
        
    async def query_cfrc_ext(self, body_names):
        request = mjc_message_pb2.QueryCfrcExtRequest(body_names=body_names)
        response = await self.stub.QueryCfrcExt(request)
        if response.success:
            return {body_cfrc_ext.body_name: list(body_cfrc_ext.cfrc_ext) for body_cfrc_ext in response.body_cfrc_exts}
        else:
            raise Exception("Failed to query cfrc_ext")        
        
    async def query_actuator_ctrl_range(self):
        request = mjc_message_pb2.QueryActuatorCtrlRangeRequest()
        response = await self.stub.QueryActuatorCtrlRange(request)
        if response.success:
            ranges = [(actuator_ctrl_range.min, actuator_ctrl_range.max) for actuator_ctrl_range in response.actuator_ctrl_ranges]
            return np.array(ranges, dtype=np.float32)  # 将列表转换为 NumPy 数组
        else:
            raise Exception("Failed to query actuator control ranges")
        
    async def set_joint_qpos(self, joint_qpos_list):
        request = mjc_message_pb2.SetJointQposRequest()
        for joint_name, qpos in joint_qpos_list.items():
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
    
    async def query_site_pos_and_mat(self, site_names):
        request = mjc_message_pb2.QuerySitePosAndMatRequest(site_names=site_names)
        response = await self.stub.QuerySitePosAndMat(request)
        site_pos_and_mat = {site.site_name: {"xpos": np.array(list(site.xpos)), "xmat": np.array(list(site.xmat))} for site in response.site_pos_and_mat}
        return site_pos_and_mat
    
    
    async def query_site_jac(self, site_names):
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