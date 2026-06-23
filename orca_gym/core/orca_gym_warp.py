import sys
import os
import grpc
import aiofiles
import xml.etree.ElementTree as ET
import tempfile
import shutil

proj_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
proto_path = os.path.abspath(os.path.join(proj_dir, "protos"))
sys.path.append(proto_path)
import mjc_message_pb2
import mjc_message_pb2_grpc

import numpy as np
from datetime import datetime

from orca_gym.log.orca_log import get_orca_logger
_logger = get_orca_logger()

from orca_gym.core.orca_gym_model import OrcaGymModel
from orca_gym.core.orca_gym_data import OrcaGymData
from orca_gym.core.orca_gym_opt_config import OrcaGymOptConfig
from orca_gym.core.orca_gym import OrcaGymBase
from orca_gym.utils.dir_utils import cleanup_zombie_locks, file_lock

import mujoco
import mujoco_warp as mjw
import warp as wp
from scipy.spatial.transform import Rotation as R


def get_qpos_size(joint_type):
    if joint_type == mujoco.mjtJoint.mjJNT_FREE:
        return 7
    elif joint_type == mujoco.mjtJoint.mjJNT_BALL:
        return 4
    elif joint_type == mujoco.mjtJoint.mjJNT_SLIDE or joint_type == mujoco.mjtJoint.mjJNT_HINGE:
        return 1
    else:
        return 0

def get_dof_size(joint_type):
    if joint_type == mujoco.mjtJoint.mjJNT_FREE:
        return 6
    elif joint_type == mujoco.mjtJoint.mjJNT_BALL:
        return 3
    elif joint_type == mujoco.mjtJoint.mjJNT_SLIDE or joint_type == mujoco.mjtJoint.mjJNT_HINGE:
        return 1
    else:
        return 0
    
class AnchorType:
    NONE = 0
    WELD = 1
    BALL = 2

class CaptureMode:
    ASYNC = 0
    SYNC = 1

def get_eq_type(anchor_type: AnchorType):
    if anchor_type == AnchorType.WELD:
        return mujoco.mjtEq.mjEQ_WELD
    elif anchor_type == AnchorType.BALL:
        return mujoco.mjtEq.mjEQ_CONNECT
    else:
        return mujoco.mjtEq.mjEQ_CONNECT


class OrcaGymWarp(OrcaGymBase):
    def __init__(self, stub, nworld: int = 1):
        super().__init__(stub = stub)

        self._timestep = 0.001
        self._mjModel = None
        self._mjData = None
        self._mjwModel = None
        self._mjwData = None
        self._nworld = nworld
        self._device = None
        self._override_ctrls : dict[int, float] = {}

        import tempfile
        temp_dir = tempfile.gettempdir()
        cleanup_zombie_locks(temp_dir)

        self._init_warp()

    def _init_warp(self):
        cuda_devices = wp.get_cuda_devices()
        if len(cuda_devices) == 0:
            _logger.warning("No CUDA device found, falling back to CPU.")
            self._device = "cpu"
        else:
            self._device = cuda_devices[0]
            _logger.info(f"Using CUDA device: {self._device}")

        wp.init()
        wp.set_device(self._device)

    async def load_model_xml(self):
        model_xml_path = await self.load_local_env()

        _logger.info(f"Model XML Path: {model_xml_path}")
        await self.process_xml_file(model_xml_path)
        return model_xml_path

    async def init_simulation(self, model_xml_path):
        self._xml_path = model_xml_path
        self._mjModel = mujoco.MjModel.from_xml_path(model_xml_path)
        self._mjData = mujoco.MjData(self._mjModel)

        size_model = mujoco.mj_sizeModel(self._mjModel)
        _logger.debug(f"size_model: {size_model}")

        self.set_opt_timestep(self._timestep)

        opt_config = self.query_opt_config()
        self.opt = OrcaGymOptConfig(opt_config)
        self.print_opt_config()

        model_info = self.query_model_info()
        self.model = OrcaGymModel(model_info)
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
        sensor_dict = self.query_all_sensors()
        self.model.init_sensor_dict(sensor_dict)
        mesh_dict = self.query_all_meshes()
        self.model.init_mesh_dict(mesh_dict)

        self.data = OrcaGymData(self.model)
        self._qpos_cache = np.array(self._mjData.qpos, copy=True)
        self._qvel_cache = np.array(self._mjData.qvel, copy=True)
        self._qacc_cache = np.array(self._mjData.qacc, copy=True)
        self.update_data()

        self._mjModel.opt.noslip_iterations = 0
        self._mjModel.opt.solver = mujoco.mjtSolver.mjSOL_NEWTON
        self._mjModel.opt.integrator = mujoco.mjtIntegrator.mjINT_EULER
        self._mjwModel = mjw.put_model(self._mjModel)
        njmax = max(self._mjModel.njmax, 2000)
        nconmax = max(self._mjModel.nconmax, 500)
        self._mjwData = mjw.make_data(self._mjModel, nworld=self._nworld,
                                      nconmax=nconmax, njmax=njmax, naconmax=nconmax)
        mjw.reset_data(self._mjwModel, self._mjwData)
        gpu_qpos_check = self._mjwData.qpos.numpy()[0]
        if np.any(np.isnan(gpu_qpos_check)) or np.any(np.isinf(gpu_qpos_check)):
            _logger.error(f"GPU qpos has NaN/Inf immediately after reset_data: "
                          f"NaN={np.sum(np.isnan(gpu_qpos_check))}, Inf={np.sum(np.isinf(gpu_qpos_check))}")
        _logger.info(f"MJWarp model and data created on device: nworld={self._nworld}, "
                     f"njmax={njmax}, nconmax={nconmax}")

    def _sync_state_to_gpu(self):
        qpos = wp.array(self._mjData.qpos.reshape(1, -1), dtype=float, device=self._device)
        qvel = wp.array(self._mjData.qvel.reshape(1, -1), dtype=float, device=self._device)
        ctrl = self._mjData.ctrl.copy()
        for aid in self._get_disabled_actuator_ids():
            ctrl[aid] = float(self._mjData.actuator_length[aid])
        gpu_ctrl = wp.array(ctrl.reshape(1, -1), dtype=float, device=self._device)
        wp.copy(self._mjwData.qpos, qpos)
        wp.copy(self._mjwData.qvel, qvel)
        wp.copy(self._mjwData.ctrl, gpu_ctrl)

        n_mocap = self._mjModel.nmocap
        if n_mocap > 0:
            mocap_pos = wp.array(self._mjData.mocap_pos.reshape(1, n_mocap, 3), dtype=float, device=self._device)
            mocap_quat = wp.array(self._mjData.mocap_quat.reshape(1, n_mocap, 4), dtype=float, device=self._device)
            wp.copy(self._mjwData.mocap_pos, mocap_pos)
            wp.copy(self._mjwData.mocap_quat, mocap_quat)

        wp.copy(self._mjwData.xfrc_applied, wp.array(self._mjData.xfrc_applied.reshape(1, self._mjModel.nbody, 6), dtype=float, device=self._device))
        wp.synchronize_device(self._device)

    def _sync_state_from_gpu(self):
        gpu_qpos = self._mjwData.qpos.numpy()[0]
        gpu_qvel = self._mjwData.qvel.numpy()[0]
        gpu_qacc = self._mjwData.qacc.numpy()[0].copy()
        gpu_time = float(self._mjwData.time.numpy()[0])
        gpu_act = self._mjwData.act.numpy()[0]

        if np.any(np.isnan(gpu_qpos)) or np.any(np.isinf(gpu_qpos)):
            nan_count = np.sum(np.isnan(gpu_qpos))
            inf_count = np.sum(np.isinf(gpu_qpos))
            _logger.error(f"GPU qpos contains {nan_count} NaN and {inf_count} Inf values after mjw.step")
            if nan_count > 0:
                nan_indices = np.where(np.isnan(gpu_qpos))[0]
                _logger.error(f"GPU qpos NaN at indices: {nan_indices[:20].tolist()}")
            if inf_count > 0:
                inf_indices = np.where(np.isinf(gpu_qpos))[0]
                _logger.error(f"GPU qpos Inf at indices: {inf_indices[:20].tolist()}")
            return

        self._mjData.qpos[:] = gpu_qpos
        self._mjData.qvel[:] = gpu_qvel
        self._mjData.time = gpu_time
        self._mjData.act[:] = gpu_act

        mujoco.mj_forward(self._mjModel, self._mjData)
        self._mjData.qacc[:] = gpu_qacc

    async def render(self):
        await self.update_local_env(self.data.qpos, self._mjData.time)

    async def update_local_env(self, qpos, time):
        request = mjc_message_pb2.UpdateLocalEnvRequest(qpos=qpos, time=time)
        response = await self.stub.UpdateLocalEnv(request)
        override_ctrls = response.override_ctrls
        self._override_ctrls.clear()
        if override_ctrls is not None and len(override_ctrls) > 0:
            for ctrl in override_ctrls:
                if ctrl.index < 0 or ctrl.index >= self._mjModel.nu:
                    _logger.warning(f"Invalid control index: {ctrl.index}, skipping.")
                    continue
                self._override_ctrls[ctrl.index] = ctrl.value

    async def load_content_file(self, content_file_name, remote_file_dir="", local_file_dir="", temp_file_path=None):
        request = mjc_message_pb2.LoadContentFileRequest(file_name=content_file_name, file_dir=remote_file_dir)
        response = await self.stub.LoadContentFile(request)

        if response.status != mjc_message_pb2.LoadContentFileResponse.SUCCESS:
            raise Exception("Load content file failed.")

        content = response.content
        if content is None or len(content) == 0:
            raise Exception("Content is empty.")
        
        if temp_file_path is not None:
            async with aiofiles.open(temp_file_path, 'wb') as f:
                await f.write(content)
            return temp_file_path
        
        if local_file_dir is None or len(local_file_dir) == 0:
            content_file_path = os.path.join(self.xml_file_dir, content_file_name)
        else:
            content_file_path = os.path.join(local_file_dir, content_file_name)

        _logger.debug(f"Content file path: {content_file_path}")

        try:
            async with file_lock(content_file_path, timeout=30):
                if not os.path.exists(content_file_path):
                    temp_file = tempfile.NamedTemporaryFile(
                        mode='wb', 
                        dir=os.path.dirname(content_file_path), 
                        delete=False,
                        prefix=f"{content_file_name}_",
                        suffix=".tmp"
                    )
                    try:
                        temp_file.write(content)
                        temp_file.flush()
                        os.fsync(temp_file.fileno())
                        temp_file.close()
                        
                        shutil.move(temp_file.name, content_file_path)
                    except Exception as e:
                        try:
                            os.unlink(temp_file.name)
                        except OSError:
                            pass
                        raise e
        except TimeoutError as e:
            _logger.warning(f"{e}")
            if os.path.exists(content_file_path):
                _logger.info(f"{content_file_path} already exists, skip download")
                return content_file_path
            else:
                raise Exception(f"Cannot acquire file lock and file not exist: {content_file_path}")

        return content_file_path

    async def process_xml_node(self, node: ET.Element):
        if node.tag == 'mesh' or node.tag == 'hfield':
            content_file_name = node.get('file')
            if content_file_name is not None:
                content_file_path = os.path.join(self.xml_file_dir, content_file_name)
                async with file_lock(content_file_path):
                    if not os.path.exists(content_file_path):
                        _logger.debug(f"Load content file: {content_file_name}")
                        await self.load_content_file(content_file_name)
        else:
            for child in node:
                await self.process_xml_node(child)
        return

    async def begin_save_video(self, file_path, capture_mode: CaptureMode = CaptureMode.ASYNC):
        request = mjc_message_pb2.BeginSaveMp4FileRequest(file_path=file_path, capture_mode=capture_mode)
        response = await self.stub.BeginSaveMp4File(request)
        if response.status == mjc_message_pb2.BeginSaveMp4FileResponse.Status.SUCCESS:
            _logger.info(f"Video saving started at {file_path}")
        else:
            _logger.error(f"Failed to start video saving: {response.error_message}")

    async def stop_save_video(self):
        request =  mjc_message_pb2.StopSaveMp4FileRequest()
        await self.stub.StopSaveMp4File(request)

    async def get_current_frame(self)-> int:
        request = mjc_message_pb2.GetCurrentFrameIndexRequest()
        response = await self.stub.GetCurrentFrameIndex(request)
        return response.current_frame

    async def get_camera_time_stamp(self, last_frame) -> dict:
        request = mjc_message_pb2.GetTimeStampRequest()
        request.last_frame_index = last_frame
        response = await self.stub.GetTimeStamp(request)
        if response.error_message != "":
            _logger.error(f"Get time stamp failed. error message: {response.error_message}")
        return {camera_name: time_stamp_list.time_stamps for camera_name, time_stamp_list in response.time_stamp_map.items()}

    async def get_frame_png(self, image_path):
        request = mjc_message_pb2.GetCameraFramePNGRequest()
        request.image_path = image_path
        response = await self.stub.GetCameraFramePNG(request)
        result = {}
        for name_transform in response.name_transform:
            result[name_transform.name] = {
                'pos': list(name_transform.pos),
                'quat': list(name_transform.quat)
            }
        return result

    @property
    def xml_file_dir(self):
        user_home = os.path.expanduser('~')
        local_dir = os.path.join(user_home, '.orcagym', 'tmp')
        if not os.path.exists(local_dir):
            os.makedirs(local_dir, exist_ok=True)
        return local_dir

    def _build_load_local_env_error(self, status=None, error_message=""):
        parts = ["Load local env failed."]
        if status is not None:
            parts.append(f"error code: {status}")
        if error_message:
            parts.append(f"error message: {error_message}")

        parts.extend([
            "",
            "Scene initialization failed on the server side.",
            "",
            "Common causes:",
            "1. Duplicate names in joint, body, geom, site, actuator, sensor",
            "2. Overlapping initial poses causing contact or constraint issues",
            "3. Unreasonable joint hierarchy, axis, range, or physics parameters",
            "4. Malformed equality / weld / connect constraints",
            "5. Missing or incorrect mesh, texture, XML resource paths",
            "6. Simulation server still initializing",
        ])
        return "\n".join(parts)

    async def load_local_env(self):
        request = mjc_message_pb2.LoadLocalEnvRequest()
        request.req_type = mjc_message_pb2.LoadLocalEnvRequest.XML_FILE_NAME
        response = await self.stub.LoadLocalEnv(request)

        if response.status != mjc_message_pb2.LoadLocalEnvResponse.SUCCESS:
            raise Exception(self._build_load_local_env_error(response.status, response.error_message))

        file_name = response.file_name
        file_path = os.path.join(self.xml_file_dir, file_name)

        async with file_lock(file_path):
            if not os.path.exists(file_path):
                request = mjc_message_pb2.LoadLocalEnvRequest()
                request.req_type = mjc_message_pb2.LoadLocalEnvRequest.XML_FILE_CONTENT
                response = await self.stub.LoadLocalEnv(request)

                if response.status != mjc_message_pb2.LoadLocalEnvResponse.SUCCESS:
                    raise Exception(self._build_load_local_env_error(response.status, response.error_message))

                xml_content = response.xml_content

                temp_file = tempfile.NamedTemporaryFile(
                    mode='wb',
                    dir=self.xml_file_dir,
                    delete=False,
                    prefix=f"{file_name}_",
                    suffix=".tmp"
                )
                try:
                    temp_file.write(xml_content)
                    temp_file.flush()
                    os.fsync(temp_file.fileno())
                    temp_file.close()

                    shutil.move(temp_file.name, file_path)
                except Exception as e:
                    try:
                        os.unlink(temp_file.name)
                    except OSError:
                        pass
                    raise e

        return os.path.abspath(file_path)

    async def process_xml_file(self, model_xml_path):
        with open(model_xml_path, 'r') as f:
            xml_content = f.read()

        root = ET.fromstring(xml_content)
        await self.process_xml_node(root)
        return

    async def get_body_manipulation_anchored(self):
        request = mjc_message_pb2.GetBodyManipulationAnchoredRequest()
        response = await self.stub.GetBodyManipulationAnchored(request)
        body_name = response.body_name
        anchor_type = response.anchor_type
        if body_name is None or len(body_name) == 0:
            return None, AnchorType.NONE
        return body_name, anchor_type

    async def get_body_manipulation_movement(self):
        request = mjc_message_pb2.GetBodyManipulationMovementRequest()
        response = await self.stub.GetBodyManipulationMovement(request)
        return {
            "delta_pos": np.array(response.delta_pos),
            "delta_quat": np.array(response.delta_quat),
        }

    async def set_timestep_remote(self, time_step):
        request = mjc_message_pb2.SetOptTimestepRequest(timestep=time_step)
        response = await self.stub.SetOptTimestep(request)
        return response

    def set_time_step(self, time_step):
        self._timestep = time_step

    def set_opt_timestep(self, timestep):
        if self._mjModel is not None:
            self._mjModel.opt.timestep = timestep
            if self._mjwModel is not None:
                self._mjwModel.opt.timestep = wp.array([[timestep]], dtype=float, device=self._device)

    def set_opt_config(self, opt_config):
        if self._mjModel is not None:
            self._mjModel.opt.timestep = opt_config['timestep']
            self._mjModel.opt.impratio = opt_config['impratio']
            self._mjModel.opt.tolerance = opt_config['tolerance']
            self._mjModel.opt.ls_tolerance = opt_config['ls_tolerance']
            self._mjModel.opt.noslip_tolerance = opt_config['noslip_tolerance']
            self._mjModel.opt.ccd_tolerance = opt_config['ccd_tolerance']
            self._mjModel.opt.gravity = opt_config['gravity']
            self._mjModel.opt.wind = opt_config['wind']
            self._mjModel.opt.magnetic = opt_config['magnetic']
            self._mjModel.opt.density = opt_config['density']
            self._mjModel.opt.viscosity = opt_config['viscosity']
            self._mjModel.opt.o_margin = opt_config['o_margin']
            self._mjModel.opt.o_solref = opt_config['o_solref']
            self._mjModel.opt.o_solimp = opt_config['o_solimp']
            self._mjModel.opt.o_friction = opt_config['o_friction']
            self._mjModel.opt.integrator = opt_config['integrator']
            self._mjModel.opt.cone = opt_config['cone']
            self._mjModel.opt.jacobian = opt_config['jacobian']
            self._mjModel.opt.solver = opt_config['solver']
            self._mjModel.opt.iterations = opt_config['iterations']
            self._mjModel.opt.ls_iterations = opt_config['ls_iterations']
            self._mjModel.opt.noslip_iterations = opt_config['noslip_iterations']
            self._mjModel.opt.ccd_iterations = opt_config['ccd_iterations']
            self._mjModel.opt.disableflags = opt_config['disableflags']
            self._mjModel.opt.enableflags = opt_config['enableflags']
            self._mjModel.opt.disableactuator = opt_config['disableactuator']
            self._mjModel.opt.sdf_initpoints = opt_config['sdf_initpoints']
            self._mjModel.opt.sdf_iterations = opt_config['sdf_iterations']

            if self._mjwModel is not None:
                self._mjwModel = mjw.put_model(self._mjModel)
                njmax = max(self._mjModel.njmax, 2000)
                nconmax = max(self._mjModel.nconmax, 500)
                self._mjwData = mjw.make_data(self._mjModel, nworld=self._nworld,
                                              nconmax=nconmax, njmax=njmax, naconmax=nconmax)
                mjw.reset_data(self._mjwModel, self._mjwData)

    def query_opt_config(self):
        if self._mjModel is None:
            return {}
        opt = self._mjModel.opt
        return {
            'timestep': opt.timestep,
            'impratio': opt.impratio,
            'tolerance': opt.tolerance,
            'ls_tolerance': opt.ls_tolerance,
            'noslip_tolerance': opt.noslip_tolerance,
            'ccd_tolerance': opt.ccd_tolerance,
            'gravity': opt.gravity,
            'wind': opt.wind,
            'magnetic': opt.magnetic,
            'density': opt.density,
            'viscosity': opt.viscosity,
            'o_margin': opt.o_margin,
            'o_solref': opt.o_solref,
            'o_solimp': opt.o_solimp,
            'o_friction': opt.o_friction,
            'integrator': opt.integrator,
            'cone': opt.cone,
            'jacobian': opt.jacobian,
            'solver': opt.solver,
            'iterations': opt.iterations,
            'ls_iterations': opt.ls_iterations,
            'noslip_iterations': opt.noslip_iterations,
            'ccd_iterations': opt.ccd_iterations,
            'disableflags': opt.disableflags,
            'enableflags': opt.enableflags,
            'disableactuator': opt.disableactuator,
            'sdf_initpoints': opt.sdf_initpoints,
            'sdf_iterations': opt.sdf_iterations,
        }

    def query_model_info(self):
        if self._mjModel is None:
            return {}
        model = self._mjModel
        info = {
            'nq': model.nq,
            'nv': model.nv,
            'nu': model.nu,
            'nbody': model.nbody,
            'njnt': model.njnt,
            'ngeom': model.ngeom,
            'nsite': model.nsite,
            'nmesh': model.nmesh,
            'ncam': model.ncam,
            'nlight': model.nlight,
            'nuser_body': model.nuser_body,
            'nuser_jnt': model.nuser_jnt,
            'nuser_geom': model.nuser_geom,
            'nuser_site': model.nuser_site,
            'nuser_tendon': model.nuser_tendon,
            'nuser_actuator': model.nuser_actuator,
            'nuser_sensor': model.nuser_sensor,
            'nconmax': model.nconmax,
            'nflex': model.nflex,
            'nflexvert': model.nflexvert,
        }
        flex_names = []
        for i in range(model.nflex):
            flex_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_FLEX, i)
            flex_names.append(flex_name if flex_name else f"flex_{i}")
        info['flex_names'] = flex_names
        return info

    def query_all_equality_constraints(self):
        model = self._mjModel
        if model is None:
            return []
        eq_list = []
        for i in range(model.neq):
            eq = {
                "obj1_id": model.eq_obj1id[i],
                "obj2_id": model.eq_obj2id[i],
                "eq_type": model.eq_type[i],
                "active": model.eq_active0[i],
                "eq_solref": model.eq_solref[i].copy(),
                "eq_solimp": model.eq_solimp[i].copy(),
                "eq_data": model.eq_data[i].copy(),
            }
            eq_list.append(eq)
        return eq_list

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

    def get_goal_bounding_box(self, goal_body_name):
        if self._mjModel is None:
            return None
        goal_geom_box_sizes = []
        goal_geom_box_positions = []
        for geom_id in range(self._mjModel.ngeom):
            body_id = self._mjModel.geom(geom_id).bodyid
            body_name = self._mjModel.body(body_id).name
            if goal_body_name not in body_name:
                continue
            geom = self._mjModel.geom(geom_id)
            xpos = self._mjData.geom_xpos[geom_id]
            if geom.type == mujoco.mjtGeom.mjGEOM_BOX:
                xmat = self._mjData.geom_xmat[geom_id].reshape(3, 3)
                half_size = geom.size
                goal_geom_box_sizes.append(half_size * 2)
                goal_geom_box_positions.append(xpos)
            elif geom.type == mujoco.mjtGeom.mjGEOM_SPHERE:
                radius = geom.size[0]
                goal_geom_box_sizes.append(np.array([radius * 2, radius * 2, radius * 2]))
                goal_geom_box_positions.append(xpos)
        if len(goal_geom_box_sizes) == 0:
            return None, None
        return np.array(goal_geom_box_sizes), np.array(goal_geom_box_positions)

    def set_actuator_trnid(self, actuator_id, trnid):
        if self._mjModel is not None:
            model = self._mjModel
            actuator = model.actuator(actuator_id)
            actuator.trnid[0] = trnid

    def disable_actuator(self, actuator_groups: list[int]):
        if self._mjModel is not None:
            model = self._mjModel
            for actuator_group in actuator_groups:
                model.opt.disableactuator |= (1 << actuator_group)

    def _get_disabled_actuator_ids(self) -> list[int]:
        if self._mjModel is None:
            return []
        disabled = self._mjModel.opt.disableactuator
        if disabled == 0:
            return []
        ids: list[int] = []
        for i in range(self._mjModel.nu):
            group = self._mjModel.actuator_group[i]
            if disabled & (1 << group):
                ids.append(i)
        return ids

    def query_all_bodies(self):
        model = self._mjModel
        body_dict = {}
        for i in range(model.nbody):
            body = model.body(i)
            body_dict[body.name] = {
                "ID": body.id,
                "ParentID": body.parentid[0],
                "WeldID": body.weldid[0],
                "RootID": body.rootid[0],
                "JntAdr": body.jntadr[0],
                "JntNum": body.jntnum[0],
                "GeomAdr": body.geomadr[0],
                "GeomNum": body.geomnum[0],
                "DofNum": body.dofnum[0],
                "DofAdr": body.dofadr[0],
                "TreeID": model.body_treeid[i],
                "MocapID": model.body_mocapid[i],
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
                "Simple": body.simple[0],
                "SameFrame": body.sameframe[0],
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
                "Frictionloss": joint.frictionloss[0],
                "Damping": joint.damping[0],
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

    def query_all_sensors(self):
        model = self._mjModel
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

    def query_all_meshes(self):
        import xml.etree.ElementTree as ET
        import os

        model = self._mjModel
        mesh_dict = {}

        mesh_files_from_xml = {}
        mesh_scales_from_xml = {}
        if hasattr(self, '_xml_path') and self._xml_path and os.path.exists(self._xml_path):
            try:
                tree = ET.parse(self._xml_path)
                root = tree.getroot()

                xml_dir = os.path.dirname(os.path.abspath(self._xml_path))

                meshdir_elem = root.find('.//compiler')
                meshdir = ""
                if meshdir_elem is not None:
                    meshdir_attr = meshdir_elem.get('meshdir')
                    if meshdir_attr:
                        meshdir = meshdir_attr

                assets = root.find('asset')
                if assets is not None:
                    for mesh_elem in assets.findall('mesh'):
                        mesh_name = mesh_elem.get('name', '')
                        mesh_file = mesh_elem.get('file', '')
                        mesh_scale_str = mesh_elem.get('scale', '1 1 1')

                        if mesh_file:
                            if meshdir and not os.path.isabs(mesh_file):
                                mesh_file = os.path.join(meshdir, mesh_file)
                                mesh_file = os.path.normpath(mesh_file)

                            mesh_files_from_xml[mesh_name] = mesh_file

                        try:
                            scale_values = [float(s) for s in mesh_scale_str.split()]
                            if len(scale_values) == 1:
                                mesh_scales_from_xml[mesh_name] = np.array(
                                    [scale_values[0], scale_values[0], scale_values[0]]
                                )
                            elif len(scale_values) == 3:
                                mesh_scales_from_xml[mesh_name] = np.array(scale_values)
                            else:
                                mesh_scales_from_xml[mesh_name] = np.ones(3)
                        except (ValueError, IndexError):
                            mesh_scales_from_xml[mesh_name] = np.ones(3)
            except Exception:
                pass

        for i in range(model.nmesh):
            mesh = model.mesh(i)
            mesh_name = mesh.name if mesh.name else f"mesh_{i}"

            mesh_file = mesh_files_from_xml.get(mesh_name, "")
            mesh_scale = mesh_scales_from_xml.get(mesh_name, np.ones(3))

            mesh_dict[mesh_name] = {
                "ID": mesh.id,
                "File": mesh_file,
                "Scale": mesh_scale,
            }

        return mesh_dict
    
    def update_data(self):
        self._qpos_cache[:] = self._mjData.qpos
        self._qvel_cache[:] = self._mjData.qvel
        self._qacc_cache[:] = self._mjData.qacc
        qfrc_bias = self.query_qfrc_bias()        
        self.data.update_qpos_qvel_qacc(self._qpos_cache, self._qvel_cache, self._qacc_cache)        
        self.data.update_qfrc_bias(qfrc_bias)
        self.data.time = self._mjData.time
        
    def update_data_external(self, qpos, qvel, qacc, qfrc_bias, time):
        self.data.update_qpos_qvel_qacc(qpos, qvel, qacc)
        self.data.update_qfrc_bias(qfrc_bias)
        self.data.time = time
    
    def query_qfrc_bias(self):
        qfrc_bias = self._mjData.qfrc_bias
        return qfrc_bias
    
    def load_initial_frame(self):
        mujoco.mj_resetData(self._mjModel, self._mjData)
        if self._mjwModel is not None and self._mjwData is not None:
            self._sync_state_to_gpu()

    def query_joint_offsets(self, joint_names):
        qpos_offsets = []
        qvel_offsets = []
        qacc_offsets = []

        for joint_name in joint_names:
            joint_id = self._mjModel.joint(joint_name).id
            qpos_offsets.append(self._mjModel.jnt_qposadr[joint_id])
            qvel_offsets.append(self._mjModel.jnt_dofadr[joint_id])
            qacc_offsets.append(self._mjModel.jnt_dofadr[joint_id])

        return qpos_offsets, qvel_offsets, qacc_offsets    
    
    def query_joint_lengths(self, joint_names):
        qpos_lengths = []
        qvel_lengths = []
        qacc_lengths = []

        for joint_name in joint_names:
            joint_id = self._mjModel.joint(joint_name).id
            qpos_lengths.append(get_qpos_size(self._mjModel.jnt_type[joint_id]))
            qvel_lengths.append(get_dof_size(self._mjModel.jnt_type[joint_id]))
            qacc_lengths.append(get_dof_size(self._mjModel.jnt_type[joint_id]))

        return qpos_lengths, qvel_lengths, qacc_lengths
    
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
            sensor_info = self.model.get_sensor(sensor_name)
            sensor_values = np.copy(self._mjData.sensordata[sensor_info['Adr']:sensor_info['Adr'] + sensor_info['Dim']])
            sensor_data_dict[sensor_name] = sensor_values

        return sensor_data_dict    
    
    def set_ctrl(self, ctrl):
        if len(self._override_ctrls) > 0:
            for actuator_id, value in self._override_ctrls.items():
                ctrl[actuator_id] = value
        self._mjData.ctrl = ctrl.copy()

    def mj_step(self, nstep):
        self._sync_state_to_gpu()
        for _ in range(nstep):
            mjw.step(self._mjwModel, self._mjwData)

        gpu_qpos = self._mjwData.qpos.numpy()[0]
        if np.any(np.isnan(gpu_qpos)) or np.any(np.isinf(gpu_qpos)):
            nan_count = np.sum(np.isnan(gpu_qpos))
            inf_count = np.sum(np.isinf(gpu_qpos))
            _logger.warning(f"GPU step produced {nan_count} NaN/{inf_count} Inf, "
                            f"falling back to CPU simulation for this step")
            for _ in range(nstep):
                mujoco.mj_step(self._mjModel, self._mjData)
            self._sync_state_to_gpu()
        else:
            self._sync_state_from_gpu()

    def mj_forward(self):
        mujoco.mj_forward(self._mjModel, self._mjData)

    def get_timer_stats(self) -> dict[str, tuple[float, int]]:
        timer = self._mjData.timer
        stats: dict[str, tuple[float, int]] = {}
        for name in dir(mujoco.mjtTimer):
            if name.startswith('mjTIMER'):
                idx = getattr(mujoco.mjtTimer, name)
                stats[name] = (float(timer.duration[idx]), int(timer.number[idx]))
        return stats

    def get_constraint_counts(self) -> dict[str, int]:
        d = self._mjData
        return {
            'nefc': int(d.nefc),
            'ne': int(d.ne),
            'nf': int(d.nf),
            'ncon': int(d.ncon),
        }

    def get_contact_sources(self) -> dict[tuple[str, str], int]:
        d = self._mjData
        m = self._mjModel
        result: dict[tuple[str, str], int] = {}
        for i in range(d.ncon):
            ct = d.contact[i]
            g1, g2 = ct.geom1, ct.geom2
            b1, b2 = m.geom_bodyid[g1], m.geom_bodyid[g2]
            name1 = m.body(b1).name
            name2 = m.body(b2).name
            if b1 > b2:
                key = (name2, name1)
            else:
                key = (name1, name2)
            result[key] = result.get(key, 0) + 1
        return result

    _AGGREGATE_TIMERS = {'mjTIMER_STEP', 'mjTIMER_FORWARD', 'mjTIMER_POSITION'}

    def log_profile(self, label: str = "") -> None:
        timer_stats = self.get_timer_stats()
        total_duration, total_count = timer_stats.get('mjTIMER_STEP', (0, 0))
        if total_duration <= 0:
            return

        non_aggregate = {
            name: (dur, cnt)
            for name, (dur, cnt) in timer_stats.items()
            if name not in self._AGGREGATE_TIMERS and dur > 0
        }
        sorted_items = sorted(
            non_aggregate.items(), key=lambda item: item[1][0], reverse=True
        )
        if not sorted_items:
            return
        top_name, (top_duration, _) = sorted_items[0]
        top_pct = top_duration / total_duration * 100

        prefix = f"[{label}] " if label else ""
        _logger.performance(
            f"{prefix}total={total_duration*1e3:.1f}ms (x{total_count} steps) | "
            f"bottleneck: {top_name}={top_duration*1e3:.1f}ms ({top_pct:.1f}%)"
        )

        constraint_dur = timer_stats.get('mjTIMER_CONSTRAINT', (0, 0))[0]
        pos_make_dur = timer_stats.get('mjTIMER_POS_MAKE', (0, 0))[0]
        pos_kinematics_dur = timer_stats.get('mjTIMER_POS_KINEMATICS', (0, 0))[0]
        col_broad_dur = timer_stats.get('mjTIMER_COL_BROAD', (0, 0))[0]
        col_narrow_dur = timer_stats.get('mjTIMER_COL_NARROW', (0, 0))[0]
        pos_col_dur = timer_stats.get('mjTIMER_POS_COLLISION', (0, 0))[0]

        _logger.performance(
            f"{prefix}CONSTRAINT={constraint_dur*1e3:.1f}ms ({constraint_dur/total_duration*100:.1f}%), "
            f"POS_MAKE={pos_make_dur*1e3:.1f}ms ({pos_make_dur/total_duration*100:.1f}%), "
            f"POS_KINEMATICS={pos_kinematics_dur*1e3:.1f}ms ({pos_kinematics_dur/total_duration*100:.1f}%), "
            f"POS_COLLISION={pos_col_dur*1e3:.1f}ms ({pos_col_dur/total_duration*100:.1f}%), "
            f"BROAD={col_broad_dur*1e3:.1f}ms, "
            f"NARROW={col_narrow_dur*1e3:.1f}ms"
        )

        cc = self.get_constraint_counts()
        contact_constraints = cc['nefc'] - cc['ne'] - cc['nf']
        per_constraint_us = (constraint_dur * 1e6) / (cc['nefc'] * total_count) if cc['nefc'] > 0 and total_count > 0 else 0
        _logger.performance(
            f"{prefix}nefc={cc['nefc']} (ne={cc['ne']}+nf={cc['nf']}+contact={contact_constraints}) "
            f"ncon={cc['ncon']} | "
            f"per_constraint={per_constraint_us:.2f}us/step"
        )

        if cc['ncon'] > 0:
            sources = self.get_contact_sources()
            sorted_sources = sorted(sources.items(), key=lambda x: x[1], reverse=True)
            _logger.performance(
                f"{prefix}Contact Sources: ncon={cc['ncon']} from {len(sorted_sources)} body pairs"
            )
            for (b1, b2), count in sorted_sources:
                _logger.performance(f"  {b1} ↔ {b2}: {count}")

    def mj_inverse(self):
        mujoco.mj_inverse(self._mjModel, self._mjData)
        
    def mj_fullM(self):
        mass_matrix = np.ndarray(shape=(self._mjModel.nv, self._mjModel.nv), dtype=np.float64, order="C")
        mujoco.mj_fullM(self._mjModel, mass_matrix, self._mjData.qM)
        mass_matrix = np.reshape(mass_matrix, (self._mjModel.nv, self._mjModel.nv))        
        return mass_matrix

    def mj_jacBody(self, jacp, jacr, body_id):
        mujoco.mj_jacBody(self._mjModel, self._mjData, jacp, jacr, body_id)

    def mj_jacSite(self, jacp, jacr, site_id):
        mujoco.mj_jacSite(self._mjModel, self._mjData, jacp, jacr, site_id)

    def mj_apply_force_at_site(self, site_name: str, force: np.ndarray, torque: np.ndarray):
        site_xpos = self._mjData.site(site_name).xpos
        body_id = self.model.get_site(site_name)['BodyID']
        body_xpos = self._mjData.xpos[body_id]
        
        r = site_xpos - body_xpos
        induced_torque = np.cross(r, force)
        total_torque = induced_torque + torque
        
        self._mjData.xfrc_applied[body_id, :3] += force
        self._mjData.xfrc_applied[body_id, 3:] += total_torque

    def mj_clear_xfrc_applied_for_site(self, site_name: str):
        try:
            site_id = mujoco.mj_name2id(self._mjModel, mujoco.mjtObj.mjOBJ_SITE, site_name)
            if site_id < 0:
                return
            
            body_id = self._mjModel.site_bodyid[site_id]
            self._mjData.xfrc_applied[body_id] = 0
        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"Failed to clear xfrc_applied for site '{site_name}': {e}")

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
            joint_type = self._mjModel.jnt_type[joint_id]
            joint_qvel_dict[joint_name] = self._mjData.qvel[self._mjModel.jnt_dofadr[joint_id]:self._mjModel.jnt_dofadr[joint_id] + get_dof_size(joint_type)]
        return joint_qvel_dict
    
    def query_joint_qacc(self, joint_names):
        joint_qacc_dict = {}
        for joint_name in joint_names:
            joint_id = self._mjModel.joint(joint_name).id
            joint_type = self._mjModel.jnt_type[joint_id]
            joint_qacc_dict[joint_name] = self._mjData.qacc[self._mjModel.jnt_dofadr[joint_id]:self._mjModel.jnt_dofadr[joint_id] + get_dof_size(joint_type)]
        return joint_qacc_dict
    
    def jnt_qposadr(self, joint_name):
        joint_id = self._mjModel.joint(joint_name).id
        return self._mjModel.jnt_qposadr[joint_id]
    
    def jnt_dofadr(self, joint_name):
        joint_id = self._mjModel.joint(joint_name).id
        return self._mjModel.jnt_dofadr[joint_id]
    
    def query_site_pos_and_mat(self, site_names: list[str]):
        site_pos_and_mat = {}
        for site_name in site_names:
            xpos = self._mjData.site(site_name).xpos
            xmat = self._mjData.site(site_name).xmat
            site_pos_and_mat[site_name] = {"xpos": xpos, "xmat": xmat}
        return site_pos_and_mat

    def query_site_size(self, site_names: list[str]):
        site_size_dict = {}
        for site_name in site_names:
            site_id = self._mjModel.site(site_name).id
            site_size = self._mjModel.site_size[site_id]
            site_size_dict[site_name] = site_size.copy()
        return site_size_dict

    def set_joint_qpos(self, joint_qpos):
        for joint_name, qpos in joint_qpos.items():
            joint_id = self._mjModel.joint(joint_name).id
            qpos_size = get_qpos_size(self._mjModel.jnt_type[joint_id])
            self._mjData.qpos[self._mjModel.jnt_qposadr[joint_id]:self._mjModel.jnt_qposadr[joint_id] + qpos_size] = qpos.copy()

    def set_joint_qvel(self, joint_qvel):
        for joint_name, qvel in joint_qvel.items():
            joint_id = self._mjModel.joint(joint_name).id
            dof_size = get_dof_size(self._mjModel.jnt_type[joint_id])
            self._mjData.qvel[self._mjModel.jnt_dofadr[joint_id]:self._mjModel.jnt_dofadr[joint_id] + dof_size] = qvel.copy()

    def mj_jac_site(self, site_names: list[str]):
        site_jacs_dict = {}
        for site_name in site_names:
            site_id = self._mjModel.site(site_name).id
            jacp = np.zeros((3, self.model.nv))
            jacr = np.zeros((3, self.model.nv))
            mujoco.mj_jacSite(self._mjModel, self._mjData, jacp, jacr, site_id)
            site_jacs_dict[site_name] = {"jacp": jacp, "jacr": jacr}
        return site_jacs_dict            

    def modify_equality_objects(self, old_obj1_id, old_obj2_id, new_obj1_id, new_obj2_id):
        for i in range(self.model.neq):
            if self._mjModel.eq_obj1id[i] == old_obj1_id and self._mjModel.eq_obj2id[i] == old_obj2_id:
                self._mjModel.eq_obj1id[i] = new_obj1_id
                self._mjModel.eq_obj2id[i] = new_obj2_id
                break

    def update_equality_constraints(self, constraint_list):
        for constraint in constraint_list:
            obj1_id = constraint['obj1_id']
            obj2_id = constraint['obj2_id']
            eq_data = constraint['eq_data']
            eq_type = constraint['eq_type']
            for i in range(self.model.neq):
                if self._mjModel.eq_obj1id[i] == obj1_id and self._mjModel.eq_obj2id[i] == obj2_id:
                    self._mjModel.eq_data[i] = eq_data.copy()
                    self._mjModel.eq_type[i] = eq_type
                    break

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
                self._mjData.mocap_pos[mocap_id] = data['pos'].copy()
                self._mjData.mocap_quat[mocap_id] = data['quat'].copy()
        
        if send_remote:
            await self._remote_set_mocap_pos_and_quat(mocap_data)

    def query_contact_simple(self):
        contact = self._mjData.contact
        contacts = []
        for i in range(self._mjData.ncon):
            if contact.geom1[i] >= 0 and contact.geom2[i] >= 0:
                contact_info = {
                    "ID": i,
                    "Dim": contact.dim[i],
                    "Geom1": contact.geom1[i],
                    "Geom2": contact.geom2[i],
                }
                contacts.append(contact_info)
        
        return contacts            
    
    def set_geom_friction(self, geom_friction_dict):
        model = self._mjModel
        for name, friction in geom_friction_dict.items():
            geom = model.geom(name)
            geom.friction = friction.copy()

    def add_extra_weight(self, random_weight_dict):
        model = self._mjModel
        for body_id, weight_info in random_weight_dict.items():
            torso = model.body(body_id)
            torso.ipos = weight_info['pos']
            torso.mass = [weight_info['weight'] + torso.mass.copy()[0]]

    def query_contact_force(self, contact_ids):
        contact_force_dict = {}
        for contact_id in contact_ids:
            contact_force = np.zeros(6)
            mujoco.mj_contactForce(self._mjModel, self._mjData, contact_id, contact_force)
            contact_force_dict[contact_id] = contact_force
        
        return contact_force_dict
    
    def get_cfrc_ext(self):
        return self._mjData.cfrc_ext.copy()

    def query_actuator_torques(self, actuator_names):
        actuator_torques = {}
        for actuator_name in actuator_names:
            actuator_id = self._mjModel.actuator(actuator_name).id

            joint_name = self._mjModel.actuator(actuator_name).trnid[0]
            joint_id = self._mjModel.joint(joint_name).id
            joint_type = self._mjModel.jnt_type[joint_id]

            torque_vector = np.zeros(6, dtype=np.float32)

            if joint_type == mujoco.mjtJoint.mjJNT_HINGE:
                gear = self._mjModel.actuator_gear[actuator_id][0]
                raw_torque = self._mjData.actuator_force[actuator_id]
                torque_vector[0] = raw_torque * gear
            else:
                gear = self._mjModel.actuator_gear[actuator_id][:3]
                raw_torque = self._mjData.actuator_force[actuator_id][:3]
                torque_vector[:3] = raw_torque * gear

            actuator_torques[actuator_name] = torque_vector

        return actuator_torques

    def query_joint_dofadrs(self, joint_names):
        dof_adrs = {}
        for joint_name in joint_names:
            joint_id = self._mjModel.joint(joint_name).id
            dof_adrs[joint_name] = self._mjModel.jnt_dofadr[joint_id]
        return dof_adrs

    def query_velocity_body_B(self, ee_body, base_body):
        base_id = self._mjModel.body(base_body).id
        ee_id = self._mjModel.body(ee_body).id

        ee_vel = np.zeros(6)
        mujoco.mj_objectVelocity(self._mjModel, self._mjData, mujoco.mjtObj.mjOBJ_BODY,
                                 ee_id, ee_vel, 0)
        base_vel = np.zeros(6)
        mujoco.mj_objectVelocity(self._mjModel, self._mjData, mujoco.mjtObj.mjOBJ_BODY,
                                 base_id, base_vel, 0)

        base_pos = self._mjData.body(base_id).xpos.copy()
        base_rot = self._mjData.body(base_id).xmat.copy().reshape(3, 3)

        linear_vel_B = base_rot.T @ (ee_vel[:3] - base_vel[:3])

        angular_vel_B = base_rot.T @ (ee_vel[3:] - base_vel[3:])

        combined_vel = np.concatenate([linear_vel_B, angular_vel_B])
        return combined_vel.astype(np.float32)

    def query_position_body_B(self, ee_body, base_body):
        base_id = self._mjModel.body(base_body).id
        base_quat = self._mjData.body(base_id).xquat.copy()
        base_pos = self._mjData.body(base_id).xpos.copy()

        ee_id = self._mjModel.body(ee_body).id
        ee_pos = self._mjData.body(ee_id).xpos.copy()

        base_rot_world = R.from_quat([base_quat[1], base_quat[2], base_quat[3], base_quat[0]])
        relative_pos_B = base_rot_world.inv().apply(ee_pos - base_pos)

        return relative_pos_B.astype(np.float32)

    def query_orientation_body_B(self, ee_body, base_body):
        base_id = self._mjModel.body(base_body).id
        base_quat = self._mjData.body(base_id).xquat.copy()

        ee_id = self._mjModel.body(ee_body).id
        ee_quat = self._mjData.body(ee_id).xquat.copy()

        rot_base = R.from_quat([base_quat[1], base_quat[2], base_quat[3], base_quat[0]])
        rot_ee = R.from_quat([ee_quat[1], ee_quat[2], ee_quat[3], ee_quat[0]])

        relative_rot_ee = rot_base.inv() * rot_ee
        relative_quat = relative_rot_ee.as_quat()[[3, 0, 1, 2]]

        return relative_quat.astype(np.float32)

    def query_joint_axes_B(self, joint_name, base_body):
        base_id = self._mjModel.body(base_body).id
        base_quat = self._mjData.body(base_id).xquat.copy()
        base_rot_world = R.from_quat([base_quat[1], base_quat[2], base_quat[3], base_quat[0]])
        base_rot_world_inv = base_rot_world.inv()

        joint_id = self._mjModel.joint(joint_name).id
        joint_axis = self._mjModel.jnt_axis[joint_id].copy()

        body_id = self._mjModel.jnt_bodyid[joint_id]
        body_quat = self._mjData.body(body_id).xquat.copy()
        body_rot = R.from_quat([body_quat[1], body_quat[2], body_quat[3], body_quat[0]])

        joint_axis_world = body_rot.apply(joint_axis)
        joint_axis_B = base_rot_world_inv.apply(joint_axis_world)

        return joint_axis_B.astype(np.float32)

    def query_robot_velocity_odom(self, base_body):
        base_id = self._mjModel.body(base_body).id
        base_pos = self._mjData.body(base_id).xpos.copy()
        base_quat = self._mjData.body(base_id).xquat.copy()

        cvel = self._mjData.body(base_id).cvel.copy()

        base_rot_world = R.from_quat([base_quat[1], base_quat[2], base_quat[3], base_quat[0]])
        linear_vel_odom = base_rot_world.inv().apply(cvel[:3])
        angular_vel_odom = base_rot_world.inv().apply(cvel[3:])

        return np.concatenate([linear_vel_odom, angular_vel_odom]).astype(np.float32)

    def query_robot_position_odom(self, base_body):
        base_id = self._mjModel.body(base_body).id
        base_pos = self._mjData.body(base_id).xpos.copy()
        return base_pos.astype(np.float32)

    def query_robot_orientation_odom(self, base_body):
        base_id = self._mjModel.body(base_body).id
        base_quat = self._mjData.body(base_id).xquat.copy()
        return base_quat.astype(np.float32)