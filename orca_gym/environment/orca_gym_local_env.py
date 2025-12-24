from os import path
from typing import Any, Dict, Optional, Tuple, Union, SupportsFloat

import numpy as np
from numpy.typing import NDArray

import gymnasium as gym
from gymnasium import error, spaces
from gymnasium.spaces import Space
from gymnasium.core import ObsType

import asyncio
import sys

from orca_gym.log.orca_log import get_orca_logger
_logger = get_orca_logger()

from orca_gym import OrcaGymLocal
from orca_gym.protos.mjc_message_pb2_grpc import GrpcServiceStub 
from orca_gym.utils.rotations import mat2quat, quat2mat, quat_mul, quat2euler, euler2quat
from orca_gym.core.orca_gym_local import AnchorType, get_eq_type, CaptureMode

from orca_gym import OrcaGymModel
from orca_gym import OrcaGymData
from . import OrcaGymBaseEnv
from scipy.spatial.transform import Rotation as R

import grpc

from datetime import datetime
import time

class OrcaGymLocalEnv(OrcaGymBaseEnv):
    metadata = {'render_modes': ['human', 'none'], 'version': '0.0.1', 'render_fps': 30}
    
    def __init__(
        self,
        frame_skip: int,
        orcagym_addr: str,
        agent_names: list[str],
        time_step: float,        
        **kwargs        
    ):
        super().__init__(
            frame_skip = frame_skip,
            orcagym_addr = orcagym_addr,
            agent_names = agent_names,
            time_step = time_step,            
            **kwargs
        )

        render_fps = self.metadata.get("render_fps")

        # 用于异步渲染
        self._render_interval = 1.0 / render_fps
        self._render_time_step = time.perf_counter()

        # 用于同步渲染
        self._render_count_interval = self.realtime_step * render_fps
        self.render_count = 0
        self._last_frame_index = -1

        self.mj_forward()

        self._body_anchored = None
        self._anchor_body_name = "ActorManipulator_Anchor"
        self._anchor_dummy_body_name = "ActorManipulator_dummy"
        body_names = self.model.get_body_names()
        if (self._anchor_body_name in body_names and self._anchor_dummy_body_name in body_names):
            self._anchor_body_id = self.model.body_name2id(self._anchor_body_name)
            self._anchor_dummy_body_id = self.model.body_name2id(self._anchor_dummy_body_name)
        else:
            self._anchor_body_id = None
            self._anchor_dummy_body_id = None
            _logger.warning(f"Anchor body {self._anchor_body_name} not found in the model. Actor manipulation is disabled.")

    def initialize_simulation(
        self,
    ) -> Tuple[OrcaGymModel, OrcaGymData]:
        _logger.info(f"Initializing simulation: Class: {self.__class__.__name__}")
        model_xml_path = self.loop.run_until_complete(self._load_model_xml())
        self.loop.run_until_complete(self._initialize_orca_sim(model_xml_path))
        model = self.gym.model
        data = self.gym.data
        return model, data
    
    async def _load_model_xml(self):
        model_xml_path = await self.gym.load_model_xml()
        return model_xml_path

    async def _initialize_orca_sim(self, model_xml_path):
        await self.gym.init_simulation(model_xml_path)
        return

    def initialize_grpc(self):
        self.channel = grpc.aio.insecure_channel(
            self.orcagym_addr,
            options=[
                ('grpc.max_receive_message_length', 1024 * 1024 * 1024),
                ('grpc.max_send_message_length', 1024 * 1024 * 1024),
            ]
        )
        self.stub = GrpcServiceStub(self.channel)
        self.gym = OrcaGymLocal(self.stub)
    
    def pause_simulation(self):
        self.loop.run_until_complete(self._pause_simulation())

    async def _pause_simulation(self):
        await self.gym.pause_simulation()

    async def _close_grpc(self):
        if self.channel:
            await self.channel.close()

    def close(self):
        self.loop.run_until_complete(self._close_grpc())

    async def _get_body_manipulation_anchored(self):
        return await self.gym.get_body_manipulation_anchored()
    
    def get_body_manipulation_anchored(self):
        return self.loop.run_until_complete(self._get_body_manipulation_anchored())
    
    async def _get_body_manipulation_movement(self):
        return await self.gym.get_body_manipulation_movement()
    
    def begin_save_video(self, file_path: str, capture_mode: CaptureMode = CaptureMode.ASYNC ):
        return self.loop.run_until_complete(self._begin_save_video(file_path, capture_mode=capture_mode))

    async def _begin_save_video(self, file_path, capture_mode: CaptureMode = CaptureMode.ASYNC ):
        return await self.gym.begin_save_video(file_path, capture_mode=capture_mode)

    def stop_save_video(self):
        return self.loop.run_until_complete(self._stop_save_video())

    async def _stop_save_video(self):
        return await self.gym.stop_save_video()

    def get_next_frame(self) -> int:
        current_frame = self.loop.run_until_complete(self._get_current_frame())
        if current_frame < 0:
            # 如果摄像头没有使能，会一直返回-1
            return current_frame
        
        for _ in range(10):
            if current_frame == self._last_frame_index:
                time.sleep(self.realtime_step)
            else:
                self._last_frame_index = current_frame
                return current_frame
        
        return current_frame
            

    def get_current_frame(self) -> int:
        return self.loop.run_until_complete(self._get_current_frame())

    async def _get_current_frame(self):
        return await self.gym.get_current_frame()

    def get_camera_time_stamp(self, last_frame_index) -> dict:
        return self.loop.run_until_complete(self._get_camera_time_stamp(last_frame_index))

    async def _get_camera_time_stamp(self, last_frame_index):
        return await self.gym.get_camera_time_stamp(last_frame_index)

    def get_frame_png(self, image_path):
        return self.loop.run_until_complete(self._get_frame_png(image_path))

    async def _get_frame_png(self, image_path):
        return await self.gym.get_frame_png(image_path)

    def get_body_manipulation_movement(self):
        actor_movement = self.loop.run_until_complete(self._get_body_manipulation_movement())
        delta_pos = actor_movement["delta_pos"]
        delta_quat = actor_movement["delta_quat"]
        return delta_pos, delta_quat

    def do_simulation(self, ctrl, n_frames) -> None:
        """
        Step the simulation n number of frames and applying a control action.
        """
        # Check control input is contained in the action space
        if np.array(ctrl).shape != (self.model.nu,):
            raise ValueError(
                f"Action dimension mismatch. Expected {(self.model.nu,)}, found {np.array(ctrl).shape}"
            )
        self._step_orca_sim_simulation(ctrl, n_frames)
        self.gym.update_data()


    @property
    def render_mode(self) -> str:
        if hasattr(self, "_render_mode"):
            return self._render_mode
        else:
            return "human"
        
    @property
    def is_subenv(self) -> bool:
        if hasattr(self, "_is_subenv"):
            return self._is_subenv
        else:
            return False

    @property
    def sync_render(self) -> bool:
        if hasattr(self, "_sync_render"):
            return self._sync_render
        else:
            return False

    def render(self):
        if self.render_mode not in ["human", "force"]:
            return

        if self.sync_render:
            self.render_count += self._render_count_interval
            if (self.render_count >= 1.0):
                self.loop.run_until_complete(self.gym.render())
                self.do_body_manipulation() # 只有在渲染时才处理锚点操作，否则也不会有场景视口交互行为
                self.render_count -= 1
        else:
            time_diff = time.perf_counter() - self._render_time_step
            if (time_diff > self._render_interval):
                self._render_time_step = time.perf_counter()
                self.loop.run_until_complete(self.gym.render())
                self.do_body_manipulation() # 只有在渲染时才处理锚点操作，否则也不会有场景视口交互行为

    def do_body_manipulation(self):
        if self._anchor_body_id is None:
            # 老版本不支持锚点操作
            return

        actor_anchored, anchor_type = self.get_body_manipulation_anchored()
        if actor_anchored is None:
            if self._body_anchored is not None:
                self.release_body_anchored()
            return
        
        if self._body_anchored is None:
            self.anchor_actor(actor_anchored, anchor_type)
        
        
        delta_pos, delta_quat = self.get_body_manipulation_movement()

        # 移动和旋转锚点
        anchor_xpos, anchor_xmat, anchor_xquat = self.get_body_xpos_xmat_xquat([self._anchor_body_name])
        if anchor_xpos is None or anchor_xmat is None or anchor_xquat is None:
            _logger.warning(f"Anchor body {self._anchor_body_name} not found in the simulation. Cannot anchor.")
            return

        # 同步锚点位置
        anchor_xpos = delta_pos
        # 同步锚点四元数
        anchor_xquat = delta_quat

        # 更新锚点的位置和四元数
        self.set_mocap_pos_and_quat({
            self._anchor_body_name: {
                "pos": anchor_xpos,
                "quat": anchor_xquat
            }
        })
        self.mj_forward()

        # print(f"Updated anchor position: {anchor_xpos}, quaternion: {anchor_xquat}")


    def release_body_anchored(self):
        if self._body_anchored is not None:
            self.update_anchor_equality_constraints(self._anchor_dummy_body_name, AnchorType.NONE)

            self.set_mocap_pos_and_quat({
                self._anchor_body_name: {
                    "pos": np.array([0.0, 0.0, -1000.0], dtype=np.float64),
                    "quat": np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)  # Reset to a far position
                }
            })
            self.mj_forward()

            # print(f"Released actor: {self._body_anchored}")
            self._body_anchored = None
        else:
            _logger.warning("No actor is currently anchored.")

    def anchor_actor(self, actor_name: str, anchor_type: AnchorType):
        assert self._body_anchored is None, "An actor is already anchored. Please release it first."
        
        # 获取actor的位姿和四元数
        actor_xpos, actor_xmat, actor_xquat = self.get_body_xpos_xmat_xquat([actor_name])
        if actor_xpos is None or actor_xmat is None or actor_xquat is None:
            _logger.warning(f"Actor {actor_name} not found in the simulation. Cannot anchor.")
            return
        
        # 将锚点位置设置为actor的位姿

        mocap_pos_and_quat_dict = {
            self._anchor_body_name: {
                "pos": actor_xpos,
                "quat": actor_xquat if anchor_type == AnchorType.WELD else np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
            }
        }
        self.set_mocap_pos_and_quat(mocap_pos_and_quat_dict)
        self.mj_forward()

        self.update_anchor_equality_constraints(actor_name, anchor_type)


        self._body_anchored = actor_name
        # print(f"Anchored actor: {self._body_anchored} at position {actor_xpos} with quaternion {actor_xquat}")

    def update_anchor_equality_constraints(self, actor_name: str, anchor_type: AnchorType):
        # 更新锚点的平衡约束
        eq_list = self.model.get_eq_list()
        if eq_list is None or len(eq_list) == 0:
            raise ValueError("No equality constraints found in the model.")
        
        for eq in eq_list:
            old_obj1_id = eq["obj1_id"]
            old_obj2_id = eq["obj2_id"]
            if eq["obj1_id"] == self._anchor_body_id:
                eq["obj2_id"] = self.model.body_name2id(actor_name)
                self.gym.modify_equality_objects(
                    old_obj1_id= old_obj1_id,
                    old_obj2_id= old_obj2_id,
                    new_obj1_id= eq["obj1_id"],
                    new_obj2_id= eq["obj2_id"]
                )
                eq["eq_type"] = get_eq_type(anchor_type)
                # print(f"Anchoring actor {actor_name} to anchor body {self._anchor_body_name}")
                break
            elif eq["obj2_id"] == self._anchor_body_id:
                eq["obj1_id"] = self.model.body_name2id(actor_name)
                self.gym.modify_equality_objects(
                    old_obj1_id= old_obj1_id,
                    old_obj2_id= old_obj2_id,
                    new_obj1_id= eq["obj1_id"],
                    new_obj2_id= eq["obj2_id"]
                )
                eq["eq_type"] = get_eq_type(anchor_type)
                # print(f"Anchoring anchor body {self._anchor_body_name} to actor {actor_name}")
                break

        self.gym.update_equality_constraints(eq_list)
        self.mj_forward()

    def set_ctrl(self, ctrl):
        self.gym.set_ctrl(ctrl)

    def mj_step(self, nstep):
        self.gym.mj_step(nstep)

    def mj_forward(self):
        self.gym.mj_forward()

    def mj_jacBody(self, jacp, jacr, body_id):
        self.gym.mj_jacBody(jacp, jacr, body_id)

    def mj_jacSite(self, jacp, jacr, site_name):
        self.gym.mj_jacSite(jacp, jacr, site_name)

    def _step_orca_sim_simulation(self, ctrl, n_frames):
        self.set_ctrl(ctrl)
        self.mj_step(nstep=n_frames)

    def set_time_step(self, time_step):
        self.time_step = time_step
        self.realtime_step = time_step * self.frame_skip
        self.gym.set_time_step(time_step)
        self.loop.run_until_complete(self.gym.set_timestep_remote(time_step))
        return

    def update_data(self):
        self.gym.update_data()
        return

    def reset_simulation(self):
        self.gym.load_initial_frame()
        self.gym.update_data()

    def init_qpos_qvel(self):
        self.gym.update_data()
        self.init_qpos = self.gym.data.qpos.ravel().copy()
        self.init_qvel = self.gym.data.qvel.ravel().copy()

    def query_joint_offsets(self, joint_names) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        qpos_offsets, qvel_offsets, qacc_offsets = self.gym.query_joint_offsets(joint_names)
        return qpos_offsets, qvel_offsets, qacc_offsets
    
    def query_joint_lengths(self, joint_names):
        qpos_lengths, qvel_lengths, qacc_lengths = self.gym.query_joint_lengths(joint_names)
        return qpos_lengths, qvel_lengths, qacc_lengths
    
    def get_body_xpos_xmat_xquat(self, body_name_list):
        body_dict = self.gym.query_body_xpos_xmat_xquat(body_name_list)
        if len(body_dict) != len(body_name_list):
            _logger.error(f"Body Name List: {body_name_list}")
            _logger.error(f"Body Dict: {body_dict}")
            raise ValueError("Some body names are not found in the simulation.")
        xpos = np.array([body_dict[body_name]['Pos'] for body_name in body_name_list]).flat.copy()
        xmat = np.array([body_dict[body_name]['Mat'] for body_name in body_name_list]).flat.copy()
        xquat = np.array([body_dict[body_name]['Quat'] for body_name in body_name_list]).flat.copy()
        return xpos, xmat, xquat
    
    def query_sensor_data(self, sensor_names):
        sensor_data_dict = self.gym.query_sensor_data(sensor_names)
        return sensor_data_dict
    
    def query_joint_qpos(self, joint_names):
        joint_qpos_dict = self.gym.query_joint_qpos(joint_names)
        return joint_qpos_dict
    
    def query_joint_qvel(self, joint_names):
        joint_qvel_dict = self.gym.query_joint_qvel(joint_names)
        return joint_qvel_dict
    
    def query_joint_qacc(self, joint_names):
        joint_qacc_dict = self.gym.query_joint_qacc(joint_names)
        return joint_qacc_dict
    
    def jnt_qposadr(self, joint_name):
        joint_qposadr = self.gym.jnt_qposadr(joint_name)
        return joint_qposadr
    
    def jnt_dofadr(self, joint_name):
        joint_dofadr = self.gym.jnt_dofadr(joint_name)
        return joint_dofadr
        
    def query_site_pos_and_mat(self, site_names):
        query_dict = self.gym.query_site_pos_and_mat(site_names)
        site_dict = {}
        for site in query_dict:
            site_dict[site] = {
                'xpos': np.array(query_dict[site]['xpos']),
                'xmat': np.array(query_dict[site]['xmat'])
            }
        return site_dict
    
    def query_site_pos_and_quat(self, site_names) -> Dict[str, Dict[str, Union[NDArray[np.float64], NDArray[np.float64]]]]:
        query_dict = self.gym.query_site_pos_and_mat(site_names)
        site_dict = {}
        for site in query_dict:
            site_dict[site] = {
                'xpos': np.array(query_dict[site]['xpos']),
                'xquat': mat2quat(np.array(query_dict[site]['xmat']).reshape(3, 3))
            }
        return site_dict
    
    def query_site_size(self, site_names):
        site_size_dict = self.gym.query_site_size(site_names)
        return site_size_dict


    def query_site_pos_and_quat_B(self, site_names, base_body_list) -> Dict[str, Dict[str, Union[NDArray[np.float32], NDArray[np.float32]]]]:
        site_dict = self.gym.query_site_pos_and_mat(site_names)
        base_pos, _, base_quat = self.get_body_xpos_xmat_xquat(base_body_list)
        site_pos_quat_B = {}
        for site_name, site_value in site_dict.items():
            ee_pos = np.array(site_value['xpos'])
            ee_quat = mat2quat(np.array(site_value['xmat']).reshape(3, 3))

            # 转换为SciPy需要的[x,y,z,w]格式
            rot_base = R.from_quat([base_quat[1], base_quat[2], base_quat[3], base_quat[0]])
            rot_base_inv = rot_base.inv()

            rot_ee = R.from_quat([ee_quat[1], ee_quat[2], ee_quat[3], ee_quat[0]])
            relative_rot_ee = rot_base_inv * rot_ee
            relative_pos_ee = rot_base_inv.apply(ee_pos - base_pos)

            site_pos_quat_B[site_name] = {}
            site_pos_quat_B[site_name]["xpos"] = relative_pos_ee
            site_pos_quat_B[site_name]["xquat"] = relative_rot_ee.as_quat()[[3, 0, 1, 2]].astype(np.float32)

        return site_pos_quat_B


    def set_joint_qpos(self, joint_qpos):
        self.gym.set_joint_qpos(joint_qpos)

    def set_joint_qvel(self, joint_qvel):
        self.gym.set_joint_qvel(joint_qvel)
    
    def query_site_xvalp_xvalr(self, site_names) -> Tuple[Dict[str, NDArray[np.float64]], Dict[str, NDArray[np.float64]]]:
        query_dict = self.gym.mj_jac_site(site_names)
        xvalp_dict = {}
        xvalr_dict = {}
        for site in query_dict:
            xvalp_dict[site] = np.array(query_dict[site]['jacp']).reshape(3, -1) @ self.data.qvel
            xvalr_dict[site] = np.array(query_dict[site]['jacr']).reshape(3, -1) @ self.data.qvel

        return xvalp_dict, xvalr_dict        
    
    def query_site_xvalp_xvalr_B(self, site_names, base_body_list) -> Tuple[Dict[str, NDArray[np.float32]], Dict[str, NDArray[np.float32]]]:
        query_dict = self.gym.mj_jac_site(site_names)
        _, base_mat, _ = self.get_body_xpos_xmat_xquat(base_body_list)
        xvalp_dict = {}
        xvalr_dict = {}
        for site in query_dict:
            ee_xvalp = np.array(query_dict[site]['jacp']).reshape(3, -1) @ self.data.qvel
            ee_xvalr = np.array(query_dict[site]['jacr']).reshape(3, -1) @ self.data.qvel

            # 目前只有固定基座，不涉及基座的速度
            base_xvalp = np.zeros(3)
            base_xvalr = np.zeros(3)

            base_mat = base_mat.reshape(3, 3)
            linear_vel_B = base_mat.T @ (ee_xvalp - base_xvalp)
            angular_vel_B = base_mat.T @ (ee_xvalr - base_xvalr)

            xvalp_dict[site] = linear_vel_B.astype(np.float32)
            xvalr_dict[site] = angular_vel_B.astype(np.float32)

        return xvalp_dict, xvalr_dict
    
    def update_equality_constraints(self, eq_list):
        self.gym.update_equality_constraints(eq_list)

    def set_mocap_pos_and_quat(self, mocap_pos_and_quat_dict):
        send_remote = self.render_mode == "human" and not self.is_subenv
        self.loop.run_until_complete(self.gym.set_mocap_pos_and_quat(mocap_pos_and_quat_dict, send_remote))

    def query_contact_simple(self):
        return self.gym.query_contact_simple()
    
    def set_geom_friction(self, geom_friction_dict):
        self.gym.set_geom_friction(geom_friction_dict)

    def add_extra_weight(self, weight_load_dict):
        self.gym.add_extra_weight(weight_load_dict)
    
    def query_contact_force(self, contact_ids):
        contact_force = self.gym.query_contact_force(contact_ids)
        return contact_force
    
    def get_cfrc_ext(self):
        cfrc_ext = self.gym.get_cfrc_ext()
        return cfrc_ext

    def query_actuator_torques(self, actuator_names):
        actuator_torques = self.gym.query_actuator_torques(actuator_names)
        return actuator_torques

    def query_joint_dofadrs(self, joint_names):
        joint_dofadrs = self.gym.query_joint_dofadrs(joint_names)
        return joint_dofadrs
        
    def get_goal_bounding_box(self, geom_name):
        geom_size = self.gym.get_goal_bounding_box(geom_name)
        return geom_size

    def query_velocity_body_B(self, ee_body, base_body):
        return self.gym.query_velocity_body_B(ee_body, base_body)

    def query_position_body_B(self, ee_body, base_body):
        position_body_B = self.gym.query_position_body_B(ee_body, base_body)
        return position_body_B

    def query_orientation_body_B(self, ee_body, base_body):
        orientation_body_B = self.gym.query_orientation_body_B(ee_body, base_body)
        return orientation_body_B

    def query_joint_axes_B(self, joint_names, base_body):
        joint_axes_B = self.gym.query_joint_axes_B(joint_names, base_body)
        return joint_axes_B

    def query_robot_velocity_odom(self, base_body, initial_base_pos, initial_base_quat):
        linear, angular = self.gym.query_robot_velocity_odom(base_body, initial_base_pos, initial_base_quat)
        return linear, angular

    def query_robot_position_odom(self, base_body, initial_base_pos, initial_base_quat):
        robot_position_odom = self.gym.query_robot_position_odom(base_body, initial_base_pos, initial_base_quat)
        return robot_position_odom

    def query_robot_orientation_odom(self, base_body, initial_base_pos, initial_base_quat):
        robot_orientation_odom = self.gym.query_robot_orientation_odom(base_body, initial_base_pos, initial_base_quat)
        return robot_orientation_odom
    
    def set_actuator_trnid(self, actuator_id, trnid):
        self.gym.set_actuator_trnid(actuator_id, trnid)
        return

    def disable_actuator(self, actuator_groups: list[int]):
        self.gym.disable_actuator(actuator_groups)
        return

    async def _load_content_file(self, content_file_name, remote_file_dir="", local_file_dir="", temp_file_path=None):
        content_file_path = await self.gym.load_content_file(content_file_name, remote_file_dir, local_file_dir, temp_file_path)
        return content_file_path

    def load_content_file(self, content_file_name, remote_file_dir="", local_file_dir="", temp_file_path=None):
        return self.loop.run_until_complete(self._load_content_file(content_file_name, remote_file_dir, local_file_dir, temp_file_path))