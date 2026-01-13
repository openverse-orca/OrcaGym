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
        """初始化仿真，加载模型并创建模型和数据对象"""
        _logger.info(f"Initializing simulation: Class: {self.__class__.__name__}")
        model_xml_path = self.loop.run_until_complete(self._load_model_xml())
        self.loop.run_until_complete(self._initialize_orca_sim(model_xml_path))
        model = self.gym.model
        data = self.gym.data
        return model, data
    
    async def _load_model_xml(self):
        """异步加载模型 XML 文件路径"""
        model_xml_path = await self.gym.load_model_xml()
        return model_xml_path

    async def _initialize_orca_sim(self, model_xml_path):
        """异步初始化 OrcaSim 仿真"""
        await self.gym.init_simulation(model_xml_path)
        return

    def initialize_grpc(self):
        """初始化 gRPC 通信通道和客户端"""
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
        """暂停仿真（采用被动模式）"""
        self.loop.run_until_complete(self._pause_simulation())

    async def _pause_simulation(self):
        """异步暂停仿真"""
        await self.gym.pause_simulation()

    async def _close_grpc(self):
        """异步关闭 gRPC 通道"""
        if self.channel:
            await self.channel.close()

    def close(self):
        """关闭环境，清理资源"""
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
        执行仿真步进：设置控制并步进 n_frames 次，然后同步数据
        
        这是环境 step() 函数的核心方法，执行一次完整的仿真步进。
        包括：设置控制输入、执行物理步进、同步最新状态。
        
        Args:
            ctrl: 控制输入数组，形状 (nu,)，nu 为执行器数量
            n_frames: 步进次数，通常等于 frame_skip
        
        使用示例 (参考 envs/legged_gym/legged_sim_env.py:179):
            ```python
            # 在 step 函数中执行仿真
            for _ in range(self._action_skip):
                # 计算扭矩控制
                torque_ctrl = agent.compute_torques(self.data.qpos, self.data.qvel)
                self.set_ctrl(torque_ctrl)
                # 执行仿真步进
                self.do_simulation(self.ctrl, self.frame_skip)
            ```
        
        使用示例 (参考 envs/xbot_gym/xbot_simple_env.py):
            ```python
            # 在 step 中执行多次物理步进（decimation）
            for _ in range(self.decimation):
                self.do_simulation(self.ctrl, 1)  # 每次步进1个物理步
            ```
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
        """
        设置控制输入（执行器命令）
        
        设置所有执行器的控制值，形状必须为 (nu,)，其中 nu 是执行器数量。
        通常在调用 mj_step() 之前设置。
        
        使用示例 (参考 envs/legged_gym/legged_sim_env.py:152-154):
            ```python
            # 在重置时清零控制
            self.ctrl = np.zeros(self.nu)
            self.set_ctrl(self.ctrl)
            self.mj_forward()
            ```
        
        使用示例 (参考 envs/xbot_gym/xbot_simple_env.py:91):
            ```python
            # 准备控制数组
            self.ctrl = np.zeros(self.nu, dtype=np.float32)
            # ... 计算控制值 ...
            self.set_ctrl(self.ctrl)
            ```
        """
        self.gym.set_ctrl(ctrl)

    def mj_step(self, nstep):
        """
        执行 MuJoCo 仿真步进
        
        执行 nstep 次物理仿真步进，每次步进的时间为 timestep。
        在调用前需要先设置控制输入 (set_ctrl)。
        
        使用示例 (参考 envs/legged_gym/legged_sim_env.py:179):
            ```python
            # 在 step 函数中执行仿真
            self.set_ctrl(self.ctrl)
            self.do_simulation(self.ctrl, self.frame_skip)  # 内部调用 mj_step
            ```
        """
        self.gym.mj_step(nstep)

    def mj_forward(self):
        """
        执行 MuJoCo 前向计算（更新动力学状态）
        
        更新所有动力学相关状态，包括位置、速度、加速度、力等。
        在设置关节状态、mocap 位置等操作后需要调用，确保状态一致。
        
        使用示例 (参考 envs/legged_gym/legged_sim_env.py:83, 154, 246):
            ```python
            # 在初始化时调用，避免 NaN 错误
            self.mj_forward()
            
            # 在设置初始状态后调用
            self.set_ctrl(self.ctrl)
            self.mj_forward()
            
            # 在重置后调用
            self.mj_forward()
            ```
        """
        self.gym.mj_forward()

    def mj_jacBody(self, jacp, jacr, body_id):
        """
        计算 body 的雅可比矩阵（位置和旋转）
        
        术语说明:
            - 雅可比矩阵 (Jacobian Matrix): 描述关节速度到 body 速度的线性映射关系
            - jacp: 位置雅可比，形状 (3, nv)，将关节速度映射到 body 的线性速度
            - jacr: 旋转雅可比，形状 (3, nv)，将关节速度映射到 body 的角速度
            - 用途: 用于逆运动学、速度控制、力控制等算法
        
        使用示例:
            ```python
            # 计算末端执行器的雅可比矩阵
            jacp = np.zeros((3, self.model.nv))
            jacr = np.zeros((3, self.model.nv))
            body_id = self.model.body_name2id("end_effector")
            self.mj_jacBody(jacp, jacr, body_id)
            # 计算末端执行器速度: v = jacp @ qvel, omega = jacr @ qvel
            ```
        """
        self.gym.mj_jacBody(jacp, jacr, body_id)

    def mj_jacSite(self, jacp, jacr, site_name):
        """
        计算 site 的雅可比矩阵（位置和旋转）
        
        术语说明:
            - 雅可比矩阵: 详见 mj_jacBody 的说明
            - Site: 标记点，详见 init_site_dict 的说明
        
        使用示例 (参考 orca_gym/environment/orca_gym_local_env.py:487-492):
            ```python
            # 计算 site 的雅可比矩阵用于速度计算
            query_dict = self.gym.mj_jac_site(["end_effector"])
            jacp = query_dict["end_effector"]["jacp"]  # (3, nv)
            jacr = query_dict["end_effector"]["jacr"]  # (3, nv)
            # 计算 site 速度: v = jacp @ self.data.qvel
            ```
        """
        self.gym.mj_jacSite(jacp, jacr, site_name)

    def _step_orca_sim_simulation(self, ctrl, n_frames):
        """执行仿真步进：设置控制并步进 n_frames 次"""
        self.set_ctrl(ctrl)
        self.mj_step(nstep=n_frames)

    def set_time_step(self, time_step):
        """设置仿真时间步长"""
        self.time_step = time_step
        self.realtime_step = time_step * self.frame_skip
        self.gym.set_time_step(time_step)
        self.loop.run_until_complete(self.gym.set_timestep_remote(time_step))
        return

    def update_data(self):
        """
        从服务器同步最新的仿真数据
        
        从 OrcaSim 服务器获取最新的 qpos、qvel、qacc 等状态数据，
        更新到本地的 self.data 中。在每次仿真步进后自动调用。
        
        使用示例 (参考 orca_gym/environment/orca_gym_local_env.py:190):
            ```python
            # 在 do_simulation 中自动调用
            self._step_orca_sim_simulation(ctrl, n_frames)
            self.gym.update_data()  # 同步最新状态
            
            # 之后可以安全访问 self.data.qpos, self.data.qvel 等
            current_qpos = self.data.qpos.copy()
            ```
        """
        self.gym.update_data()
        return

    def reset_simulation(self):
        """
        重置仿真到初始状态
        
        加载初始帧，同步数据，并重新设置时间步长。
        在环境 reset() 时调用，将仿真恢复到初始状态。
        
        使用示例 (参考 orca_gym/environment/orca_gym_env.py:165):
            ```python
            # 在 reset 函数中调用
            def reset(self, seed=None, options=None):
                self.reset_simulation()  # 重置到初始状态
                obs, info = self.reset_model()  # 重置模型特定状态
                return obs, info
            ```
        """
        self.gym.load_initial_frame()
        self.gym.update_data()
        self.set_time_step(self.time_step)

    def init_qpos_qvel(self):
        """
        初始化并保存初始关节位置和速度
        
        在环境初始化时调用，保存初始状态用于后续重置。
        保存的值可以通过 self.init_qpos 和 self.init_qvel 访问。
        
        使用示例 (参考 orca_gym/environment/orca_gym_env.py:68):
            ```python
            # 在 __init__ 中调用
            self.model, self.data = self.initialize_simulation()
            self.reset_simulation()
            self.init_qpos_qvel()  # 保存初始状态
            
            # 在 reset_model 中使用
            self.data.qpos[:] = self.init_qpos  # 恢复到初始位置
            self.data.qvel[:] = self.init_qvel  # 恢复到初始速度
            ```
        """
        self.gym.update_data()
        self.init_qpos = self.gym.data.qpos.ravel().copy()
        self.init_qvel = self.gym.data.qvel.ravel().copy()

    def query_joint_offsets(self, joint_names) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        """查询关节在状态数组中的偏移量（位置、速度、加速度）"""
        qpos_offsets, qvel_offsets, qacc_offsets = self.gym.query_joint_offsets(joint_names)
        return qpos_offsets, qvel_offsets, qacc_offsets
    
    def query_joint_lengths(self, joint_names):
        """查询关节状态数组的长度（位置、速度、加速度维度）"""
        qpos_lengths, qvel_lengths, qacc_lengths = self.gym.query_joint_lengths(joint_names)
        return qpos_lengths, qvel_lengths, qacc_lengths
    
    def get_body_xpos_xmat_xquat(self, body_name_list):
        """
        获取 body 的位姿（位置、旋转矩阵、四元数）
        
        返回指定 body 在世界坐标系中的位置、旋转矩阵和四元数。
        这是最常用的位姿查询方法，用于获取机器人基座、末端执行器等关键 body 的位姿。
        
        术语说明:
            - 位姿 (Pose): 物体的位置和姿态的组合
            - 位置 (Position): 物体在空间中的坐标 [x, y, z]
            - 旋转矩阵 (Rotation Matrix): 3x3 矩阵，描述物体的旋转姿态
            - 四元数 (Quaternion): [w, x, y, z] 格式，用于表示旋转，避免万向锁问题
            - 世界坐标系: 固定的全局坐标系，所有位置和姿态都相对于此坐标系
        
        Args:
            body_name_list: body 名称列表，如 ["base_link", "end_effector"]
        
        Returns:
            xpos: 位置数组，形状 (len(body_name_list)*3,)，每3个元素为一个 body 的 [x, y, z]
            xmat: 旋转矩阵数组，形状 (len(body_name_list)*9,)，每9个元素为一个 body 的 3x3 矩阵（按行展开）
            xquat: 四元数数组，形状 (len(body_name_list)*4,)，每4个元素为一个 body 的 [w, x, y, z]
        
        使用示例 (参考 examples/replicator/cameras_env.py:159, 168):
            ```python
            # 获取相机 body 的位姿
            camera_pos, _, camera_quat = self.get_body_xpos_xmat_xquat([self._camera_body_name])
            # camera_pos: [x, y, z]
            # camera_quat: [w, x, y, z]
            ```
        
        使用示例 (参考 envs/xbot_gym/xbot_simple_env.py:456):
            ```python
            # 获取基座位置用于计算高度
            base_pos, _, _ = self.get_body_xpos_xmat_xquat([self.base_body_name])
            real_base_z = float(base_pos[2])  # z 坐标
            ```
        
        使用示例 (参考 orca_gym/environment/orca_gym_local_env.py:249, 292):
            ```python
            # 获取锚点 body 的位姿用于物体操作
            anchor_xpos, anchor_xmat, anchor_xquat = self.get_body_xpos_xmat_xquat([self._anchor_body_name])
            ```
        """
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
        """查询传感器数据"""
        sensor_data_dict = self.gym.query_sensor_data(sensor_names)
        return sensor_data_dict
    
    def query_joint_qpos(self, joint_names):
        """
        查询关节位置
        
        返回指定关节的当前位置，字典格式，键为关节名称，值为位置值或数组。
        
        使用示例:
            ```python
            # 查询特定关节位置
            joint_pos = self.query_joint_qpos(["joint1", "joint2", "joint3"])
            # 返回: {"joint1": value1, "joint2": value2, "joint3": value3}
            
            # 用于构建观测空间
            obs["joint_pos"] = np.array([joint_pos[name] for name in joint_names])
            ```
        """
        joint_qpos_dict = self.gym.query_joint_qpos(joint_names)
        return joint_qpos_dict
    
    def query_joint_qvel(self, joint_names):
        """
        查询关节速度
        
        返回指定关节的当前速度，字典格式，键为关节名称，值为速度值或数组。
        
        使用示例:
            ```python
            # 查询关节速度用于观测
            joint_vel = self.query_joint_qvel(["joint1", "joint2"])
            # 返回: {"joint1": vel1, "joint2": vel2}
            
            # 用于计算奖励（速度惩罚）
            vel_penalty = sum(abs(v) for v in joint_vel.values())
            ```
        """
        joint_qvel_dict = self.gym.query_joint_qvel(joint_names)
        return joint_qvel_dict
    
    def query_joint_qacc(self, joint_names):
        """
        查询关节加速度
        
        返回指定关节的当前加速度，字典格式，键为关节名称，值为加速度值或数组。
        
        使用示例:
            ```python
            # 查询关节加速度
            joint_acc = self.query_joint_qacc(["joint1", "joint2"])
            # 用于分析运动状态或计算动力学
            ```
        """
        joint_qacc_dict = self.gym.query_joint_qacc(joint_names)
        return joint_qacc_dict
    
    def jnt_qposadr(self, joint_name):
        """
        获取关节在 qpos 数组中的起始地址
        
        返回关节在全局 qpos 数组中的起始索引，用于访问特定关节的位置数据。
        不同关节类型占用的位置数量不同（旋转关节1个，自由关节7个等）。
        
        使用示例:
            ```python
            # 获取关节在 qpos 中的地址
            joint_addr = self.jnt_qposadr("joint1")
            joint_nq = self.model.get_joint_byname("joint1")["JointNq"]
            # 提取该关节的位置
            joint_qpos = self.data.qpos[joint_addr:joint_addr+joint_nq]
            ```
        """
        joint_qposadr = self.gym.jnt_qposadr(joint_name)
        return joint_qposadr
    
    def jnt_dofadr(self, joint_name):
        """
        获取关节在 qvel 数组中的起始地址
        
        返回关节在全局 qvel 数组中的起始索引，用于访问特定关节的速度数据。
        通常等于自由度数量（旋转关节1个，自由关节6个等）。
        
        使用示例:
            ```python
            # 获取关节在 qvel 中的地址
            joint_dofadr = self.jnt_dofadr("joint1")
            joint_nv = self.model.get_joint_byname("joint1")["JointNv"]
            # 提取该关节的速度
            joint_qvel = self.data.qvel[joint_dofadr:joint_dofadr+joint_nv]
            ```
        """
        joint_dofadr = self.gym.jnt_dofadr(joint_name)
        return joint_dofadr
        
    def query_site_pos_and_mat(self, site_names):
        """查询 site 的位置和旋转矩阵"""
        query_dict = self.gym.query_site_pos_and_mat(site_names)
        site_dict = {}
        for site in query_dict:
            site_dict[site] = {
                'xpos': np.array(query_dict[site]['xpos']),
                'xmat': np.array(query_dict[site]['xmat'])
            }
        return site_dict
    
    def query_site_pos_and_quat(self, site_names) -> Dict[str, Dict[str, Union[NDArray[np.float64], NDArray[np.float64]]]]:
        """
        查询 site 的位置和四元数（从旋转矩阵转换）
        
        返回指定 site 在世界坐标系中的位置和四元数。
        Site 通常用于标记末端执行器、目标点等关键位置。
        
        Args:
            site_names: site 名称列表，如 ["end_effector", "target"]
        
        Returns:
            字典，键为 site 名称，值为包含 'xpos' 和 'xquat' 的字典
            - xpos: 位置数组 [x, y, z]
            - xquat: 四元数 [w, x, y, z]
        
        使用示例:
            ```python
            # 查询末端执行器位姿
            ee_site = self.query_site_pos_and_quat(["end_effector"])
            ee_pos = ee_site["end_effector"]["xpos"]  # [x, y, z]
            ee_quat = ee_site["end_effector"]["xquat"]  # [w, x, y, z]
            
            # 用于计算到目标的距离
            target_site = self.query_site_pos_and_quat(["target"])
            distance = np.linalg.norm(ee_pos - target_site["target"]["xpos"])
            ```
        """
        query_dict = self.gym.query_site_pos_and_mat(site_names)
        site_dict = {}
        for site in query_dict:
            site_dict[site] = {
                'xpos': np.array(query_dict[site]['xpos']),
                'xquat': mat2quat(np.array(query_dict[site]['xmat']).reshape(3, 3))
            }
        return site_dict
    
    def query_site_size(self, site_names):
        """查询 site 的尺寸"""
        site_size_dict = self.gym.query_site_size(site_names)
        return site_size_dict


    def query_site_pos_and_quat_B(self, site_names, base_body_list) -> Dict[str, Dict[str, Union[NDArray[np.float32], NDArray[np.float32]]]]:
        """
        查询 site 相对于基座 body 的位置和四元数（基座坐标系）
        
        术语说明:
            - 基座坐标系 (Base Frame): 以机器人基座为原点的局部坐标系
            - 相对位姿: 相对于基座的位置和姿态，而不是世界坐标系
            - 用途: 在机器人控制中，通常需要知道末端执行器相对于基座的位置
        
        使用示例:
            ```python
            # 查询末端执行器相对于基座的位置
            ee_pos_B, ee_quat_B = self.query_site_pos_and_quat_B(
                ["end_effector"], 
                ["base_link"]
            )
            # 返回的是相对于基座的位置，用于逆运动学计算
            ```
        """
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
        """
        设置关节位置
        
        直接设置关节位置，用于重置或初始化机器人姿态。
        设置后需要调用 mj_forward() 更新动力学状态。
        
        使用示例:
            ```python
            # 在重置时设置初始关节位置
            initial_qpos = np.array([0.0, 0.5, -1.0, ...])  # 初始姿态
            self.set_joint_qpos(initial_qpos)
            self.mj_forward()  # 更新状态
            ```
        """
        self.gym.set_joint_qpos(joint_qpos)

    def set_joint_qvel(self, joint_qvel):
        """
        设置关节速度
        
        直接设置关节速度，用于重置或初始化机器人运动状态。
        设置后需要调用 mj_forward() 更新动力学状态。
        
        使用示例:
            ```python
            # 在重置时清零速度
            initial_qvel = np.zeros(self.model.nv)
            self.set_joint_qvel(initial_qvel)
            self.mj_forward()  # 更新状态
            ```
        """
        self.gym.set_joint_qvel(joint_qvel)
    
    def query_site_xvalp_xvalr(self, site_names) -> Tuple[Dict[str, NDArray[np.float64]], Dict[str, NDArray[np.float64]]]:
        """查询 site 的线速度和角速度（世界坐标系）"""
        query_dict = self.gym.mj_jac_site(site_names)
        xvalp_dict = {}
        xvalr_dict = {}
        for site in query_dict:
            xvalp_dict[site] = np.array(query_dict[site]['jacp']).reshape(3, -1) @ self.data.qvel
            xvalr_dict[site] = np.array(query_dict[site]['jacr']).reshape(3, -1) @ self.data.qvel

        return xvalp_dict, xvalr_dict        
    
    def query_site_xvalp_xvalr_B(self, site_names, base_body_list) -> Tuple[Dict[str, NDArray[np.float32]], Dict[str, NDArray[np.float32]]]:
        """
        查询 site 相对于基座 body 的线速度和角速度（基座坐标系）
        
        术语说明:
            - 线速度 (Linear Velocity): 物体在空间中的移动速度 [vx, vy, vz]
            - 角速度 (Angular Velocity): 物体绕轴旋转的速度 [wx, wy, wz]
            - 基座坐标系: 详见 query_site_pos_and_quat_B 的说明
        
        使用示例:
            ```python
            # 查询末端执行器相对于基座的速度
            linear_vel_B, angular_vel_B = self.query_site_xvalp_xvalr_B(
                ["end_effector"],
                ["base_link"]
            )
            # 用于速度控制或计算速度误差
            ```
        """
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
        """更新等式约束列表"""
        self.gym.update_equality_constraints(eq_list)

    def set_mocap_pos_and_quat(self, mocap_pos_and_quat_dict):
        """
        设置 mocap body 的位置和四元数（用于物体操作）
        
        Mocap body 是用于物体操作的虚拟 body，通过设置其位姿可以控制被锚定的物体。
        常用于实现抓取、拖拽等操作。
        
        Args:
            mocap_pos_and_quat_dict: 字典，键为 mocap body 名称，值为包含 'pos' 和 'quat' 的字典
                - pos: 位置数组 [x, y, z]
                - quat: 四元数 [w, x, y, z]
        
        使用示例 (参考 orca_gym/environment/orca_gym_local_env.py:260-265, 275-280):
            ```python
            # 设置锚点位置用于物体操作
            self.set_mocap_pos_and_quat({
                self._anchor_body_name: {
                    "pos": np.array([0.5, 0.0, 0.8], dtype=np.float64),
                    "quat": np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
                }
            })
            self.mj_forward()  # 更新状态
            
            # 释放物体时移动到远处
            self.set_mocap_pos_and_quat({
                self._anchor_body_name: {
                    "pos": np.array([0.0, 0.0, -1000.0], dtype=np.float64),
                    "quat": np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
                }
            })
            ```
        """
        send_remote = self.render_mode == "human" and not self.is_subenv
        self.loop.run_until_complete(self.gym.set_mocap_pos_and_quat(mocap_pos_and_quat_dict, send_remote))

    def query_contact_simple(self):
        """查询简单接触信息"""
        return self.gym.query_contact_simple()
    
    def set_geom_friction(self, geom_friction_dict):
        """设置几何体的摩擦系数"""
        self.gym.set_geom_friction(geom_friction_dict)

    def add_extra_weight(self, weight_load_dict):
        """为 body 添加额外重量"""
        self.gym.add_extra_weight(weight_load_dict)
    
    def query_contact_force(self, contact_ids):
        """查询接触力"""
        contact_force = self.gym.query_contact_force(contact_ids)
        return contact_force
    
    def get_cfrc_ext(self):
        """获取外部约束力"""
        cfrc_ext = self.gym.get_cfrc_ext()
        return cfrc_ext

    def query_actuator_torques(self, actuator_names):
        """查询执行器扭矩"""
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