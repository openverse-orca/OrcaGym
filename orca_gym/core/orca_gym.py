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


class OrcaGymBase:
    """
    gRPC 基础封装（local backend 的基类）。

    用途：
    - 作为 `OrcaGymLocal` 的基类，封装最基础的 gRPC 调用与 model/opt/data 指针。
    - 提供通用的异步接口（pause_simulation、set_qpos/qvel、mj_forward/inverse/step 等）。

    关键属性（由子类填充）：
    - `stub`：gRPC 服务存根，用于与服务端通信
    - `model`：`OrcaGymModel` 对象（静态模型信息）
    - `opt`：`OrcaGymOptConfig` 对象（MuJoCo 优化配置）
    - `data`：`OrcaGymData` 对象（动态仿真状态）

    术语速查（面向首次接触 gRPC/异步编程的读者）：
    - gRPC / stub：gRPC 是远程过程调用框架；stub 是客户端存根，用于调用远程服务（类似"函数代理"）
    - 异步方法（async/await）：异步方法需要 `await` 调用，不会阻塞当前线程；常用于网络 I/O 操作
    - 被动模式：OrcaGym 采用"被动模式"，仿真状态初始为 PAUSED，由 Gym 的 `step()` 主动驱动物理步进
    - `mj_forward/inverse/step`：MuJoCo 的核心计算函数；`mj_forward` 更新运动学/传感器，`mj_inverse` 计算逆动力学，`mj_step` 执行物理步进

    注意：
    - 该基类通常不直接使用；实际开发中通过 `OrcaGymLocal`（本地 backend）访问。
    - 所有异步方法（如 `pause_simulation`、`set_qpos`）需要在 `async` 函数中 `await` 调用。
    """
    def __init__(self, stub):
        self.stub = stub
        self.model = None
        self.opt = None
        self.data = None

    async def pause_simulation(self):
        """
        将仿真状态设置为 PAUSED（暂停）。

        说明：
        - OrcaGym 采用“被动模式”，仿真状态初始为 PAUSED，由 Gym 的 `step()` 主动驱动物理步进。
        - 该方法通过 gRPC 调用服务端，设置仿真状态为暂停。

        注意：
        - 异步方法，需要在 `async` 函数中 `await` 调用。
        """
        request = mjc_message_pb2.SetSimulationStateRequest(state=mjc_message_pb2.PAUSED)
        response = await self.stub.SetSimulationState(request)
        return response


    def print_opt_config(self):
        """
        打印优化配置信息（用于调试）。

        输出包含：timestep、iterations、noslip_iterations、ccd_iterations、sdf_iterations、gravity 等。
        """
        _logger.info(f"Opt config: timestep:{self.opt.timestep}, iterations:{self.opt.iterations}, noslip_iterations:{self.opt.noslip_iterations}, ccd_iterations:{self.opt.ccd_iterations}, sdf_iterations:{self.opt.sdf_iterations}, gravity:{self.opt.gravity}")
        
    def print_model_info(self, model_info):
        """
        打印模型基本信息（用于调试）。

        参数：
        - `model_info`：模型信息字典，包含 nq、nv、nu、nbody、njnt、ngeom、nsite 等维度参数。
        """
        _logger.info(f"Model info: nq:{model_info['nq']}, nv:{model_info['nv']}, nu:{model_info['nu']}, nbody:{model_info['nbody']}, njnt:{model_info['njnt']}, ngeom:{model_info['ngeom']}, nsite:{model_info['nsite']}")

    async def set_qpos(self, qpos):
        """
        设置广义坐标 `qpos`（远程调用版本）。

        参数：
        - `qpos`：`(nq,)` 广义坐标数组

        说明：
        - 该方法通过 gRPC 调用服务端，设置关节位置。
        - 修改状态后通常需要调用 `mj_forward()` 更新派生量。

        注意：
        - 异步方法，需要在 `async` 函数中 `await` 调用。
        """
        request = mjc_message_pb2.SetQposRequest(qpos=qpos)
        response = await self.stub.SetQpos(request)
        return response

    async def mj_forward(self):
        """
        执行 MuJoCo 前向计算（远程调用版本）。

        说明：
        - 更新所有动力学相关状态（位置、速度、加速度、力等）。
        - 在设置关节状态、mocap 位置等操作后需要调用，确保状态一致。

        注意：
        - 异步方法，需要在 `async` 函数中 `await` 调用。
        """
        request = mjc_message_pb2.MJ_ForwardRequest()
        response = await self.stub.MJ_Forward(request)
        return response

    async def mj_inverse(self):
        """
        执行 MuJoCo 逆动力学计算（远程调用版本）。

        说明：
        - 根据给定的加速度计算所需的力和力矩。
        - 用于计算实现特定运动所需的控制输入。

        注意：
        - 异步方法，需要在 `async` 函数中 `await` 调用。
        """
        request = mjc_message_pb2.MJ_InverseRequest()
        response = await self.stub.MJ_Inverse(request)
        return response
    
    async def mj_step(self, nstep):
        """
        执行 MuJoCo 仿真步进（远程调用版本）。

        参数：
        - `nstep`：步进次数，通常为 1 或 frame_skip

        说明：
        - 执行 nstep 次物理仿真步进，每次步进的时间为 timestep。
        - 在调用前需要先设置控制输入。

        注意：
        - 异步方法，需要在 `async` 函数中 `await` 调用。
        """
        request = mjc_message_pb2.MJ_StepRequest(nstep=nstep)
        response = await self.stub.MJ_Step(request)
        return response    
    
    async def set_qvel(self, qvel):
        """
        设置广义速度 `qvel`（远程调用版本）。

        参数：
        - `qvel`：`(nv,)` 广义速度数组

        说明：
        - 该方法通过 gRPC 调用服务端，设置关节速度。
        - 修改状态后通常需要调用 `mj_forward()` 更新派生量。

        注意：
        - 异步方法，需要在 `async` 函数中 `await` 调用。
        """
        request = mjc_message_pb2.SetQvelRequest(qvel=qvel)
        response = await self.stub.SetQvel(request)
        return response    