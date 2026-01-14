"""
OrcaGymOptConfig - MuJoCo 仿真器优化配置

本模块提供 MuJoCo 仿真器优化参数的封装类，用于配置物理仿真器的各种参数。
这些参数影响仿真的精度、稳定性和性能。

使用场景:
    - 在环境初始化时从服务器获取配置
    - 通过 env.gym.opt 访问配置对象
    - 调整物理仿真精度和性能平衡

典型用法:
    ```python
    # 配置通过 OrcaGymLocal 的初始化自动获取
    env = OrcaGymLocalEnv(...)
    # 访问配置
    timestep = env.gym.opt.timestep
    gravity = env.gym.opt.gravity
    solver = env.gym.opt.solver
    ```
"""

import sys
import os
import grpc
import numpy as np
import json
from datetime import datetime

class OrcaGymOptConfig:
    """
    MuJoCo 仿真器优化配置容器
    
    存储 MuJoCo 仿真器的所有配置参数，包括时间步长、求解器、积分器、
    碰撞检测、物理参数等。这些参数影响仿真的精度、稳定性和性能。
    
    配置参数分类:
        1. 时间相关: timestep, apirate
        2. 求解器相关: solver, iterations, tolerance
        3. 物理参数: gravity, density, viscosity, wind, magnetic
        4. 接触参数: o_margin, o_solref, o_solimp, o_friction
        5. 积分器: integrator, impratio
        6. 碰撞检测: ccd_tolerance, ccd_iterations
        7. 其他: jacobian, cone, disableflags, enableflags
    
    使用示例:
        ```python
        # 访问时间步长
        dt = self.gym.opt.timestep * self.frame_skip
        
        # 访问重力
        gravity = self.gym.opt.gravity  # [x, y, z]
        ```
    """

    def __init__(self, opt_config: dict):
        """
        初始化配置对象，从字典中提取所有配置参数
        
        Args:
            opt_config: 配置字典，包含所有 MuJoCo 优化参数
        
        术语说明:
            - timestep: 物理仿真时间步长，越小越精确但计算越慢
            - solver: 求解器类型，用于求解约束优化问题
            - integrator: 积分器类型，用于数值积分计算运动
            - tolerance: 求解器容差，越小越精确但迭代次数越多
            - contact: 接触约束，处理物体之间的碰撞和接触
        """
        self.opt_config = opt_config.copy()
        
        # ========== 时间相关参数 ==========
        self.timestep = opt_config['timestep']
        """物理仿真时间步长 (秒)，通常为 0.001-0.01，越小越精确但计算越慢"""
        
        self.apirate = opt_config['apirate']
        """API 调用频率，控制与外部系统的通信频率"""
        
        # ========== 求解器参数 ==========
        self.impratio = opt_config['impratio']
        """阻抗比例，用于约束求解的数值稳定性"""
        
        self.tolerance = opt_config['tolerance']
        """主求解器容差，控制求解精度，越小越精确但迭代次数越多"""
        
        self.ls_tolerance = opt_config['ls_tolerance']
        """线搜索容差，用于优化算法中的线搜索步骤"""
        
        self.noslip_tolerance = opt_config['noslip_tolerance']
        """无滑动约束容差，用于处理接触时的滑动约束"""
        
        self.ccd_tolerance = opt_config['ccd_tolerance']
        """连续碰撞检测容差，用于检测快速移动物体的碰撞"""
        
        # ========== 物理环境参数 ==========
        self.gravity = opt_config['gravity']
        """重力加速度 [x, y, z]，通常为 [0, 0, -9.81] m/s²"""
        
        self.wind = opt_config['wind']
        """风力 [x, y, z]，用于模拟风对物体的作用力"""
        
        self.magnetic = opt_config['magnetic']
        """磁场强度 [x, y, z]，用于模拟磁场对物体的作用"""
        
        self.density = opt_config['density']
        """空气密度 (kg/m³)，影响空气阻力和浮力计算"""
        
        self.viscosity = opt_config['viscosity']
        """空气粘度，影响流体阻力计算"""
        
        # ========== 接触约束参数 ==========
        self.o_margin = opt_config['o_margin']
        """接触边距，接触点之间的最小距离阈值"""
        
        self.o_solref = opt_config['o_solref']
        """接触求解器参考参数 [timeconst, dampratio]，控制接触的刚度和阻尼"""
        
        self.o_solimp = opt_config['o_solimp']
        """接触求解器阻抗参数 [dmin, dmax, width, mid, power]，控制接触力的非线性特性"""
        
        self.o_friction = opt_config['o_friction']
        """接触摩擦参数 [滑动摩擦, 扭转摩擦, 滚动摩擦]"""
        
        # ========== 积分器和数值方法 ==========
        self.integrator = opt_config['integrator']
        """积分器类型: 'Euler' (欧拉法) 或 'RK4' (四阶龙格-库塔法)，RK4 更精确但更慢"""
        
        self.cone = opt_config['cone']
        """摩擦锥类型，用于摩擦力的计算模型"""
        
        self.jacobian = opt_config['jacobian']
        """雅可比矩阵计算方式: 'dense' (稠密) 或 'sparse' (稀疏)，稀疏矩阵更高效"""
        
        # ========== 求解器配置 ==========
        self.solver = opt_config['solver']
        """求解器类型: 'Newton' (牛顿法), 'PGS' (投影高斯-赛德尔), 'CG' (共轭梯度法)"""
        
        self.iterations = opt_config['iterations']
        """主求解器迭代次数，越多越精确但计算越慢，通常为 10-100"""
        
        self.ls_iterations = opt_config['ls_iterations']
        """线搜索迭代次数，用于优化算法"""
        
        self.noslip_iterations = opt_config['noslip_iterations']
        """无滑动约束迭代次数，用于处理接触滑动"""
        
        self.ccd_iterations = opt_config['ccd_iterations']
        """连续碰撞检测迭代次数，用于检测快速移动物体的碰撞"""
        
        # ========== 功能开关 ==========
        self.disableflags = opt_config['disableflags']
        """禁用的功能标志位，用于关闭某些仿真特性以提高性能"""
        
        self.enableflags = opt_config['enableflags']
        """启用的功能标志位，用于开启某些仿真特性"""
        
        self.disableactuator = opt_config['disableactuator']
        """禁用的执行器组，用于临时禁用某些执行器"""
        
        # ========== SDF (符号距离场) 参数 ==========
        self.sdf_initpoints = opt_config['sdf_initpoints']
        """SDF 初始化点数，用于复杂几何体的碰撞检测"""
        
        self.sdf_iterations = opt_config['sdf_iterations']
        """SDF 迭代次数，用于 SDF 计算"""
        
        # ========== 其他参数 ==========
        self.filterparent = opt_config.get('filterparent', True)
        """是否过滤父级碰撞，避免父子 body 之间的无效碰撞检测"""

