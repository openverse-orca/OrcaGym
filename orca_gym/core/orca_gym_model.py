import sys
import os
import grpc
import numpy as np
import json
from datetime import datetime
import mujoco
from typing import Union

from orca_gym.log.orca_log import get_orca_logger
_logger = get_orca_logger()


class OrcaGymModel:
    mjEQ_CONNECT = 0       # connect two bodies at a point (ball joint)
    mjEQ_WELD = 1          # fix relative position and orientation of two bodies
    mjEQ_JOINT = 2         # couple the values of two scalar joints with cubic
    mjEQ_TENDON = 3        # couple the lengths of two tendons with cubic
    mjEQ_FLEX = 4          # fix all edge lengths of a flex
    mjEQ_DISTANCE = 5      # unsupported, will cause an error if used

    PRINT_INIT_INFO = False
    PRINT_FORMATTED_INFO = False

    def __init__(self, model_info):
        self.init_model_info(model_info)                
        self._eq_list = None
        self._mocap_dict = None
        self._actuator_dict = None
        self._body_dict = None
        self._joint_dict = None
        self._site_dict = None
        self._init_time = datetime.now()
        
    def init_model_info(self, model_info):
        """初始化模型基本信息（维度参数）"""
        self.model_info = model_info
        if self.PRINT_INIT_INFO:
            if self.PRINT_FORMATTED_INFO:
                formatted_dict = json.dumps(model_info, indent=4)
                _logger.debug(f"Model info: {formatted_dict}")
            else:
                _logger.debug(f"Model info: {model_info}")

        self.nq = model_info["nq"]
        self.nv = model_info["nv"]
        self.nu = model_info["nu"]
        self.ngeom = model_info["ngeom"]
    
    def init_eq_list(self, eq_list):
        """
        初始化等式约束列表
        
        术语说明:
            - 等式约束 (Equality Constraint): 在 MuJoCo 中用于连接两个 body 的约束关系
            - 常见类型: CONNECT (球关节连接)、WELD (焊接固定)、JOINT (关节耦合) 等
            - 用途: 实现抓取、固定物体等操作，通过约束将两个 body 连接在一起
        
        使用示例:
            ```python
            # 获取等式约束列表用于物体操作
            eq_list = self.model.get_eq_list()
            # 修改约束以连接物体
            eq["obj2_id"] = self.model.body_name2id(actor_name)
            ```
        """
        self._eq_list = eq_list
        if self.PRINT_INIT_INFO:
            if self.PRINT_FORMATTED_INFO:
                formatted_dict = json.dumps(eq_list, indent=4)
                _logger.debug(f"Equality constraints: {formatted_dict}")
            else:
                _logger.debug(f"Equality constraints: {eq_list}")

        self.neq = len(eq_list)

    def get_eq_list(self):
        """
        获取等式约束列表
        
        术语说明:
            - 等式约束: 用于连接两个 body 的约束关系，详见 init_eq_list 的说明
        
        使用示例:
            ```python
            # 获取约束列表用于修改
            eq_list = self.model.get_eq_list()
            for eq in eq_list:
                if eq["obj1_id"] == self._anchor_body_id:
                    # 修改约束目标
                    eq["obj2_id"] = self.model.body_name2id(actor_name)
            ```
        """
        return self._eq_list
    
    def init_mocap_dict(self, mocap_dict):
        """
        初始化 mocap body 字典
        
        术语说明:
            - Mocap Body (Motion Capture Body): 虚拟的、可自由移动的 body，不受物理约束
            - 用途: 用于物体操作，通过等式约束将 mocap body 与真实物体连接，移动 mocap body 即可控制物体
            - 常见应用: 抓取、拖拽、移动物体等操作
        
        使用示例:
            ```python
            # 设置 mocap body 位置用于物体操作
            self.set_mocap_pos_and_quat({
                "ActorManipulator_Anchor": {
                    "pos": np.array([0.5, 0.0, 0.8]),
                    "quat": np.array([1.0, 0.0, 0.0, 0.0])
                }
            })
            ```
        """
        self._mocap_dict = mocap_dict
        if self.PRINT_INIT_INFO:
            if self.PRINT_FORMATTED_INFO:
                formatted_dict = json.dumps(mocap_dict, indent=4)
                _logger.debug(f"Mocap dict: {formatted_dict}")
            else:
                _logger.debug(f"Mocap dict: {mocap_dict}")

        self.nmocap = len(mocap_dict)

    def init_actuator_dict(self, actuator_dict):
        """
        初始化执行器字典，建立名称和ID的映射关系
        
        术语说明:
            - 执行器 (Actuator): 机器人的驱动元件，如电机、液压缸等，用于产生力和力矩
            - 控制输入: 发送给执行器的命令值，通常对应期望的扭矩、位置或速度
            - nu: 执行器数量，等于动作空间的维度
        
        使用示例:
            ```python
            # 执行器在模型加载时自动初始化
            # 可以通过以下方式访问:
            actuator_dict = self.model.get_actuator_dict()
            actuator_id = self.model.actuator_name2id("joint1_actuator")
            ```
        """
        self._actuaton_id2name_map = {}
        for i, (actuator_name, actuator) in enumerate(actuator_dict.items()):
            actuator["ActuatorId"] = i
            self._actuaton_id2name_map[i] = actuator_name
        self._actuator_dict = actuator_dict.copy()
        if self.PRINT_INIT_INFO:
            if self.PRINT_FORMATTED_INFO:
                formatted_dict = json.dumps(actuator_dict, indent=4)
                _logger.debug(f"Actuator dict: {formatted_dict}")
            else:
                _logger.debug(f"Actuator dict: {actuator_dict}")

    def get_actuator_dict(self):
        """获取所有执行器字典"""
        return self._actuator_dict
    
    def get_actuator_byid(self, id : int):
        """根据ID获取执行器信息"""
        actuator_name = self._actuaton_id2name_map[id]
        return self._actuator_dict[actuator_name]
    
    def get_actuator_byname(self, name : str):
        """根据名称获取执行器信息"""
        return self._actuator_dict[name]
    
    def actuator_name2id(self, actuator_name):
        """
        执行器名称转ID
        
        将执行器名称转换为对应的 ID，用于设置控制输入。
        
        使用示例:
            ```python
            # 获取执行器 ID 列表用于控制
            self._arm_actuator_id = [
                self.model.actuator_name2id(actuator_name) 
                for actuator_name in self._arm_moto_names
            ]
            ```
        """
        return self._actuator_dict[actuator_name]["ActuatorId"]
    
    def actuator_id2name(self, actuator_id):
        """执行器ID转名称"""
        return self._actuaton_id2name_map[actuator_id]
    
    def init_body_dict(self, body_dict):
        """
        初始化 body 字典，建立名称和ID的映射关系
        
        术语说明:
            - Body: MuJoCo 中的刚体，是物理仿真的基本单元
            - 每个 body 有质量、惯性、位置、姿态等属性
            - Body 之间通过关节 (Joint) 连接，形成运动链
        
        使用示例:
            ```python
            # Body 在模型加载时自动初始化
            # 可以通过以下方式访问:
            body_names = list(self.model.get_body_names())
            body_id = self.model.body_name2id("base_link")
            ```
        """
        self._body_id2name_map = {}
        for i, (body_name, body) in enumerate(body_dict.items()):
            body["BodyId"] = i
            self._body_id2name_map[i] = body_name
        self._body_dict = body_dict.copy()
        if self.PRINT_INIT_INFO:
            if self.PRINT_FORMATTED_INFO:
                formatted_dict = json.dumps(body_dict, indent=4)
                _logger.debug(f"Body dict: {formatted_dict}")
            else:
                _logger.debug(f"Body dict: {body_dict}")

    def get_body_dict(self):
        """获取所有 body 字典"""
        return self._body_dict
    
    def get_body_byid(self, id : int):
        """根据ID获取 body 信息"""
        body_name = self._body_id2name_map[id]
        return self._body_dict[body_name]
    
    def get_body_byname(self, name : str):
        """根据名称获取 body 信息"""
        return self._body_dict[name]
    
    def body_name2id(self, body_name):
        """
        Body 名称转ID
        
        将 body 名称转换为对应的 ID，用于需要 ID 的底层操作。
        
        使用示例:
            ```python
            # 在更新等式约束时使用
            body_id = self.model.body_name2id(actor_name)
            eq["obj2_id"] = body_id
            ```
        """
        return self._body_dict[body_name]["BodyId"]
    
    def body_id2name(self, body_id):
        """Body ID转名称"""
        return self._body_id2name_map[body_id]

    def init_joint_dict(self, joint_dict):
        """
        初始化关节字典，建立名称和ID的映射关系
        
        术语说明:
            - 关节 (Joint): 连接两个 body 的约束，定义它们之间的相对运动
            - 关节类型: 旋转关节 (revolute)、滑动关节 (prismatic)、自由关节 (free) 等
            - 关节自由度: 关节允许的运动维度，旋转关节1个，滑动关节1个，自由关节6个
        
        使用示例:
            ```python
            # 关节在模型加载时自动初始化
            # 可以通过以下方式访问:
            joint_dict = self.model.get_joint_dict()
            joint_id = self.model.joint_name2id("joint1")
            ```
        """
        self._joint_id2name_map = {}
        for i, (joint_name, joint) in enumerate(joint_dict.items()):
            joint["JointId"] = i
            self._joint_id2name_map[i] = joint_name
        self._joint_dict = joint_dict.copy()
        if self.PRINT_INIT_INFO:
            if self.PRINT_FORMATTED_INFO:
                formatted_dict = json.dumps(joint_dict, indent=4)
                _logger.debug(f"Joint dict: {formatted_dict}")
            else:
                _logger.debug(f"Joint dict: {joint_dict}")

    def get_joint_dict(self):
        """获取所有关节字典"""
        return self._joint_dict
    
    def get_joint_byid(self, id : int):
        """根据ID获取关节信息"""
        joint_name = self._joint_id2name_map[id]
        return self._joint_dict[joint_name]
    
    def get_joint_byname(self, name : str):
        """根据名称获取关节信息"""
        return self._joint_dict[name]
    
    def joint_name2id(self, joint_name):
        """关节名称转ID"""
        return self._joint_dict[joint_name]["JointId"]
    
    def joint_id2name(self, joint_id):
        """关节ID转名称"""
        return self._joint_id2name_map[joint_id]

    def init_geom_dict(self, geom_dict):
        """初始化几何体字典，建立名称和ID的映射关系"""
        self._geom_id2name_map = {}
        for i, (geom_name, geom) in enumerate(geom_dict.items()):
            geom["GeomId"] = i
            self._geom_id2name_map[i] = geom_name
        self._geom_dict = geom_dict.copy()
        if self.PRINT_INIT_INFO:
            if self.PRINT_FORMATTED_INFO:
                formatted_dict = json.dumps(geom_dict, indent=4)
                _logger.debug(f"Geom dict: {formatted_dict}")
            else:
                _logger.debug(f"Geom dict: {geom_dict}")

    def get_geom_dict(self):
        """获取所有几何体字典"""
        return self._geom_dict
    
    def get_geom_byid(self, id : int):
        """根据ID获取几何体信息"""
        geom_name = self._geom_id2name_map[id]
        return self._geom_dict[geom_name]
    
    def get_geom_byname(self, name : str):
        """根据名称获取几何体信息"""
        return self._geom_dict[name]
    
    def geom_name2id(self, geom_name):
        """几何体名称转ID"""
        return self._geom_dict[geom_name]["GeomId"]
    
    def geom_id2name(self, geom_id):
        """几何体ID转名称"""
        return self._geom_id2name_map[geom_id]

    def get_body_names(self):
        """
        获取所有 body 名称列表
        
        返回可迭代的 body 名称集合，用于查找特定 body 或遍历所有 body。
        
        使用示例:
            ```python
            # 查找包含特定关键词的 body
            all_bodies = self.model.get_body_names()
            for body in all_bodies:
                if "base" in body.lower() and "link" in body.lower():
                    self.base_body_name = body
                    break
            ```
        
        使用示例:
            ```python
            # 遍历所有 body 进行查询
            for body_name in self.model.get_body_names():
                pos, _, quat = self.get_body_xpos_xmat_xquat([body_name])
            ```
        """
        return self._body_dict.keys()
    
    def get_geom_body_name(self, geom_id):
        """根据几何体ID获取其所属的 body 名称"""
        geom_name = self.geom_id2name(geom_id)
        return self._geom_dict[geom_name]["BodyName"]
    
    def get_geom_body_id(self, geom_id):
        """根据几何体ID获取其所属的 body ID"""
        body_name = self.get_geom_body_name(geom_id)
        return self.body_name2id(body_name)
    
    def get_actuator_ctrlrange(self):
        """
        获取所有执行器的控制范围（用于定义动作空间）
        
        返回形状为 (nu, 2) 的数组，每行包含 [min, max] 控制范围。
        常用于在环境初始化时定义 action_space。
        
        术语说明:
            - 动作空间 (Action Space): 强化学习中智能体可以执行的所有动作的集合
            - 控制范围: 执行器能够接受的最小和最大控制值，超出范围会被截断
            - nu: 执行器数量，等于动作空间的维度
        
        使用示例:
            ```python
            # 获取执行器控制范围并定义动作空间
            all_actuator_ctrlrange = self.model.get_actuator_ctrlrange()
            # ctrlrange 形状: (nu, 2)，每行为 [min, max]
            self.action_space = self.generate_action_space(all_actuator_ctrlrange)
            ```
        """
        actuator_ctrlrange = {}
        for actuator_name, actuator in self._actuator_dict.items():
            actuator_ctrlrange[actuator_name] = actuator["CtrlRange"]
        ctrlrange = np.array(list(actuator_ctrlrange.values()))
        return ctrlrange
    
    def get_joint_qposrange(self, joint_names : list):
        """获取指定关节的位置范围"""
        joint_range = {}
        for joint_name in joint_names:
            joint = self.get_joint_byname(joint_name)
            joint_range[joint_name] = joint["Range"]
        qposrange = np.array(list(joint_range.values()))
        return qposrange
    
    def init_site_dict(self, site_dict):
        """
        初始化 site 字典
        
        术语说明:
            - Site: MuJoCo 中的标记点，用于标记特定位置（如末端执行器、目标点）
            - Site 不参与物理仿真，仅用于查询位置和姿态
            - 常用于: 查询末端执行器位姿、定义目标位置、计算距离等
        
        使用示例:
            ```python
            # Site 在模型加载时自动初始化
            # 可以通过以下方式查询:
            site_pos, site_quat = self.query_site_pos_and_quat(["end_effector"])
            ```
        """
        for i, (site_name, site) in enumerate(site_dict.items()):
            site["SiteId"] = i
        self._site_dict = site_dict.copy()
        if self.PRINT_INIT_INFO:
            if self.PRINT_FORMATTED_INFO:
                formatted_dict = json.dumps(site_dict, indent=4)
                _logger.debug(f"Site dict: {formatted_dict}")
            else:
                _logger.debug(f"Site dict: {site_dict}")

    def get_site_dict(self):
        """获取所有 site 字典"""
        return self._site_dict
    
    def get_site(self, name_or_id: Union[str, int]):
        """根据名称或ID获取 site 信息"""
        if isinstance(name_or_id, str):
            site_name = name_or_id
            return self._site_dict[site_name]
        elif isinstance(name_or_id, int):
            site_id = name_or_id
            site_name = self.site_id2name(site_id)
            if site_name is not None:
                return self._site_dict[site_name]
        return None
    
    def site_name2id(self, site_name):
        """Site 名称转ID"""
        return self._site_dict[site_name]["SiteId"]
    
    def site_id2name(self, site_id):
        """Site ID转名称"""
        for site_name, site in self._site_dict.items():
            if site["SiteId"] == site_id:
                return site_name
        return None
        

    def init_sensor_dict(self, sensor_dict):
        """
        初始化传感器字典，识别传感器类型
        
        术语说明:
            - 传感器 (Sensor): 用于测量物理量的虚拟设备
            - 常见类型:
                - accelerometer: 加速度计，测量线性加速度
                - gyro: 陀螺仪，测量角速度
                - touch: 触觉传感器，测量接触力
                - velocimeter: 速度计，测量线性速度
                - framequat: 框架四元数，测量姿态
        
        使用示例:
            ```python
            # 传感器在模型加载时自动初始化
            # 可以通过以下方式查询:
            sensor_data = self.query_sensor_data(["imu_accelerometer", "imu_gyro"])
            ```
        """
        for i, (sensor_name, sensor) in enumerate(sensor_dict.items()):
            sensor["SensorId"] = i
            sensor_type = sensor["Type"]
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
            sensor["SensorTypeStr"] = sensor_type_str

        self._sensor_dict = sensor_dict.copy()
        if self.PRINT_INIT_INFO:
            if self.PRINT_FORMATTED_INFO:
                formatted_dict = json.dumps(sensor_dict, indent=4)
                _logger.debug(f"Sensor dict: {formatted_dict}")
            else:
                _logger.debug(f"Sensor dict: {sensor_dict}")

    def gen_sensor_dict(self):
        """获取所有传感器字典"""
        return self._sensor_dict
    
    def get_sensor(self, name_or_id: Union[str, int]):
        """根据名称或ID获取传感器信息"""
        if isinstance(name_or_id, str):
            sensor_name = name_or_id
            return self._sensor_dict[sensor_name]
        elif isinstance(name_or_id, int):
            sensor_id = name_or_id
            sensor_name = self.sensor_id2name(sensor_id)
            if sensor_name is not None:
                return self._sensor_dict[sensor_name]
        return None
    
    def sensor_name2id(self, sensor_name):
        """传感器名称转ID"""
        return self._sensor_dict[sensor_name]["SensorId"]
    
    def sensor_id2name(self, sensor_id):
        """传感器ID转名称"""
        for sensor_name, sensor in self._sensor_dict.items():
            if sensor["SensorId"] == sensor_id:
                return sensor_name
        return None

    # def stip_agent_name(self, org_name):
    #     for agent in self.agent_names:
    #         if org_name.startswith(agent + "_"):
    #             return agent, org_name[len(agent + "_"):]
    #     return "", org_name     # No agent name prefix found, use empty string

    # def update_actuator_info(self, actuator_info):
    #     for actuator in actuator_info:
    #         agent_name, actuator_name = self.stip_agent_name(actuator["ActuatorName"])
    #         agent = next((r for r in self.agents if r.name == agent_name), None)
    #         if agent is not None:
    #             agent.add_actuator(actuator_name, actuator["JointName"], actuator["GearRatio"])

    # def update_body_info(self, body_info_list):
    #     for body_info in body_info_list:
    #         agent_name, body_name = self.stip_agent_name(body_info["body_name"])
    #         agent = next((r for r in self.agents if r.name == agent_name), None)
    #         if agent is not None:
    #             agent.add_body(body_name, body_info["body_id"])

    # def update_joint_info(self, joint_info_list):
    #     for joint_info in joint_info_list:
    #         agent_name, joint_name = self.stip_agent_name(joint_info["joint_name"])
    #         agent = next((r for r in self.agents if r.name == agent_name), None)
    #         if agent is not None:
    #             agent.add_joint(joint_name, joint_info["joint_id"], joint_info["joint_body_id"], joint_info["joint_type"])

    # def agent(self, name):
    #     agent = next((r for r in self.agents if r.name == name), None)
    #     if (agent is None):
    #         raise ValueError("Agent " + name + " not found")
    #     return agent
