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
        self._eq_list = eq_list
        if self.PRINT_INIT_INFO:
            if self.PRINT_FORMATTED_INFO:
                formatted_dict = json.dumps(eq_list, indent=4)
                _logger.debug(f"Equality constraints: {formatted_dict}")
            else:
                _logger.debug(f"Equality constraints: {eq_list}")

        self.neq = len(eq_list)

    def get_eq_list(self):
        return self._eq_list
    
    def init_mocap_dict(self, mocap_dict):
        self._mocap_dict = mocap_dict
        if self.PRINT_INIT_INFO:
            if self.PRINT_FORMATTED_INFO:
                formatted_dict = json.dumps(mocap_dict, indent=4)
                _logger.debug(f"Mocap dict: {formatted_dict}")
            else:
                _logger.debug(f"Mocap dict: {mocap_dict}")

        self.nmocap = len(mocap_dict)

    def init_actuator_dict(self, actuator_dict):
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
        return self._actuator_dict
    
    def get_actuator_byid(self, id : int):
        actuator_name = self._actuaton_id2name_map[id]
        return self._actuator_dict[actuator_name]
    
    def get_actuator_byname(self, name : str):
        return self._actuator_dict[name]
    
    def actuator_name2id(self, actuator_name):
        return self._actuator_dict[actuator_name]["ActuatorId"]
    
    def actuator_id2name(self, actuator_id):
        return self._actuaton_id2name_map[actuator_id]
    
    def init_body_dict(self, body_dict):
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
        return self._body_dict
    
    def get_body_byid(self, id : int):
        body_name = self._body_id2name_map[id]
        return self._body_dict[body_name]
    
    def get_body_byname(self, name : str):
        return self._body_dict[name]
    
    def body_name2id(self, body_name):
        return self._body_dict[body_name]["BodyId"]
    
    def body_id2name(self, body_id):
        return self._body_id2name_map[body_id]

    def init_joint_dict(self, joint_dict):
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
        return self._joint_dict
    
    def get_joint_byid(self, id : int):
        joint_name = self._joint_id2name_map[id]
        return self._joint_dict[joint_name]
    
    def get_joint_byname(self, name : str):
        return self._joint_dict[name]
    
    def joint_name2id(self, joint_name):
        return self._joint_dict[joint_name]["JointId"]
    
    def joint_id2name(self, joint_id):
        return self._joint_id2name_map[joint_id]

    def init_geom_dict(self, geom_dict):
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
        return self._geom_dict
    
    def get_geom_byid(self, id : int):
        geom_name = self._geom_id2name_map[id]
        return self._geom_dict[geom_name]
    
    def get_geom_byname(self, name : str):
        return self._geom_dict[name]
    
    def geom_name2id(self, geom_name):
        return self._geom_dict[geom_name]["GeomId"]
    
    def geom_id2name(self, geom_id):
        return self._geom_id2name_map[geom_id]

    def get_body_names(self):
        return self._body_dict.keys()
    
    def get_geom_body_name(self, geom_id):
        geom_name = self.geom_id2name(geom_id)
        return self._geom_dict[geom_name]["BodyName"]
    
    def get_geom_body_id(self, geom_id):
        body_name = self.get_geom_body_name(geom_id)
        return self.body_name2id(body_name)
    
    def get_actuator_ctrlrange(self):
        actuator_ctrlrange = {}
        for actuator_name, actuator in self._actuator_dict.items():
            actuator_ctrlrange[actuator_name] = actuator["CtrlRange"]
        ctrlrange = np.array(list(actuator_ctrlrange.values()))
        return ctrlrange
    
    def get_joint_qposrange(self, joint_names : list):
        joint_range = {}
        for joint_name in joint_names:
            joint = self.get_joint_byname(joint_name)
            joint_range[joint_name] = joint["Range"]
        qposrange = np.array(list(joint_range.values()))
        return qposrange
    
    def init_site_dict(self, site_dict):
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
        return self._site_dict
    
    def get_site(self, name_or_id: Union[str, int]):
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
        return self._site_dict[site_name]["SiteId"]
    
    def site_id2name(self, site_id):
        for site_name, site in self._site_dict.items():
            if site["SiteId"] == site_id:
                return site_name
        return None
        

    def init_sensor_dict(self, sensor_dict):
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
        return self._sensor_dict
    
    def get_sensor(self, name_or_id: Union[str, int]):
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
        return self._sensor_dict[sensor_name]["SensorId"]
    
    def sensor_id2name(self, sensor_id):
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
