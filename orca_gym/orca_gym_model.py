import sys
import os
import grpc
import numpy as np
import json
from datetime import datetime


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
                print("Model info: ", formatted_dict)
            else:
                print("Model info: ", model_info)

        self.nq = model_info["nq"]
        self.nv = model_info["nv"]
        self.nu = model_info["nu"]
        self.ngeom = model_info["ngeom"]
    
    def init_eq_list(self, eq_list):
        self._eq_list = eq_list
        if self.PRINT_INIT_INFO:
            if self.PRINT_FORMATTED_INFO:
                formatted_dict = json.dumps(eq_list, indent=4)
                print("Equality constraints: ", formatted_dict)
            else:
                print("Equality constraints: ", eq_list)

        self.neq = len(eq_list)

    def get_eq_list(self):
        return self._eq_list
    
    def init_mocap_dict(self, mocap_dict):
        self._mocap_dict = mocap_dict
        if self.PRINT_INIT_INFO:
            if self.PRINT_FORMATTED_INFO:
                formatted_dict = json.dumps(mocap_dict, indent=4)
                print("Mocap dict: ", formatted_dict)
            else:
                print("Mocap dict: ", mocap_dict)

        self.nmocap = len(mocap_dict)

    def init_actuator_dict(self, actuator_dict):
        for i, (actuator_name, actuator) in enumerate(actuator_dict.items()):
            actuator["ActuatorId"] = i
        self._actuator_dict = actuator_dict.copy()
        if self.PRINT_INIT_INFO:
            if self.PRINT_FORMATTED_INFO:
                formatted_dict = json.dumps(actuator_dict, indent=4)
                print("Actuator dict: ", formatted_dict)
            else:
                print("Actuator dict: ", actuator_dict)

    def get_actuator_dict(self):
        return self._actuator_dict
    
    def get_actuator(self, actuator_name:str):
        return self._actuator_dict[actuator_name]
    
    def get_actuator(self, actuator_id:int):
        actuator_name = self.actuator_id2name(actuator_id)
        if actuator_name is not None:
            return self._actuator_dict[actuator_name]
        return None
    
    def actuator_name2id(self, actuator_name):
        return self._actuator_dict[actuator_name]["ActuatorId"]
    
    def actuator_id2name(self, actuator_id):
        for actuator_name, actuator in self._actuator_dict.items():
            if actuator["ActuatorId"] == actuator_id:
                return actuator_name
        return None

    def init_body_dict(self, body_dict):
        for i, (body_name, body) in enumerate(body_dict.items()):
            body["BodyId"] = i
        self._body_dict = body_dict.copy()
        if self.PRINT_INIT_INFO:
            if self.PRINT_FORMATTED_INFO:
                formatted_dict = json.dumps(body_dict, indent=4)
                print("Body dict: ", formatted_dict)
            else:
                print("Body dict: ", body_dict)

    def get_body_dict(self):
        return self._body_dict
    
    def get_body(self, body_name:str):
        return self._body_dict[body_name]
    
    def get_body(self, body_id:int):
        body_name = self.body_id2name(body_id)
        if body_name is not None:
            return self._body_dict[body_name]
        return None
    
    def body_name2id(self, body_name):
        return self._body_dict[body_name]["BodyId"]
    
    def body_id2name(self, body_id):
        for body_name, body in self._body_dict.items():
            if body["BodyId"] == body_id:
                return body_name
        return None

    def init_joint_dict(self, joint_dict):
        for i, (joint_name, joint) in enumerate(joint_dict.items()):
            joint["JointId"] = i
        self._joint_dict = joint_dict.copy()
        if self.PRINT_INIT_INFO:
            if self.PRINT_FORMATTED_INFO:
                formatted_dict = json.dumps(joint_dict, indent=4)
                print("Joint dict: ", formatted_dict)
            else:
                print("Joint dict: ", joint_dict)

    def get_joint_dict(self):
        return self._joint_dict
    
    def get_joint(self, joint_name:str):
        return self._joint_dict[joint_name]
    
    def get_joint(self, joint_id:int):
        joint_name = self.joint_id2name(joint_id)
        if joint_name is not None:
            return self._joint_dict[joint_name]
        return None
    
    def joint_name2id(self, joint_name):
        return self._joint_dict[joint_name]["JointId"]
    
    def joint_id2name(self, joint_id):
        for joint_name, joint in self._joint_dict.items():
            if joint["JointId"] == joint_id:
                return joint_name
        return None

    def init_geom_dict(self, geom_dict):
        for i, (geom_name, geom) in enumerate(geom_dict.items()):
            geom["GeomId"] = i
        self._geom_dict = geom_dict.copy()
        if self.PRINT_INIT_INFO:
            if self.PRINT_FORMATTED_INFO:
                formatted_dict = json.dumps(geom_dict, indent=4)
                print("Geom dict: ", formatted_dict)
            else:
                print("Geom dict: ", geom_dict)

    def get_geom_dict(self):
        return self._geom_dict
    
    def get_geom(self, geom_name:str):
        return self._geom_dict[geom_name]
    
    def get_geom(self, geom_id:int):
        geom_name = self.geom_id2name(geom_id)
        if geom_name is not None:
            return self._geom_dict[geom_name]
        return None
    
    def geom_name2id(self, geom_name):
        return self._geom_dict[geom_name]["GeomId"]
    
    def geom_id2name(self, geom_id):
        for geom_name, geom in self._geom_dict.items():
            if geom["GeomId"] == geom_id:
                return geom_name
        return None

    def get_body_names(self):
        return self._body_dict.keys()
    
    def get_geom_bodyname(self, geom_name):
        return self._geom_dict[geom_name]["BodyName"]
    
    def get_geom_bodyid(self, geom_id):
        geom_name = self.geom_id2name(geom_id)
        body_name = self.get_geom_bodyname(geom_name)
        return self.body_name2id(body_name)
    
    def get_actuator_ctrlrange(self):
        actuator_ctrlrange = {}
        for actuator_name, actuator in self._actuator_dict.items():
            actuator_ctrlrange[actuator_name] = actuator["CtrlRange"]
        ctrlrange = np.array(list(actuator_ctrlrange.values()))
        return ctrlrange
    
    def init_site_dict(self, site_dict):
        for i, (site_name, site) in enumerate(site_dict.items()):
            site["SiteId"] = i
        self._site_dict = site_dict.copy()
        if self.PRINT_INIT_INFO:
            if self.PRINT_FORMATTED_INFO:
                formatted_dict = json.dumps(site_dict, indent=4)
                print("Site dict: ", formatted_dict)
            else:
                print("Site dict: ", site_dict)

    def get_site_dict(self):
        return self._site_dict
    
    def get_site(self, site_name:str):
        return self._site_dict[site_name]
    
    def get_site(self, site_id:int):
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
