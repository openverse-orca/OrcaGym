import sys
import os
import grpc


class OrcaGymModel:
    mjEQ_CONNECT = 0       # connect two bodies at a point (ball joint)
    mjEQ_WELD = 1          # fix relative position and orientation of two bodies
    mjEQ_JOINT = 2         # couple the values of two scalar joints with cubic
    mjEQ_TENDON = 3        # couple the lengths of two tendons with cubic
    mjEQ_FLEX = 4          # fix all edge lengths of a flex
    mjEQ_DISTANCE = 5      # unsupported, will cause an error if used

    def __init__(self, model_info):


        self.init_model_info(model_info)                

        self._eq_list = None
        self._mocap_dict = None
        self._actuator_dict = None
        self._actuator_ctrlrange = None
        self._body_dict = None
        self._joint_dict = None
        
    def init_model_info(self, model_info):
        self.model_info = model_info
        print("Model info: ", model_info)

        self.nq = model_info["nq"]
        self.nv = model_info["nv"]
        self.nu = model_info["nu"]
    
    def init_eq_list(self, eq_list):
        self._eq_list = eq_list
        print("Equality constraints: ", eq_list)

        self.neq = len(eq_list)

    def init_mocap_dict(self, mocap_dict):
        self._mocap_dict = mocap_dict
        print("Mocap dict: ", mocap_dict)

        self.nmocap = len(mocap_dict)

    def init_actuator_dict(self, actuator_dict):
        self._actuator_dict = actuator_dict
        print("Actuator dict: ", actuator_dict)

    def init_actuator_ctrlrange(self, actuator_ctrlrange):
        self._actuator_ctrlrange = actuator_ctrlrange
        print("Actuator control range: ", actuator_ctrlrange)

    def init_body_dict(self, body_dict):
        self._body_dict = body_dict
        print("Body dict: ", body_dict)

    def init_joint_dict(self, joint_dict):
        self._joint_dict = joint_dict
        print("Joint dict: ", joint_dict)


    def get_eq_list(self):
        return self._eq_list
    

    def get_body_names(self):
        return self._body_dict.keys()
    
    def get_actuator_ctrlrange(self):
        return self._actuator_ctrlrange

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
