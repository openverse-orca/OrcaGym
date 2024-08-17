import sys
import os
import grpc
from enum import Enum



class Body:
    def __init__(self, name, id):
        self.name = name
        self.id = id
    
class Joint:
    # 定义关节类型常量
    mjJNT_FREE = 0
    mjJNT_BALL = 1
    mjJNT_SLIDE = 2
    mjJNT_HINGE = 3

    def __init__(self, name, id, body_id, type):
        self.name = name
        self.id = id
        self.body_id = body_id
        self.type = type

    # 获取 joint 的 qpos size
    def qpos_size(joint_type):
        if joint_type == Joint.mjJNT_FREE:
            return 7
        elif joint_type == Joint.mjJNT_BALL:
            return 4
        elif joint_type in (Joint.mjJNT_SLIDE, Joint.mjJNT_HINGE):
            return 1
        else:
            return 0

    # 获取 joint 的 qvel size
    def qvel_size(joint_type):
        if joint_type == Joint.mjJNT_FREE:
            return 6
        elif joint_type == Joint.mjJNT_BALL:
            return 3
        elif joint_type in (Joint.mjJNT_SLIDE, Joint.mjJNT_HINGE):
            return 1
        else:
            return 0

    # 获取 joint 的 dof size
    def dof_size(joint_type):
        if joint_type == Joint.mjJNT_FREE:
            return 6  # 一个自由关节有6个DOF
        elif joint_type == Joint.mjJNT_BALL:
            return 3  # 一个球关节有3个DOF
        elif joint_type == Joint.mjJNT_SLIDE:
            return 1  # 一个滑动关节有1个DOF
        elif joint_type == Joint.mjJNT_HINGE:
            return 1  # 一个铰链关节有1个DOF
        else:
            return 0  # 其他类型的关节可能没有DOF

class Actuator:
    def __init__(self, name, joint_name, gear_ratio):
        self.name = name
        self.joint_name = joint_name
        self.gear_ratio = gear_ratio


class Agent:
    def __init__(self, name):
        self.name = name
        self.body_names = []    # 保证顺序
        self.joint_names = []
        self.bodies = {}
        self.joints = {}
        self.actuators = {}

    def add_body(self, body_name, body_id):
        self.body_names.append(body_name)
        body = Body(body_name, body_id)
        self.bodies[body.name] = body
        
    def add_joint(self, joint_name, joint_id, joint_body_id, joint_type):
        self.joint_names.append(joint_name)
        joint = Joint(joint_name, joint_id, joint_body_id, joint_type)
        self.joints[joint.name] = joint

    def add_actuator(self, actuator_name, joint_name, gear_ratio):
        actuator = Actuator(actuator_name, joint_name, gear_ratio)
        self.actuators[actuator.name] = actuator

    def body(self, body_name):
        body = self.bodies.get(body_name, None)
        if (body is None):
            raise ValueError("Body " + body_name + " not found")
        return body
    
    def joint(self, joint_name):
        joint = self.joints.get(joint_name, None)
        if (joint is None):
            raise ValueError("Joint " + joint_name + " not found")
        return joint
    
    def actuator(self, actuator_name):
        actuator = self.actuators.get(actuator_name, None)
        if (actuator is None):
            raise ValueError("Actuator " + actuator_name + " not found")
        return actuator
            