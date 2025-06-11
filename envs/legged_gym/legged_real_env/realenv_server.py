from concurrent import futures
import logging

import grpc
import realenv_pb2
import realenv_pb2_grpc

from google.protobuf.json_format import MessageToDict

from legged_real_env import Lite3RealAgent

lite3_real_agent = Lite3RealAgent()

def create_proto_Action(fl_positions, fr_positions, hl_positions, hr_positions, 
                 kp=45.0, kd=0.7, velocities=None, torques=None):
    """
    Create an Action protobuf message from leg positions.
    
    Args:
        fl_positions (list): Front left leg joint positions [hip, thigh, calf] in degrees
        fr_positions (list): Front right leg joint positions [hip, thigh, calf] in degrees
        hl_positions (list): Hind left leg joint positions [hip, thigh, calf] in degrees
        hr_positions (list): Hind right leg joint positions [hip, thigh, calf] in degrees
        kp (float): Position gain for all joints (default: 45.0)
        kd (float): Velocity gain for all joints (default: 0.7)
        velocities (list of lists): Optional velocities for each joint. If None, sets to 0
        torques (list of lists): Optional torques for each joint. If None, sets to 0
    
    Returns:
        realenv_pb2.Action: The protobuf Action message
    """

    def create_joint_cmd(position, velocity=0.0, torque=0.0):
        joint = realenv_pb2.Action.JointCmd()
        joint.position = position
        joint.velocity = velocity
        joint.torque = torque
        joint.kp = kp
        joint.kd = kd
        return joint

    # Create the action message
    action = realenv_pb2.Action()

    # Create robot command
    robot_cmd = realenv_pb2.Action.RobotCmd()
    
    # Helper function to process each leg
    def process_leg(positions, velocities=None, torques=None):
        leg_cmds = []
        for i in range(3):  # 3 joints per leg
            vel = velocities[i] if velocities is not None else 0.0
            torq = torques[i] if torques is not None else 0.0
            leg_cmds.append(create_joint_cmd(positions[i], vel, torq))
        return leg_cmds

    # Process each leg
    robot_cmd.fl_leg.extend(process_leg(fl_positions))
    robot_cmd.fr_leg.extend(process_leg(fr_positions))
    robot_cmd.hl_leg.extend(process_leg(hl_positions))
    robot_cmd.hr_leg.extend(process_leg(hr_positions))
    
    # Set the robot command
    action.robot_cmd.CopyFrom(robot_cmd)
    return action


class PolicyService(realenv_pb2_grpc.PolicyServiceServicer):
    def GetAction(self, request, context):
        d = MessageToDict(request)
        # print(d)
        action = lite3_real_agent.get_action(d)
        print(action)
        return create_proto_Action(
            fl_positions=action[0:3], 
            fr_positions=action[3:6], 
            hl_positions=action[6:9], 
            hr_positions=action[9:12]
        )

def serve():
    port = "50051"
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    realenv_pb2_grpc.add_PolicyServiceServicer_to_server(PolicyService(), server)
    server.add_insecure_port("[::]:" + port)
    server.start()
    print("Server started, listening on " + port)
    server.wait_for_termination()


if __name__ == "__main__":
    logging.basicConfig()
    serve()
