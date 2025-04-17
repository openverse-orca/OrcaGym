from __future__ import print_function

import logging

import grpc
import realenv_pb2
import realenv_pb2_grpc


def run():
    # NOTE(gRPC Python Team): .close() is possible on a channel and should be
    # used in circumstances in which the with statement does not fit the needs
    # of the code.
        # Create a fake observation
    fake_observation = realenv_pb2.Observation(
        tick=12345,
        imu=realenv_pb2.Observation.ImuData(
            angle_roll=1.0,
            angle_pitch=2.0,
            angle_yaw=3.0,
            angular_velocity_roll=0.1,
            angular_velocity_pitch=0.2,
            angular_velocity_yaw=0.3,
            acc_x=9.8,
            acc_y=0.0,
            acc_z=-9.8,
        ),
        joint_data=realenv_pb2.Observation.LegData(
            fl_leg=[
                realenv_pb2.Observation.JointData(position=0.1, velocity=0.2, torque=0.3, temperature=40.0),
                realenv_pb2.Observation.JointData(position=0.4, velocity=0.5, torque=0.6, temperature=41.0),
                realenv_pb2.Observation.JointData(position=0.7, velocity=0.8, torque=0.9, temperature=42.0),
            ],
            fr_leg=[
                realenv_pb2.Observation.JointData(position=0.1, velocity=0.2, torque=0.3, temperature=40.0),
                realenv_pb2.Observation.JointData(position=0.4, velocity=0.5, torque=0.6, temperature=41.0),
                realenv_pb2.Observation.JointData(position=0.7, velocity=0.8, torque=0.9, temperature=42.0),
            ],
            hl_leg=[
                realenv_pb2.Observation.JointData(position=0.1, velocity=0.2, torque=0.3, temperature=40.0),
                realenv_pb2.Observation.JointData(position=0.4, velocity=0.5, torque=0.6, temperature=41.0),
                realenv_pb2.Observation.JointData(position=0.7, velocity=0.8, torque=0.9, temperature=42.0),
            ],
            hr_leg=[
                realenv_pb2.Observation.JointData(position=0.1, velocity=0.2, torque=0.3, temperature=40.0),
                realenv_pb2.Observation.JointData(position=0.4, velocity=0.5, torque=0.6, temperature=41.0),
                realenv_pb2.Observation.JointData(position=0.7, velocity=0.8, torque=0.9, temperature=42.0),
            ],
        ),
        contact_force=realenv_pb2.Observation.ContactForce(
            fl_leg=[1.0, 2.0, 3.0],
            fr_leg=[4.0, 5.0, 6.0],
            hl_leg=[7.0, 8.0, 9.0],
            hr_leg=[10.0, 11.0, 12.0],
        ),
    )
    # Send the fake observation to the server
    with grpc.insecure_channel("localhost:50051") as channel:
        stub = realenv_pb2_grpc.PolicyServiceStub(channel)
        response = stub.GetAction(fake_observation)
    print("Greeter client received: " + str(response))


if __name__ == "__main__":
    logging.basicConfig()
    run()