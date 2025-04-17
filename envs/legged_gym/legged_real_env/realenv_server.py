from concurrent import futures
import logging

import grpc
import realenv_pb2
import realenv_pb2_grpc

from google.protobuf.json_format import MessageToDict

from legged_real_env import Lite3RealAgent

lite3_real_agent = Lite3RealAgent()


class PolicyService(realenv_pb2_grpc.PolicyServiceServicer):
    def GetAction(self, request, context):
        d = MessageToDict(request)
        action = lite3_real_agent.get_action(d)
        print(action)
        return realenv_pb2.Action(action)

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
