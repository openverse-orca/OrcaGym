import grpc
from orca_gym.orca_lab.math import Transform
import orca_gym.orca_lab.protos.edit_service_pb2_grpc as edit_service_pb2_grpc
import orca_gym.orca_lab.protos.edit_service_pb2 as edit_service_pb2
import orca_gym.protos.mjc_message_pb2_grpc as mjc_message_pb2_grpc
import orca_gym.protos.mjc_message_pb2 as mjc_message_pb2

from orca_gym.orca_lab.path import Path
from orca_gym.orca_lab.actor import BaseActor, GroupActor, AssetActor

from typing import List


Success = edit_service_pb2.StatusCode.Success
Error = edit_service_pb2.StatusCode.Error


# 由于Qt是异步的，所以这里只提供异步接口。
class RemoteScene:
    def __init__(self, edit_grpc_addr: str, sim_grpc_addr: str):
        super().__init__()

        self.edit_grpc_addr = edit_grpc_addr
        self.sim_grpc_addr = sim_grpc_addr

    async def init_grpc(self):
        options = [
            ("grpc.max_receive_message_length", 1024 * 1024 * 1024),
            ("grpc.max_send_message_length", 1024 * 1024 * 1024),
        ]
        self.edit_channel = grpc.aio.insecure_channel(
            self.edit_grpc_addr,
            options=options,
        )
        self.sim_channel = grpc.aio.insecure_channel(
            self.sim_grpc_addr,
            options=options,
        )
        self.edit_stub = edit_service_pb2_grpc.GrpcServiceStub(self.edit_channel)
        self.sim_stub = mjc_message_pb2_grpc.GrpcServiceStub(self.sim_channel)

        self.timeout = 3

        if not await self.aloha():
            raise Exception("Failed to connect to server.")

    async def destroy_grpc(self):
        await self._stop_query_pending_operation_loop()

        if self.edit_channel:
            await self.edit_channel.close()
        self.edit_stub = None
        self.edit_channel = None

        if self.sim_channel:
            await self.sim_channel.close()
        self.sim_stub = None
        self.sim_channel = None

    def _create_transform_message(self, transform: Transform):
        msg = edit_service_pb2.Transform(
            pos=transform.position,
            quat=transform.rotation,
            scale=transform.scale,
        )
        return msg

    def _get_transform_from_message(self, msg) -> Transform:
        transform = Transform()
        transform.position = msg.pos
        transform.rotation = msg.quat
        transform.scale = msg.scale
        return transform

    def _check_response(self, response):
        if response.status_code != Success:
            raise Exception(f"Request failed. {response.error_message}")

    async def aloha(self) -> bool:
        request = edit_service_pb2.AlohaRequest(value=1)
        response = await self.edit_stub.Aloha(request)
        self._check_response(response)
        return response.value == 2

    async def query_pending_operation_loop(self) -> List[str]:
        request = edit_service_pb2.GetPendingOperationsRequest()
        response = await self.edit_stub.GetPendingOperations(request)
        self._check_response(response)
        return response.operations

    async def get_pending_actor_transform(self, path: Path, local: bool) -> Transform:
        space = edit_service_pb2.Space.Local if local else edit_service_pb2.Space.World

        request = edit_service_pb2.GetPendingActorTransformRequest(
            actor_path=path.string(),
            space=space,
        )

        response = await self.edit_stub.GetPendingActorTransform(request)
        self._check_response(response)

        transform = response.transform
        return self._get_transform_from_message(transform)

    async def add_actor(self, actor: BaseActor, parent_path: Path):
        transform_msg = self._create_transform_message(actor.transform)

        if isinstance(actor, GroupActor):
            request = edit_service_pb2.AddGroupRequest(
                actor_name=actor.name,
                parent_actor_path=parent_path.string(),
                transform=transform_msg,
                space=edit_service_pb2.Space.Local,
            )
            response = await self.edit_stub.AddGroup(request)
        elif isinstance(actor, AssetActor):
            request = edit_service_pb2.AddActorRequest(
                actor_name=actor.name,
                spawnable_name=actor.spawnable_name,
                parent_actor_path=parent_path.string(),
                transform=transform_msg,
                space=edit_service_pb2.Space.Local,
            )
            response = await self.edit_stub.AddActor(request)
        else:
            raise Exception(f"Unsupported actor type: {type(actor)}")

        self._check_response(response)

    async def set_actor_transform(self, path: Path, transform: Transform, local: bool):
        transform_msg = self._create_transform_message(transform)
        space = edit_service_pb2.Space.Local if local else edit_service_pb2.Space.World

        request = edit_service_pb2.SetActorTransformRequest(
            actor_path=path.string(),
            transform=transform_msg,
            space=space,
        )

        response = await self.edit_stub.SetActorTransform(request, timeout=self.timeout)
        self._check_response(response)

    async def publish_scene(self):
        print(f"publish_scene")
        request = mjc_message_pb2.PublishSceneRequest()
        response = await self.sim_stub.PublishScene(request)
        if response.status != mjc_message_pb2.PublishSceneResponse.SUCCESS:
            print("Publish scene failed: ", response.error_message)
            raise Exception("Publish scene failed.")
        print("done")

    async def forward_scene(self):
        print(f"forward_scene")
        request = mjc_message_pb2.MJ_ForwardRequest()
        response = await self.sim_stub.MJ_Forward(request)
        print("done")

    async def get_sync_from_mujoco_to_scene(self) -> bool:
        request = edit_service_pb2.GetSyncFromMujocoToSceneRequest()
        response = await self.edit_stub.GetSyncFromMujocoToScene(request)
        self._check_response(response)
        return response.value

    async def set_sync_from_mujoco_to_scene(self, value: bool):
        print(f"set_sync_from_mujoco_to_scene {value}")
        request = edit_service_pb2.SetSyncFromMujocoToSceneRequest(value=value)
        response = await self.edit_stub.SetSyncFromMujocoToScene(request)
        self._check_response(response)
        print("done")

    async def clear_scene(self):
        request = edit_service_pb2.ClearSceneRequest()
        response = await self.edit_stub.ClearScene(request)
        self._check_response(response)

    async def get_pending_selection_change(self) -> list[str]:
        request = edit_service_pb2.GetPendingSelectionChangeRequest()
        response = await self.edit_stub.GetPendingSelectionChange(request)
        self._check_response(response)
        return response.actor_paths

    async def set_selection(self, actor_paths: list[Path]):
        paths = []
        for p in actor_paths:
            if not isinstance(p, Path):
                raise Exception(f"Invalid path: {p}")
            paths.append(p.string())

        request = edit_service_pb2.SetSelectionRequest(actor_paths=paths)
        response = await self.edit_stub.SetSelection(request)
        self._check_response(response)

    async def get_actor_assets(self):
        request = edit_service_pb2.GetActorAssetsRequest()
        response = await self.edit_stub.GetActorAssets(request)
        self._check_response(response)
        return response.actor_asset_names

    async def save_body_transform(self):
        request = edit_service_pb2.SaveBodyTransformRequest()
        response = await self.edit_stub.SaveBodyTransform(request)
        self._check_response(response)

    async def restore_body_transform(self):
        request = edit_service_pb2.RestoreBodyTransformRequest()
        response = await self.edit_stub.RestoreBodyTransform(request)
        self._check_response(response)

    async def delete_actor(self, actor_path: Path):
        request = edit_service_pb2.DeleteActorRequest(actor_path=actor_path.string())
        response = await self.edit_stub.DeleteActor(request)
        self._check_response(response)

    async def rename_actor(self, actor_path: Path, new_name: str):
        request = edit_service_pb2.RenameActorRequest(
            actor_path=actor_path.string(),
            new_name=new_name,
        )
        response = await self.edit_stub.RenameActor(request)
        self._check_response(response)

    async def reparent_actor(self, actor_path: Path, new_parent_path: Path):
        request = edit_service_pb2.ReParentActorRequest(
            actor_path=actor_path.string(),
            new_parent_path=new_parent_path.string(),
        )
        response = await self.edit_stub.ReParentActor(request)
        self._check_response(response)
