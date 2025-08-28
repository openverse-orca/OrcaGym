import asyncio
import random
from typing import Dict
from PySide6 import QtCore, QtWidgets, QtGui
from orca_gym.orca_lab.actor import AssetActor, BaseActor, GroupActor
from orca_gym.orca_lab.path import Path
from orca_gym.orca_lab.scene import OrcaLabScene
import numpy as np
from scipy.spatial.transform import Rotation
import subprocess

from orca_gym.orca_lab.ui.actor_outline import ActorOutline
from orca_gym.orca_lab.ui.actor_outline_model import ActorOutlineModel
from orca_gym.orca_lab.ui.asset_browser import AssetBrowser
from orca_gym.orca_lab.ui.tool_bar import ToolBar
from orca_gym.orca_lab.math import Transform, Vec3

import PySide6.QtAsyncio as QtAsyncio


class MainWindow(QtWidgets.QWidget):

    def __init__(self):
        super().__init__()

    async def init(self):
        # 作为根节点，不可见， 路径是"/"。下面挂着所有的顶层Actor。
        self.root_actor = GroupActor(name="root", parent=None)
        self._actors: Dict[Path, BaseActor] = {}
        self._actors[Path.root_path()] = self.root_actor

        self.edit_grpc_addr = "localhost:50151"
        self.sim_grpc_addr = "localhost:50051"
        self.scene = OrcaLabScene(self.edit_grpc_addr, self.sim_grpc_addr)

        self._sim_process_check_lock = asyncio.Lock()
        self.sim_process_running = False

        await self.scene.init_grpc()
        await self.scene.set_sync_from_mujoco_to_scene(False)
        await self.scene.set_selection([])
        await self.scene.clear_scene()

        self._query_pending_operation_lock = asyncio.Lock()
        self._query_pending_operation_running = False
        await self._start_query_pending_operation_loop()

        await self._init_ui()

        self.resize(400, 200)
        self.show()

        # For testing ...

        # # Add a actor, for testing.
        # rot = Rotation.from_euler("xyz", [90, 45, 30], degrees=True)
        # q = rot.as_quat()  # x,y,z,w

        transform = Transform()
        transform.position = Vec3(0, 0, 2)

        actor = AssetActor(name=f"box1", spawnable_name="box")
        actor.transform = transform

        await self.add_actor(actor, Path.root_path())

        self.asset_map_counter = {}
        self.asset_map_counter[actor.spawnable_name] = 2

    async def _init_ui(self):

        self.tool_bar = ToolBar()
        self.tool_bar.action_start.triggered.connect(
            lambda: asyncio.ensure_future(self.run_sim())
        )
        self.tool_bar.action_stop.triggered.connect(
            lambda: asyncio.ensure_future(self.stop_sim())
        )

        self.actor_outline = ActorOutline()
        self.actor_outline_model = ActorOutlineModel()
        self.actor_outline.set_actor_model(self.actor_outline_model)
        self.actor_outline_model.set_root_group(self.root_actor)
        self.actor_outline.actor_selection_changed.connect(
            lambda actors: asyncio.ensure_future(self.set_scene_selection(actors))
        )
        self.actor_outline.request_delete.connect(
            lambda actor: asyncio.ensure_future(self.delete_actor(actor))
        )

        self.asset_browser = AssetBrowser()
        assets = await self.scene.get_actor_assets()
        self.asset_browser.set_assets(assets)
        self.asset_browser.add_item.connect(
            lambda item_name: asyncio.ensure_future(self.add_item_to_scene(item_name))
        )

        self.menu_bar = QtWidgets.QMenuBar()

        layout1 = QtWidgets.QHBoxLayout()
        layout1.addWidget(self.actor_outline)
        layout1.addWidget(self.asset_browser)
        layout1.addWidget(self.tool_bar)

        layout2 = QtWidgets.QVBoxLayout()
        layout2.addWidget(self.menu_bar)
        layout2.addWidget(self.tool_bar)
        layout2.addLayout(layout1)

        self.setLayout(layout2)

    async def _start_query_pending_operation_loop(self):
        async with self._query_pending_operation_lock:
            if self._query_pending_operation_running:
                return
            asyncio.create_task(self._query_pending_operation_loop())
            self._query_pending_operation_running = True

    async def _stop_query_pending_operation_loop(self):
        async with self._query_pending_operation_lock:
            self._query_pending_operation_running = False

    async def _query_pending_operation_loop(self):
        async with self._query_pending_operation_lock:
            if not self._query_pending_operation_running:
                return

            operations = await self.scene.query_pending_operation_loop()
            for op in operations:
                await self._process_pending_operation(op)

        # frequency = 30  # Hz
        # await asyncio.sleep(1 / frequency)
        asyncio.create_task(self._query_pending_operation_loop())

    async def _process_pending_operation(self, op: str):
        local_transform_change = "local_transform_change:"
        if op.startswith(local_transform_change):
            actor_path = Path(op[len(local_transform_change) :])

            if not actor_path in self._actors:
                raise Exception(f"actor not exist")

            transform = await self.scene.get_pending_actor_transform(actor_path, True)

            await self.scene.set_actor_transform(actor_path, transform, True)

            actor = self._actors[actor_path]
            actor.transform = transform

        world_transform_change = "world_transform_change:"
        if op.startswith(world_transform_change):
            actor_path = op[len(world_transform_change) :]

            if not actor_path in self._actors:
                raise Exception(f"actor not exist")

            transform = await self.scene.get_pending_actor_transform(actor_path, False)

            await self.scene.set_actor_transform(actor_path, transform, False)

            actor = self._actors[actor_path]
            actor.transform = transform

        selection_change = "selection_change"
        if op.startswith(selection_change):
            actor_paths = await self._get_pending_selection_change()

            paths = []
            for p in actor_paths:
                paths.append(Path(p))

            self.set_actor_outline_selection(paths)

    async def run_sim(self):
        if self.sim_process_running:
            return

        await self.scene.publish_scene()
        await self.scene.save_body_transform()

        cmd = [
            "python",
            "-m",
            "orca_gym.orca_lab.sim_process",
            "--sim_addr",
            self.sim_grpc_addr,
        ]
        self.sim_process = subprocess.Popen(cmd)
        self.sim_process_running = True
        asyncio.create_task(self._sim_process_check_loop())

        # await asyncio.sleep(2)
        await self.scene.set_sync_from_mujoco_to_scene(True)

    async def stop_sim(self):
        if not self.sim_process_running:
            return

        async with self._sim_process_check_lock:
            await self.scene.set_sync_from_mujoco_to_scene(False)
            self.sim_process_running = False
            self.sim_process.terminate()
            await self.scene.restore_body_transform()

    async def _sim_process_check_loop(self):
        async with self._sim_process_check_lock:
            if not self.sim_process_running:
                return

            code = self.sim_process.poll()
            if code is not None:
                print("Simulation process exit with {code}")
                self.sim_process_running = False
                # TODO notify ui.

        frequency = 0.5  # Hz
        await asyncio.sleep(1 / frequency)
        asyncio.create_task(self._sim_process_check_loop())

    async def add_item_to_scene(self, item_name):
        print(f"Adding {item_name} to the scene...")

        index = self.asset_map_counter[item_name]
        transform = Transform()
        transform.position = Vec3(1 * index, 0, 2)
        new_item_name = item_name + str(index)
        actor = AssetActor(name=new_item_name, spawnable_name=item_name)
        actor.transform = transform

        await self.add_actor(actor, Path.root_path())

        self.asset_map_counter[item_name] = self.asset_map_counter[item_name] + 1
        print(f"{item_name} added to the scene!")

    def find_actor_by_path(self, path: Path) -> BaseActor | None:
        if path in self._actors:
            return self._actors[path]
        return None

    def get_actor_path(self, actor: BaseActor) -> Path | None:
        for path, a in self._actors.items():
            if a == actor:
                return path
        return None

    async def set_actor_outline_selection(self, actor_paths: list[Path]):
        actors = []
        for path in actor_paths:
            actor = self.find_actor_by_path(path)
            if actor is None:
                raise Exception(f"Actor doest not exist at path: {path}")

            actors.append(actor)

        self.actor_outline.set_actor_selection(actors)

    async def set_scene_selection(self, actors: list[AssetActor]):
        actor_paths = []
        for actor in actors:
            path = self.get_actor_path(actor)
            if path is None:
                raise Exception(f"Invalid actor: {actor}")

            actor_paths.append(path)
        await self.scene.set_selection(actor_paths)

    async def add_actor(self, actor: BaseActor, parent_path: Path):
        parent_actor = self.find_actor_by_path(parent_path)
        if parent_actor is None:
            raise Exception(f"Invalid parent path: {parent_path}")

        await self.scene.add_actor(actor, parent_path)

        model = self.actor_outline_model
        parent_index = model.get_index_from_actor(parent_actor)
        child_count = len(parent_actor.children)

        model.beginInsertRows(parent_index, child_count, child_count)

        actor.parent = parent_actor

        path = parent_path / actor.name
        self._actors[path] = actor

        model.endInsertRows()

    async def delete_actor(self, actor):
        if actor is None:
            return

        actor_path = self.get_actor_path(actor)
        if actor_path is None:
            raise Exception("Invalid actor.")

        if actor_path == actor_path.root_path():
            raise Exception("Cannot delete root actor.")

        await self.scene.delete_actor(actor_path)

        model = self.actor_outline_model
        index = model.get_index_from_actor(actor)
        parent_index = index.parent()

        model.beginRemoveRows(parent_index, index.row(), index.row())

        actor.parent = None

        actors_to_delete = [actor_path]
        for p in self._actors.keys():
            if p.is_descendant_of(actor_path):
                actors_to_delete.append(p)

        for p in actors_to_delete:
            del self._actors[p]

        model.endRemoveRows()


if __name__ == "__main__":

    q_app = QtWidgets.QApplication([])

    main_window = MainWindow()

    # 在这之后，Qt的event_loop变成asyncio的event_loop。
    # 这是目前统一Qt和asyncio最好的方法。
    # 所以不要保存loop，统一使用asyncio.xxx()。
    # https://doc.qt.io/qtforpython-6/PySide6/QtAsyncio/index.html
    QtAsyncio.run(main_window.init())

    # magic!
    # AttributeError: 'NoneType' object has no attribute 'POLLER'
    # https://github.com/google-gemini/deprecated-generative-ai-python/issues/207#issuecomment-2601058191
    exit(0)
