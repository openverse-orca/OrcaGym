import asyncio
import random
from PySide6 import QtCore, QtWidgets, QtGui
from orca_gym.orca_lab.path import Path
from orca_gym.orca_lab.scene import OrcaLabScene, AssetActor
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
        self.edit_grpc_addr = "localhost:50151"
        self.sim_grpc_addr = "localhost:50051"
        self.scene = OrcaLabScene(self.edit_grpc_addr, self.sim_grpc_addr)
        self._sim_process_check_lock = asyncio.Lock()
        self.sim_process_running = False

        await self.scene.init_grpc()
        await self.scene.set_sync_from_mujoco_to_scene(False)
        await self.scene.set_selection([])
        await self.scene.clear_scene()

        # # Add a actor, for testing.
        # rot = Rotation.from_euler("xyz", [90, 45, 30], degrees=True)
        # q = rot.as_quat()  # x,y,z,w

        transform = Transform()
        transform.position = Vec3(0, 0, 2)

        actor = AssetActor(name=f"box1", spawnable_name="box")
        actor.transform = transform

        await self.scene.add_actor(actor, Path.root_path())

        self.asset_map_counter = {}
        self.asset_map_counter[actor.spawnable_name] = 2

        await self._init_ui()

        self.resize(400, 200)
        self.show()

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
        self.actor_outline_model.set_root_group(self.scene.root_actor)
        self.actor_outline.actor_selection_changed.connect(
            lambda actors: asyncio.ensure_future(self.set_scene_selection(actors))
        )
        self.scene.selection_changed.connect(
            lambda actor_paths: asyncio.ensure_future(
                self.set_actor_outline_selection(actor_paths)
            )
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

        await self.scene.add_actor(actor, Path.root_path())

        self.asset_map_counter[item_name] = self.asset_map_counter[item_name] + 1
        print(f"{item_name} added to the scene!")

    async def set_actor_outline_selection(self, actor_paths: list[Path]):
        actors = []
        for path in actor_paths:
            actor = self.scene.find_actor_by_path(path)
            if actor is None:
                raise Exception(f"Actor doest not exist at path: {path}")

            actors.append(actor)

        self.actor_outline.set_actor_selection(actors)

    async def set_scene_selection(self, actors: list[AssetActor]):
        actor_paths = []
        for actor in actors:
            path = self.scene.get_actor_path(actor)
            if path is None:
                raise Exception(f"Invalid actor: {actor}")

            actor_paths.append(path)
        await self.scene.set_selection(actor_paths)


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
