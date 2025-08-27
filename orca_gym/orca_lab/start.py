import asyncio
import random
from PySide6 import QtCore, QtWidgets, QtGui
from orca_gym.orca_lab.path import Path
from orca_gym.orca_lab.scene import OrcaLabScene, AssetActor
import numpy as np
from scipy.spatial.transform import Rotation
import subprocess

from orca_gym.orca_lab.ui.actor_outline import ActorOutline
from orca_gym.orca_lab.ui.asset_browser import AssetBrowser
from orca_gym.orca_lab.math import Transform, Vec3

import PySide6.QtAsyncio as QtAsyncio


async def empty_task():
    pass


class UIWindow(QtWidgets.QWidget):
    def __init__(self, parent=None, add_item_to_scene_callback=None):
        super().__init__(parent)
        self.add_item_to_scene = add_item_to_scene_callback
        self.layout = QtWidgets.QHBoxLayout()
        self.setLayout(self.layout)

        self.asset_list = AssetBrowser()
        self.tool_bar = ToolBar()

        self.layout.addWidget(self.asset_list)
        self.layout.addWidget(self.tool_bar)

        self.resize(400, 200)


class ToolBar(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.layout = QtWidgets.QHBoxLayout()
        self.setLayout(self.layout)

        self.run_button = QtWidgets.QPushButton("Run Sim")
        self.stop_button = QtWidgets.QPushButton("Stop Sim")

        self.layout.addWidget(self.run_button)
        self.layout.addWidget(self.stop_button)

        self.resize(400, 200)


class App:

    def __init__(self):

        self.sim_process_running = False

        self._init_ui()
        self.asset_map_counter = {}

    async def _init_scene(self):
        self.edit_grpc_addr = "localhost:50151"
        self.sim_grpc_addr = "localhost:50051"
        self.scene = OrcaLabScene(self.edit_grpc_addr, self.sim_grpc_addr)
        self._sim_process_check_lock = asyncio.Lock()

        await self.scene.init_grpc()
        await self.scene.set_sync_from_mujoco_to_scene(False)
        await self.scene.clear_scene()

        # # Add a actor, for testing.
        # rot = Rotation.from_euler("xyz", [90, 45, 30], degrees=True)
        # q = rot.as_quat()  # x,y,z,w

        transform = Transform()
        transform.position = Vec3(0, 0, 2)

        actor = AssetActor(name=f"box1", spawnable_name="box")
        actor.transform = transform

        await self.scene.add_actor(actor, Path.root_path())

        assets = await self.scene.get_actor_assets()
        self.myUIWindow.asset_list.set_assets(assets)
        self.asset_map_counter[actor.spawnable_name] = 2
        #self.myUIWindow.add_item_to_scene


    def _init_ui(self):
        self.q_app = QtWidgets.QApplication([])

        self.myUIWindow = UIWindow(add_item_to_scene_callback=self.add_item_to_scene)
        
        # self.actor_outline = ActorOutline()
        # self.asset_browser = AssetBrowser()
        # self.actor_outline.show()
        # self.asset_browser.show()
        '''
        self.tool_bar = ToolBar()
        self.tool_bar.run_button.clicked.connect(
            lambda: asyncio.ensure_future(self.run_sim())
        )
        self.tool_bar.stop_button.clicked.connect(
            lambda: asyncio.ensure_future(self.stop_sim())
        )
        # self.tool_bar.show()
        layout.addWidget(self.tool_bar)
        layout.addWidget(self.asset_browser)
        '''
        self.myUIWindow.tool_bar.run_button.clicked.connect(
            lambda: asyncio.ensure_future(self.run_sim())
        )
        self.myUIWindow.tool_bar.stop_button.clicked.connect(
            lambda: asyncio.ensure_future(self.stop_sim())
        )
        self.myUIWindow.asset_list.add_item.connect(
            lambda item_name: asyncio.ensure_future( self.add_item_to_scene(item_name))
        )
        
        self.myUIWindow.show()

    def exec(self) -> int:

        # asyncio.run_forever()
        # code = self.q_app.exe()

        # 在这之后，Qt的event_loop变成asyncio的event_loop。
        # 这是目前统一Qt和asyncio最好的方法。
        # 所以不要保存loop，统一使用asyncio.xxx()。
        # https://doc.qt.io/qtforpython-6/PySide6/QtAsyncio/index.html
        QtAsyncio.run(self._init_scene())
        self.scene.destroy_grpc()

        return 0

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
        asyncio.sleep(1 / frequency)
        asyncio.create_task(self._sim_process_check_loop())

    async def add_item_to_scene(self, item_name):
        print(f"Adding {item_name} to the scene...")

        index = self.asset_map_counter[item_name]
        transform = Transform()
        transform.position = Vec3(1*index, 0, 2)
        new_item_name = item_name+str(index)
        actor = AssetActor(name=new_item_name, spawnable_name=item_name)
        actor.transform = transform
        
        await self.scene.add_actor(actor, Path.root_path())
        
        self.asset_map_counter[item_name] = self.asset_map_counter[item_name] + 1
        print(f"{item_name} added to the scene!")
        


if __name__ == "__main__":

    app = App()

    code = app.exec()

    # magic!
    # AttributeError: 'NoneType' object has no attribute 'POLLER'
    # https://github.com/google-gemini/deprecated-generative-ai-python/issues/207#issuecomment-2601058191
    exit(code)
