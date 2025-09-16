import asyncio
from copy import deepcopy
import random

from typing import Dict, Tuple
from PySide6 import QtCore, QtWidgets, QtGui
from orca_gym.orca_lab import sim_process
from orca_gym.orca_lab.actor import AssetActor, BaseActor, GroupActor
from orca_gym.orca_lab.local_scene import LocalScene
from orca_gym.orca_lab.path import Path
from orca_gym.orca_lab.pyside_util import connect
from orca_gym.orca_lab.remote_scene import RemoteScene
import numpy as np
from scipy.spatial.transform import Rotation
import subprocess

from orca_gym.orca_lab.ui.actor_editor import ActorEditor
from orca_gym.orca_lab.ui.actor_outline import ActorOutline
from orca_gym.orca_lab.ui.rename_dialog import RenameDialog
from orca_gym.orca_lab.ui.actor_outline_model import ActorOutlineModel
from orca_gym.orca_lab.ui.asset_browser import AssetBrowser
from orca_gym.orca_lab.ui.tool_bar import ToolBar
from orca_gym.orca_lab.math import Transform

import PySide6.QtAsyncio as QtAsyncio

class MainWindow(QtWidgets.QWidget):

    def __init__(self):
        super().__init__()

    async def init(self):
        self.local_scene = LocalScene()

        self.edit_grpc_addr = "localhost:50151"
        self.sim_grpc_addr = "localhost:50051"
        self.remote_scene = RemoteScene(self.edit_grpc_addr, self.sim_grpc_addr)

        self._sim_process_check_lock = asyncio.Lock()
        self.sim_process_running = False

        await self.remote_scene.init_grpc()
        await self.remote_scene.set_sync_from_mujoco_to_scene(False)
        await self.remote_scene.set_selection([])
        await self.remote_scene.clear_scene()

        self._query_pending_operation_lock = asyncio.Lock()
        self._query_pending_operation_running = False
        await self._start_query_pending_operation_loop()

        respone = await self.remote_scene.get_window_id()
        self.hwnd = respone.window_id
        self.start_transform = None
        self.end_transform = None

    async def _init_ui(self):
        self.tool_bar = ToolBar()
        connect(self.tool_bar.action_start.triggered, self.run_sim)
        connect(self.tool_bar.action_stop.triggered, self.stop_sim)

        self.actor_outline_model = ActorOutlineModel(self.local_scene)
        self.actor_outline_model.set_root_group(self.local_scene.root_actor)

        self.actor_outline = ActorOutline()
        self.actor_outline.set_actor_model(self.actor_outline_model)

        self.actor_editor = ActorEditor()

        self.asset_browser = AssetBrowser(hwnd_target=self.hwnd)
        assets = await self.remote_scene.get_actor_assets()
        self.asset_browser.set_assets(assets)

        self.menu_bar = QtWidgets.QMenuBar()

        self.menu_file = self.menu_bar.addMenu("File")
        self.menu_edit = self.menu_bar.addMenu("Edit")

        layout1 = QtWidgets.QHBoxLayout()
        layout1.addWidget(self.actor_outline, 1)
        layout1.addWidget(self.actor_editor, 1)
        layout1.addWidget(self.asset_browser, 1)

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

            operations = await self.remote_scene.query_pending_operation_loop()
            for op in operations:
                await self._process_pending_operation(op)

        # frequency = 30  # Hz
        # await asyncio.sleep(1 / frequency)
        asyncio.create_task(self._query_pending_operation_loop())

    async def _process_pending_operation(self, op: str):
        sltc = "start_local_transform_change:"
        if op.startswith(sltc):
            actor_path = Path(op[len(sltc) :])

            if actor_path not in self.local_scene:
                raise Exception(f"actor not exist")
            
            actor = self.local_scene[actor_path]
            self.start_transform = actor.transform
        
        eltc = "end_local_transform_change:"
        if op.startswith(eltc):
            actor_path = Path(op[len(eltc) :])

            if actor_path not in self.local_scene:
                raise Exception(f"actor not exist")
            
            actor = self.local_scene[actor_path]
            self.end_transform = actor.transform
            self.transform_change.emit(actor_path, True)

        swtc = "start_world_transform_change:"
        if op.startswith(swtc):
            actor_path = Path(op[len(swtc) :])

            if actor_path not in self.local_scene:
                raise Exception(f"actor not exist")
            
            actor = self.local_scene[actor_path]
            self.start_transform = actor.world_transform
        
        ewtc = "end_world_transform_change:"
        if op.startswith(ewtc):
            actor_path = Path(op[len(ewtc) :])

            if actor_path not in self.local_scene:
                raise Exception(f"actor not exist")
            
            actor = self.local_scene[actor_path]
            self.end_transform = actor.world_transform
            self.transform_change.emit(actor_path, False)

        local_transform_change = "local_transform_change:"
        if op.startswith(local_transform_change):
            actor_path = Path(op[len(local_transform_change) :])

            if actor_path not in self.local_scene:
                raise Exception(f"actor not exist")

            transform = await self.remote_scene.get_pending_actor_transform(
                actor_path, True
            )
            self.set_transform_from_scene(actor_path, transform, True)
            await self.remote_scene.set_actor_transform(actor_path, transform, True)

        world_transform_change = "world_transform_change:"
        if op.startswith(world_transform_change):
            actor_path = Path(op[len(world_transform_change) :])

            if not actor_path in self.local_scene:
                raise Exception(f"actor not exist")

            transform = await self.remote_scene.get_pending_actor_transform(
                actor_path, False
            )

            self.set_transform_from_scene(actor_path, transform, False)
            await self.remote_scene.set_actor_transform(actor_path, transform, False)

        selection_change = "selection_change"
        if op.startswith(selection_change):
            actor_paths = await self.remote_scene.get_pending_selection_change()

            paths = []
            for p in actor_paths:
                paths.append(Path(p))

            await self.set_selection_from_remote_scene(paths)

        add_item = "add_item"
        if op.startswith(add_item):
            [transform, name] = await self.remote_scene.get_pending_add_item()
            self.add_item_by_drag.emit(name, transform)

    async def run_sim(self):
        if self.sim_process_running:
            return

        await self.remote_scene.publish_scene()
        await self.remote_scene.save_body_transform()

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
        await self.remote_scene.set_sync_from_mujoco_to_scene(True)

    async def stop_sim(self):
        if not self.sim_process_running:
            return

        async with self._sim_process_check_lock:
            await self.remote_scene.set_sync_from_mujoco_to_scene(False)
            self.sim_process_running = False
            self.sim_process.terminate()
            await self.remote_scene.restore_body_transform()

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

    async def set_selection(self, actors: list[BaseActor | Path], source: str = ""):
        actors, actor_paths = self.local_scene.get_actor_and_path_list(actors)

        if source != "outline":
            self.actor_outline.set_actor_selection(actors)

        if source != "remote":
            await self.remote_scene.set_selection(actor_paths)

        self.local_scene.selection = actor_paths

        # sync editor
        if len(actors) == 0:
            self.actor_editor.actor = None
        else:
            self.actor_editor.actor = actors[0]

    def make_unique_name(self, base_name: str, parent: BaseActor | Path) -> str:
        parent, _ = self.local_scene.get_actor_and_path(parent)
        assert isinstance(parent, GroupActor)

        existing_names = {child.name for child in parent.children}

        counter = 1
        new_name = f"{base_name}_{counter}"
        while new_name in existing_names:
            counter += 1
            new_name = f"{base_name}_{counter}"

        return new_name

    async def add_actor(self, actor: BaseActor, parent_actor: GroupActor | Path):
        ok, err = self.local_scene.can_add_actor(actor, parent_actor)
        if not ok:
            raise Exception(err)

        parent_actor, parent_actor_path = self.local_scene.get_actor_and_path(
            parent_actor
        )

        model = self.actor_outline_model
        parent_index = model.get_index_from_actor(parent_actor)
        child_count = len(parent_actor.children)

        model.beginInsertRows(parent_index, child_count, child_count)

        self.local_scene.add_actor(actor, parent_actor_path)

        model.endInsertRows()

        await self.remote_scene.add_actor(actor, parent_actor_path)

    async def delete_actor(self, actor) -> Tuple[BaseActor, GroupActor | None]:
        ok, err = self.local_scene.can_delete_actor(actor)
        if not ok:
            return

        actor, actor_path = self.local_scene.get_actor_and_path(actor)

        model = self.actor_outline_model
        index = model.get_index_from_actor(actor)
        parent_index = index.parent()

        model.beginRemoveRows(parent_index, index.row(), index.row())

        self.local_scene.delete_actor(actor)

        model.endRemoveRows()

        await self.remote_scene.delete_actor(actor_path)

    async def rename_actor(self, actor: BaseActor, new_name: str):
        ok, err = self.local_scene.can_rename_actor(actor, new_name)
        if not ok:
            raise Exception(err)

        actor, actor_path = self.local_scene.get_actor_and_path(actor)

        model = self.actor_outline_model
        index = model.get_index_from_actor(actor)

        self.local_scene.rename_actor(actor, new_name)

        model.dataChanged.emit(index, index)

        await self.remote_scene.rename_actor(actor_path, new_name)

    async def reparent_actor(
        self, actor: BaseActor | Path, new_parent: BaseActor | Path, row: int
    ):
        ok, err = self.local_scene.can_reparent_actor(actor, new_parent)
        if not ok:
            raise Exception(err)

        actor, actor_path = self.local_scene.get_actor_and_path(actor)
        new_parent, new_parent_path = self.local_scene.get_actor_and_path(new_parent)

        model = self.actor_outline_model

        model.beginResetModel()
        self.local_scene.reparent_actor(actor, new_parent, row)
        model.endResetModel()

        await self.remote_scene.reparent_actor(actor_path, new_parent_path)

    async def on_transform_edit(self):
        actor = self.actor_editor.actor
        if actor is None:
            return

        transform = self.actor_editor.transform
        if transform is None:
            return

        actor.transform = transform

        actor_path = self.local_scene.get_actor_path(actor)
        if actor_path is None:
            raise Exception("Invalid actor.")

        await self.remote_scene.set_actor_transform(actor_path, transform, True)

    def set_transform_from_scene(
        self, actor_path: Path, transform: Transform, local: bool
    ):
        actor = self.local_scene[actor_path]

        if local == True:
            actor.transform = transform
        else:
            actor.world_transform = transform

        if self.actor_editor.actor == actor:
            self.actor_editor.update_ui()


# 不要存Actor对象，只存Path。
# Actor可能被删除和创建，前后的Actor是不相等的。
# DeleteActorCommand中存的Actor不会再次放到LocalScene中，
# 而是作为模板使用。


class SelectionCommand:
    def __init__(self):
        self.old_selection = []
        self.new_selection = []

    def __repr__(self):
        return f"SelectionCommand(old_selection={self.old_selection}, new_selection={self.new_selection})"


class CreateGroupCommand:
    def __init__(self):
        self.path: Path | None = None

    def __repr__(self):
        return f"CreateGroupCommand(path={self.path})"


class CreateActorCommand:
    def __init__(self):
        self.actor = None
        self.path: Path = None
        self.row = -1

    def __repr__(self):
        return f"CreteActorCommand(path={self.path})"


class DeleteActorCommand:
    def __init__(self):
        self.actor: BaseActor = None
        self.path: Path = None
        self.row = -1

    def __repr__(self):
        return f"DeleteActorCommand(path={self.path})"


class RenameActorCommand:
    def __init__(self):
        self.old_path: Path = None
        self.new_path: Path = None

    def __repr__(self):
        return f"RenameActorCommand(old_path={self.old_path}, new_path={self.new_path})"


class ReparentActorCommand:
    def __init__(self):
        self.old_path = None
        self.old_row = -1
        self.new_path = None
        self.new_row = -1

    def __repr__(self):
        return f"ReparentActorCommand(old_path={self.old_path}, old_row={self.old_row}, new_path={self.new_path}, new_row={self.new_row})"


class TransformCommand:
    def __init__(self):
        self.actor_path = None
        self.old_transform = None
        self.new_transform = None
        self.local = None

    def __repr__(self):
        return f"TransformCommand(actor_path={self.actor_path})"


# Add undo/redo functionality
class MainWindow1(MainWindow):

    add_item_by_drag = QtCore.Signal(str, Transform)
    transform_change = QtCore.Signal(Path, bool)

    def __init__(self):
        super().__init__()

        self.command_history = []
        self.command_history_index = -1

    async def init(self):
        await super().init()

        await super()._init_ui()

        connect(self.actor_outline_model.request_reparent, self.reparent_from_outline)
        connect(self.actor_outline_model.add_item, self.add_item_to_scene)

        connect(
            self.actor_outline.actor_selection_changed,
            self.set_selection_from_outline,
        )
        connect(self.actor_outline.request_add_group, self.add_group_actor_from_outline)
        connect(self.actor_outline.request_delete, self.delete_actor_from_outline)
        connect(self.actor_outline.request_rename, self.open_rename_dialog)

        connect(self.actor_editor.transform_changed, self.on_transform_edit)

        connect(self.asset_browser.add_item, self.add_item_to_scene)

        connect(self.menu_file.aboutToShow, self.prepare_file_menu)
        connect(self.menu_edit.aboutToShow, self.prepare_edit_menu)

        connect(self.add_item_by_drag, self.add_item_drag)
        connect(self.transform_change, self.transform_change_command)

        # Window actions.

        action_undo = QtGui.QAction("Undo")
        action_undo.setShortcut(QtGui.QKeySequence("Ctrl+Z"))
        action_undo.setShortcutContext(QtCore.Qt.ShortcutContext.ApplicationShortcut)
        connect(action_undo.triggered, self.undo)

        action_redo = QtGui.QAction("Redo")
        action_redo.setShortcut(QtGui.QKeySequence("Ctrl+Shift+Z"))
        action_redo.setShortcutContext(QtCore.Qt.ShortcutContext.ApplicationShortcut)
        connect(action_redo.triggered, self.redo)

        self.addActions([action_undo, action_redo])

        self.resize(800, 400)
        self.show()

    def prepare_file_menu(self):
        self.menu_file.clear()

        action_exit = self.menu_file.addAction("Exit")
        connect(action_exit.triggered, self.close)

    def prepare_edit_menu(self):
        self.menu_edit.clear()

        action_undo = self.menu_edit.addAction("Undo")
        connect(action_undo.triggered, self.undo)

        if self.command_history_index >= 0:
            action_undo.setEnabled(True)
        else:
            action_undo.setEnabled(False)

        action_redo = self.menu_edit.addAction("Redo")
        connect(action_redo.triggered, self.redo)

        if self.command_history_index + 1 < len(self.command_history):
            action_redo.setEnabled(True)
        else:
            action_redo.setEnabled(False)

    def add_command(self, command):
        # Remove commands after the current index
        self.command_history = self.command_history[: self.command_history_index + 1]
        self.command_history.append(command)

        self.command_history_index = self.command_history_index + 1

        print(f"Added command: {command}")

    async def undo(self):
        if self.command_history_index < 0:
            return

        command = self.command_history[self.command_history_index]
        self.command_history_index -= 1

        match command:
            case SelectionCommand():
                await self.set_selection(command.old_selection)
            case CreateGroupCommand():
                await self.delete_actor(command.path)
            case CreateActorCommand():
                await self.delete_actor(command.path)
            case DeleteActorCommand():
                actor = command.actor
                parent_path = command.path.parent()
                await self.undo_delete_recursive(actor, parent_path)
            case RenameActorCommand():
                actor, _ = self.local_scene.get_actor_and_path(command.new_path)
                await self.rename_actor(actor, command.old_path.name())
            case ReparentActorCommand():
                actor, _ = self.local_scene.get_actor_and_path(command.new_path)
                old_parent_path = command.old_path.parent()
                await self.reparent_actor(actor, old_parent_path, command.old_row)
            case TransformCommand():
                self.set_transform_from_scene(command.actor_path, command.old_transform, command.local)
                await self.remote_scene.set_actor_transform(command.actor_path, command.old_transform, command.local)
            case _:
                raise Exception("Unknown command type.")

    async def redo(self):
        if self.command_history_index + 1 >= len(self.command_history):
            return

        command = self.command_history[self.command_history_index + 1]
        self.command_history_index += 1

        match command:
            case SelectionCommand():
                await self.set_selection(command.new_selection)
            case CreateGroupCommand():
                parent = command.path.parent()
                name = command.path.name()
                actor = GroupActor(name=name)
                await self.add_actor(actor, parent)
            case CreateActorCommand():
                parent = command.path.parent()
                actor = deepcopy(command.actor)
                await self.add_actor(actor, parent)
            case DeleteActorCommand():
                await self.delete_actor(command.path)
            case RenameActorCommand():
                actor, _ = self.local_scene.get_actor_and_path(command.old_path)
                await self.rename_actor(actor, command.new_path.name())
            case ReparentActorCommand():
                actor, _ = self.local_scene.get_actor_and_path(command.old_path)
                new_parent_path = command.new_path.parent()
                await self.reparent_actor(actor, new_parent_path, command.new_row)
            case TransformCommand():
                self.set_transform_from_scene(command.actor_path, command.new_transform, command.local)
                await self.remote_scene.set_actor_transform(command.actor_path, command.new_transform, command.local)
            case _:
                raise Exception("Unknown command type.")

    async def undo_delete_recursive(self, actor: BaseActor, parent_path: Path):
        if isinstance(actor, GroupActor):
            new_actor = GroupActor(name=actor.name)
            new_actor.transform = actor.transform

            await self.add_actor(new_actor, parent_path)

            this_path = parent_path / actor.name
            for child in actor.children:
                await self.undo_delete_recursive(child, this_path)
        else:
            new_actor = deepcopy(actor)
            await self.add_actor(new_actor, parent_path)

    async def add_group_actor_from_outline(self, parent_actor: BaseActor | Path):
        parent_actor, parent_actor_path = self.local_scene.get_actor_and_path(
            parent_actor
        )

        if not isinstance(parent_actor, GroupActor):
            parent_actor = parent_actor.parent
            parent_actor_path = parent_actor_path.parent()

        assert isinstance(parent_actor, GroupActor)

        new_group_name = self.make_unique_name("group", parent_actor)
        actor = GroupActor(name=new_group_name)
        await self.add_actor(actor, parent_actor)

        command = CreateGroupCommand()
        command.path = parent_actor_path / new_group_name
        self.add_command(command)

    async def delete_actor_from_outline(self, actor: BaseActor | Path):
        actor, actor_path = self.local_scene.get_actor_and_path(actor)

        parent_actor = actor.parent
        index = parent_actor.children.index(actor)
        assert index != -1

        command = DeleteActorCommand()
        command.actor = actor
        command.path = actor_path
        command.row = index

        await self.delete_actor(actor)

        self.add_command(command)

    async def set_selection_from_outline(self, actors):
        _, actor_paths = self.local_scene.get_actor_and_path_list(actors)
        if self.local_scene.selection != actor_paths:
            command = SelectionCommand()
            command.new_selection = actor_paths
            command.old_selection = self.local_scene.selection  
            self.add_command(command)
            await self.set_selection(actor_paths, "outline")

    async def set_selection_from_remote_scene(self, actor_paths: list[Path]):
        if self.local_scene.selection != actor_paths:
            command = SelectionCommand()
            command.new_selection = actor_paths
            command.old_selection = self.local_scene.selection
            self.add_command(command)
            await self.set_selection(actor_paths, "remote")

    async def reparent_from_outline(
        self, actor: BaseActor | Path, new_parent: BaseActor | Path, row: int
    ):
        actor, actor_path = self.local_scene.get_actor_and_path(actor)
        new_parent, new_parent_path = self.local_scene.get_actor_and_path(new_parent)
        old_parent = actor.parent
        old_index = old_parent.children.index(actor)
        assert old_index != -1

        command = ReparentActorCommand()
        command.old_path = actor_path
        command.old_row = old_index
        command.new_path = new_parent_path / actor.name
        command.new_row = row

        await self.reparent_actor(actor, new_parent, row)
        self.add_command(command)

    async def open_rename_dialog(self, actor: BaseActor):
        actor_path = self.local_scene.get_actor_path(actor)
        if actor_path is None:
            raise Exception("Invalid actor.")

        dialog = RenameDialog(actor_path, self.local_scene.can_rename_actor, self)
        if dialog.exec() == QtWidgets.QDialog.DialogCode.Accepted:
            new_name = dialog.new_name
            await self.rename_undoable(actor, new_name)

    async def rename_undoable(self, actor: BaseActor | Path, new_name: str):
        _, actor_path = self.local_scene.get_actor_and_path(actor)
        command = RenameActorCommand()
        command.old_path = actor_path
        command.new_path = actor_path.parent() / new_name

        await self.rename_actor(actor, new_name)

        self.add_command(command)

    async def add_item_to_scene(self, item_name, parent_actor=None):
        if parent_actor is None:
            parent_path = Path.root_path()
        else:
            parent_path = self.local_scene.get_actor_path(parent_actor)
        name = self.make_unique_name(item_name, parent_path)
        actor = AssetActor(name=name, spawnable_name=item_name)

        await self.add_actor(actor, parent_path)

        command = CreateActorCommand()
        command.actor = deepcopy(actor)
        command.path = parent_path / name
        self.add_command(command)

    async def add_item_drag(self, item_name, transform):
        name = self.make_unique_name(item_name, Path.root_path())
        actor = AssetActor(name=name, spawnable_name=item_name)

        pos = np.array([transform.pos[0], transform.pos[1], transform.pos[2]])
        quat = np.array(
            [transform.quat[0], transform.quat[1], transform.quat[2], transform.quat[3]]
        )
        scale = transform.scale
        actor.transform = Transform(pos, quat, scale)

        await self.add_actor(actor, Path.root_path())

        command = CreateActorCommand()
        command.actor = deepcopy(actor)
        command.path = Path.root_path() / name
        self.add_command(command)

    async def transform_change_command(self, actor_path, local):
        command = TransformCommand()
        command.actor_path = actor_path
        command.old_transform = self.start_transform
        command.new_transform = self.end_transform
        command.local = local
        self.add_command(command)

if __name__ == "__main__":
    
    q_app = QtWidgets.QApplication([])

    main_window = MainWindow1()

    # 在这之后，Qt的event_loop变成asyncio的event_loop。
    # 这是目前统一Qt和asyncio最好的方法。
    # 所以不要保存loop，统一使用asyncio.xxx()。
    # https://doc.qt.io/qtforpython-6/PySide6/QtAsyncio/index.html
    QtAsyncio.run(main_window.init())

    # magic!
    # AttributeError: 'NoneType' object has no attribute 'POLLER'
    # https://github.com/google-gemini/deprecated-generative-ai-python/issues/207#issuecomment-2601058191
    exit(0)
