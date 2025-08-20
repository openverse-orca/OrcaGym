# import sys
# import random
# from PySide6 import QtCore, QtWidgets, QtGui
from orca_gym.orca_lab.path import Path
from orca_gym.orca_lab.protos.edit_service_pb2_grpc import GrpcServiceStub
from orca_gym.orca_lab.scene import OrcaLabScene, Actor
import numpy as np

from scipy.spatial.transform import Rotation

# class AppWidget(QtWidgets.QWidget):
#     def __init__(self):
#         super().__init__()

#         self.hello = ["Hallo Welt", "Hei maailma", "Hola Mundo", "Привет мир"]

#         self.button = QtWidgets.QPushButton("Click me!")
#         self.text = QtWidgets.QLabel("Hello World", alignment=QtCore.Qt.AlignCenter)

#         self.layout = QtWidgets.QVBoxLayout(self)
#         self.layout.addWidget(self.text)
#         self.layout.addWidget(self.button)

#         self.button.clicked.connect(self.magic)

#     @QtCore.Slot()
#     def magic(self):
#         self.text.setText(random.choice(self.hello))


# class App:

#     def __init__(self, grpc_addr: str):
#         self.q_app = QtWidgets.QApplication([])
#         self.scene = OrcaLabScene(grpc_addr)
#         self.scene.publish_scene()

#     def active(self):
#         self.app_widget = AppWidget()
#         self.app_widget.resize(800, 600)
#         self.app_widget.show()

#     def deactivate(self):
#         self.scene.close_grpc()

#     def exec(self) -> int:
#         return self.q_app.exec()


if __name__ == "__main__":
    grpc_addr = "localhost:50151"
    scene = OrcaLabScene(grpc_addr)

    rot = Rotation.from_euler("xyz", [90, 45, 30], degrees=True)
    q = rot.as_quat()  # x,y,z,w

    actor = Actor(
        name=f"box1",
        spawnable_name="box",
        position=np.array([0, 0, 2]),
        rotation=np.array([1, 0, 0, 0]),
        scale=1.0,
    )

    scene.add_actor(actor, Path.root_path())

    scene.loop.run_forever()

    scene.close_grpc()

    # magic!
    # AttributeError: 'NoneType' object has no attribute 'POLLER'
    # https://github.com/google-gemini/deprecated-generative-ai-python/issues/207#issuecomment-2601058191
    exit()

    # sys.exit(0)
    # app = App(grpc_addr)

    # app.active()
    # code = app.exec()
    # sys.exit(code)
