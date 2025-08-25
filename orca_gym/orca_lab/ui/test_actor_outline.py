from PySide6 import QtCore, QtWidgets, QtGui

from actor_outline import ActorOutline
from actor_outline_model import ActorOutlineModel
from orca_gym.orca_lab.actor import GroupActor, AssetActor


app = QtWidgets.QApplication([])
actor_outline = ActorOutline()
model = ActorOutlineModel()


group1 = GroupActor("g1")
group2 = GroupActor("g2", group1)
group3 = GroupActor("g3", group1)
group4 = GroupActor("g4", group3)
asset1 = AssetActor("a1", "spw_name", group1)


model.set_root_group(group1)
actor_outline.setModel(model)
actor_outline.show()

app.exec()
