from typing import List, override

from orca_gym.orca_lab.path import Path
from orca_gym.orca_lab.math import Transform

type ParentActor = GroupActor | None


class BaseActor:
    def __init__(self, name: str, parent: ParentActor):
        self._name = ""
        self._parent = None
        self._transform = Transform()
        self.name = name
        self.parent = parent

    def __repr__(self):
        return f"BaseActor(name={self._name})"

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, value):
        if not isinstance(value, str) or not Path.is_valid_name(value):
            raise ValueError(f"Invalid name: {value}")
        self._name = value

    @property
    def parent(self) -> ParentActor:
        return self._parent

    @parent.setter
    def parent(self, parent_actor):
        if parent_actor is not None and not isinstance(parent_actor, GroupActor):
            raise TypeError("parent must be an instance of GroupActor or None.")

        if parent_actor == self._parent:
            return

        if self._parent is not None:
            self._parent.remove_child(self)

        self._parent = parent_actor
        parent_actor.add_child(self)

    @property
    def transform(self):
        return self._transform

    @transform.setter
    def transform(self, value):
        if not isinstance(value, Transform):
            raise TypeError("transform must be an instance of Transform.")
        self._transform = value


class GroupActor(BaseActor):
    def __init__(self, name: str, parent: ParentActor = None):
        self._children: List[BaseActor] = []
        super().__init__(name, parent)

    def __repr__(self):
        return f"GroupActor(name={self.name}, children_count={len(self._children)})"

    @property
    def children(self):
        return self._children.copy()

    def add_child(self, child: BaseActor):
        if not isinstance(child, BaseActor):
            raise TypeError("Child must be an instance of GroupActor or AssetActor.")

        if child in self._children:
            raise ValueError(f"{child.name} is already a child of {self.name}")

        self._children.append(child)
        child.parent = self

    def remove_child(self, child: BaseActor):
        if child in self._children:
            self.children.remove(child)
            child.parent = None
        else:
            raise ValueError(f"{child.name} is not a child of {self.name}")


class AssetActor(BaseActor):
    def __init__(self, name: str, spawnable_name: str, parent: GroupActor = None):
        super().__init__(name, parent)
        self.spawnable_name = spawnable_name

    def __repr__(self):
        return f"AssetActor(name={self.name})"

    @property
    def spawnable_name(self):
        return self._spawnable_name

    @spawnable_name.setter
    def spawnable_name(self, value):
        if not isinstance(value, str) or len(value) == 0:
            raise ValueError("spawnable_name name must be non-empty string")
        self._spawnable_name = value
