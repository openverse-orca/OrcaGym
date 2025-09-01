from pyparsing import Dict
from orca_gym.orca_lab.actor import BaseActor, GroupActor
from orca_gym.orca_lab.path import Path


class LocalScene:
    def __init__(self):
        # 作为根节点，不可见， 路径是"/"。下面挂着所有的顶层Actor。
        self.root_actor = GroupActor(name="root", parent=None)
        self._actors: Dict[Path, BaseActor] = {}
        self._actors[Path.root_path()] = self.root_actor

    def __contains__(self, path: Path) -> bool:
        return path in self._actors

    def __getitem__(self, path: Path) -> BaseActor:
        if path not in self._actors:
            raise KeyError(f"No actor at path {path}")
        return self._actors[path]

    @property
    def pseudo_root_actor(self):
        return self.root_actor

    def find_actor_by_path(self, path: Path):
        if path in self._actors:
            return self._actors[path]
        return None

    def get_actor_path(self, actor):
        for path, a in self._actors.items():
            if a == actor:
                return path
        return None

    def add_actor(self, actor: BaseActor, parent_path: Path):
        if not isinstance(actor, BaseActor):
            raise Exception("Invalid actor.")

        parent_actor = self.find_actor_by_path(parent_path)
        if parent_actor is None:
            raise Exception(f"Invalid parent path: {parent_path}")

        actor.parent = parent_actor
        path = parent_path / actor.name
        self._actors[path] = actor

    def delete_actor(self, actor: BaseActor):
        actor_path = self.get_actor_path(actor)

        if actor_path is None:
            raise Exception("Invalid actor.")

        if actor_path == actor_path.root_path():
            raise Exception("Cannot delete root actor.")

        actor.parent = None

        actors_to_delete = [actor_path]
        for p in self._actors.keys():
            if p.is_descendant_of(actor_path):
                actors_to_delete.append(p)

        for p in actors_to_delete:
            del self._actors[p]

    def rename_actor(self, actor: BaseActor, new_name) -> bool:
        actor_path = self.get_actor_path(actor)
        if actor_path is None:
            raise Exception("Invalid actor.")

        if actor_path == actor_path.root_path():
            raise Exception("Cannot delete root actor.")

        if Path.is_valid_name(new_name) == False:
            raise Exception("Invalid name.")

        actor.name = new_name
        new_actor_path = actor_path.parent() / new_name

        paths_to_update = [actor_path]

        for p in self._actors.keys():
            if p.is_descendant_of(actor_path):
                paths_to_update.append(p)

        prefix = actor_path.string()
        for p in paths_to_update:
            relative_path = p.string()[len(prefix) :]
            updated_path = Path(new_actor_path.string() + relative_path)
            self._actors[updated_path] = self._actors[p]
            del self._actors[p]

    def reparent_actor(self, actor, new_parent):
        raise NotImplementedError
