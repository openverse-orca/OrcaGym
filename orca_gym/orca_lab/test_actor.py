import unittest

from orca_gym.orca_lab.actor import BaseActor, GroupActor


class TestGroupActor(unittest.TestCase):
    def test_add_child(self):
        group = GroupActor("Group1")
        actor = BaseActor("Actor1", None)
        group.add_child(actor)
        self.assertIn(actor, group.children)
        self.assertIs(actor.parent, group)

    def test_set_parent_when_construct(self):
        group = GroupActor("Group1")
        actor = BaseActor("Actor1", group)
        self.assertIn(actor, group.children)
        self.assertIs(actor.parent, group)

    def test_set_parent_none(self):
        group = GroupActor("Group1")
        actor = BaseActor("Actor1", group)
        actor.parent = None
        self.assertEqual(len(group.children), 0)
        self.assertEqual(actor.parent, None)

    def test_add_child_wrong_type(self):
        group = GroupActor("Group1")
        with self.assertRaises(TypeError):
            group.add_child("not_an_actor")

    def test_remove_child(self):
        group = GroupActor("Group1")
        actor = BaseActor("Actor1", None)
        group.add_child(actor)
        group.remove_child(actor)
        self.assertNotIn(actor, group.children)
        self.assertIsNone(actor.parent)

    def test_remove_child_not_present(self):
        group = GroupActor("Group1")
        actor = BaseActor("Actor1", None)
        with self.assertRaises(ValueError):
            group.remove_child(actor)

    def test_children_property(self):
        group = GroupActor("Group1")
        actor1 = BaseActor("Actor1", None)
        actor2 = BaseActor("Actor2", None)
        group.add_child(actor1)
        group.add_child(actor2)
        children = group.children
        self.assertEqual(len(children), 2)
        self.assertListEqual(children, [actor1, actor2])
        # Ensure it's a copy
        children.append("dummy")
        self.assertEqual(len(group.children), 2)

    def test_repr(self):
        group = GroupActor("Group1")
        self.assertIn("GroupActor(name=Group1", repr(group))


if __name__ == "__main__":
    unittest.main()
