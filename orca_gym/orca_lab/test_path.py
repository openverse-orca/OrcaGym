from path import Path

import unittest


class TestAddFunction(unittest.TestCase):
    def test_empty_path_is_invalid(self):
        with self.assertRaises(Exception):
            p = Path("")

    def test_default_construct_path_is_root_path(self):
        p = Path()
        self.assertEqual(p, Path.root_path())

    def test_equal(self):
        self.assertEqual(Path("/a"), Path("/a"))
        self.assertEqual(Path(), Path())


if __name__ == "__main__":
    unittest.main()
