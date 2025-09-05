
from orca_gym.orca_lab.math import Transform

import unittest
import numpy as np
import math


class TestTransform(unittest.TestCase):
    def setUp(self):
        self.identity = Transform()
        self.position = np.array([1.0, 2.0, 3.0])
        self.rotation = np.array([1.0, 0.0, 0.0, 0.0])  # Identity quaternion (w, x, y, z)
        self.scale = 2.0
        self.t = Transform(position=self.position, rotation=self.rotation, scale=self.scale)

    def test_default_constructor(self):
        t = Transform()
        np.testing.assert_array_equal(t.position, np.zeros(3))
        np.testing.assert_array_equal(t.rotation, np.array([1, 0, 0, 0]))
        self.assertEqual(t.scale, 1.0)

    def test_position_setter_type(self):
        with self.assertRaises(TypeError):
            Transform(position=[1, 2, 3])

    def test_rotation_setter_type(self):
        with self.assertRaises(TypeError):
            Transform(rotation=[1, 0, 0, 0])

    def test_rotation_setter_shape(self):
        with self.assertRaises(TypeError):
            Transform(rotation=np.array([1, 0, 0]))

    def test_rotation_unit_quaternion(self):
        with self.assertRaises(ValueError):
            Transform(rotation=np.array([2, 0, 0, 0]))

    def test_scale_setter_type(self):
        with self.assertRaises(TypeError):
            Transform(scale="not_a_float")

    def test_transform_point_identity(self):
        point = np.array([1.0, 2.0, 3.0])
        result = self.identity.transform_point(point)
        np.testing.assert_array_almost_equal(result, point)

    def test_transform_point_translation(self):
        t = Transform(position=np.array([1, 2, 3]), rotation=np.array([1, 0, 0, 0]), scale=1.0)
        point = np.array([0.0, 0.0, 0.0])
        result = t.transform_point(point)
        np.testing.assert_array_almost_equal(result, np.array([1, 2, 3]))

    def test_transform_point_scale(self):
        t = Transform(position=np.zeros(3), rotation=np.array([1, 0, 0, 0]), scale=2.0)
        point = np.array([1.0, 1.0, 1.0])
        result = t.transform_point(point)
        np.testing.assert_array_almost_equal(result, np.array([2.0, 2.0, 2.0]))

    def test_transform_vector(self):
        t = Transform(position=np.zeros(3), rotation=np.array([1, 0, 0, 0]), scale=3.0)
        vector = np.array([1.0, 0.0, 0.0])
        result = t.transform_vector(vector)
        np.testing.assert_array_almost_equal(result, np.array([3.0, 0.0, 0.0]))

    def test_transform_direction(self):
        t = Transform(position=np.zeros(3), rotation=np.array([1, 0, 0, 0]), scale=5.0)
        direction = np.array([0.0, 1.0, 0.0])
        result = t.transform_direction(direction)
        np.testing.assert_array_almost_equal(result, direction)

    def test_transform_point_type_error(self):
        t = Transform()
        with self.assertRaises(TypeError):
            t.transform_point([1, 2, 3])

    def test_transform_vector_type_error(self):
        t = Transform()
        with self.assertRaises(TypeError):
            t.transform_vector([1, 2, 3])

    def test_transform_direction_type_error(self):
        t = Transform()
        with self.assertRaises(TypeError):
            t.transform_direction([1, 2, 3])

    def test_multiply_identity(self):
        t = Transform(position=np.array([1, 2, 3]), rotation=np.array([1, 0, 0, 0]), scale=2.0)
        result = t.multiply(Transform())
        self.assertTrue(np.allclose(result.position, t.position))
        self.assertTrue(np.allclose(result.rotation, t.rotation))
        self.assertTrue(math.isclose(result.scale, t.scale))

    def test_multiply_type_error(self):
        t = Transform()
        with self.assertRaises(TypeError):
            t.multiply("not_a_transform")

    def test_inverse_identity(self):
        t = Transform()
        inv = t.inverse()
        self.assertTrue(np.allclose(inv.position, np.zeros(3)))
        self.assertTrue(np.allclose(inv.rotation, np.array([1, 0, 0, 0])))
        self.assertTrue(math.isclose(inv.scale, 1.0))

    def test_inverse_roundtrip(self):
        t = Transform(position=np.array([1, 2, 3]), rotation=np.array([1, 0, 0, 0]), scale=2.0)
        inv = t.inverse()
        roundtrip = t.multiply(inv)
        np.testing.assert_allclose(roundtrip.position, np.zeros(3), atol=1e-6)
        np.testing.assert_allclose(roundtrip.rotation, np.array([1, 0, 0, 0]), atol=1e-6)
        self.assertTrue(math.isclose(roundtrip.scale, 1.0, abs_tol=1e-6))

    def test_eq_and_ne(self):
        t1 = Transform()
        t2 = Transform()
        t3 = Transform(position=np.array([1, 0, 0]))
        self.assertTrue(t1 == t2)
        self.assertFalse(t1 != t2)
        self.assertFalse(t1 == t3)
        self.assertTrue(t1 != t3)

    def test_hash(self):
        t1 = Transform()
        t2 = Transform()
        self.assertEqual(hash(t1), hash(t2))

if __name__ == "__main__":
    unittest.main()
