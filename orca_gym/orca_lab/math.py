import numpy as np


eps = 1e-9  # Small value for numerical stability in comparisons


def is_close(a, b, eps=eps) -> bool:
    """Check if two floating-point things are close to each other."""
    return abs(a - b) <= eps


class Vec3:
    def __init__(self, x=0.0, y=0.0, z=0.0):
        self.data = np.array([x, y, z], dtype=np.float64)

    def __repr__(self):
        return f"Vec({self.data[0]}, {self.data[1]}, {self.data[2]})"

    def __str__(self):
        return f"Vec({self.data[0]}, {self.data[1]}, {self.data[2]})"

    def __add__(self, other):
        if isinstance(other, Vec3):
            return Vec3(*(self.data + other.data))
        raise TypeError(
            "Unsupported operand type(s) for +: 'Vec' and '{}'".format(
                type(other).__name__
            )
        )

    def __sub__(self, other):
        if isinstance(other, Vec3):
            return Vec3(*(self.data - other.data))
        raise TypeError(
            "Unsupported operand type(s) for -: 'Vec' and '{}'".format(
                type(other).__name__
            )
        )

    def __mul__(self, scalar):
        if isinstance(scalar, (int, float)):
            return Vec3(*(self.data * scalar))
        raise TypeError(
            "Unsupported operand type(s) for *: 'Vec' and '{}'".format(
                type(scalar).__name__
            )
        )

    def __truediv__(self, scalar):
        if isinstance(scalar, (int, float)):
            return Vec3(*(self.data / scalar))
        raise TypeError(
            "Unsupported operand type(s) for /: 'Vec' and '{}'".format(
                type(scalar).__name__
            )
        )

    def __eq__(self, other):
        if isinstance(other, Vec3):
            return np.array_equal(self.data, other.data)
        return False

    def __ne__(self, other):
        if isinstance(other, Vec3):
            return not np.array_equal(self.data, other.data)
        return True

    def __hash__(self):
        return hash(tuple(self.data))

    def __neg__(self):
        return Vec3(*(-self.data))

    def __abs__(self):
        return Vec3(*(np.abs(self.data)))

    def __getitem__(self, index):
        if isinstance(index, int) and 0 <= index < 3:
            return self.data[index]
        raise IndexError("Index out of range. Valid indices are 0, 1, 2.")

    def __setitem__(self, index, value):
        if isinstance(index, int) and 0 <= index < 3:
            self.data[index] = value
        else:
            raise IndexError("Index out of range. Valid indices are 0, 1, 2.")

    def __iter__(self):
        return iter(self.data)

    def dot(self, other):
        if isinstance(other, Vec3):
            return np.dot(self.data, other.data)
        raise TypeError(
            "Unsupported operand type(s) for dot: 'Vec' and '{}'".format(
                type(other).__name__
            )
        )

    def cross(self, other):
        if isinstance(other, Vec3):
            return Vec3(*(np.cross(self.data, other.data)))
        raise TypeError(
            "Unsupported operand type(s) for cross: 'Vec' and '{}'".format(
                type(other).__name__
            )
        )

    def length(self):
        return np.linalg.norm(self.data)

    def normalize(self):
        norm = self.length()
        if norm == 0:
            raise ValueError("Cannot normalize a zero vector.")
        return self / norm

    def to_array(self):
        return self.data.copy()

    def from_array(self, array):
        if isinstance(array, (list, np.ndarray)) and len(array) == 3:
            self.data = np.array(array, dtype=float)
        else:
            raise ValueError("Input must be a list or array of length 3.")


class Transform:
    def __init__(self, position=None, rotation=None, scale=None):
        self.position = Vec3() if position is None else Vec3(*position)
        self.rotation = np.array([1,0,0,0])
        self.scale = 1.0

    def __repr__(self):
        return f"Transform(position={self.position}, rotation={self.rotation}, scale={self.scale})"

    def __str__(self):
        return f"Transform(position={self.position}, rotation={self.rotation}, scale={self.scale})"

    def __eq__(self, other):
        if isinstance(other, Transform):
            return self.position == other.position and self.rotation == other.rotation and self.scale == other.scale
        return False

    def __ne__(self, other):
        if isinstance(other, Transform):
            return not (
                self.position == other.position and self.rotation == other.rotation and self.scale == other.scale
            )
        return True

    def __hash__(self):
        return hash((self.position, self.rotation, self.scale))

