import numpy as np
from scipy.spatial.transform import Rotation
import math


class Transform:
    """
    Represents a 3D transformation including position, rotation (as a quaternion), and uniform scale.
    When apply to a vector, the order of operations is scale, then rotate, then translate.
    Vector is treated as **column vector**. i.e. v' = M * v, where M is the transformation matrix.
    When combining two transforms T1 and T2 (T = T1 * T2), T2 is applied first, then T1.
    This is consistent with our Renderer convention.

    Attributes:
        position (np.array): 3D position as a numpy array of shape (3,).
        rotation (np.array): Rotation is a uint quaternion as a numpy array of shape (4,) in (w, x, y, z) format.
        scale (float): Uniform scale factor.

    Raises:
        TypeError: If input types for position, rotation, scale, or point are incorrect.
    """

    def __init__(
        self, position=np.array([0, 0, 0]), rotation=np.array([1, 0, 0, 0]), scale=1.0
    ):
        self.position = position
        self.rotation = rotation
        self.scale = scale

    @property
    def position(self) -> np.array:
        return self._position

    @position.setter
    def position(self, value: np.array):
        if not isinstance(value, np.ndarray) or value.shape != (3,):
            raise TypeError("position must be a numpy array of shape (3,).")
        self._position = value

    @property
    def rotation(self) -> np.array:
        return self._rotation

    @rotation.setter
    def rotation(self, value: np.array):
        if not isinstance(value, np.ndarray) or value.shape != (4,):
            raise TypeError("rotation must be a numpy array of shape (4,).")

        norm = np.linalg.norm(value)
        if not math.isclose(norm, 1.0):
            raise ValueError("rotation must be a unit quaternion.")
        self._rotation = value

    @property
    def scale(self) -> float:
        return self._scale

    @scale.setter
    def scale(self, value: float):
        if not isinstance(value, (int, float)):
            raise TypeError("scale must be a float.")
        self._scale = float(value)

    def __repr__(self):
        return f"Transform(position={self.position}, rotation={self.rotation}, scale={self.scale})"

    def __str__(self):
        return f"Transform(position={self.position}, rotation={self.rotation}, scale={self.scale})"

    def __eq__(self, other):
        if isinstance(other, Transform):
            return (
                np.allclose(self.position, other.position)
                and np.allclose(self.rotation, other.rotation)
                and math.isclose(self.scale, other.scale)
            )
        return False

    def __ne__(self, other):
        if isinstance(other, Transform):
            return not self.__eq__(other)
        return True

    def __hash__(self):
        return hash((tuple(self.position), tuple(self.rotation), self.scale))

    def __mul__(self, other: "Transform") -> "Transform":
        return self.multiply(other)

    def transform_point(self, point: np.array) -> np.array:
        """Apply the transform to a point."""

        if not isinstance(point, np.ndarray) or point.shape != (3,):
            raise TypeError("point must be a numpy array of shape (3,).")

        # Scale
        scaled_point = point * self.scale

        # Rotate
        r = Rotation.from_quat(self.rotation, scalar_first=True)
        rotated_point = r.apply(scaled_point)

        # Translate
        translated_point = rotated_point + self.position

        return translated_point

    def transform_vector(self, vector: np.array) -> np.array:
        """Apply the transform to a vector (ignoring translation)."""

        if not isinstance(vector, np.ndarray) or vector.shape != (3,):
            raise TypeError("vector must be a numpy array of shape (3,).")

        # Scale
        scaled_vector = vector * self.scale

        # Rotate
        r = Rotation.from_quat(self.rotation, scalar_first=True)
        rotated_vector = r.apply(scaled_vector)

        return rotated_vector

    def transform_direction(self, direction: np.array) -> np.array:
        """Apply the transform to a direction (ignoring translation and scale)."""

        if not isinstance(direction, np.ndarray) or direction.shape != (3,):
            raise TypeError("direction must be a numpy array of shape (3,).")

        # Rotate
        r = Rotation.from_quat(self.rotation, scalar_first=True)
        rotated_direction = r.apply(direction)

        return rotated_direction

    def multiply(self, other: "Transform") -> "Transform":
        """Combine this transform with another transform. (self * other)"""

        if not isinstance(other, Transform):
            raise TypeError("other must be an instance of Transform.")

        # Combined scale
        combined_scale = self.scale * other.scale

        # Combined rotation
        r1 = Rotation.from_quat(self.rotation, scalar_first=True)
        r2 = Rotation.from_quat(other.rotation, scalar_first=True)
        combined_rotation = (r1 * r2).as_quat(scalar_first=True)

        # Combined position
        combined_position = self.transform_point(other.position)

        return Transform(
            position=combined_position,
            rotation=combined_rotation,
            scale=combined_scale,
        )

    def inverse(self):
        """Return the inverse of the transform."""
        inv_scale = 1.0 / max(self.scale, 1e-9)  # Avoid division by zero

        # Invert of a unit quaternion is its conjugate.
        inv_rotation = np.array(
            [self.rotation[0], -self.rotation[1], -self.rotation[2], -self.rotation[3]]
        )

        inv_r = Rotation.from_quat(inv_rotation, scalar_first=True)
        inv_position = -inv_scale * inv_r.apply(self.position)

        return Transform(position=inv_position, rotation=inv_rotation, scale=inv_scale)
