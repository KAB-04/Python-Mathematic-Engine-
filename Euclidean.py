# A Euclidean Mathematical Engine

from __future__ import annotations
import math


class Point:
    """Represents a point in 2D Euclidean space.
    The most fundamental primitive — all geometry builds on points.
    """

    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y

    def distance_to(self, other: Point) -> float:
        """Returns the straight-line distance to another point.
        Uses math.hypot for numerical stability over manual sqrt.
        """
        return math.hypot(other.x - self.x, other.y - self.y)

    def __eq__(self, other: object) -> bool:
        """Compares two points with floating-point tolerance.
        Raw == is avoided because geometry involves real numbers where
        floating-point rounding silently breaks exact equality.
        """
        if not isinstance(other, Point):
            return NotImplemented
        return math.isclose(self.x, other.x) and math.isclose(self.y, other.y)

    def __repr__(self):
        return f"Point({self.x}, {self.y})"


class Segment:
    """Represents a bounded line segment between two points.
    Unlike Line, a Segment has a definite length and midpoint.
    """

    def __init__(self, p1: Point, p2: Point):
        self.p1 = p1
        self.p2 = p2

    def length(self) -> float:
        """Returns the length of the segment."""
        return self.p1.distance_to(self.p2)

    def midpoint(self) -> Point:
        """Returns the point exactly halfway between the two endpoints."""
        return Point(
            (self.p1.x + self.p2.x) / 2,
            (self.p1.y + self.p2.y) / 2
        )

    def __repr__(self):
        return f"Segment({self.p1}, {self.p2})"


class Line:
    """Represents an infinite straight line in general form: ax + by + c = 0.

    This form is preferred over slope-intercept (y = mx + b) because it
    handles vertical lines without special cases (slope would be undefined).
    """

    def __init__(self, a: float, b: float, c: float):
        self.a = a
        self.b = b
        self.c = c

    @classmethod
    def from_two_points(cls, p1: Point, p2: Point) -> Line:
        """Constructs the unique line passing through two given points."""
        a = p2.y - p1.y
        b = p1.x - p2.x
        c = a * p1.x + b * p1.y
        return cls(a, b, -c)

    def contains_point(self, p: Point, tol=1e-9) -> bool:
        """Returns True if the point lies on this line (within tolerance).
        Substitutes the point into ax + by + c and checks if result ≈ 0.
        """
        return abs(self.a * p.x + self.b * p.y + self.c) < tol

    def __repr__(self):
        return f"Line({self.a}x + {self.b}y + {self.c} = 0)"


class Angle:
    """Represents the angle at vertex B, formed by rays BA and BC."""

    def __init__(self, A: Point, B: Point, C: Point):
        self.A = A
        self.B = B  # vertex
        self.C = C

    def measure_radians(self) -> float:
        """Measures the angle in radians using the dot product formula.
        Clamps the cosine to [-1, 1] to guard against floating-point drift
        that would cause math.acos to raise a ValueError.
        """
        v1 = (self.A.x - self.B.x, self.A.y - self.B.y)
        v2 = (self.C.x - self.B.x, self.C.y - self.B.y)
        mag1 = math.hypot(*v1)
        mag2 = math.hypot(*v2)

        if math.isclose(mag1, 0) or math.isclose(mag2, 0):
            raise ValueError("Angle undefined: two of the three points are coincident.")

        dot = v1[0]*v2[0] + v1[1]*v2[1]
        cos_angle = max(-1.0, min(1.0, dot / (mag1 * mag2)))
        return math.acos(cos_angle)

    def measure_degrees(self) -> float:
        """Measures the angle in degrees."""
        return math.degrees(self.measure_radians())

    def is_right(self, tol=1e-9) -> bool:
        """Returns True if the angle is exactly 90° (within tolerance)."""
        return math.isclose(self.measure_radians(), math.pi / 2, abs_tol=tol)

    def is_acute(self) -> bool:
        """Returns True if the angle is less than 90°."""
        return self.measure_radians() < math.pi / 2

    def is_obtuse(self) -> bool:
        """Returns True if the angle is greater than 90°."""
        return self.measure_radians() > math.pi / 2

    def __repr__(self):
        return f"Angle(A={self.A}, vertex={self.B}, C={self.C})"