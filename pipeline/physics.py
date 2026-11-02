"""Small deterministic analytical models for introductory mechanics.

Physical time is expressed in seconds and is independent of Manim playback time.
The initial model is projectile motion with constant gravity and no air drag.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable


@dataclass(frozen=True)
class Projectile:
    x0: float
    y0: float
    speed: float
    angle_degrees: float
    gravity: float = 9.81
    assumptions: tuple[str, ...] = (
        "constant gravitational acceleration",
        "negligible air resistance",
        "flat ground at y=0 when y0 is zero",
    )

    def __post_init__(self) -> None:
        vals = (self.x0, self.y0, self.speed, self.angle_degrees, self.gravity)
        if not all(math.isfinite(v) for v in vals):
            raise ValueError("Projectile parameters must be finite numbers.")
        if self.speed < 0:
            raise ValueError("Initial speed must be non-negative (m/s).")
        if self.gravity <= 0:
            raise ValueError("Gravity must be positive (m/s^2).")
        if not -90 <= self.angle_degrees <= 90:
            raise ValueError("Launch angle must be between -90 and 90 degrees.")

    @property
    def vx0(self) -> float:
        return self.speed * math.cos(math.radians(self.angle_degrees))

    @property
    def vy0(self) -> float:
        return self.speed * math.sin(math.radians(self.angle_degrees))

    def position(self, time_seconds: float) -> tuple[float, float]:
        t = _time(time_seconds)
        return (
            self.x0 + self.vx0 * t,
            self.y0 + self.vy0 * t - 0.5 * self.gravity * t * t,
        )

    def velocity(self, time_seconds: float) -> tuple[float, float]:
        t = _time(time_seconds)
        return self.vx0, self.vy0 - self.gravity * t

    def time_to_ground(self, ground_y: float = 0.0) -> float | None:
        if not math.isfinite(ground_y):
            raise ValueError("Ground height must be finite (m).")
        discriminant = self.vy0**2 + 2 * self.gravity * (self.y0 - ground_y)
        if discriminant < 0:
            return None
        impact = (self.vy0 + math.sqrt(discriminant)) / self.gravity
        return impact if impact >= 0 else None

    def trajectory(self, samples: int = 101, *, ground_y: float = 0.0) -> list[tuple[float, float]]:
        if isinstance(samples, bool) or samples < 2:
            raise ValueError("Trajectory requires at least two samples.")
        end = self.time_to_ground(ground_y)
        if end is None:
            raise ValueError("Projectile does not reach the requested ground height.")
        return [self.position(end * i / (samples - 1)) for i in range(samples)]


def _time(value: float) -> float:
    if not math.isfinite(value) or value < 0:
        raise ValueError("Physical time must be a finite non-negative number of seconds.")
    return value

