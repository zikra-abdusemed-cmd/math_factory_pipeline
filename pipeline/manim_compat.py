"""Small compatibility helpers for common generated Manim calls."""

from __future__ import annotations

from typing import Any

from manim import Arc


def arc_with_dimensions(
    *,
    width: float | None = None,
    height: float | None = None,
    radius: float | None = None,
    **kwargs: Any,
) -> Arc:
    """Build a Manim Arc and optionally fit its bounding-box dimensions.

    Manim's Arc constructor accepts ``radius`` but does not accept ``width``
    or ``height``. Generated diagrams often use those names to draw an
    elliptical cylinder rim, so apply the requested dimensions after creation.
    """
    if radius is None:
        if width is not None:
            radius = width / 2
        elif height is not None:
            radius = height / 2
        else:
            radius = 1.0

    arc = Arc(radius=radius, **kwargs)
    if width is not None:
        arc.stretch_to_fit_width(width)
    if height is not None:
        arc.stretch_to_fit_height(height)
    return arc
