"""Generate the Grade 9 projectile lesson through the existing pipeline CLI.

Usage: python examples/generate_projectile_reference.py

The normal text model, TTS provider, validation, rendering, cache and stitching
workflow are used. This wrapper only supplies the fixed reference brief.
"""

from __future__ import annotations

from pipeline.orchestrator import run_pipeline


TITLE = "Projectile Motion: Why the Path Curves"
BRIEF = (
    "Grade 9. Hook with a launched ball and an intriguing question. Use launch "
    "speed 12 m/s at 45 degrees, g=9.81 m/s^2, negligible air resistance. "
    "Show the observed trajectory first, then velocity and downward gravity "
    "vectors, pause for the force explanation, transition from the physical "
    "path to its trajectory graph, progressively reveal x(t)=x0+v0 cos(theta)t "
    "and y(t)=y0+v0 sin(theta)t-1/2 gt^2, work a numerical range example, and "
    "return to the opening question for a concise recap. Keep camera movement "
    "purposeful and use one consistent object identity and coordinate system."
)


if __name__ == "__main__":
    print(run_pipeline(TITLE, extra_instructions=BRIEF))
