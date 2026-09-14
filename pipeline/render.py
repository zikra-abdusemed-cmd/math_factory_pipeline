"""
Step 4a: render each validated scene .py file to an actual .mp4 (with the
narration audio already muxed in by manim-voiceover/manim itself).
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

from pipeline.scene_codegen import GeneratedScene

QUALITY_FLAGS = {
    "ql": "-ql",
    "qm": "-qm",
    "qh": "-qh",
    "qk": "-qk",
}


def render_scene(scene: GeneratedScene, media_dir: Path, project_root: Path, quality: str) -> Path:
    """Renders one scene and returns the path to its output .mp4."""
    if quality not in QUALITY_FLAGS:
        raise ValueError(f"Unknown quality {quality!r}, expected one of {list(QUALITY_FLAGS)}")

    media_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "manim",
        QUALITY_FLAGS[quality],
        "--disable_caching",
        "--media_dir",
        str(media_dir),
        str(scene.file_path),
        scene.class_name,
    ]
    env = {**os.environ, "PYTHONPATH": str(project_root) + os.pathsep + os.environ.get("PYTHONPATH", "")}
    result = subprocess.run(cmd, cwd=str(project_root), capture_output=True, text=True, timeout=900, env=env)
    if result.returncode != 0:
        raise RuntimeError(
            f"Final render failed for {scene.class_name}:\n{result.stdout[-3000:]}\n{result.stderr[-3000:]}"
        )

    candidates = sorted(
        media_dir.glob(f"**/{scene.class_name}.mp4"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(
            f"manim reported success but no {scene.class_name}.mp4 was found under {media_dir}"
        )
    return candidates[0]
