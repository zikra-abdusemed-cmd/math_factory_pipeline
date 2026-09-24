"""
Step 4a: render each validated scene .py file to an actual .mp4 (with the
narration audio already muxed in by manim-voiceover/manim itself).
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import config
from pipeline.scene_codegen import GeneratedScene

QUALITY_FLAGS = {
    "ql": "-ql",
    "qm": "-qm",
    "qh": "-qh",
    "qk": "-qk",
}

# Soft per-scene ceilings. A healthy ql beat should finish in a few minutes;
# if Manim is still going after these limits, something pathological slipped
# through (historically: corrupt streaming WAV headers → multi-hour scenes).
QUALITY_TIMEOUT_SECONDS = {
    "ql": 900,   # 15 min
    "qm": 1800,  # 30 min
    "qh": 3600,  # 60 min
    "qk": 7200,  # 120 min
}


def _scene_timeout_seconds(quality: str) -> int:
    configured = config.pipeline.render_timeout_seconds
    soft = QUALITY_TIMEOUT_SECONDS.get(quality, configured)
    return min(configured, soft)


def render_scene(scene: GeneratedScene, media_dir: Path, project_root: Path, quality: str) -> Path:
    """Renders one scene and returns the path to its output .mp4."""
    if quality not in QUALITY_FLAGS:
        raise ValueError(f"Unknown quality {quality!r}, expected one of {list(QUALITY_FLAGS)}")

    media_dir.mkdir(parents=True, exist_ok=True)
    timeout = _scene_timeout_seconds(quality)
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
    # Manim MathTex needs latex/dvisvgm; GUI-launched servers often lack TeX PATH.
    from pipeline.math_rendering import ensure_tex_bin_on_path

    ensure_tex_bin_on_path()
    env["PATH"] = os.environ.get("PATH", "")
    try:
        result = subprocess.run(
            cmd,
            cwd=str(project_root),
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"Final render timed out after {timeout} seconds for "
            f"{scene.class_name} at quality={quality}. A healthy draft scene should finish "
            "well under this. Check generated/_voiceover_cache for corrupt WAVs, or use "
            "--quality ql. Increase RENDER_TIMEOUT_SECONDS only if you intentionally need longer."
        ) from exc
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
