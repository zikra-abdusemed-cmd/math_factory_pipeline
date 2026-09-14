"""
Step 4b: stitch the per-beat clips (each already has its narration audio
muxed in, courtesy of manim-voiceover) into one polished final video.

We try a fast stream-copy concat first. If the clips don't share identical
codecs/parameters (which can happen since each is rendered by a separate
manim invocation) we fall back to a re-encoding concat, which is slower but
always works.
"""

from __future__ import annotations

import subprocess
from pathlib import Path


def concat_videos(clip_paths: list[Path], output_path: Path, tmp_dir: Path) -> Path:
    if not clip_paths:
        raise ValueError("No clips to concatenate.")

    tmp_dir.mkdir(parents=True, exist_ok=True)
    list_file = tmp_dir / "concat_list.txt"
    list_file.write_text("\n".join(f"file '{p.resolve()}'" for p in clip_paths), encoding="utf-8")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Attempt 1: stream copy (fast, lossless).
    copy_cmd = [
        "ffmpeg", "-y", "-f", "concat", "-safe", "0",
        "-i", str(list_file), "-c", "copy", str(output_path),
    ]
    result = subprocess.run(copy_cmd, capture_output=True, text=True)
    if result.returncode == 0 and output_path.exists():
        return output_path

    # Attempt 2: re-encode (robust fallback).
    reencode_cmd = [
        "ffmpeg", "-y", "-f", "concat", "-safe", "0",
        "-i", str(list_file),
        "-c:v", "libx264", "-preset", "medium", "-crf", "18",
        "-c:a", "aac", "-b:a", "192k",
        str(output_path),
    ]
    result2 = subprocess.run(reencode_cmd, capture_output=True, text=True)
    if result2.returncode != 0 or not output_path.exists():
        raise RuntimeError(
            "ffmpeg concat failed.\n"
            f"stream-copy attempt stderr:\n{result.stderr[-1500:]}\n\n"
            f"re-encode attempt stderr:\n{result2.stderr[-1500:]}"
        )
    return output_path
