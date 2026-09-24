"""
Step 4b: stitch the per-beat clips (each already has its narration audio
muxed in, courtesy of manim-voiceover) into one polished final video.

Each clip is first normalized so its video and audio streams share one
duration (audio is padded if the picture ran slightly long after frame
rounding). Then we concat. Stream-copy is attempted first; if codecs differ
we fall back to a re-encode.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

_FFMPEG_CANDIDATES = (
    Path("/usr/local/bin/ffmpeg"),
    Path("/opt/homebrew/bin/ffmpeg"),
    Path("/usr/bin/ffmpeg"),
)

_DURATION_RE = re.compile(r"Duration:\s+(\d+):(\d+):(\d+(?:\.\d+)?)")


def _ffmpeg_bin() -> str:
    found = shutil.which("ffmpeg")
    if found:
        return found
    for candidate in _FFMPEG_CANDIDATES:
        if candidate.is_file():
            return str(candidate)
    raise RuntimeError(
        "ffmpeg was not found on PATH. Install it (e.g. `brew install ffmpeg`) "
        "and restart the pipeline."
    )


def _run_ffmpeg(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run([_ffmpeg_bin(), *args], capture_output=True, text=True)


def _hms_to_seconds(match: re.Match[str]) -> float:
    hours, minutes, seconds = match.groups()
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)


def _probe_container_duration(path: Path) -> float | None:
    """Read duration from `ffmpeg -i` (no ffprobe required)."""
    result = _run_ffmpeg(["-hide_banner", "-i", str(path)])
    text = (result.stderr or "") + (result.stdout or "")
    match = _DURATION_RE.search(text)
    if not match:
        return None
    return _hms_to_seconds(match)


def _probe_mapped_duration(path: Path, stream_map: str) -> float | None:
    """Duration of one stream by decoding it to null with ffmpeg."""
    result = _run_ffmpeg(
        [
            "-hide_banner",
            "-i",
            str(path),
            "-map",
            stream_map,
            "-f",
            "null",
            "-",
        ]
    )
    text = (result.stderr or "") + (result.stdout or "")
    # The last "time=HH:MM:SS.xx" in the encode progress line is the stream length.
    times = re.findall(r"time=(\d+):(\d+):(\d+(?:\.\d+)?)", text)
    if times:
        hours, minutes, seconds = times[-1]
        return int(hours) * 3600 + int(minutes) * 60 + float(seconds)
    match = _DURATION_RE.search(text)
    if match:
        return _hms_to_seconds(match)
    return None


def _probe_duration(path: Path, stream_selector: str) -> float | None:
    stream_map = "0:v:0" if stream_selector.startswith("v") else "0:a:0"
    duration = _probe_mapped_duration(path, stream_map)
    if duration is not None:
        return duration
    return _probe_container_duration(path)


def _normalize_clip_av(clip: Path, output: Path) -> Path:
    """Make video and audio the same length without truncating narration."""
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        video_dur = _probe_duration(clip, "v:0")
        audio_dur = _probe_duration(clip, "a:0")
    except RuntimeError:
        return clip
    if video_dur is None or audio_dur is None:
        return clip
    if abs(video_dur - audio_dur) < 0.04:
        return clip

    target = max(video_dur, audio_dur)
    video_pad = max(0.0, target - video_dur)
    audio_pad = max(0.0, target - audio_dur)
    result = _run_ffmpeg(
        [
            "-y",
            "-i",
            str(clip),
            "-filter_complex",
            f"[0:v]tpad=stop_mode=clone:stop_duration={video_pad:.4f}[v];"
            f"[0:a]apad=pad_dur={audio_pad:.4f}[a]",
            "-map",
            "[v]",
            "-map",
            "[a]",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "18",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-movflags",
            "+faststart",
            str(output),
        ]
    )
    if result.returncode != 0 or not output.exists():
        return clip
    return output


def concat_videos(clip_paths: list[Path], output_path: Path, tmp_dir: Path) -> Path:
    if not clip_paths:
        raise ValueError("No clips to concatenate.")

    tmp_dir.mkdir(parents=True, exist_ok=True)
    normalized: list[Path] = []
    for index, clip in enumerate(clip_paths, start=1):
        target = tmp_dir / f"norm_{index:02d}_{clip.stem}.mp4"
        normalized.append(_normalize_clip_av(clip, target))

    list_file = tmp_dir / "concat_list.txt"
    list_file.write_text("\n".join(f"file '{p.resolve()}'" for p in normalized), encoding="utf-8")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    copy_result = _run_ffmpeg(
        ["-y", "-f", "concat", "-safe", "0", "-i", str(list_file), "-c", "copy", str(output_path)]
    )
    if copy_result.returncode == 0 and output_path.exists():
        return output_path

    reencode_result = _run_ffmpeg(
        [
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(list_file),
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "18",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-movflags",
            "+faststart",
            str(output_path),
        ]
    )
    if reencode_result.returncode != 0 or not output_path.exists():
        raise RuntimeError(
            "ffmpeg concat failed.\n"
            f"stream-copy attempt stderr:\n{(copy_result.stderr or '')[-1500:]}\n\n"
            f"re-encode attempt stderr:\n{(reencode_result.stderr or '')[-1500:]}"
        )
    return output_path
