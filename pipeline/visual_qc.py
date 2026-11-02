"""Geometric and preview-frame quality checks for generated scenes."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import numpy as np

from pipeline.layout import is_annotation, mobject_aabb, out_of_frame, overlapping_pairs
from pipeline.visual_plan import VisualPlan


class VisualQualityError(ValueError):
    """Raised when composition or camera instructions cannot be auto-corrected."""


def validate_visual_plan_quality(plan: VisualPlan | None) -> list[str]:
    issues: list[str] = []
    if plan is None:
        return issues
    ids = {obj.get("id") for obj in plan.objects if isinstance(obj, dict)}
    follow_targets = [event.target for event in plan.camera if event.command == "follow_object"]
    if len(set(follow_targets)) > 1:
        issues.append("Multiple follow_object targets in one beat; use sequential shots.")
    for event in plan.camera:
        if event.duration < 0.1:
            issues.append(f"Camera command {event.command!r} has an invalid duration.")
        if event.command in {"push_in", "pan_to", "follow_object", "freeze_and_annotate"}:
            if not event.target:
                issues.append(f"Camera command {event.command!r} is missing a target.")
            elif event.target not in ids:
                issues.append(f"Camera target {event.target!r} is not a declared object.")
        if plan.mode != "3d" and event.command == "orbit_scene":
            issues.append("orbit_scene requires visual_plan.mode='3d'.")
    commands = [event.command for event in plan.camera]
    if commands.count("follow_object") > 1:
        issues.append("More than one follow_object shot in a beat will stack tracking; keep a single tracking shot.")
    for first, second in zip(plan.camera, plan.camera[1:]):
        if first.start + first.duration > second.start + 0.05:
            issues.append(
                f"Camera shots overlap ({first.command} @ {first.start} and {second.command} @ {second.start})."
            )
    if plan.main_subject and plan.main_subject not in ids:
        issues.append(f"main_subject {plan.main_subject!r} is not a declared object.")
    for object_id in plan.keep_readable:
        if object_id not in ids:
            issues.append(f"keep_readable id {object_id!r} is not a declared object.")
    return issues


def validate_scene_composition(
    scene: Any,
    namespace: dict[str, Any],
    *,
    controller: Any | None = None,
    raise_on_error: bool = False,
) -> list[str]:
    from pipeline.layout import _SKIP_NAMES

    items = {
        name: value
        for name, value in namespace.items()
        if isinstance(name, str) and name not in _SKIP_NAMES and mobject_aabb(value) is not None
    }
    issues: list[str] = []
    clipped = out_of_frame({name: mob for name, mob in items.items() if is_annotation(mob) or type(mob).__name__ in {"Axes", "NumberPlane"}})
    if clipped:
        issues.append(f"Annotations or graphs extend outside the safe frame: {clipped}.")
    overlaps = overlapping_pairs({name: mob for name, mob in items.items() if is_annotation(mob)})
    if overlaps:
        issues.append(f"Independent labels/equations still overlap: {overlaps}.")
    if controller is not None:
        attached = bool(getattr(controller, "_attached", False))
        frame = getattr(getattr(scene, "camera", None), "frame", None)
        updater_count = len(getattr(frame, "updaters", []) or []) if frame is not None else 0
        if updater_count > 1:
            issues.append("Multiple camera-frame updaters are active; only CameraController may move the frame.")
        if attached and updater_count == 0:
            issues.append("CameraController is attached but the frame has no updater.")
    if raise_on_error and issues:
        raise VisualQualityError(" ".join(issues))
    return issues


def inspect_preview_frames(paths: Iterable[Path], *, jump_threshold: float = 0.42) -> list[str]:
    """Inspect extracted PNG frames for emptiness and abrupt cuts.

    Geometry cannot see camera jitter; sampled pixels can. This does not use an
    external vision API.
    """
    frames: list[np.ndarray] = []
    issues: list[str] = []
    for path in paths:
        path = Path(path)
        if not path.is_file():
            issues.append(f"Missing preview frame {path}.")
            continue
        try:
            from PIL import Image
        except ImportError:
            return issues
        image = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
        frames.append(image)
        occupancy = float(np.mean(np.max(image, axis=2) > 0.08))
        # Sparse physics shots (a tracked particle on a dark field) occupy
        # far less than 1% of pixels; treat a frame as empty only if it has
        # essentially no content at all.
        if occupancy < 1e-5 and float(np.max(image)) < 0.12:
            issues.append(f"Frame {path.name} is nearly empty.")
        edge = np.zeros(image.shape[:2], dtype=bool)
        edge[:4, :] = True
        edge[-4:, :] = True
        edge[:, :4] = True
        edge[:, -4:] = True
        bright = np.max(image, axis=2) > 0.18
        if bright.any() and float(np.mean(bright[edge])) > 0.55 and occupancy < 0.2:
            issues.append(f"Frame {path.name} looks clipped against the frame edge.")
    for index in range(1, len(frames)):
        delta = float(np.mean(np.abs(frames[index] - frames[index - 1])))
        if delta > jump_threshold:
            issues.append(f"Abrupt visual jump between sampled frames {index - 1} and {index} (delta={delta:.2f}).")
    return issues


def extract_video_frames(video_path: Path, output_dir: Path, timestamps: Iterable[float]) -> list[Path]:
    from pipeline.concat import _ffmpeg_bin, _run_ffmpeg

    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for index, stamp in enumerate(timestamps):
        dest = output_dir / f"qc_{index:02d}_{stamp:.2f}s.png"
        _run_ffmpeg([
            "-y", "-ss", f"{max(0.0, stamp):.3f}", "-i", str(video_path),
            "-frames:v", "1", "-q:v", "3", str(dest),
        ])
        if dest.is_file():
            paths.append(dest)
    return paths
