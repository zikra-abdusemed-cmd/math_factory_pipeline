"""Validated, constrained visual-director plans for generated scenes."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import json
import re
from typing import Any

CAMERA_COMMANDS = {"push_in", "pull_back", "pan_to", "follow_object", "orbit_scene", "freeze_and_annotate", "transition_to_graph"}
SCENE_MODES = {"2d", "3d"}
_ID = re.compile(r"^[A-Za-z][A-Za-z0-9_-]{0,39}$")


@dataclass(frozen=True)
class CameraEvent:
    command: str
    target: str | None = None
    duration: float = 1.0
    start: float = 0.0
    reason: str = ""


@dataclass(frozen=True)
class VisualPlan:
    objective: str = ""
    learning_purpose: str = ""
    theme: str = "dark"
    mode: str = "2d"
    duration: float | None = None
    objects: tuple[dict[str, Any], ...] = ()
    camera: tuple[CameraEvent, ...] = ()
    events: tuple[dict[str, Any], ...] = ()
    narration_cues: tuple[dict[str, Any], ...] = ()
    assumptions: tuple[str, ...] = ()
    parameters: dict[str, float] = field(default_factory=dict)
    main_subject: str = ""
    secondary: tuple[str, ...] = ()
    keep_readable: tuple[str, ...] = ()
    intentional_overlaps: tuple[str, ...] = ()


def validate_visual_plan(raw: Any, *, narration_duration: float | None = None) -> VisualPlan:
    """Validate optional plan data. A missing plan is a safe static 2D plan."""
    if raw is None:
        return VisualPlan()
    if not isinstance(raw, dict):
        raise ValueError("visual_plan must be an object when provided.")
    mode = raw.get("mode", "2d")
    if mode not in SCENE_MODES:
        raise ValueError(f"Unsupported visual_plan mode {mode!r}; use '2d' or '3d'.")
    duration = raw.get("duration")
    if duration is not None:
        duration = _number(duration, "visual_plan.duration", minimum=0.1, maximum=600)
    if narration_duration is not None and duration is not None and duration > narration_duration + 0.05:
        raise ValueError("visual_plan.duration cannot exceed its narration duration.")
    objects = raw.get("objects", [])
    if not isinstance(objects, list):
        raise ValueError("visual_plan.objects must be an array.")
    ids: set[str] = set()
    for index, obj in enumerate(objects):
        if not isinstance(obj, dict) or not isinstance(obj.get("id"), str) or not _ID.fullmatch(obj["id"]):
            raise ValueError(f"visual_plan.objects[{index}] needs a valid string id.")
        if obj["id"] in ids:
            raise ValueError(f"Duplicate visual object id {obj['id']!r}.")
        ids.add(obj["id"])

    def register_reference(target: Any, source: str, *, required: bool = False) -> str | None:
        """Repair an omitted declaration for an otherwise valid referenced ID."""
        if target is None:
            return None
        if isinstance(target, str) and _ID.fullmatch(target):
            if target not in ids:
                ids.add(target)
                objects.append({
                    "id": target,
                    "kind": "referenced_visual",
                    "initial_state": f"Referenced by {source}; infer from the scene storyboard.",
                })
            return target
        if required:
            raise ValueError(f"{source} target must be a valid object id.")
        return None

    cameras: list[CameraEvent] = []
    for index, item in enumerate(raw.get("camera", [])):
        if not isinstance(item, dict):
            raise ValueError(f"visual_plan.camera[{index}] must be an object.")
        command = item.get("command")
        if command not in CAMERA_COMMANDS:
            raise ValueError(f"Unknown camera command {command!r}; supported: {sorted(CAMERA_COMMANDS)}.")
        target = item.get("target")
        if command in {"pan_to", "follow_object", "push_in", "freeze_and_annotate"}:
            if target is None:
                raise ValueError(f"Camera command {command!r} requires a target matching a declared object id.")
            target = register_reference(target, f"camera[{index}]", required=True)
        else:
            target = register_reference(target, f"camera[{index}]")
        if mode == "2d" and command == "orbit_scene":
            raise ValueError("orbit_scene requires visual_plan.mode='3d'.")
        if mode == "3d" and command in {"pan_to", "follow_object"}:
            # Supported but target moves are translated to the relevant ThreeDScene camera API by toolkit.
            pass
        start = _number(item.get("start", 0), f"camera[{index}].start", minimum=0, maximum=600)
        duration_s = _number(item.get("duration", 1), f"camera[{index}].duration", minimum=0.1, maximum=20)
        if duration is not None and start + duration_s > duration + 0.05:
            raise ValueError(f"Camera event {index} extends past the scene duration.")
        if narration_duration is not None and start + duration_s > narration_duration + 0.05:
            raise ValueError(f"Camera event {index} extends past narration duration.")
        cameras.append(CameraEvent(command, target, duration_s, start, str(item.get("reason", ""))))
    if any(cameras[i].start > cameras[i + 1].start for i in range(len(cameras) - 1)):
        raise ValueError("visual_plan.camera timestamps must be ordered.")
    for first, second in zip(cameras, cameras[1:]):
        if first.start + first.duration > second.start + 0.05:
            raise ValueError("visual_plan.camera events cannot overlap; use sequential purposeful moves.")
    normalized_sections: dict[str, Any] = {}
    for section in ("events", "narration_cues"):
        refs = raw.get(section, [])
        if isinstance(refs, list):
            normalized_refs = []
            for index, item in enumerate(refs):
                if isinstance(item, dict) and item.get("target") is not None:
                    item = dict(item)
                    normalized_target = register_reference(item.get("target"), f"{section}[{index}]")
                    if normalized_target is None:
                        item["target_label"] = str(item.pop("target"))
                    else:
                        item["target"] = normalized_target
                normalized_refs.append(item)
            normalized_sections[section] = normalized_refs
        else:
            normalized_sections[section] = refs
    events = _validate_timed_items(normalized_sections["events"], "events", duration, narration_duration, ids)
    cues = _validate_timed_items(
        normalized_sections["narration_cues"], "narration_cues", duration,
        narration_duration, ids, allow_text=True,
    )
    # Explicit camera wins. Otherwise a motion event gets one restrained
    # push → follow → pull sequence so tracking is target-aware and sequential.
    if not cameras:
        cameras = _camera_sequence_for_motion(events, duration, narration_duration)
    theme = raw.get("theme", "dark")
    parameters_raw = raw.get("parameters", {})
    if parameters_raw in (None, ""):
        parameters_raw = {}
    if not isinstance(parameters_raw, dict):
        raise ValueError("visual_plan.parameters must be an object of numeric values.")
    parameters_raw = dict(parameters_raw)
    # Older or loosely-following model responses sometimes put a descriptive
    # theme under `parameters`. Migrate it to the dedicated visual field.
    if "theme" in parameters_raw:
        parameter_theme = parameters_raw.pop("theme")
        if theme == "dark":
            theme = parameter_theme
    theme = _normalize_theme(theme)
    parameters, parameter_notes = _numeric_parameters(parameters_raw)
    main_subject = raw.get("main_subject") or raw.get("subject") or ""
    if main_subject in (None, ""):
        # Prefer the follow target / first moving object when the model omits it.
        follow = next((event.target for event in cameras if event.command == "follow_object" and event.target), None)
        main_subject = follow or ""
    elif not isinstance(main_subject, str) or not _ID.fullmatch(main_subject):
        raise ValueError("visual_plan.main_subject must be a declared object id.")
    else:
        register_reference(main_subject, "main_subject")
    secondary = _id_list(raw.get("secondary", []), "secondary", ids, register_reference)
    keep_readable = _id_list(raw.get("keep_readable", []), "keep_readable", ids, register_reference)
    overlaps_raw = raw.get("intentional_overlaps", [])
    if overlaps_raw in (None, ""):
        overlaps_raw = []
    if not isinstance(overlaps_raw, list) or not all(isinstance(item, str) for item in overlaps_raw):
        raise ValueError("visual_plan.intentional_overlaps must be an array of strings such as 'velocity+ball'.")
    assumptions = raw.get("assumptions", [])
    if assumptions in (None, ""):
        assumptions = []
    if not isinstance(assumptions, list) or not all(isinstance(v, str) for v in assumptions):
        raise ValueError("visual_plan.assumptions must be an array of strings.")
    assumptions = list(assumptions)
    for note in parameter_notes:
        if note not in assumptions:
            assumptions.append(note)
    return VisualPlan(
        objective=_string(raw.get("objective", ""), "objective"),
        learning_purpose=_string(raw.get("learning_purpose", ""), "learning_purpose"),
        theme=theme, mode=mode, duration=duration, objects=tuple(objects), camera=tuple(cameras),
        events=tuple(events), narration_cues=tuple(cues), assumptions=tuple(assumptions), parameters=parameters,
        main_subject=main_subject, secondary=tuple(secondary),
        keep_readable=tuple(keep_readable), intentional_overlaps=tuple(str(item) for item in overlaps_raw),
    )


_MOTION_KINDS = {
    "move", "motion", "follow", "projectile", "fall", "launch", "travel", "slide", "throw",
}


def _camera_sequence_for_motion(
    events: list[dict[str, Any]],
    duration: float | None,
    narration_duration: float | None,
) -> list[CameraEvent]:
    """Build one push/follow/pull sequence from the first motion event."""
    motion = next(
        (
            event for event in events
            if isinstance(event.get("kind"), str)
            and event["kind"].lower() in _MOTION_KINDS
            and isinstance(event.get("target"), str)
            and _ID.fullmatch(event["target"])
        ),
        None,
    )
    if motion is None:
        return []
    target = motion["target"]
    bound = min(v for v in (duration, narration_duration) if v is not None) if duration is not None or narration_duration is not None else 8.0
    at = float(motion.get("at", 0.0))
    push_start = max(0.0, min(at, bound) - 1.0)
    push_dur = max(0.4, min(1.0, at - push_start if at > push_start else 0.8))
    follow_start = max(push_start + push_dur, at)
    remaining_after_follow_start = max(0.8, bound - follow_start)
    follow_dur = max(1.0, min(3.0, remaining_after_follow_start - 1.0))
    pull_start = follow_start + follow_dur
    pull_dur = max(0.5, min(1.0, bound - pull_start))
    if pull_start + pull_dur > bound + 0.05:
        pull_dur = max(0.4, bound - pull_start)
    if follow_start + follow_dur + 0.05 > pull_start and pull_dur < 0.4:
        # Scene too short for a pull-back; keep push + follow only.
        return [
            CameraEvent("push_in", target, push_dur, push_start, "establish moving subject"),
            CameraEvent("follow_object", target, follow_dur, follow_start, "keep subject framed"),
        ]
    return [
        CameraEvent("push_in", target, push_dur, push_start, "establish moving subject"),
        CameraEvent("follow_object", target, follow_dur, follow_start, "keep subject framed"),
        CameraEvent("pull_back", target, pull_dur, pull_start, "restore wider composition"),
    ]


def _validate_timed_items(
    items: Any,
    name: str,
    duration: float | None,
    narration_duration: float | None,
    object_ids: set[str],
    *,
    allow_text: bool = False,
) -> list[dict[str, Any]]:
    if not isinstance(items, list):
        raise ValueError(f"visual_plan.{name} must be an array.")
    result = []
    previous = -1.0
    for index, item in enumerate(items):
        if allow_text and isinstance(item, str):
            # Some model responses serialize short cues as strings despite
            # the requested {at, cue} objects. Preserve them as untimed cues
            # at the beginning of the beat rather than failing the whole job.
            item = {"at": 0, "cue": item}
        if not isinstance(item, dict):
            raise ValueError(f"visual_plan.{name}[{index}] must be an object.")
        at = _number(item.get("at", 0), f"{name}[{index}].at", minimum=0, maximum=600)
        target = item.get("target")
        if target is not None and target not in object_ids:
            raise ValueError(f"visual_plan.{name}[{index}] references undeclared object id {target!r}.")
        if at < previous:
            raise ValueError(f"visual_plan.{name} timestamps must be ordered.")
        bound = min(v for v in (duration, narration_duration) if v is not None) if duration is not None or narration_duration is not None else None
        if bound is not None and at > bound + 0.05:
            raise ValueError(f"visual_plan.{name}[{index}] timestamp exceeds scene/narration duration.")
        previous = at
        result.append(dict(item))
    return result


def _id_list(value: Any, label: str, ids: set[str], register_reference: Any) -> list[str]:
    if value in (None, ""):
        return []
    if not isinstance(value, list):
        raise ValueError(f"visual_plan.{label} must be an array of object ids.")
    result = []
    for index, item in enumerate(value):
        if not isinstance(item, str) or not _ID.fullmatch(item):
            raise ValueError(f"visual_plan.{label}[{index}] must be a valid object id.")
        register_reference(item, f"{label}[{index}]")
        result.append(item)
    return result


_NON_NUMERIC_PARAMETER_KEYS = {
    "physical_motion", "motion", "easing", "profile", "palette", "mode",
    "style", "camera", "notes", "description", "learning_purpose", "objective",
}


def _numeric_parameters(raw: dict[str, Any]) -> tuple[dict[str, float], list[str]]:
    """Keep only finite numeric physics values; descriptive keys become notes.

    Models often put flags such as ``physical_motion: "linear"`` next to ``g``.
    Those must not abort a lesson.
    """
    numbers: dict[str, float] = {}
    notes: list[str] = []
    for key, value in raw.items():
        name = str(key)
        if name.lower() in _NON_NUMERIC_PARAMETER_KEYS:
            notes.append(f"{name}={value}")
            continue
        if isinstance(value, bool):
            notes.append(f"{name}={value}")
            continue
        if isinstance(value, str):
            text = value.strip()
            try:
                value = float(text)
            except ValueError:
                notes.append(f"{name}={text}")
                continue
        try:
            numbers[name] = _number(value, f"visual_plan.parameters.{name}", minimum=-1e9, maximum=1e9)
        except ValueError:
            notes.append(f"{name}={value}")
    return numbers, notes


def _number(value: Any, label: str, *, minimum: float, maximum: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not minimum <= value <= maximum:
        raise ValueError(f"{label} must be a finite number in [{minimum}, {maximum}].")
    return float(value)


def _string(value: Any, label: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"visual_plan.{label} must be a string.")
    return value


def _normalize_theme(value: Any) -> str:
    """Convert optional model-generated theme metadata into bounded text.

    Theme is descriptive direction, not executable configuration. Tolerate
    JSON scalars, lists, and objects instead of failing an otherwise valid
    lesson, while keeping the value bounded before passing it to codegen.
    """
    if value is None or value == "":
        return "dark"
    if isinstance(value, str):
        text = value.strip() or "dark"
    else:
        try:
            text = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        except (TypeError, ValueError):
            text = str(value)
    return text[:240]

