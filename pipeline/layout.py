"""Deterministic composition helpers for generated Manim scenes.

Language-model code often guesses coordinates. This module measures bounding
boxes after mobjects exist and then repositions independent labels, titles,
and equations. Physics objects that define the demonstration stay locked so
intentional geometric relationships (a vector rooted on a particle, a trail
through an object) are preserved.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

import numpy as np
from manim import DOWN, LEFT, ORIGIN, RIGHT, UL, UR, DL, DR, UP, config


SAFE_MARGIN = 0.55
MIN_LABEL_GAP = 0.22
MIN_INDEPENDENT_GAP = 0.28
MIN_TEXT_SCALE = 0.72
MAX_NUDGE_STEPS = 12
_REPEL_ITERS = 14
_ANNOTATION_TYPES = {"Text", "MathTex", "Tex", "MarkupText", "SingleStringMathTex"}
_GRAPH_TYPES = {"Axes", "ThreeDAxes", "NumberPlane", "ComplexPlane"}
_SKIP_NAMES = {
    "self", "tracker", "camera_frame", "_camera_controller", "np", "math",
    "ORIGIN", "UP", "DOWN", "LEFT", "RIGHT",
}


@dataclass
class AABB:
    left: float
    right: float
    bottom: float
    top: float

    @property
    def width(self) -> float:
        return self.right - self.left

    @property
    def height(self) -> float:
        return self.top - self.bottom

    @property
    def center(self) -> np.ndarray:
        return np.array([(self.left + self.right) / 2, (self.bottom + self.top) / 2, 0.0])

    def inflate(self, padding: float) -> "AABB":
        return AABB(self.left - padding, self.right + padding, self.bottom - padding, self.top + padding)

    def intersects(self, other: "AABB") -> bool:
        return not (
            self.right < other.left
            or self.left > other.right
            or self.top < other.bottom
            or self.bottom > other.top
        )

    def contains(self, other: "AABB") -> bool:
        return (
            self.left <= other.left
            and self.right >= other.right
            and self.bottom <= other.bottom
            and self.top >= other.top
        )

    def translate(self, dx: float, dy: float) -> "AABB":
        return AABB(self.left + dx, self.right + dx, self.bottom + dy, self.top + dy)


@dataclass
class LayoutRegion:
    name: str
    box: AABB


@dataclass
class LayoutReport:
    moved: list[str] = field(default_factory=list)
    scaled: list[str] = field(default_factory=list)
    issues: list[str] = field(default_factory=list)
    overcrowded: bool = False


def frame_aabb(*, margin: float = SAFE_MARGIN) -> AABB:
    half_w = float(config.frame_width) / 2 - margin
    half_h = float(config.frame_height) / 2 - margin
    return AABB(-half_w, half_w, -half_h, half_h)


def default_regions(*, margin: float = SAFE_MARGIN) -> dict[str, LayoutRegion]:
    """Flexible semantic regions; physics scenes may still use the full frame."""
    frame = frame_aabb(margin=margin)
    title = LayoutRegion("title", AABB(frame.left, frame.right, frame.top - 0.95, frame.top))
    equation = LayoutRegion(
        "equation",
        AABB(frame.right - min(4.6, frame.width * 0.34), frame.right, frame.bottom + 0.2, frame.top - 1.05),
    )
    graph = LayoutRegion("graph", AABB(frame.left + 0.15, frame.right - 0.15, frame.bottom + 0.15, frame.top - 1.05))
    main = LayoutRegion(
        "main",
        AABB(frame.left, max(frame.left + 0.2, equation.box.left - 0.25), frame.bottom + 0.15, title.box.bottom - 0.12),
    )
    return {"title": title, "equation": equation, "graph": graph, "main": main, "frame": LayoutRegion("frame", frame)}


def mobject_aabb(mobject: Any) -> AABB | None:
    if mobject is None or not hasattr(mobject, "get_corner"):
        return None
    try:
        corners = [np.asarray(mobject.get_corner(corner), dtype=float) for corner in (UL, UR, DL, DR)]
    except Exception:
        return None
    xs = [p[0] for p in corners]
    ys = [p[1] for p in corners]
    if not xs or not all(np.isfinite(v) for v in xs + ys):
        return None
    return AABB(min(xs), max(xs), min(ys), max(ys))


def is_annotation(mobject: Any) -> bool:
    name = type(mobject).__name__
    if name in _ANNOTATION_TYPES:
        return True
    if name in {"VGroup", "Group"} and hasattr(mobject, "submobjects"):
        children = [child for child in mobject.submobjects if child is not None]
        return bool(children) and all(is_annotation(child) for child in children)
    return False


def is_graph(mobject: Any) -> bool:
    return type(mobject).__name__ in _GRAPH_TYPES


def is_vector_like(mobject: Any) -> bool:
    return type(mobject).__name__ in {"Arrow", "Vector", "Line", "DashedLine", "Arc"}


def intentional_overlap(a: Any, b: Any) -> bool:
    """Vectors, trails, and braces may legally intersect their hosts."""
    if is_vector_like(a) or is_vector_like(b):
        return True
    names = {type(a).__name__, type(b).__name__}
    if names & {"VMobject"} and ({"Dot", "Circle", "Sphere"} & names):
        return True
    return False


def _shift(mobject: Any, delta: np.ndarray) -> None:
    if hasattr(mobject, "shift"):
        mobject.shift(delta)


def _scale(mobject: Any, factor: float) -> None:
    if factor == 1 or not hasattr(mobject, "scale"):
        return
    mobject.scale(factor)


def _nearest_parent(label: Any, candidates: list[Any]) -> Any | None:
    box = mobject_aabb(label)
    if box is None:
        return None
    best = None
    best_dist = 2.4
    center = box.center
    for host in candidates:
        host_box = mobject_aabb(host)
        if host_box is None:
            continue
        dist = float(np.linalg.norm(center - host_box.center))
        if dist < best_dist:
            best = host
            best_dist = dist
    return best


def _placement_blockers(label: Any, host: Any | None, candidates: list[Any]) -> list[Any]:
    """Graphs that contain the host are not hard blockers; labels may sit in the plot."""
    host_box = mobject_aabb(host) if host is not None else None
    blockers = []
    for other in candidates:
        if other is label:
            continue
        if is_graph(other) and host_box is not None:
            graph_box = mobject_aabb(other)
            if graph_box is not None and graph_box.inflate(0.2).contains(host_box):
                continue
        blockers.append(other)
    return blockers


def _region_slots(region: LayoutRegion, count: int) -> list[np.ndarray]:
    count = max(1, int(count))
    box = region.box
    if count == 1:
        return [box.center.copy()]
    usable = max(box.height - 0.3, 0.4)
    top = box.top - 0.15
    return [
        np.array([box.center[0], top - usable * (i + 0.5) / count, 0.0])
        for i in range(count)
    ]


def _repel_annotations(annotations: dict[str, Any], frame: AABB, report: LayoutReport) -> None:
    for _ in range(_REPEL_ITERS):
        pairs = overlapping_pairs(annotations)
        if not pairs:
            return
        for left, right in pairs:
            a, b = annotations[left], annotations[right]
            box_a, box_b = mobject_aabb(a), mobject_aabb(b)
            if box_a is None or box_b is None:
                continue
            delta = box_a.center - box_b.center
            if float(np.linalg.norm(delta[:2])) < 1e-6:
                delta = np.array([0.35, 0.2, 0.0])
            else:
                delta = np.array([delta[0], delta[1], 0.0])
                delta = delta / (np.linalg.norm(delta) or 1.0) * 0.22
            _shift(a, delta)
            _shift(b, -delta)
            _clamp_to_frame(a, frame)
            _clamp_to_frame(b, frame)
            if left not in report.moved:
                report.moved.append(left)
            if right not in report.moved:
                report.moved.append(right)


def _nudge_label(label: Any, host: Any | None, blockers: list[Any], frame: AABB) -> bool:
    """Try a ring of offsets around the host (or current position)."""
    box = mobject_aabb(label)
    if box is None:
        return False
    origin = host.get_center() if host is not None and hasattr(host, "get_center") else box.center
    radius = 0.45
    if host is not None:
        host_box = mobject_aabb(host)
        if host_box is not None:
            radius = max(0.4, 0.5 * max(host_box.width, host_box.height) + MIN_LABEL_GAP + 0.5 * max(box.width, box.height))
    directions = [
        UP, DOWN, LEFT, RIGHT, UL, UR, DL, DR,
        UP + 0.4 * LEFT, UP + 0.4 * RIGHT, DOWN + 0.4 * LEFT, DOWN + 0.4 * RIGHT,
    ]
    original = np.asarray(label.get_center(), dtype=float)
    for step in range(1, MAX_NUDGE_STEPS + 1):
        for direction in directions:
            target = np.asarray(origin, dtype=float) + np.asarray(direction, dtype=float) * (radius * (0.65 + 0.2 * step))
            _shift(label, target - np.asarray(label.get_center(), dtype=float))
            new_box = mobject_aabb(label)
            if new_box is None or not frame.contains(new_box):
                continue
            if not any(
                other_box is not None and new_box.inflate(MIN_INDEPENDENT_GAP / 2).intersects(other_box.inflate(MIN_INDEPENDENT_GAP / 2))
                for other in blockers
                if other is not label
                for other_box in [mobject_aabb(other)]
            ):
                return True
    _shift(label, original - np.asarray(label.get_center(), dtype=float))
    return False


def _clamp_to_frame(mobject: Any, frame: AABB) -> bool:
    box = mobject_aabb(mobject)
    if box is None:
        return False
    dx = 0.0
    dy = 0.0
    if box.left < frame.left:
        dx = frame.left - box.left
    elif box.right > frame.right:
        dx = frame.right - box.right
    if box.bottom < frame.bottom:
        dy = frame.bottom - box.bottom
    elif box.top > frame.top:
        dy = frame.top - box.top
    if dx or dy:
        _shift(mobject, np.array([dx, dy, 0.0]))
        return True
    return False


def _move_to_region(mobject: Any, region: LayoutRegion, frame: AABB) -> None:
    box = mobject_aabb(mobject)
    if box is None:
        return
    target = region.box.center.copy()
    # Keep the item inside the region by aligning its center, then clamp.
    _shift(mobject, target - box.center)
    _clamp_to_frame(mobject, AABB(
        max(frame.left, region.box.left),
        min(frame.right, region.box.right),
        max(frame.bottom, region.box.bottom),
        min(frame.top, region.box.top),
    ))


def overlapping_pairs(items: dict[str, Any], *, ignore_intentional: bool = True) -> list[tuple[str, str]]:
    names = list(items)
    pairs: list[tuple[str, str]] = []
    for i, name_a in enumerate(names):
        box_a = mobject_aabb(items[name_a])
        if box_a is None:
            continue
        for name_b in names[i + 1 :]:
            mob_a, mob_b = items[name_a], items[name_b]
            if ignore_intentional and intentional_overlap(mob_a, mob_b):
                continue
            # Associated label sitting next to a host is not an independent collision
            # unless the boxes heavily overlap.
            box_b = mobject_aabb(mob_b)
            if box_b is None:
                continue
            a_ann, b_ann = is_annotation(mob_a), is_annotation(mob_b)
            if a_ann ^ b_ann:
                padding = MIN_LABEL_GAP / 3
            elif a_ann and b_ann:
                padding = MIN_INDEPENDENT_GAP / 2
            else:
                continue
            if box_a.inflate(padding).intersects(box_b.inflate(padding)):
                pairs.append((name_a, name_b))
    return pairs


def out_of_frame(items: dict[str, Any], *, margin: float = SAFE_MARGIN) -> list[str]:
    frame = frame_aabb(margin=margin)
    clipped = []
    for name, mob in items.items():
        box = mobject_aabb(mob)
        if box is None:
            continue
        if box.right < frame.left or box.left > frame.right or box.top < frame.bottom or box.bottom > frame.top:
            clipped.append(name)
        elif not frame.contains(box) and (is_annotation(mob) or is_graph(mob)):
            clipped.append(name)
    return clipped


def compose_named_mobjects(
    scene: Any,
    namespace: dict[str, Any],
    *,
    locked_ids: Iterable[str] | None = None,
    readable_ids: Iterable[str] | None = None,
    title_ids: Iterable[str] | None = None,
    equation_ids: Iterable[str] | None = None,
    raise_if_overcrowded: bool = False,
) -> LayoutReport:
    """Arrange independent labels/equations. Locked physics objects stay put."""
    del scene  # Scene is accepted so callers can pass `self` uniformly.
    items = {
        name: value
        for name, value in namespace.items()
        if isinstance(name, str)
        and name not in _SKIP_NAMES
        and not name.startswith("__")
        and mobject_aabb(value) is not None
    }
    report = LayoutReport()
    if len(items) < 2:
        return report

    locked = set(locked_ids or ())
    readable = set(readable_ids or ())
    title_ids = set(title_ids or ())
    equation_ids = set(equation_ids or ())
    for name, mob in items.items():
        if name not in locked and not is_annotation(mob) and not is_graph(mob):
            locked.add(name)
        if is_annotation(mob) and name.lower() in {"title", "heading"}:
            title_ids.add(name)

    regions = default_regions()
    frame = regions["frame"].box
    annotations = {name: mob for name, mob in items.items() if is_annotation(mob) and name not in locked}
    hosts = [mob for name, mob in items.items() if name in locked or not is_annotation(mob)]

    # 1. Titles occupy the top band.
    for name in title_ids:
        if name not in annotations:
            continue
        _move_to_region(annotations[name], regions["title"], frame)
        report.moved.append(name)

    # 2. Equations stack in the annotation region, not over the demonstration.
    equation_names = [name for name in annotations if name in equation_ids]
    for slot, name in zip(_region_slots(regions["equation"], max(1, len(equation_names))), equation_names):
        label = annotations[name]
        box = mobject_aabb(label)
        if box is not None:
            _shift(label, slot - box.center)
        _clamp_to_frame(label, regions["equation"].box)
        _clamp_to_frame(label, frame)
        report.moved.append(name)

    # 3. Remaining labels sit near their hosts, ignoring plot-area graphs.
    for name, label in list(annotations.items()):
        if name in title_ids or name in equation_ids:
            continue
        host = _nearest_parent(label, hosts)
        blockers = _placement_blockers(
            label,
            host,
            [mob for other, mob in items.items() if other != name and mob is not host],
        )
        if _nudge_label(label, host, blockers, frame) or _clamp_to_frame(label, frame):
            report.moved.append(name)

    # 4. Separate remaining annotation/annotation overlaps without collapsing to one slot.
    _repel_annotations(annotations, frame, report)
    for left, right in overlapping_pairs(annotations):
        if left in locked or right in locked:
            continue
        prefer = left if left not in readable else right
        host = _nearest_parent(items[prefer], hosts)
        blockers = _placement_blockers(items[prefer], host, [mob for n, mob in items.items() if n != prefer])
        _nudge_label(items[prefer], host, blockers, frame)
        report.moved.append(prefer)

    # 5. Unassociated secondary text that still covers a moving particle goes aside.
    for name, label in annotations.items():
        if name in title_ids or name in equation_ids:
            continue
        box = mobject_aabb(label)
        if box is None:
            continue
        host = _nearest_parent(label, hosts)
        motion_hit = False
        for candidate in hosts:
            if is_graph(candidate) or is_vector_like(candidate) or is_annotation(candidate):
                continue
            host_box = mobject_aabb(candidate)
            if host_box is None or intentional_overlap(label, candidate):
                continue
            # Associated labels may sit beside their host; only coverings of
            # unrelated motion objects are relocated.
            if host is candidate:
                continue
            if box.intersects(host_box.inflate(0.05)):
                motion_hit = True
                break
        if motion_hit:
            _move_to_region(label, regions["equation"], frame)
            _nudge_label(label, None, _placement_blockers(label, None, [mob for n, mob in items.items() if n != name]), frame)
            _clamp_to_frame(label, frame)
            if name not in report.moved:
                report.moved.append(name)
    _repel_annotations(annotations, frame, report)

    # 6. Scale readable text only as a last local fix, never below MIN_TEXT_SCALE.
    for name, label in annotations.items():
        box = mobject_aabb(label)
        if box is None:
            continue
        still_hits = any(
            other_box is not None
            and not intentional_overlap(label, other)
            and box.inflate(0.05).intersects(other_box)
            for other_name, other in items.items()
            if other_name != name and is_annotation(other)
            for other_box in [mobject_aabb(other)]
        )
        if still_hits:
            current = getattr(label, "height", 1.0) or 1.0
            factor = max(MIN_TEXT_SCALE, 0.88)
            if current * factor >= 0.28:
                _scale(label, factor)
                report.scaled.append(name)
                _clamp_to_frame(label, frame)

    remaining = overlapping_pairs(annotations)
    clipped = out_of_frame(items)
    if remaining or clipped:
        readable_problems = [pair for pair in remaining if pair[0] in readable or pair[1] in readable]
        message = (
            "Scene composition is overcrowded after automatic layout. "
            f"Overlapping annotations: {remaining or 'none'}; clipped: {clipped or 'none'}. "
            "Split the demonstration, equation, and graph into sequential shots."
        )
        report.issues.append(message)
        report.overcrowded = bool(readable_problems or clipped)
        if raise_if_overcrowded and report.overcrowded:
            raise ValueError(message)
    return report


def compose_scene_locals(scene: Any, namespace: dict[str, Any], **kwargs: Any) -> LayoutReport:
    return compose_named_mobjects(scene, namespace, **kwargs)
