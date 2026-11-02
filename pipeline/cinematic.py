"""Camera, timing, and physics visual helpers for Manim Community 0.22.

Camera motion is owned by :class:`CameraController`. Object motion that comes
from an analytical model must use :func:`animate_physical` (linear in
simulation time). Cinematic easing never writes the tracked object's position.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, Sequence

import numpy as np
from manim import Arrow, Axes, BLUE, Create, FadeIn, FadeOut, GREEN, ORANGE, ORIGIN, ValueTracker, YELLOW, config, linear, smooth
from manim.utils.rate_functions import ease_in_out_cubic, ease_out_cubic

# Installed Manim 0.22 rate functions. Avoid back/elastic/bounce for camera.
MOTION_PROFILES: dict[str, Callable[[float], float]] = {
    "camera_smooth": ease_in_out_cubic,
    "camera_linear": linear,
    "object_physical": linear,
    "reveal_smooth": smooth,
    "hold": lambda t: 1.0 if t >= 1.0 else 0.0,
}


def rate_func_for(profile: str) -> Callable[[float], float]:
    if profile not in MOTION_PROFILES:
        raise ValueError(f"Unknown motion profile {profile!r}; use one of {sorted(MOTION_PROFILES)}.")
    return MOTION_PROFILES[profile]


def _bounded_duration(duration: float) -> float:
    duration = float(duration)
    if not np.isfinite(duration) or not 0.1 <= duration <= 20:
        raise ValueError("Camera/visual duration must be between 0.1 and 20 seconds.")
    return duration


def _as_point(value: Any) -> np.ndarray:
    if hasattr(value, "get_center"):
        value = value.get_center()
    point = np.asarray(value, dtype=float).reshape(-1)
    if point.size == 2:
        point = np.array([point[0], point[1], 0.0])
    elif point.size != 3:
        raise ValueError("Camera targets must resolve to a 2D or 3D point.")
    if not np.all(np.isfinite(point)):
        raise ValueError("Camera target point is not finite.")
    return point


@dataclass
class CameraState:
    center: np.ndarray
    width: float
    phi: float | None = None
    theta: float | None = None

    def copy(self) -> "CameraState":
        return CameraState(self.center.copy(), float(self.width), self.phi, self.theta)

    def interpolate(self, other: "CameraState", alpha: float) -> "CameraState":
        alpha = float(np.clip(alpha, 0.0, 1.0))
        center = self.center * (1.0 - alpha) + other.center * alpha
        width = self.width * (1.0 - alpha) + other.width * alpha
        phi = theta = None
        if self.phi is not None and other.phi is not None:
            phi = self.phi * (1.0 - alpha) + other.phi * alpha
        if self.theta is not None and other.theta is not None:
            theta = self.theta * (1.0 - alpha) + other.theta * alpha
        return CameraState(center, width, phi, theta)


@dataclass
class CameraShot:
    command: str
    target: str | None = None
    start: float = 0.0
    duration: float = 1.0
    reason: str = ""
    profile: str = "camera_smooth"
    margin: float = 1.18
    look_ahead: float = 0.9
    hold: float = 0.15


class CameraController:
    """Single owner of 2D MovingCamera frame position and width.

    Lifecycle: ``ensure`` → ``register`` → ``attach`` → ``bind_tracker`` /
    ``set_plan`` → updater runs one strategy → ``finish``.
    """

    def __init__(self, scene: Any):
        if not hasattr(scene, "camera") or not hasattr(scene.camera, "frame"):
            raise TypeError("CameraController requires a MovingCameraScene (2D camera.frame).")
        self.scene = scene
        self.targets: dict[str, Any] = {}
        self.plan: list[CameraShot] = []
        self.mode = "hold"
        self.active_target_id: str | None = None
        self.smoothing_tau = 0.22
        self.track_tau = 0.18
        self._attached = False
        self._tracker: Any = None
        self._velocity = np.zeros(3)
        self._last_target_pos: np.ndarray | None = None
        self._home: CameraState | None = None
        self._from_state: CameraState | None = None
        self._to_state: CameraState | None = None
        self._move_elapsed = 0.0
        self._move_duration = 0.0
        self._move_profile = "camera_smooth"
        self._settling = 0.0
        self._finished_shot: CameraShot | None = None
        self._look_ahead = 0.9

    @classmethod
    def ensure(cls, scene: Any) -> "CameraController":
        existing = getattr(scene, "_pipeline_camera_controller", None)
        if isinstance(existing, cls):
            return existing
        controller = cls(scene)
        scene._pipeline_camera_controller = controller
        return controller

    def capture_state(self) -> CameraState:
        frame = self.scene.camera.frame
        return CameraState(frame.get_center().copy(), float(frame.width))

    def restore_state(self, state: CameraState) -> None:
        self._apply_state(state)

    def register(self, object_id: str, mobject: Any) -> None:
        if not object_id:
            raise ValueError("Camera target ids must be non-empty.")
        if mobject is None or not hasattr(mobject, "get_center"):
            raise ValueError(f"Camera target {object_id!r} is not a mobject with get_center().")
        self.targets[object_id] = mobject

    def resolve(self, object_id: str | None) -> Any:
        if not object_id:
            raise ValueError("Camera command is missing a tracking target.")
        if object_id not in self.targets:
            raise ValueError(
                f"Camera target {object_id!r} is not registered. Known ids: {sorted(self.targets)}."
            )
        target = self.targets[object_id]
        if target is None or not hasattr(target, "get_center"):
            raise ValueError(f"Camera target {object_id!r} disappeared or is invalid.")
        return target

    def set_plan(self, events: Iterable[Any]) -> None:
        shots: list[CameraShot] = []
        for event in events:
            if isinstance(event, CameraShot):
                shots.append(event)
                continue
            if isinstance(event, dict):
                command = event["command"]
                target = event.get("target")
                start = float(event.get("start", 0))
                duration = float(event.get("duration", 1))
                reason = str(event.get("reason", ""))
                profile = str(event.get("profile", "camera_smooth"))
                margin = float(event.get("margin", 1.18))
                look_ahead = float(event.get("look_ahead", 0.9))
                hold = float(event.get("hold", 0.15))
            else:
                command = event.command
                target = event.target
                start = float(event.start)
                duration = float(event.duration)
                reason = str(getattr(event, "reason", ""))
                profile = str(getattr(event, "profile", "camera_smooth"))
                margin = float(getattr(event, "margin", 1.18))
                look_ahead = float(getattr(event, "look_ahead", 0.9))
                hold = float(getattr(event, "hold", 0.15))
            if profile not in MOTION_PROFILES:
                profile = "camera_smooth"
            shots.append(CameraShot(
                command=command,
                target=target,
                start=start,
                duration=duration,
                reason=reason,
                profile=profile,
                margin=margin,
                look_ahead=look_ahead,
                hold=hold,
            ))
        shots.sort(key=lambda shot: shot.start)
        for first, second in zip(shots, shots[1:]):
            if first.start + first.duration > second.start + 0.05:
                raise ValueError("Camera shots overlap; only one controller strategy may run at a time.")
        self.plan = shots
        self._finished_shot = None

    def attach(self) -> None:
        if self._attached:
            return
        frame = self.scene.camera.frame
        self._home = self.capture_state()
        frame.add_updater(self._update)
        self._attached = True
        self.mode = "hold"

    def bind_tracker(self, tracker: Any) -> None:
        self._tracker = tracker

    def start_tracking(self, target: Any, *, object_id: str | None = None, look_ahead: float = 0.9) -> None:
        if object_id:
            self.register(object_id, target)
            self.active_target_id = object_id
        else:
            self.targets["__live__"] = target
            self.active_target_id = "__live__"
        self.mode = "track"
        self._look_ahead = look_ahead
        self._last_target_pos = _as_point(target)
        self._velocity = np.zeros(3)
        self._from_state = None
        self._to_state = None

    def stop_tracking(self) -> None:
        self.mode = "hold"
        self.active_target_id = None
        self._last_target_pos = None

    def begin_move(self, destination: CameraState, *, duration: float, profile: str = "camera_smooth") -> None:
        self.mode = "move"
        self._from_state = self.capture_state()
        self._to_state = destination
        self._move_elapsed = 0.0
        self._move_duration = _bounded_duration(duration)
        self._move_profile = profile
        self._settling = 0.0

    def restore_home(self, *, duration: float = 0.8) -> None:
        if self._home is None:
            self._home = CameraState(ORIGIN.copy() if hasattr(ORIGIN, "copy") else np.array(ORIGIN, dtype=float), float(config.frame_width))
        self.stop_tracking()
        self.begin_move(self._home.copy(), duration=duration, profile="camera_smooth")

    def finish(self) -> None:
        self.stop_tracking()
        self._tracker = None
        if self._attached:
            self.scene.camera.frame.remove_updater(self._update)
            self._attached = False
        self.mode = "hold"

    def drive_elapsed(self, elapsed: float, dt: float) -> None:
        shot = self._shot_at(elapsed)
        if shot is None:
            # Between planned shots: finish tracking cleanly, continue an in-flight move.
            if self.mode == "track":
                self.stop_tracking()
            elif self.mode == "move":
                self._move_step(dt)
            return
        if shot is not self._finished_shot:
            # End the previous strategy before transferring ownership.
            if self.mode == "track" and shot.command != "follow_object":
                self.stop_tracking()
            self._enter_shot(shot)
            self._finished_shot = shot
        local = elapsed - shot.start
        if shot.command == "follow_object":
            if self.mode != "track":
                self.mode = "track"
            self._track_step(dt)
            if local >= shot.duration - max(shot.hold, 0.0):
                self.stop_tracking()
        elif shot.command in {"push_in", "pan_to", "pull_back", "transition_to_graph"}:
            if self.mode == "hold" and self._from_state is not None and self._move_elapsed >= self._move_duration:
                return
            if self.mode != "move":
                self.mode = "move"
            self._move_step(dt)
        elif shot.command == "freeze_and_annotate":
            self.stop_tracking()
            self.mode = "hold"
        else:
            self.mode = "hold"

    def _enter_shot(self, shot: CameraShot) -> None:
        current = self.capture_state()
        if shot.command == "follow_object":
            target = self.resolve(shot.target)
            self.start_tracking(target, object_id=shot.target, look_ahead=shot.look_ahead)
            return
        if shot.command == "push_in":
            target = self.resolve(shot.target)
            dest = self._framed_state(target, current, zoom=1.22, margin=shot.margin, look_ahead=False)
            self.begin_move(dest, duration=shot.duration, profile=shot.profile)
            return
        if shot.command == "pan_to":
            target = self.resolve(shot.target)
            dest = CameraState(_as_point(target), current.width)
            self.begin_move(dest, duration=shot.duration, profile=shot.profile)
            return
        if shot.command == "pull_back":
            home = self._home.copy() if self._home is not None else CameraState(np.array([0.0, 0.0, 0.0]), float(config.frame_width))
            self.stop_tracking()
            self.begin_move(home, duration=shot.duration, profile=shot.profile)
            return
        if shot.command == "transition_to_graph":
            self.stop_tracking()
            home = self._home.copy() if self._home is not None else current
            self.begin_move(home, duration=min(shot.duration, 1.0), profile="camera_smooth")

    def _shot_at(self, elapsed: float) -> CameraShot | None:
        active: CameraShot | None = None
        for shot in self.plan:
            if shot.start - 1e-6 <= elapsed <= shot.start + shot.duration + 1e-6:
                active = shot
        return active

    def _update(self, frame: Any, dt: float) -> None:
        del frame
        dt = float(dt)
        if not np.isfinite(dt) or dt < 0:
            dt = 0.0
        if self._tracker is not None:
            remaining = float(self._tracker.get_remaining_duration())
            elapsed = max(0.0, float(self._tracker.duration) - remaining)
            self.drive_elapsed(elapsed, dt)
            return
        if self.mode == "track":
            self._track_step(dt)
        elif self.mode == "move":
            self._move_step(dt)

    def _apply_state(self, state: CameraState) -> None:
        frame = self.scene.camera.frame
        frame.move_to(state.center)
        width = max(4.0, min(float(state.width), float(config.frame_width) * 1.35))
        frame.set(width=width)

    def _framed_state(
        self,
        target: Any,
        current: CameraState,
        *,
        zoom: float,
        margin: float,
        look_ahead: bool,
    ) -> CameraState:
        center = _as_point(target)
        if look_ahead:
            center = center + self._look_ahead_offset(center, current.width)
        width = max(4.0, current.width / max(1.0, float(zoom)))
        width = max(width, self._width_for_context(target, current.width, margin))
        return CameraState(center, width)

    def _width_for_context(self, target: Any, current_width: float, margin: float) -> float:
        companions = [target]
        for mob in self.targets.values():
            if mob is target:
                continue
            if is_nearby_annotation(mob, target):
                companions.append(mob)
        min_w, min_h = 0.0, 0.0
        for mob in companions:
            try:
                min_w = max(min_w, float(mob.width))
                min_h = max(min_h, float(mob.height))
            except Exception:
                continue
        aspect = float(config.frame_width) / float(config.frame_height)
        needed = max(min_w * margin, min_h * margin * aspect, 4.0)
        # Tracking must not zoom-to-fit every frame; only enlarge if the subject is clipped.
        if needed > current_width * 1.05:
            return min(needed, current_width * 1.12)
        return current_width

    def _look_ahead_offset(self, position: np.ndarray, frame_width: float) -> np.ndarray:
        speed = float(np.linalg.norm(self._velocity[:2]))
        if speed < 0.12:
            return np.zeros(3)
        direction = self._velocity.copy()
        direction[2] = 0.0
        direction = direction / (np.linalg.norm(direction) or 1.0)
        distance = float(np.clip(0.35 * frame_width * min(1.0, speed / 4.0), 0.35, 1.6))
        return direction * distance * float(getattr(self, "_look_ahead", 0.9))

    def _track_step(self, dt: float) -> None:
        target_id = self.active_target_id or ("__live__" if "__live__" in self.targets else None)
        if target_id is None:
            self.stop_tracking()
            return
        try:
            target = self.resolve(target_id) if target_id != "__live__" else self.targets["__live__"]
        except ValueError:
            self.stop_tracking()
            return
        pos = _as_point(target)
        step = float(dt) if np.isfinite(dt) and dt > 0 else 0.0
        if step <= 1e-9:
            # Apply the current framing without inventing a time step.
            current = self.capture_state()
            desired_center = pos + self._look_ahead_offset(pos, current.width)
            desired_center = self._keep_target_in_frame(pos, desired_center, current.width)
            self._apply_state(CameraState(desired_center, current.width))
            return
        if self._last_target_pos is not None:
            raw_v = (pos - self._last_target_pos) / step
            blend = 1.0 - float(np.exp(-step / 0.12))
            self._velocity = self._velocity + blend * (raw_v - self._velocity)
        self._last_target_pos = pos.copy()
        current = self.capture_state()
        desired_center = pos + self._look_ahead_offset(pos, current.width)
        desired_center = self._keep_target_in_frame(pos, desired_center, current.width)
        tau = self.track_tau
        alpha = 1.0 - float(np.exp(-step / max(tau, 1e-3)))
        center = current.center + alpha * (desired_center - current.center)
        width = current.width
        needed = self._width_for_context(target, width, 1.12)
        if needed > width:
            width = current.width + alpha * (needed - current.width)
        self._apply_state(CameraState(center, width))

    def _keep_target_in_frame(self, target_pos: np.ndarray, camera_center: np.ndarray, width: float) -> np.ndarray:
        height = width * float(config.frame_height) / float(config.frame_width)
        # Inner safe rectangle so the subject never sits on the cut line.
        half_w = width / 2 * 0.62
        half_h = height / 2 * 0.62
        delta = target_pos - camera_center
        shift = np.zeros(3)
        if delta[0] > half_w:
            shift[0] = delta[0] - half_w
        elif delta[0] < -half_w:
            shift[0] = delta[0] + half_w
        if delta[1] > half_h:
            shift[1] = delta[1] - half_h
        elif delta[1] < -half_h:
            shift[1] = delta[1] + half_h
        return camera_center + shift

    def _move_step(self, dt: float) -> None:
        if self._from_state is None or self._to_state is None:
            self.mode = "hold"
            return
        step = float(dt) if np.isfinite(dt) and dt > 0 else 0.0
        if step <= 1e-9:
            alpha = rate_func_for(self._move_profile)(
                min(max(self._move_elapsed / max(self._move_duration, 1e-6), 0.0), 1.0)
            )
            self._apply_state(self._from_state.interpolate(self._to_state, alpha))
            return
        self._move_elapsed += step
        progress = min(max(self._move_elapsed / max(self._move_duration, 1e-6), 0.0), 1.0)
        alpha = rate_func_for(self._move_profile)(progress)
        self._apply_state(self._from_state.interpolate(self._to_state, alpha))
        if progress >= 1.0:
            self._settling += step
            self.mode = "hold" if self._settling >= 0.05 else "move"


def is_nearby_annotation(mobject: Any, target: Any, radius: float = 2.2) -> bool:
    from pipeline.layout import is_annotation, mobject_aabb

    if not is_annotation(mobject):
        return False
    box = mobject_aabb(mobject)
    if box is None:
        return False
    return float(np.linalg.norm(box.center - _as_point(target))) <= radius


def push_in(scene: Any, target: Any, *, scale: float = 1.25, duration: float = 1.2) -> None:
    """Move a 2D MovingCameraScene frame toward a target, within safe bounds."""
    if not 1.0 <= float(scale) <= 2.0:
        raise ValueError("push_in scale must be between 1.0 and 2.0.")
    controller = CameraController.ensure(scene)
    controller.attach()
    current = controller.capture_state()
    destination = controller._framed_state(target, current, zoom=float(scale), margin=1.18, look_ahead=False)
    controller.begin_move(destination, duration=_bounded_duration(duration), profile="camera_smooth")
    scene.wait(_bounded_duration(duration))


def pull_back(scene: Any, *, scale: float = 1.25, duration: float = 1.2) -> None:
    if not 1.0 <= float(scale) <= 2.0:
        raise ValueError("pull_back scale must be between 1.0 and 2.0.")
    controller = CameraController.ensure(scene)
    controller.attach()
    controller.restore_home(duration=_bounded_duration(duration))
    scene.wait(_bounded_duration(duration))


def pan_to(scene: Any, target: Any, *, duration: float = 1.0) -> None:
    controller = CameraController.ensure(scene)
    controller.attach()
    current = controller.capture_state()
    controller.begin_move(CameraState(_as_point(target), current.width), duration=_bounded_duration(duration), profile="camera_smooth")
    scene.wait(_bounded_duration(duration))


def follow_object(scene: Any, target: Any, *, duration: float, samples: int = 24) -> None:
    """Track a moving mobject with the shared 2D camera controller."""
    del samples
    controller = CameraController.ensure(scene)
    controller.attach()
    controller.start_tracking(target)
    scene.wait(_bounded_duration(duration))
    controller.stop_tracking()


class _FollowFrame:
    """Deprecated lerp adapter retained for import compatibility."""

    def __new__(cls, frame: Any, target: Any, start: Any, samples: int):
        raise RuntimeError(
            "Direct _FollowFrame camera animations conflict with CameraController. "
            "Use follow_object() or CameraController.start_tracking()."
        )


def orbit_scene(scene: Any, *, phi: float, theta: float, duration: float = 1.5) -> None:
    """Move a ThreeDScene camera to a deliberate orientation."""
    if not hasattr(scene, "move_camera"):
        raise TypeError("orbit_scene requires a Manim ThreeDScene.")
    scene.move_camera(phi=float(phi), theta=float(theta), run_time=_bounded_duration(duration), rate_func=ease_in_out_cubic)


def freeze_and_annotate(scene: Any, annotations: list[Any], *, duration: float = 0.8) -> None:
    """Hold the current composition while labels/vectors/equations appear."""
    controller = getattr(scene, "_pipeline_camera_controller", None)
    if isinstance(controller, CameraController):
        controller.stop_tracking()
        controller.mode = "hold"
    scene.play(*[FadeIn(item) for item in annotations], run_time=_bounded_duration(duration), rate_func=smooth)


def reveal_system(scene: Any, elements: list[Any], *, duration: float = 1.2) -> None:
    if not elements:
        return
    scene.play(*[Create(item) for item in elements], run_time=_bounded_duration(duration), rate_func=smooth)


def transition_to_graph(scene: Any, physical_group: Any, graph_group: Any, *, duration: float = 1.2) -> None:
    duration = _bounded_duration(duration)
    controller = getattr(scene, "_pipeline_camera_controller", None)
    if isinstance(controller, CameraController):
        controller.stop_tracking()
        controller.restore_home(duration=max(0.35, duration * 0.4))
        scene.wait(max(0.35, duration * 0.4))
    remaining = max(0.4, duration * 0.6)
    scene.play(FadeOut(physical_group), FadeIn(graph_group), run_time=remaining, rate_func=smooth)


def trajectory_trail(points: list[Any], *, color: Any = BLUE) -> Any:
    from manim import VMobject
    trail = VMobject(color=color, stroke_width=3)
    if len(points) < 2:
        raise ValueError("A trajectory trail requires at least two points.")
    trail.set_points_as_corners(points)
    return trail


def vector_arrow(origin: Any, vector: Any, *, color: Any = YELLOW, scale: float = 1.0) -> Arrow:
    vector = np.asarray(vector, dtype=float) * float(scale)
    if vector.shape != (3,) or not np.all(np.isfinite(vector)) or np.linalg.norm(vector) == 0:
        raise ValueError("Vector arrow requires a finite, nonzero 3D vector.")
    return Arrow(origin, np.asarray(origin, dtype=float) + vector, color=color, buff=0)


def acceleration_arrow(origin: Any, gravity: float = 9.81, *, scale: float = 0.35) -> Arrow:
    if not np.isfinite(gravity) or gravity <= 0 or not 0 < scale <= 2:
        raise ValueError("Gravity must be positive and arrow scale must be in (0, 2].")
    return vector_arrow(origin, np.array([0.0, -1.0, 0.0]), color=ORANGE, scale=scale)


def trajectory_graph(points: list[tuple[float, float]], *, x_range: tuple[float, float] | None = None,
                     y_range: tuple[float, float] | None = None) -> tuple[Axes, Any]:
    if len(points) < 2 or not all(np.isfinite(v) for p in points for v in p):
        raise ValueError("Graph requires at least two finite trajectory points.")
    xs, ys = zip(*points)
    x_range = x_range or (min(xs), max(xs) + 0.1)
    y_range = y_range or (min(0.0, min(ys)), max(ys) + 0.1)
    axes = Axes(
        x_range=[x_range[0], x_range[1], max((x_range[1] - x_range[0]) / 5, 0.1)],
        y_range=[y_range[0], y_range[1], max((y_range[1] - y_range[0]) / 5, 0.1)],
        tips=False,
        # Number labels use TeX in Manim CE; keep graphs usable on BasicTeX.
        axis_config={"include_numbers": False},
    )
    return axes, axes.plot_line_graph(list(xs), list(ys), add_vertex_dots=False, line_color=GREEN)


def animate_physical(
    scene: Any,
    mobject: Any,
    points: Sequence[Any],
    *,
    duration: float,
) -> None:
    """Move ``mobject`` along sampled physical points with linear simulation time.

    Camera easing must not be passed here. Playback duration is independent of
    the physical seconds represented by ``points``.
    """
    if len(points) < 2:
        raise ValueError("Physical motion requires at least two sampled positions.")
    coords = [_as_point(point) for point in points]
    tracker = ValueTracker(0)
    count = len(coords) - 1

    def update_mobject(mob: Any, _dt: float = 0.0) -> None:
        alpha = float(np.clip(tracker.get_value(), 0.0, 1.0))
        index = alpha * count
        lo = int(np.floor(index))
        hi = min(lo + 1, count)
        frac = index - lo
        mob.move_to(coords[lo] * (1.0 - frac) + coords[hi] * frac)

    mobject.add_updater(update_mobject)
    update_mobject(mobject)
    scene.play(tracker.animate.set_value(1.0), run_time=_bounded_duration(duration), rate_func=linear)
    mobject.remove_updater(update_mobject)
    mobject.move_to(coords[-1])
