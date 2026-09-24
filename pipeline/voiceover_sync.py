"""Keep Manim animation timelines from overrunning voiceover audio.

manim-voiceover starts audio when a ``voiceover`` block opens and waits for
any *underfill* when the block exits.  It does **not** stop animations that
overrun ``tracker.duration``.  Equal splits like ``run_time = duration / N``
often round up to whole frames, so N plays last longer than the narration and
the picture keeps changing after the voice has finished.

:class:`SyncedVoiceoverScene` caps every ``play`` / ``wait`` to the active
tracker's remaining duration so video cannot run ahead of audio.  Generated
scenes are rewritten to subclass this automatically.
"""

from __future__ import annotations

from typing import Any

from manim import config
from manim_voiceover import VoiceoverScene
from manim_voiceover.tracker import VoiceoverTracker


def _frame_duration() -> float:
    rate = float(getattr(config, "frame_rate", 0) or 0)
    if rate <= 0:
        rate = 15.0
    return 1.0 / rate


class SyncedVoiceoverScene(VoiceoverScene):
    """VoiceoverScene that never lets animations outlast the spoken audio."""

    def play(self, *args: Any, **kwargs: Any) -> None:
        tracker = getattr(self, "current_tracker", None)
        if tracker is not None:
            kwargs = dict(kwargs)
            kwargs["run_time"] = self._capped_run_time(tracker, kwargs.get("run_time"))
            if kwargs["run_time"] is None:
                return
        super().play(*args, **kwargs)

    def wait(self, duration: float = 1.0, **kwargs: Any) -> None:
        tracker = getattr(self, "current_tracker", None)
        if tracker is not None:
            capped = self._capped_run_time(tracker, duration)
            if capped is None:
                return
            duration = capped
        super().wait(duration, **kwargs)

    def _capped_run_time(self, tracker: VoiceoverTracker, requested: float | None) -> float | None:
        """Return a run_time that fits in the remaining narration, or None to skip."""
        remaining = float(tracker.get_remaining_duration())
        frame = _frame_duration()
        if remaining < frame:
            return None

        if requested is None:
            # Manim's default animation length is 1 second when unspecified.
            requested = 1.0

        capped = min(float(requested), remaining)
        if capped < frame:
            return None
        return capped


def play_remaining(
    scene: VoiceoverScene,
    tracker: VoiceoverTracker,
    *animations: Any,
    steps_left: int = 1,
    **kwargs: Any,
) -> None:
    """Play animations using an equal share of whatever narration time remains."""
    steps_left = max(1, int(steps_left))
    remaining = float(tracker.get_remaining_duration())
    frame = _frame_duration()
    run_time = remaining / steps_left
    if run_time < frame or not animations:
        return
    scene.play(*animations, run_time=run_time, **kwargs)
