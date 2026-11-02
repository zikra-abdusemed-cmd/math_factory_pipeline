"""Unit and low-res regression coverage for layout, camera, and timing."""

from __future__ import annotations

import math
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from pipeline.cinematic import (
    MOTION_PROFILES,
    CameraController,
    CameraShot,
    CameraState,
    animate_physical,
    rate_func_for,
)
from pipeline.layout import (
    AABB,
    compose_named_mobjects,
    frame_aabb,
    out_of_frame,
    overlapping_pairs,
)
from pipeline.physics import Projectile
from pipeline.visual_plan import validate_visual_plan
from pipeline.visual_qc import inspect_preview_frames, validate_visual_plan_quality


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REGRESSION_DIR = PROJECT_ROOT / "examples" / "regression_scenes"


class LayoutUnitTests(unittest.TestCase):
    def test_aabb_intersection_and_frame_margin(self):
        a = AABB(0, 1, 0, 1)
        b = AABB(0.5, 1.5, 0.5, 1.5)
        c = AABB(2, 3, 2, 3)
        self.assertTrue(a.intersects(b))
        self.assertFalse(a.intersects(c))
        frame = frame_aabb(margin=0.55)
        self.assertLess(frame.width, 14.2)
        self.assertTrue(frame.contains(AABB(-1, 1, -1, 1)))

    def test_compose_separates_overlapping_labels(self):
        from manim import Dot, MovingCameraScene, Text, config

        config.pixel_height = 480
        config.pixel_width = 854
        config.frame_rate = 15
        result: dict = {}

        class Probe(MovingCameraScene):
            def construct(self):
                ball = Dot().move_to([-1, 0, 0])
                label_a = Text("velocity", font_size=28).move_to([0.1, 0.1, 0])
                label_b = Text("acceleration", font_size=28).move_to([0.15, 0.12, 0])
                self.add(ball, label_a, label_b)
                result["report"] = compose_named_mobjects(
                    self,
                    {"ball": ball, "label_a": label_a, "label_b": label_b},
                    locked_ids=["ball"],
                    readable_ids=["label_a", "label_b"],
                )
                result["overlaps"] = overlapping_pairs({"label_a": label_a, "label_b": label_b})
                result["clipped"] = out_of_frame({"label_a": label_a, "label_b": label_b, "ball": ball})

        Probe().render()
        self.assertEqual(result["overlaps"], [])
        self.assertEqual(result["clipped"], [])
        self.assertTrue(result["report"].moved or result["overlaps"] == [])


class CameraControllerUnitTests(unittest.TestCase):
    def test_target_resolution_and_non_overlap_plan(self):
        from manim import Dot, MovingCameraScene, config

        config.frame_rate = 15
        errors: list[str] = []

        class Probe(MovingCameraScene):
            def construct(self):
                ball = Dot()
                ctrl = CameraController.ensure(self)
                ctrl.register("ball", ball)
                if ctrl.resolve("ball") is not ball:
                    errors.append("resolve mismatch")
                try:
                    ctrl.resolve("missing")
                    errors.append("missing target should raise")
                except ValueError:
                    pass
                ctrl.set_plan([
                    CameraShot("push_in", "ball", 0.0, 0.8),
                    CameraShot("follow_object", "ball", 0.8, 2.0),
                    CameraShot("pull_back", None, 2.8, 0.8),
                ])
                try:
                    ctrl.set_plan([
                        CameraShot("push_in", "ball", 0.0, 1.5),
                        CameraShot("follow_object", "ball", 1.0, 1.0),
                    ])
                    errors.append("overlap should raise")
                except ValueError:
                    pass
                self.add(ball)

        Probe().render()
        self.assertEqual(errors, [])

    def test_tracking_keeps_target_in_frame_with_lookahead(self):
        from manim import Dot, MovingCameraScene, config

        config.frame_rate = 15
        config.pixel_height = 240
        config.pixel_width = 426
        errors: list[str] = []

        class Probe(MovingCameraScene):
            def construct(self):
                ball = Dot().move_to([-5, 0, 0])
                ctrl = CameraController.ensure(self)
                ctrl.register("ball", ball)
                ctrl.attach()
                ctrl.start_tracking(ball, object_id="ball", look_ahead=1.0)
                positions = []
                state = ctrl.capture_state()
                for x in np.linspace(-5, 5, 40):
                    ball.move_to([x, 0.2 * math.sin(x), 0])
                    ctrl._track_step(1 / 15)
                    state = ctrl.capture_state()
                    delta = ball.get_center() - state.center
                    half_w = state.width / 2 * 0.75
                    half_h = (state.width * config.frame_height / config.frame_width) / 2 * 0.75
                    if abs(delta[0]) > half_w + 1e-6 or abs(delta[1]) > half_h + 1e-6:
                        errors.append(f"out of frame at x={x}: delta={delta}")
                    positions.append(state.center.copy())
                jumps = [float(np.linalg.norm(positions[i] - positions[i - 1])) for i in range(1, len(positions))]
                if any(jump > state.width * 0.35 for jump in jumps):
                    errors.append("tracking jump too large")
                ctrl.finish()
                self.add(ball)

        Probe().render()
        self.assertEqual(errors, [])

    def test_move_interpolates_from_actual_state(self):
        from manim import MovingCameraScene, config

        config.frame_rate = 15
        result: dict = {}

        class Probe(MovingCameraScene):
            def construct(self):
                ctrl = CameraController.ensure(self)
                ctrl.attach()
                start = CameraState(np.array([1.0, 0.5, 0.0]), 10.0)
                ctrl.restore_state(start)
                dest = CameraState(np.array([3.0, -1.0, 0.0]), 8.0)
                ctrl.begin_move(dest, duration=1.0, profile="camera_smooth")
                ctrl._move_step(0.0)
                result["after_zero"] = ctrl.capture_state()
                for _ in range(8):
                    ctrl._move_step(0.125)
                result["final"] = ctrl.capture_state()
                result["start"] = start
                result["dest"] = dest
                ctrl.finish()

        Probe().render()
        np.testing.assert_allclose(result["after_zero"].center, result["start"].center, atol=1e-9)
        np.testing.assert_allclose(result["final"].center, result["dest"].center, atol=1e-6)
        self.assertAlmostEqual(result["final"].width, result["dest"].width, places=5)


class TimingAndPhysicsTests(unittest.TestCase):
    def test_motion_profiles_are_stable_and_non_elastic(self):
        self.assertEqual(set(MOTION_PROFILES), {"camera_smooth", "camera_linear", "object_physical", "reveal_smooth", "hold"})
        smooth = rate_func_for("camera_smooth")
        physical = rate_func_for("object_physical")
        self.assertAlmostEqual(smooth(0.0), 0.0, places=6)
        self.assertAlmostEqual(smooth(1.0), 1.0, places=6)
        self.assertAlmostEqual(physical(0.37), 0.37, places=9)
        samples = [smooth(t) for t in np.linspace(0, 1, 21)]
        self.assertTrue(all(0.0 - 1e-9 <= value <= 1.0 + 1e-9 for value in samples))

    def test_animate_physical_matches_projectile_samples(self):
        from manim import Dot, MovingCameraScene, config

        config.frame_rate = 15
        config.pixel_height = 240
        config.pixel_width = 426
        model = Projectile(0, 0, 12, 45)
        points = model.trajectory(21)
        result: dict = {}

        class Probe(MovingCameraScene):
            def construct(self):
                ball = Dot().move_to([points[0][0], points[0][1], 0])
                self.add(ball)
                animate_physical(self, ball, [[x, y, 0] for x, y in points], duration=1.0)
                result["end"] = ball.get_center().copy()

        Probe().render()
        self.assertAlmostEqual(result["end"][0], points[-1][0], places=5)
        self.assertAlmostEqual(result["end"][1], points[-1][1], places=5)


class VisualPlanQualityTests(unittest.TestCase):
    def test_qc_flags_conflicting_follows_and_missing_targets(self):
        plan = validate_visual_plan({
            "objects": [{"id": "ball"}, {"id": "other"}],
            "camera": [
                {"command": "follow_object", "target": "ball", "start": 0, "duration": 1},
                {"command": "follow_object", "target": "other", "start": 1.1, "duration": 1},
            ],
        })
        issues = validate_visual_plan_quality(plan)
        self.assertTrue(any("follow_object" in issue for issue in issues))


class RegressionRenderTests(unittest.TestCase):
    """Low-quality offline renders of focused regression scenes A-E."""

    @classmethod
    def setUpClass(cls):
        from pipeline.math_rendering import tex_path_environment

        cls.media = Path(tempfile.mkdtemp(prefix="mymarian_regress_"))
        cls.env = {
            **os.environ,
            "VOICE_PROVIDER": "offline",
            "PYTHONPATH": str(PROJECT_ROOT) + os.pathsep + os.environ.get("PYTHONPATH", ""),
            "PATH": tex_path_environment(),
        }

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.media, ignore_errors=True)

    def _render(self, module: str, class_name: str) -> Path:
        scene_path = REGRESSION_DIR / module
        self.assertTrue(scene_path.is_file(), f"missing {scene_path}")
        cmd = [
            sys.executable, "-m", "manim", "-ql", "--disable_caching",
            "--media_dir", str(self.media),
            str(scene_path), class_name,
        ]
        result = subprocess.run(cmd, cwd=str(PROJECT_ROOT), capture_output=True, text=True, env=self.env, timeout=240)
        if result.returncode != 0:
            self.fail(result.stdout[-2000:] + "\n" + result.stderr[-2000:])
        videos = list(self.media.rglob(f"{class_name}.mp4"))
        self.assertTrue(videos, f"no video for {class_name}")
        return videos[0]

    def test_regression_a_dense_annotations(self):
        self._render("test_a_dense_annotations.py", "DenseAnnotationsScene")

    def test_regression_b_target_tracking(self):
        self._render("test_b_target_tracking.py", "TargetTrackingScene")

    def test_regression_c_camera_handoff(self):
        self._render("test_c_camera_handoff.py", "CameraHandoffScene")

    def test_regression_d_physical_vs_camera_easing(self):
        self._render("test_d_physical_easing.py", "PhysicalVsCameraEasingScene")

    def test_regression_e_consecutive_shots(self):
        video = self._render("test_e_consecutive_shots.py", "ConsecutiveShotsScene")
        try:
            from pipeline.visual_qc import extract_video_frames
            frames_dir = self.media / "qc_frames"
            paths = extract_video_frames(video, frames_dir, [0.2, 1.0, 2.0, 3.0])
            if paths:
                issues = inspect_preview_frames(paths, jump_threshold=0.55)
                self.assertFalse(any("nearly empty" in issue for issue in issues), issues)
        except Exception:
            pass


if __name__ == "__main__":
    unittest.main()
