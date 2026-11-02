"""Regression D: camera easing must not distort physical projectile motion."""

from manim import *
from pipeline.cinematic import CameraController, CameraState, animate_physical
from pipeline.physics import Projectile


class PhysicalVsCameraEasingScene(MovingCameraScene):
    def construct(self):
        model = Projectile(0, 0, 12, 45)
        samples = model.trajectory(31)
        coords = [np.array([x * 0.5 - 3.0, y * 0.5 - 1.2, 0.0]) for x, y in samples]
        ball = Dot(color=YELLOW).move_to(coords[0])
        self.add(ball)

        controller = CameraController.ensure(self)
        controller.register("ball", ball)
        controller.attach()
        current = controller.capture_state()
        # Independent camera move while the ball follows linear simulation time.
        controller.begin_move(
            CameraState(np.array([0.5, 0.2, 0.0]), max(6.0, current.width * 0.85)),
            duration=1.5,
            profile="camera_smooth",
        )
        animate_physical(self, ball, coords, duration=1.5)
        controller.finish()

        end = ball.get_center()
        expected = coords[-1]
        if np.linalg.norm(end - expected) > 1e-3:
            raise ValueError(f"Physical endpoint drifted under camera easing: {end} vs {expected}")
        t_end = model.time_to_ground()
        mid_t = t_end / 2
        analytical = model.position(mid_t)
        expected_y = model.vy0 * mid_t - 0.5 * model.gravity * mid_t * mid_t
        if abs(analytical[1] - expected_y) > 1e-9:
            raise ValueError("Analytical mid-flight position inconsistent.")
        self.wait(0.3)
