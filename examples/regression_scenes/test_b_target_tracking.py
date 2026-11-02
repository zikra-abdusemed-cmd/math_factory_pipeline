"""Regression B: camera tracks the declared projectile with look-ahead."""

from manim import *
from pipeline.cinematic import CameraController, animate_physical
from pipeline.physics import Projectile


class TargetTrackingScene(MovingCameraScene):
    def construct(self):
        model = Projectile(x0=-5.5, y0=-1.5, speed=11, angle_degrees=38, gravity=9.81)
        points = model.trajectory(61)
        coords = [np.array([x, y, 0.0]) for x, y in points]
        ground = Line(LEFT * 7, RIGHT * 7).shift(DOWN * 1.6).set_color(GREY)
        ball = Dot(color=YELLOW, radius=0.12).move_to(coords[0])
        self.add(ground, ball)

        controller = CameraController.ensure(self)
        controller.register("ball", ball)
        controller.attach()
        controller.start_tracking(ball, object_id="ball", look_ahead=1.0)
        animate_physical(self, ball, coords, duration=2.4)
        controller.stop_tracking()
        controller.finish()

        # After tracking ends, the last applied state should still contain the ball.
        state = CameraController.ensure(self).capture_state() if hasattr(self, "_pipeline_camera_controller") else None
        # Re-capture from the live frame after finish restored hold mode.
        center = self.camera.frame.get_center()
        width = float(self.camera.frame.width)
        delta = ball.get_center() - center
        half_w = width / 2 * 0.85
        half_h = (width * config.frame_height / config.frame_width) / 2 * 0.85
        if abs(delta[0]) > half_w or abs(delta[1]) > half_h:
            raise ValueError(f"Tracked projectile left the safe frame: delta={delta}, width={width}")
        del state
        self.wait(0.3)
