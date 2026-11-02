"""Regression E: wide → push → track → pull → equation from live camera state."""

from manim import *
from pipeline.cinematic import CameraController, CameraShot, animate_physical
from pipeline.layout import compose_named_mobjects
from pipeline.math_rendering import create_math, ensure_pipeline_tex_template
from pipeline.physics import Projectile

ensure_pipeline_tex_template()


class ConsecutiveShotsScene(MovingCameraScene):
    def construct(self):
        model = Projectile(x0=-4.5, y0=-1.2, speed=9.5, angle_degrees=42)
        points = model.trajectory(45)
        coords = [np.array([x, y, 0.0]) for x, y in points]
        ball = Dot(color=YELLOW, radius=0.11).move_to(coords[0])
        ground = Line(LEFT * 7, RIGHT * 7).shift(DOWN * 1.4).set_color(GREY)
        equation = create_math(r"y=y_0+v_0\sin\theta t-\frac{1}{2}gt^2").scale(0.75)
        equation.set_opacity(0)
        self.add(ground, ball, equation)

        compose_named_mobjects(
            self,
            {"ball": ball, "ground": ground, "equation": equation},
            locked_ids=["ball", "ground"],
            equation_ids=["equation"],
        )

        controller = CameraController.ensure(self)
        controller.register("ball", ball)
        controller.register("equation", equation)
        controller.attach()
        widths = [controller.capture_state().width]

        class _Tracker:
            duration = 4.0
            def __init__(self):
                self._elapsed = 0.0
            def get_remaining_duration(self):
                return max(0.0, self.duration - self._elapsed)
            def advance(self, dt):
                self._elapsed += dt

        tracker = _Tracker()
        controller.bind_tracker(tracker)
        controller.set_plan([
            CameraShot("push_in", "ball", 0.0, 0.7, profile="camera_smooth"),
            CameraShot("follow_object", "ball", 0.7, 2.0, look_ahead=1.0),
            CameraShot("pull_back", None, 2.7, 0.7, profile="camera_smooth"),
            CameraShot("pan_to", "equation", 3.4, 0.5, profile="camera_smooth"),
        ])

        def advance(dt):
            tracker.advance(dt)
            widths.append(controller.capture_state().width)

        self.add_updater(advance)
        animate_physical(self, ball, coords, duration=2.6)
        equation.set_opacity(1)
        equation.move_to(RIGHT * 3.2 + UP * 1.5)
        self.wait(1.2)
        self.remove_updater(advance)
        controller.finish()

        for previous, current in zip(widths, widths[1:]):
            if abs(current - previous) > 2.5:
                raise ValueError(f"Sudden zoom change detected: {previous} -> {current}")
        if equation.height < 0.25:
            raise ValueError("Final equation is not legible.")
        self.wait(0.2)
