"""Regression C: stop tracking, then hand off to a graph without a camera jump."""

from manim import *
from pipeline.cinematic import CameraController, animate_physical, trajectory_graph, transition_to_graph
from pipeline.math_rendering import ensure_pipeline_tex_template
from pipeline.physics import Projectile

ensure_pipeline_tex_template()


class CameraHandoffScene(MovingCameraScene):
    def construct(self):
        model = Projectile(0, 0, 10, 40)
        points = model.trajectory(41)
        coords = [np.array([p[0] * 0.55 - 3.5, p[1] * 0.55 - 1.0, 0.0]) for p in points]
        ball = Dot(color=YELLOW).move_to(coords[0])
        physical = VGroup(ball)
        self.add(physical)

        controller = CameraController.ensure(self)
        controller.register("ball", ball)
        controller.attach()
        home = controller.capture_state()
        controller.start_tracking(ball, object_id="ball")
        animate_physical(self, ball, coords, duration=1.6)
        controller.stop_tracking()

        axes, graph = trajectory_graph([(p[0], p[1]) for p in points])
        graph_group = VGroup(axes, graph).scale(0.75).to_edge(RIGHT)
        before = controller.capture_state()
        transition_to_graph(self, physical, graph_group, duration=1.2)
        after = controller.capture_state()
        # No abrupt reset: width should ease toward home, not snap to an unrelated zoom.
        if abs(after.width - before.width) > max(4.0, home.width):
            raise ValueError(f"Camera width jumped during handoff: {before.width} -> {after.width}")
        if abs(after.width - home.width) > 0.35 * home.width:
            raise ValueError(f"Handoff did not settle near home width: {after.width} vs {home.width}")
        controller.finish()
        self.wait(0.4)
