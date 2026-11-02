"""Regression A: dense annotations must stay readable without overlaps."""

from manim import *
from pipeline.cinematic import acceleration_arrow, vector_arrow
from pipeline.layout import compose_named_mobjects, out_of_frame, overlapping_pairs
from pipeline.math_rendering import create_content, create_math, ensure_pipeline_tex_template
from pipeline.visual_qc import validate_scene_composition

ensure_pipeline_tex_template()


class DenseAnnotationsScene(MovingCameraScene):
    def construct(self):
        axes = Axes(x_range=[0, 8, 2], y_range=[0, 5, 1], x_length=7, y_length=4, tips=False, axis_config={"include_numbers": False}).to_edge(LEFT, buff=0.6)
        ball = Dot(color=YELLOW).move_to(axes.c2p(1.5, 2.2))
        trail = axes.plot(lambda x: 2.2 + 0.35 * x - 0.05 * x * x, x_range=[1.5, 6.5], color=BLUE)
        velocity = vector_arrow(ball.get_center(), np.array([1.2, 0.6, 0.0]), color=GREEN)
        accel = acceleration_arrow(ball.get_center(), scale=0.55)
        v_label = create_content("v", font_size=28, color=GREEN).next_to(velocity, UP, buff=0.12)
        a_label = create_content("a", font_size=28, color=ORANGE).next_to(accel, RIGHT, buff=0.12)
        title = create_content("Dense annotations", font_size=32).to_edge(UP)
        eq = create_math(r"\vec{a}=d\vec{v}/dt").scale(0.9).to_edge(RIGHT)
        self.add(axes, trail, ball, velocity, accel, v_label, a_label, title, eq)
        report = compose_named_mobjects(
            self,
            {
                "axes": axes, "ball": ball, "trail": trail, "velocity": velocity, "accel": accel,
                "v_label": v_label, "a_label": a_label, "title": title, "eq": eq,
            },
            locked_ids=["axes", "ball", "trail", "velocity", "accel"],
            readable_ids=["v_label", "a_label", "eq", "title"],
            title_ids=["title"],
            equation_ids=["eq"],
        )
        issues = validate_scene_composition(self, {
            "v_label": v_label, "a_label": a_label, "title": title, "eq": eq, "axes": axes,
        })
        overlaps = overlapping_pairs({"v_label": v_label, "a_label": a_label, "title": title, "eq": eq})
        clipped = out_of_frame({"v_label": v_label, "a_label": a_label, "title": title, "eq": eq, "axes": axes})
        if overlaps or clipped:
            raise ValueError(f"Dense annotation layout failed: overlaps={overlaps} clipped={clipped} report={report} qc={issues}")
        self.wait(0.8)
