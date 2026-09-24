"""Manual/renderable regression scene for the universal math renderer.

Run with: venv/bin/manim -ql tests/manim_math_validation.py UniversalMathValidation
"""

from manim import DOWN, Scene, VGroup

from pipeline.math_rendering import create_content, ensure_pipeline_tex_template


class UniversalMathValidation(Scene):
    def construct(self):
        ensure_pipeline_tex_template()
        samples = [
            r"A \cup B \qquad A \cap B",
            r"x^2 + 5x + 6 = 0",
            r"\frac{a+b}{c}",
            r"\angle ABC = 60^\circ",
            r"f(x)=x^2+1",
            r"\sin^2\theta+\cos^2\theta=1",
            r"\bar{x}=\frac{\sum x}{n}",
            r"\frac{dy}{dx}=2x",
            r"Therefore, the area is A = \pi r^2.",
        ]
        rows = VGroup(*(create_content(sample, font_size=34) for sample in samples))
        rows.arrange(DOWN, buff=0.18).scale_to_fit_height(6.7)
        self.add(rows)
