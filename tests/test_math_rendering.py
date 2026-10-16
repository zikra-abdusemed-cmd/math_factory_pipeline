import ast
import tempfile
import unittest
from pathlib import Path

from manim import config
from manim import Line

from pipeline.math_rendering import (
    create_content,
    create_math,
    line_intersection,
    ensure_math_renderer_available,
    ensure_pipeline_tex_template,
    is_math_content,
    split_mixed_content,
)
from pipeline.scene_codegen import normalize_visible_content


class MathRenderingTests(unittest.TestCase):
    def test_cross_topic_expressions_are_math(self):
        expressions = [
            r"A \cup B",
            r"A \cap B",
            "x^2 + 5x + 6 = 0",
            r"\frac{a+b}{c}",
            r"\angle ABC = 60^\circ",
            "f(x)=x^2+1",
            r"\sin^2\theta+\cos^2\theta=1",
            r"\bar{x}=\frac{\sum x}{n}",
            r"\frac{dy}{dx}=2x",
        ]
        for expression in expressions:
            with self.subTest(expression=expression):
                self.assertTrue(is_math_content(expression))
                self.assertEqual(split_mixed_content(expression)[0].kind, "math")

    def test_mixed_sentence_is_split(self):
        parts = split_mixed_content(r"Therefore, the area is A = \pi r^2.")
        self.assertEqual([part.kind for part in parts], ["text", "math", "text"])
        self.assertEqual(parts[1].value, r"A = \pi r^2")

    def test_generated_text_calls_are_rewritten_even_when_model_disobeys(self):
        source = (
            "from manim import *\n"
            "from manim_voiceover import VoiceoverScene\n"
            "from tts_services.active_service import get_speech_service\n"
            "class Demo(VoiceoverScene):\n"
            "    def construct(self):\n"
            "        x = Text(r'\\frac{a}{b}')\n"
            "        y = Text('ordinary')\n"
        )
        normalized = normalize_visible_content(source)
        tree = ast.parse(normalized)
        calls = [
            node.func.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        ]
        self.assertNotIn("Text", calls)
        self.assertGreaterEqual(calls.count("create_content"), 2)
        self.assertIn("ensure_pipeline_tex_template", calls)
        self.assertIn("from math import", normalized)
        self.assertIn("import numpy as np", normalized)
        class_defs = [node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
        self.assertEqual(class_defs[0].bases[0].id, "SyncedVoiceoverScene")
        self.assertIn("from pipeline.voiceover_sync import SyncedVoiceoverScene", normalized)
        self.assertNotIn("from manim_voiceover import VoiceoverScene", normalized)

    def test_mathtex_calls_are_rewritten_to_create_math(self):
        source = "from manim import *\nx = MathTex(r'x^2')\n"
        normalized = normalize_visible_content(source)
        tree = ast.parse(normalized)
        calls = [
            node.func.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        ]
        self.assertNotIn("MathTex", calls)
        self.assertIn("create_math", calls)

    def test_sector_outer_radius_is_normalized_for_manim(self):
        normalized = normalize_visible_content("from manim import *\nwedge = Sector(outer_radius=2, angle=1)\n")
        tree = ast.parse(normalized)
        sector_call = next(
            node for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "Sector"
        )
        self.assertIn("radius", [keyword.arg for keyword in sector_call.keywords])
        self.assertNotIn("outer_radius", [keyword.arg for keyword in sector_call.keywords])

    def test_get_intersection_calls_use_pipeline_helper(self):
        normalized = normalize_visible_content(
            "from manim import *\np = first.get_intersection(second)\n"
        )
        tree = ast.parse(normalized)
        call = next(
            node for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "pipeline_line_intersection"
        )
        self.assertEqual(len(call.args), 2)

    def test_line_intersection_returns_crossing_point(self):
        point = line_intersection(Line([-1, -1, 0], [1, 1, 0]), Line([-1, 1, 0], [1, -1, 0]))
        self.assertAlmostEqual(point[0], 0)
        self.assertAlmostEqual(point[1], 0)

    def test_python_consumed_tex_escapes_are_restored(self):
        source = r'''from manim import *
x = MathTex("\frac{a}{b}")
y = MathTex("\theta + \frac{1}{2}")
'''
        normalized = normalize_visible_content(source)
        tree = ast.parse(normalized)
        calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "create_math"]
        self.assertEqual([call.args[0].value for call in calls], [r"\frac{a}{b}", r"\theta + \frac{1}{2}"])

    def test_invalid_unicode_tex_escapes_are_made_raw_before_parsing(self):
        source = r'''from manim import *
x = Tex("\xi + \underbrace{a}_{b}")
'''
        normalized = normalize_visible_content(source)
        tree = ast.parse(normalized)
        calls = [
            node for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "create_math"
        ]
        self.assertEqual(calls[0].args[0].value, r"\xi + \underbrace{a}_{b}")

    def test_create_math_accepts_multiple_mathtex_parts(self):
        ensure_pipeline_tex_template()
        mob = create_math(r"f(x)=b^x", r"\qquad b>0,\ b\ne 1", font_size=32)
        self.assertEqual(len(mob), 2)
        self.assertGreater(mob.width, 0)

    def test_future_import_stays_first(self):
        normalized = normalize_visible_content(
            '"""scene"""\nfrom __future__ import annotations\nfrom manim import *\nx = Text("x^2")\n'
        )
        compile(normalized, "<generated-scene>", "exec")

    def test_pipeline_template_avoids_standalone_and_renders(self):
        ensure_math_renderer_available()
        media = Path(tempfile.mkdtemp())
        config.media_dir = str(media)
        ensure_pipeline_tex_template()
        samples = [
            r"A \cup B",
            r"A \cap B",
            r"x^2 + 5x + 6 = 0",
            r"\frac{a+b}{c}",
            r"\angle ABC = 60^\circ",
            r"f(x)=x^2+1",
            r"\sin^2\theta+\cos^2\theta=1",
            r"\bar{x}=\frac{\sum x}{n}",
            r"\frac{dy}{dx}=2x",
            r"Therefore, the area is A = \pi r^2.",
        ]
        for sample in samples:
            with self.subTest(sample=sample):
                mob = create_content(sample)
                self.assertGreater(mob.width, 0)


class VoiceoverSyncTests(unittest.TestCase):
    def test_capped_run_time_never_exceeds_remaining(self):
        from pipeline.voiceover_sync import SyncedVoiceoverScene

        class DummyTracker:
            def get_remaining_duration(self, buff: float = 0.0) -> float:
                return 0.5 + buff

        scene = SyncedVoiceoverScene.__new__(SyncedVoiceoverScene)
        self.assertAlmostEqual(scene._capped_run_time(DummyTracker(), 2.0), 0.5)

        class TinyTracker:
            def get_remaining_duration(self, buff: float = 0.0) -> float:
                return 0.01 + buff

        self.assertIsNone(scene._capped_run_time(TinyTracker(), 1.0))


if __name__ == "__main__":
    unittest.main()
