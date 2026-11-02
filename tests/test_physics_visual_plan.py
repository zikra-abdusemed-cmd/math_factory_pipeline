import ast
import math
import unittest

from pipeline.physics import Projectile
from pipeline.visual_plan import validate_visual_plan
from pipeline.storyboard_gen import STORYBOARD_USER_PROMPT
from pipeline import openai_client
from pipeline.scene_codegen import inject_camera_plan, normalize_visible_content
from unittest.mock import patch


class ProjectileTests(unittest.TestCase):
    def test_45_degree_projectile_matches_analytical_results(self):
        model = Projectile(0, 0, 12, 45, 9.81)
        t = model.time_to_ground()
        self.assertAlmostEqual(t, 12 * math.sqrt(2) / 9.81)
        self.assertAlmostEqual(model.position(t)[1], 0, places=9)
        self.assertAlmostEqual(model.position(t)[0], 12**2 / 9.81, places=9)
        self.assertAlmostEqual(model.velocity(t)[1], -model.vy0, places=9)

    def test_time_and_parameters_are_validated(self):
        with self.assertRaisesRegex(ValueError, "positive"):
            Projectile(0, 0, 1, 0, 0)
        with self.assertRaisesRegex(ValueError, "non-negative"):
            Projectile(0, 0, 1, 0).position(-0.1)

    def test_trajectory_is_ordered_and_ends_on_ground(self):
        points = Projectile(0, 0, 10, 30).trajectory(11)
        self.assertEqual(len(points), 11)
        self.assertGreater(points[5][1], points[0][1])
        self.assertAlmostEqual(points[-1][1], 0, places=9)


class VisualPlanTests(unittest.TestCase):
    def test_video_package_f_string_formats_visual_plan_schema(self):
        class Response:
            output_text = "{}"

        with patch.object(openai_client, "_create_response_with_retry", return_value=Response()) as request:
            result = openai_client.generate_video_package("Format diagnostic")
        self.assertEqual(result, {})
        prompt = request.call_args.kwargs["input"]
        self.assertIn('objects [{"id":...', prompt)
        self.assertIn('camera [{"command": supported command', prompt)

    def test_storyboard_prompt_formats_json_visual_plan_example(self):
        prompt = STORYBOARD_USER_PROMPT.format(title="Projectile motion", beats_block="- [b1] Observe")
        self.assertIn('"visual_plan": {', prompt)
        self.assertIn('"command": "push_in"', prompt)

    def test_math_storyboard_without_plan_uses_safe_default(self):
        self.assertEqual(validate_visual_plan(None).mode, "2d")
        self.assertEqual(validate_visual_plan(None).camera, ())

    def test_valid_camera_instruction_and_ordered_cues(self):
        plan = validate_visual_plan({
            "duration": 8,
            "objects": [{"id": "ball", "kind": "particle"}],
            "camera": [{"command": "follow_object", "target": "ball", "start": 1, "duration": 3}],
            "narration_cues": [{"at": 1}, {"at": 5}],
        }, narration_duration=9)
        self.assertEqual(plan.camera[0].target, "ball")
        self.assertEqual([cue["at"] for cue in plan.narration_cues], [1, 5])

    def test_undeclared_event_targets_are_repaired_as_referenced_objects(self):
        plan = validate_visual_plan({
            "events": [{"at": 2, "kind": "highlight", "target": "mango"}],
        })
        self.assertEqual(plan.events[0]["target"], "mango")
        self.assertIn("mango", [obj["id"] for obj in plan.objects])
        self.assertEqual(plan.objects[0]["kind"], "referenced_visual")

    def test_undeclared_camera_target_is_registered_before_validation(self):
        plan = validate_visual_plan({
            "camera": [{"command": "follow_object", "target": "ball", "start": 0, "duration": 1}],
        })
        self.assertEqual(plan.camera[0].target, "ball")
        self.assertIn("ball", [obj["id"] for obj in plan.objects])

    def test_motion_event_adds_camera_sequence_and_scene_executes_it(self):
        plan = validate_visual_plan({
            "duration": 8,
            "objects": [{"id": "mango", "kind": "Circle"}],
            "events": [{"at": 3, "kind": "move", "target": "mango"}],
        })
        self.assertEqual([event.command for event in plan.camera], ["push_in", "follow_object", "pull_back"])
        self.assertEqual(plan.main_subject, "mango")
        source = '''from manim import *
from manim_voiceover import VoiceoverScene
class Demo(VoiceoverScene):
    def construct(self):
        self.set_speech_service(get_speech_service())
        mango = Dot()
        with self.voiceover(text="The mango falls.") as tracker:
            self.play(mango.animate.shift(DOWN), run_time=tracker.get_remaining_duration())
            self.play(self.camera.frame.animate.move_to(mango), run_time=0.5)
'''
        rewritten = normalize_visible_content(inject_camera_plan(source, plan))
        tree = ast.parse(rewritten)
        class_node = next(node for node in tree.body if isinstance(node, ast.ClassDef))
        self.assertEqual(class_node.bases[0].id, "SyncedMovingCameraVoiceoverScene")
        self.assertIn("CameraController.ensure(self)", rewritten)
        self.assertIn("_camera_controller.register('mango', mango)", rewritten)
        self.assertIn("_camera_controller.set_plan(", rewritten)
        self.assertIn("_camera_controller.attach()", rewritten)
        self.assertIn("_camera_controller.bind_tracker(tracker)", rewritten)
        self.assertIn("_camera_controller.finish()", rewritten)
        self.assertIn("compose_named_mobjects(", rewritten)
        self.assertIn("validate_scene_composition(", rewritten)
        # Conflicting camera.frame.animate plays are stripped; object motion remains.
        self.assertNotIn("self.camera.frame.animate", rewritten)
        self.assertEqual(rewritten.count("self.play("), 1)

        already_normalized = source.replace("VoiceoverScene)", "SyncedVoiceoverScene)")
        rewritten_existing = normalize_visible_content(inject_camera_plan(already_normalized, plan))
        existing_class = next(node for node in ast.parse(rewritten_existing).body if isinstance(node, ast.ClassDef))
        self.assertEqual(existing_class.bases[0].id, "SyncedMovingCameraVoiceoverScene")
        # Re-injection must not stack controllers.
        reinjected = inject_camera_plan(rewritten, plan)
        self.assertEqual(reinjected.count("CameraController.ensure(self)"), 1)
        self.assertEqual(reinjected.count("_pipeline_layout = compose_named_mobjects"), 1)

    def test_string_narration_cues_are_normalized_without_failing_job(self):
        plan = validate_visual_plan({"narration_cues": ["say the key idea", {"at": 2, "cue": "show equation"}]})
        self.assertEqual(plan.narration_cues[0], {"at": 0.0, "cue": "say the key idea"})
        self.assertEqual(plan.narration_cues[1]["at"], 2)

    def test_non_numeric_parameter_flags_are_moved_to_assumptions(self):
        plan = validate_visual_plan({
            "parameters": {"g": 9.81, "physical_motion": "linear", "speed": "12"},
        })
        self.assertEqual(plan.parameters["g"], 9.81)
        self.assertEqual(plan.parameters["speed"], 12.0)
        self.assertNotIn("physical_motion", plan.parameters)
        self.assertTrue(any("physical_motion=linear" in item for item in plan.assumptions))

    def test_descriptive_theme_is_supported_and_migrated_from_legacy_parameters(self):
        direct = validate_visual_plan({"theme": "dark blue with orange accents", "parameters": {"g": 9.81}})
        legacy = validate_visual_plan({"parameters": {"theme": "dark blue", "g": 9.81}})
        self.assertEqual(direct.theme, "dark blue with orange accents")
        self.assertEqual(legacy.theme, "dark blue")
        self.assertEqual(legacy.parameters, {"g": 9.81})

    def test_theme_normalizes_long_and_structured_model_values(self):
        long_theme = "cinematic dark blue palette " * 12
        plan = validate_visual_plan({"theme": {"palette": ["blue", "orange"], "mood": "calm"}})
        truncated = validate_visual_plan({"theme": long_theme})
        self.assertIn('"mood":"calm"', plan.theme)
        self.assertEqual(len(truncated.theme), 240)


    def test_unknown_command_missing_target_and_2d_orbit_fail(self):
        for raw, message in [
            ({"camera": [{"command": "zoom_magic"}]}, "Unknown camera"),
            ({"objects": [{"id": "ball"}], "camera": [{"command": "follow_object"}]}, "requires a target"),
            ({"camera": [{"command": "orbit_scene"}]}, "requires visual_plan.mode='3d'"),
        ]:
            with self.subTest(raw=raw), self.assertRaisesRegex(ValueError, message):
                validate_visual_plan(raw)

    def test_bad_timing_and_duplicate_ids_fail(self):
        with self.assertRaisesRegex(ValueError, "timestamps must be ordered"):
            validate_visual_plan({"events": [{"at": 3}, {"at": 2}]})
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            validate_visual_plan({"objects": [{"id": "ball"}, {"id": "ball"}]})


if __name__ == "__main__":
    unittest.main()
