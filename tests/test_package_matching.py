import unittest
from unittest.mock import patch

import config
from pipeline.openai_client import generate_missing_video_scenes
from pipeline.orchestrator import (
    _build_storyboard,
    _match_scenes_to_beats,
    _parse_script_beats,
    _prepare_missing_scene_request,
)
from pipeline.script_gen import Script
from pipeline import script_gen
from webapp.server import get_config


class PackageMatchingTests(unittest.TestCase):
    def test_matches_exact_ids_even_when_scenes_are_reordered(self):
        scenes = [{"id": "b2", "code": "second"}, {"id": "b1", "code": "first"}]
        matched = _match_scenes_to_beats(["b1", "b2"], scenes)
        self.assertEqual(matched["b1"]["code"], "first")
        self.assertEqual(matched["b2"]["code"], "second")

    def test_falls_back_to_order_when_complete_response_has_bad_ids(self):
        scenes = [{"id": "beat-one", "code": "first"}, {"id": "beat-two", "code": "second"}]
        matched = _match_scenes_to_beats(["b1", "b2"], scenes)
        self.assertEqual(matched["b1"]["code"], "first")
        self.assertEqual(matched["b2"]["code"], "second")

    def test_missing_scene_reports_ids_and_counts(self):
        with self.assertRaisesRegex(ValueError, "Expected 2 scenes; received 1.*Missing beat IDs: \\['b2'\\]"):
            _match_scenes_to_beats(["b1", "b2"], [{"id": "b1", "code": "first"}])

    def test_missing_scene_preparation_keeps_only_one_valid_scene(self):
        beats = _parse_script_beats([
            {"id": "b1", "narration": "First."},
            {"id": "b2", "narration": "Second."},
            {"id": "b3", "narration": "Third."},
        ])
        retained, missing = _prepare_missing_scene_request(beats, [
            {"id": "b1", "class_name": "FirstScene", "code": "first"},
            {"id": "b1", "class_name": "DuplicateScene", "code": "duplicate"},
            {"id": "unexpected", "class_name": "ExtraScene", "code": "extra"},
        ])
        self.assertEqual([scene["id"] for scene in retained], ["b1"])
        self.assertEqual([beat["id"] for beat in missing], ["b2", "b3"])
        self.assertEqual(missing[0]["class_name"], "SupplementalSceneB2")

    def test_script_validation_rejects_unsafe_duplicate_and_out_of_range_beats(self):
        with self.assertRaisesRegex(ValueError, "only letters, numbers"):
            _parse_script_beats([{"id": "../scene", "narration": "Unsafe path."}])
        with self.assertRaisesRegex(ValueError, "duplicate ID"):
            _parse_script_beats([
                {"id": "b1", "narration": "First."},
                {"id": "b1", "narration": "Duplicate."},
            ])
        with self.assertRaisesRegex(ValueError, "at least one"):
            _parse_script_beats([])

    def test_script_prompt_uses_effective_minimum_scene_limit(self):
        with patch.object(config.pipeline, "max_scenes", 3), patch.object(
            script_gen, "generate_json", return_value={
                "beats": [{"id": "b1", "narration": "One."}]
            }
        ) as generate:
            script_gen.generate_script("Title")
        self.assertIn("Produce between 4 and 4 beats total", generate.call_args.args[0])

    def test_web_config_reports_effective_scene_limit(self):
        with patch.object(config.pipeline, "max_scenes", 3):
            self.assertEqual(get_config()["max_scenes"], 4)

    def test_storyboard_checks_scene_code_and_repairs_optional_fields(self):
        beats = _parse_script_beats([
            {"id": f"b{i}", "narration": f"Beat {i}."}
            for i in range(1, config.pipeline.max_scenes + 1)
        ])
        script = Script("Title", "Audience", beats)
        scene_data = {
            beat.id: {"class_name": f"Scene{beat.id}", "code": "class Scene: pass", "steps": None}
            for beat in beats
        }
        storyboard = _build_storyboard(script, scene_data)
        self.assertEqual(storyboard[0].steps, [])
        scene_data["b1"]["code"] = " "
        with self.assertRaisesRegex(ValueError, "non-empty Python 'code'"):
            _build_storyboard(script, scene_data)

    @patch("pipeline.openai_client._create_response_with_retry")
    def test_supplemental_response_checks_ids_and_assigns_requested_class(self, mock_response):
        class Response:
            output_text = '{"scenes":[{"id":"b2","class_name":"WrongName","code":"class WrongName: pass"}]}'

        mock_response.return_value = Response()
        result = generate_missing_video_scenes(
            "Title", "Audience",
            [{"id": "b2", "beat_title": "Second", "narration": "Second.", "class_name": "SupplementalSceneB2"}],
            [],
        )
        self.assertEqual(result[0]["class_name"], "SupplementalSceneB2")
        instructions = mock_response.call_args.kwargs["instructions"]
        self.assertIn("return a JSON object", instructions)

    @patch("pipeline.openai_client._create_response_with_retry")
    def test_supplemental_response_reports_duplicate_ids(self, mock_response):
        class Response:
            output_text = '{"scenes":[{"id":"b2"},{"id":"b2"}]}'

        mock_response.return_value = Response()
        with self.assertRaisesRegex(ValueError, "duplicates: \\[.*b2.*\\]"):
            generate_missing_video_scenes(
                "Title", "Audience",
                [{"id": "b2", "beat_title": "Second", "narration": "Second.", "class_name": "SupplementalSceneB2"}],
                [],
            )


if __name__ == "__main__":
    unittest.main()
