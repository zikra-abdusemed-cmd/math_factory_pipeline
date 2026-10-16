import unittest

from pipeline.orchestrator import _match_scenes_to_beats


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
        with self.assertRaisesRegex(ValueError, "Expected 2 scenes; received 1.*Missing beat IDs: \['b2'\]"):
            _match_scenes_to_beats(["b1", "b2"], [{"id": "b1", "code": "first"}])


if __name__ == "__main__":
    unittest.main()
