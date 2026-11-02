import unittest
from pathlib import Path

from pipeline.olx import brief_for_section, brief_for_unit, html_to_text, load_olx


FIXTURE = Path(__file__).resolve().parent / "fixtures" / "olx_mini"


class OlxLoaderTests(unittest.TestCase):
    def test_html_to_text_skips_script_and_style(self):
        text = html_to_text("<style>.x{}</style><script>alert(1)</script><p>Hello <em>world</em></p>")
        self.assertIn("Hello world", text)
        self.assertNotIn("alert", text)
        self.assertNotIn(".x", text)

    def test_loads_course_and_follows_static_iframe(self):
        course = load_olx(FIXTURE)
        self.assertEqual(course.course_id, "course-v1:MYMARIAN+PHY09+2026")
        self.assertEqual(course.display_name, "Physics Grade 9")
        self.assertEqual(course.chapters[0].url_name, "chapter_u01")
        intro = course.chapters[0].children[0].children[0]
        self.assertIn("mango falling", intro.text.lower())
        self.assertNotIn("alert", intro.text)
        self.assertNotIn("color:red", intro.text)

    def test_unit_brief_skips_quiz_and_keeps_lesson_source(self):
        course = load_olx(FIXTURE)
        brief = brief_for_unit(course, "chapter_u01")
        self.assertEqual(brief.section_ids, ["vert_intro"])
        instructions = brief.extra_instructions()
        self.assertIn("mango falling", instructions.lower())
        self.assertIn("Use ONLY facts", instructions)
        self.assertNotIn("Unit Quiz", instructions)

    def test_section_brief_uses_vertical_title(self):
        course = load_olx(FIXTURE)
        brief = brief_for_section(course, "vert_intro")
        self.assertEqual(brief.title, "Physics Grade 9: Section 01: Intro")


if __name__ == "__main__":
    unittest.main()
