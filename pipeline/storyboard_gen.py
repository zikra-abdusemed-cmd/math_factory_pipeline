"""
Step 2: script -> Manim storyboard.

For each narration beat we ask Gemini for a concrete visual plan: what
objects appear, what happens to them, and roughly how that maps onto the
sentences being spoken. This storyboard is what the code-generation step
(pipeline/scene_codegen.py) turns into an actual Manim scene, so it's kept
structured (JSON) rather than free prose.
"""

from __future__ import annotations

from dataclasses import dataclass

from pipeline.gemini_client import generate_json
from pipeline.script_gen import Script

STORYBOARD_SYSTEM_PROMPT = """You are a Manim (Community Edition) storyboard \
artist who plans math explainer visuals in the style of 3Blue1Brown.

For each narration beat you are given, design a visual plan that:
- Uses only things Manim can actually draw: shapes (Circle, Square, Dot, \
Line, Arrow, Polygon), MathTex/Tex equations, Axes/NumberPlane graphs, \
Text labels, and simple transforms (Create, Transform, FadeIn/Out, \
Write, animate.shift/scale/rotate).
- Has 2-4 concrete visual "beats" (sub-steps) that a Manim scene can animate \
in the same order the narration would naturally reveal them.
- Never invents external images, icons, logos, or video/audio assets -- \
everything must be drawable with Manim's own primitives.
- Stays visually simple enough to render in well under a minute of scene time.
"""

STORYBOARD_USER_PROMPT = """Video title: "{title}"

Here are the narration beats, in order:
{beats_block}

Return STRICT JSON with this exact shape:
{{
  "scenes": [
    {{
      "id": "b1",
      "class_name": "PascalCaseSceneClassName",
      "visual_summary": "one or two sentences describing the overall visual for this beat",
      "steps": [
        "short imperative description of sub-step 1, e.g. 'Draw a unit circle centered at the origin'",
        "short imperative description of sub-step 2"
      ],
      "on_screen_text": ["any short labels/equations that should appear as text/MathTex, or [] if none"]
    }}
  ]
}}

There must be exactly one scene object per beat id listed above, in the same
order, with matching "id". class_name must be a unique valid Python class
name. Do not include anything outside the JSON object.
"""


@dataclass
class StoryboardScene:
    id: str
    class_name: str
    visual_summary: str
    steps: list[str]
    on_screen_text: list[str]


def generate_storyboard(script: Script) -> list[StoryboardScene]:
    beats_block = "\n".join(f"- [{b.id}] {b.narration}" for b in script.beats)
    prompt = STORYBOARD_USER_PROMPT.format(title=script.title, beats_block=beats_block)

    data = generate_json(prompt, system_instruction=STORYBOARD_SYSTEM_PROMPT, temperature=0.8)

    scenes = [
        StoryboardScene(
            id=s["id"],
            class_name=s["class_name"],
            visual_summary=s.get("visual_summary", ""),
            steps=list(s.get("steps", [])),
            on_screen_text=list(s.get("on_screen_text", [])),
        )
        for s in data["scenes"]
    ]

    # Defensive: make sure ids line up 1:1 with the script beats, and that
    # class names are unique (Gemini occasionally repeats a name).
    by_id = {s.id: s for s in scenes}
    missing = [b.id for b in script.beats if b.id not in by_id]
    if missing:
        raise ValueError(f"Storyboard is missing scenes for beat ids: {missing}")

    seen_names: set[str] = set()
    ordered: list[StoryboardScene] = []
    for b in script.beats:
        s = by_id[b.id]
        if s.class_name in seen_names:
            s.class_name = f"{s.class_name}{len(seen_names)}"
        seen_names.add(s.class_name)
        ordered.append(s)
    return ordered
