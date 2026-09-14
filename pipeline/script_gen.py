"""
Step 1: title -> narration script.

The script is what gets sent to the voice engine (Gemini TTS today,
OmniVoice later). It's split into "beats": short narration chunks
(~20-30 seconds of spoken audio each) that each become one Manim scene.
Keeping beats short is what makes the audio/animation sync in
manim-voiceover work well -- a scene that only has to illustrate one
idea is much easier to storyboard and code-generate correctly.
"""

from __future__ import annotations

from dataclasses import dataclass

import config
from pipeline.gemini_client import generate_json

SCRIPT_SYSTEM_PROMPT = """You are an expert math educator and scriptwriter, \
in the style of 3Blue1Brown / Numberphile. You write narration scripts meant \
to be read aloud by a text-to-speech engine over a Manim animation.

Rules:
- Spoken narration only. No stage directions, no "[pause]", no emojis, no markdown.
- Plain sentences a TTS engine can read naturally.
- Each beat should be speakable in roughly {target_seconds} seconds \
(around {min_words}-{max_words} words).
- Build one clear idea per beat, in a logical teaching order: hook / motivation, \
setup and definitions, the core idea worked through step by step, \
a concrete example, and a memorable takeaway.
- Be precise about math. Do not state anything mathematically incorrect.
- Assume a curious high-school-to-early-college audience with no special background.
"""

SCRIPT_USER_PROMPT = """Write a narration script for a short math education video.

Video title: "{title}"

Return STRICT JSON with this exact shape:
{{
  "title": "{title}",
  "audience": "one short sentence describing the target viewer",
  "beats": [
    {{
      "id": "b1",
      "beat_title": "short internal label, e.g. 'Hook'",
      "narration": "the exact text to be spoken aloud for this beat"
    }}
  ]
}}

Produce between 4 and {max_scenes} beats total. Do not include anything
outside the JSON object.
"""


@dataclass
class ScriptBeat:
    id: str
    beat_title: str
    narration: str


@dataclass
class Script:
    title: str
    audience: str
    beats: list[ScriptBeat]


def generate_script(title: str) -> Script:
    words_per_second = 2.5  # ~150 wpm conversational pace
    target_seconds = config.pipeline.target_seconds_per_beat
    min_words = int(target_seconds * words_per_second * 0.7)
    max_words = int(target_seconds * words_per_second * 1.3)

    system_prompt = SCRIPT_SYSTEM_PROMPT.format(
        target_seconds=int(target_seconds), min_words=min_words, max_words=max_words
    )
    user_prompt = SCRIPT_USER_PROMPT.format(title=title, max_scenes=config.pipeline.max_scenes)

    data = generate_json(user_prompt, system_instruction=system_prompt, temperature=0.8)

    beats = [
        ScriptBeat(id=b["id"], beat_title=b.get("beat_title", b["id"]), narration=b["narration"].strip())
        for b in data["beats"]
    ]
    return Script(title=data.get("title", title), audience=data.get("audience", ""), beats=beats)
