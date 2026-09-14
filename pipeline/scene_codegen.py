"""
Step 3: (narration beat + storyboard scene) -> a working Manim scene file.

We ask Gemini to write a full, self-contained VoiceoverScene subclass. Then,
instead of trusting it blindly, we actually try to render it at low quality
in a temp location. If that fails, the traceback is fed straight back to
Gemini with a "fix this" prompt, up to `max_codegen_retries` times. This
self-repair loop is what makes fully-automated generation viable -- LLM-written
Manim code is right more often than not, but rarely right 100% of the time.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from pipeline.gemini_client import generate_text
from pipeline.script_gen import ScriptBeat
from pipeline.storyboard_gen import StoryboardScene
import config

CODEGEN_SYSTEM_PROMPT = """You are a senior Manim (Community Edition, v0.19+) \
developer who writes clean, CORRECT, self-contained scenes for math \
explainer videos, using the manim-voiceover plugin to sync narration audio \
with animation.

Hard requirements for every scene you write:
- `from manim import *`
- `from manim_voiceover import VoiceoverScene`
- `from tts_services.active_service import get_speech_service`
- Exactly one class, subclassing VoiceoverScene, named EXACTLY the class \
name you are given.
- In construct(), the FIRST line must be: \
`self.set_speech_service(get_speech_service())`
- Wrap narration in `with self.voiceover(text="...") as tracker:` blocks. \
The text inside each block must be EXACTLY one of the narration sentences \
you were given (you may split the beat's narration across multiple \
consecutive `with self.voiceover(...)` blocks, one sentence or clause per \
block, but concatenating all of them back together must reproduce the full \
narration text exactly, in order, with nothing added, removed, or reworded).
- Every `self.play(...)` call inside a `with self.voiceover(...)` block \
should pass `run_time=tracker.duration` (or divide tracker.duration across \
multiple plays in that block if there is more than one) so the animation \
exactly fills the spoken audio. Use `self.wait(tracker.duration)` instead \
of `self.play` if a block has no animation.
- Only use Manim's own primitives: geometric Mobjects (Circle, Square, Dot, \
Line, Arrow, Polygon, Rectangle, Triangle, Dot, ...), Text/MathTex/Tex, \
Axes/NumberPlane/graphing, VGroup, and standard animations (Create, Write, \
FadeIn, FadeOut, Transform, ReplacementTransform, Indicate, \
.animate.shift/scale/rotate/move_to, etc). NEVER reference external image, \
audio, video, or font files.
- Keep every on-screen element within the default frame (do not manually \
change camera/frame size). Clean up each visual beat with FadeOut before \
introducing an unrelated one so the screen doesn't get cluttered.
- No print(), no input(), no network calls, no file I/O other than what \
manim/manim-voiceover already do internally.
- Output ONLY the Python source code. No markdown fences, no commentary, \
no explanations before or after the code.
"""

CODEGEN_USER_PROMPT = """Scene class name: {class_name}

Narration for this beat (must be reproduced exactly, split across
`self.voiceover(...)` blocks as described in the system rules):
\"\"\"{narration}\"\"\"

Storyboard for this beat:
- Visual summary: {visual_summary}
- Steps:
{steps_block}
- On-screen text/equations to include somewhere: {on_screen_text}

Write the complete Python file now.
"""

CODEGEN_FIX_PROMPT = """The Manim scene you previously wrote failed to render.

--- PREVIOUS CODE ---
{previous_code}

--- ERROR OUTPUT ---
{error}

Fix the code so it renders successfully, keeping all the same hard \
requirements from the system prompt (same class name, same narration text \
reproduced exactly across self.voiceover blocks, run_time=tracker.duration, \
only Manim's own primitives). Output ONLY the corrected Python source code, \
no markdown fences, no commentary.
"""


@dataclass
class GeneratedScene:
    class_name: str
    file_path: Path
    module_stem: str


def _strip_code_fences(code: str) -> str:
    code = code.strip()
    if code.startswith("```"):
        lines = code.splitlines()
        lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        code = "\n".join(lines)
    return code.strip() + "\n"


def _try_render(file_path: Path, class_name: str, project_root: Path) -> tuple[bool, str]:
    """Low-quality dry-run render used only to validate the generated code."""
    cmd = [
        "manim",
        "-ql",
        "--disable_caching",
        "--media_dir",
        str(project_root / "generated" / "_codegen_preview_media"),
        str(file_path),
        class_name,
    ]
    env = {**os.environ, "PYTHONPATH": str(project_root) + os.pathsep + os.environ.get("PYTHONPATH", "")}
    result = subprocess.run(
        cmd,
        cwd=str(project_root),
        capture_output=True,
        text=True,
        timeout=300,
        env=env,
    )
    if result.returncode == 0:
        return True, ""
    return False, (result.stdout[-3000:] + "\n" + result.stderr[-3000:])


def generate_scene_code(
    beat: ScriptBeat,
    storyboard: StoryboardScene,
    scenes_dir: Path,
    project_root: Path,
    validate: bool = True,
) -> GeneratedScene:
    steps_block = "\n".join(f"  {i + 1}. {s}" for i, s in enumerate(storyboard.steps))
    user_prompt = CODEGEN_USER_PROMPT.format(
        class_name=storyboard.class_name,
        narration=beat.narration,
        visual_summary=storyboard.visual_summary,
        steps_block=steps_block,
        on_screen_text=storyboard.on_screen_text or "none",
    )

    code = generate_text(user_prompt, system_instruction=CODEGEN_SYSTEM_PROMPT, temperature=0.6)
    code = _strip_code_fences(code)

    scenes_dir.mkdir(parents=True, exist_ok=True)
    module_stem = f"scene_{beat.id}"
    file_path = scenes_dir / f"{module_stem}.py"
    file_path.write_text(code, encoding="utf-8")

    if not validate:
        return GeneratedScene(class_name=storyboard.class_name, file_path=file_path, module_stem=module_stem)

    last_error = ""
    for attempt in range(config.pipeline.max_codegen_retries + 1):
        ok, error = _try_render(file_path, storyboard.class_name, project_root)
        if ok:
            return GeneratedScene(class_name=storyboard.class_name, file_path=file_path, module_stem=module_stem)
        last_error = error
        if attempt == config.pipeline.max_codegen_retries:
            break
        fix_prompt = CODEGEN_FIX_PROMPT.format(previous_code=code, error=error)
        code = generate_text(fix_prompt, system_instruction=CODEGEN_SYSTEM_PROMPT, temperature=0.4)
        code = _strip_code_fences(code)
        file_path.write_text(code, encoding="utf-8")

    # Clean up the throwaway low-quality preview media either way.
    shutil.rmtree(project_root / "generated" / "_codegen_preview_media", ignore_errors=True)

    raise RuntimeError(
        f"Scene {storyboard.class_name} failed to render after "
        f"{config.pipeline.max_codegen_retries + 1} attempts.\nLast error:\n{last_error}"
    )
