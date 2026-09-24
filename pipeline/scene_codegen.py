"""
Step 3: (narration beat + storyboard scene) -> a working Manim scene file.

We ask OpenAI to write a full, self-contained VoiceoverScene subclass. Then,
instead of trusting it blindly, we actually try to render it at low quality
in a temp location. If that fails, the traceback is fed straight back to
the model with a "fix this" prompt, up to `max_codegen_retries` times. This
self-repair loop is what makes fully-automated generation viable -- LLM-written
Manim code is right more often than not, but rarely right 100% of the time.
"""

from __future__ import annotations

import os
import ast
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from pipeline.openai_client import generate_text
from pipeline.script_gen import ScriptBeat
from pipeline.storyboard_gen import StoryboardScene
from pipeline.math_rendering import ensure_math_renderer_available
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
- Use EXACTLY ONE `with self.voiceover(text="...") as tracker:` block for \
the entire beat. Its text must reproduce the supplied narration exactly, \
without additions, removals, or rewording. This minimizes live TTS API calls.
- AUDIO/VIDEO SYNC (critical): Build ALL mobjects BEFORE the voiceover block. \
Inside the block only call self.play / self.wait — no create_content/create_math. \
Never let animations overrun the narration. Prefer remaining-duration splits so \
frame rounding cannot push video past audio, e.g.:
    n = 5
    for i in range(n):
        self.play(..., run_time=tracker.get_remaining_duration() / (n - i))
Do NOT precompute `step = tracker.duration / n` and reuse it for every play \
(that overshoots after frame rounding). Underfilling is fine — VoiceoverScene \
waits for the audio to finish. If there is no animation, use \
`self.wait(tracker.duration)`.
- Keep scenes computationally light: prefer FadeIn/FadeOut/Create over Write \
for multi-word Text (Write is expensive). Avoid dense NumberPlane grids, \
hundreds of mobjects, updaters, and nested VGroups larger than ~20 items. \
Aim for 4-8 play() calls max per beat.
- Only use Manim's own primitives: geometric Mobjects (Circle, Square, Dot, \
Line, Arrow, Polygon, Rectangle, Triangle, Dot, ...), Text, \
Axes/NumberPlane/graphing, VGroup, and standard animations (Create, Write, \
FadeIn, FadeOut, Transform, ReplacementTransform, Indicate, \
.animate.shift/scale/rotate/move_to, etc). NEVER reference external image, \
audio, video, or font files.
- Use Text only for ordinary natural language. Prefer create_content(...) \
from pipeline.math_rendering for any on-screen label whose content may mix \
prose and math. Use MathTex / create_math only for pure equations. Never put \
raw LaTeX, scripts (^ or _), or mathematical source inside Text.
- Keep every on-screen element within the default frame (do not manually \
change camera/frame size). Clean up each visual beat with FadeOut before \
introducing an unrelated one so the screen doesn't get cluttered.
- No print(), no input(), no network calls, no file I/O other than what \
manim/manim-voiceover already do internally.
- Output ONLY the Python source code. No markdown fences, no commentary, \
no explanations before or after the code.
"""

CODEGEN_USER_PROMPT = """Scene class name: {class_name}

Narration for this beat (must be reproduced exactly in the one required
`self.voiceover(...)` block):
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
reproduced exactly in one self.voiceover block, remaining-duration sync so \
animations never overrun audio, only Manim's own primitives). Do NOT work \
around MathTex failures by putting LaTeX into Text(); fix the TeX expression \
instead. Output ONLY the corrected Python source code, no markdown fences, \
no commentary.
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


class _UniversalTextTransformer(ast.NodeTransformer):
    """Route generated Text/MathTex/VoiceoverScene through pipeline helpers."""

    def visit_Call(self, node: ast.Call) -> ast.AST:
        self.generic_visit(node)
        func_name: str | None = None
        if isinstance(node.func, ast.Name):
            func_name = node.func.id
        elif (
            isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "manim"
        ):
            func_name = node.func.attr

        if func_name == "Text":
            node.func = ast.Name(id="create_content", ctx=ast.Load())
        elif func_name == "MathTex":
            node.func = ast.Name(id="create_math", ctx=ast.Load())
        return node

    def visit_ClassDef(self, node: ast.ClassDef) -> ast.AST:
        self.generic_visit(node)
        new_bases: list[ast.expr] = []
        for base in node.bases:
            if isinstance(base, ast.Name) and base.id == "VoiceoverScene":
                new_bases.append(ast.Name(id="SyncedVoiceoverScene", ctx=ast.Load()))
            else:
                new_bases.append(base)
        node.bases = new_bases
        return node

    def visit_ImportFrom(self, node: ast.ImportFrom) -> ast.AST | None:
        # Drop the stock VoiceoverScene import; we inject SyncedVoiceoverScene.
        if node.module == "manim_voiceover":
            kept = [alias for alias in node.names if alias.name != "VoiceoverScene"]
            if not kept:
                return None
            node.names = kept
        return node


def normalize_visible_content(code: str) -> str:
    """Rewrite generated visible text and sync base class at the pipeline boundary.

    This is deterministic enforcement, independent of prompt compliance.
    Syntax errors are reported before Manim is launched.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        raise ValueError(f"generated scene is not valid Python: {exc}") from exc
    tree = _UniversalTextTransformer().visit(tree)
    # Remove any ImportFrom nodes that the transformer deleted (returned None).
    tree.body = [node for node in tree.body if node is not None]
    ast.fix_missing_locations(tree)

    helpers = ast.parse(
        "from pipeline.math_rendering import create_content, create_math, ensure_pipeline_tex_template\n"
        "from pipeline.voiceover_sync import SyncedVoiceoverScene\n"
        "ensure_pipeline_tex_template()\n"
    ).body

    insert_at = 1 if tree.body and isinstance(tree.body[0], ast.Expr) and isinstance(tree.body[0].value, ast.Constant) else 0
    while (
        insert_at < len(tree.body)
        and isinstance(tree.body[insert_at], ast.ImportFrom)
        and tree.body[insert_at].module == "__future__"
    ):
        insert_at += 1
    for offset, node in enumerate(helpers):
        tree.body.insert(insert_at + offset, node)
    return ast.unparse(tree) + "\n"


def _manim_env(project_root: Path) -> dict[str, str]:
    """Environment for Manim subprocesses, including TeX binaries on PATH."""
    from pipeline.math_rendering import ensure_tex_bin_on_path

    ensure_tex_bin_on_path()
    env = {
        **os.environ,
        "PYTHONPATH": str(project_root) + os.pathsep + os.environ.get("PYTHONPATH", ""),
    }
    return env


def _try_render(file_path: Path, class_name: str, project_root: Path) -> tuple[bool, str]:
    """Low-quality, offline dry-run used only to validate generated Manim code."""
    cmd = [
        "manim",
        "-ql",
        "--disable_caching",
        "--media_dir",
        str(project_root / "generated" / "_codegen_preview_media"),
        str(file_path),
        class_name,
    ]
    # Validation should verify generated Manim code, not call the paid TTS API
    # for every codegen attempt. Final renders use the configured VOICE_PROVIDER
    # against the shared pre-warmed voiceover cache.
    env = _manim_env(project_root)
    env["VOICE_PROVIDER"] = "offline"
    try:
        result = subprocess.run(
            cmd,
            cwd=str(project_root),
            capture_output=True,
            text=True,
            timeout=300,
            env=env,
        )
    except subprocess.TimeoutExpired as exc:
        output = (exc.stdout or "") + "\n" + (exc.stderr or "")
        return False, f"Offline validation timed out after 300 seconds.\n{output[-3000:]}"
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
    ensure_math_renderer_available()
    steps_block = "\n".join(f"  {i + 1}. {s}" for i, s in enumerate(storyboard.steps))
    user_prompt = CODEGEN_USER_PROMPT.format(
        class_name=storyboard.class_name,
        narration=beat.narration,
        visual_summary=storyboard.visual_summary,
        steps_block=steps_block,
        on_screen_text=storyboard.on_screen_text or "none",
    )

    code = generate_text(user_prompt, system_instruction=CODEGEN_SYSTEM_PROMPT, temperature=0.6)
    code = normalize_visible_content(_strip_code_fences(code))

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
        code = normalize_visible_content(_strip_code_fences(code))
        file_path.write_text(code, encoding="utf-8")

    # Clean up the throwaway low-quality preview media either way.
    shutil.rmtree(project_root / "generated" / "_codegen_preview_media", ignore_errors=True)

    raise RuntimeError(
        f"Scene {storyboard.class_name} failed to render after "
        f"{config.pipeline.max_codegen_retries + 1} attempts.\nLast error:\n{last_error}"
    )
