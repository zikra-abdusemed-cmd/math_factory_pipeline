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
import io
import re
import shutil
import subprocess
import sys
import tokenize
from dataclasses import dataclass
from pathlib import Path

from pipeline.openai_client import generate_text
from pipeline.script_gen import ScriptBeat
from pipeline.storyboard_gen import StoryboardScene
from pipeline.math_rendering import ensure_math_renderer_available, tex_path_environment
import config

CODEGEN_SYSTEM_PROMPT = """You are a senior Manim (Community Edition, v0.19+) \
developer who writes clean, CORRECT, self-contained scenes for math \
explainer videos, using the manim-voiceover plugin to sync narration audio \
with animation.

Hard requirements for every scene you write:
- `from manim import *`
- `from manim_voiceover import VoiceoverScene`
- `from tts_services.active_service import get_speech_service`
- Cinematic 2D scenes may subclass `MovingCameraScene`; 3D scenes may subclass
  `ThreeDScene`. The pipeline normalizes voiceover scenes to synced variants.
- Name each declared visual_plan object id as the Python variable for that
  mobject. The pipeline registers those names with CameraController.
- If visual_plan.camera is non-empty, subclass MovingCameraScene (2D) or
  ThreeDScene (3D). Do not write custom camera updaters, frame.animate chains,
  or extra camera plays; the pipeline injects CameraController and executes
  the validated shot list. Do not animate camera.frame yourself.
- Request camera tracking only when a moving subject would leave a readable
  static frame. Prefer a still composition for equations, graphs, and recaps.
  Never combine unrelated pans, zooms, and orbits in one shot.
- Physical trajectories must use pipeline.physics samples with
  `animate_physical(...)` or `rate_func=linear` on a ValueTracker. Do not
  apply smooth/ease camera rate functions to the object's path.
- For planned camera operations, import helpers from `pipeline.cinematic` only
  if you call them, and use only operations and targets in the validated plan.
  Keep movement purposeful, bounded, and synchronized inside the voiceover block.
- Keep labels, equations, and diagrams from overlapping. Place titles in the
  top band and secondary equations to the right or below the demonstration.
  Do not park explanatory text on top of a moving particle or its trail.
- Derive physical values from pipeline.physics when applicable; never invent
  unsupported parameters.
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
- Optional camera movement must only implement declared visual_plan operations.
  Account for each event's narration-relative start and duration inside the
  voiceover; do not just play every camera event at full duration sequentially.
- Keep scenes computationally light: prefer FadeIn/FadeOut/Create over Write \
for multi-word Text (Write is expensive). Avoid dense NumberPlane grids, \
hundreds of mobjects, updaters, and nested VGroups larger than ~20 items. \
Aim for 4-8 play() calls max per beat.
- Apply `GrowArrow` only to one `Arrow` at a time. For a group of arrows, \
use `Create(group)` or `LaggedStart(*[GrowArrow(arrow) for arrow in arrows])`; \
`GrowArrow(VGroup(...))` fails because a VGroup has no own start point.
- Only use Manim's own primitives: geometric Mobjects (Circle, Square, Dot, \
Line, Arrow, Polygon, Rectangle, Triangle, Dot, ...), Text, \
Axes/NumberPlane/graphing, VGroup, and standard animations (Create, Write, \
FadeIn, FadeOut, Transform, ReplacementTransform, Indicate, \
.animate.shift/scale/rotate/move_to, etc). NEVER reference external image, \
audio, video, or font files.
- For spheres, cones, cylinders, prisms, and other solids, use Manim's \
`ThreeDScene` with `ThreeDAxes` only when axes help explain coordinates. Set \
the camera orientation once before the voiceover block; use a stable oblique \
view (for example `phi=65*DEGREES, theta=-45*DEGREES`). Center and scale the \
solid so the full shape and labels fit the frame. Keep the camera still during \
the beat; avoid ambient rotation and moving the camera while drawing edges.
- Build solids from one primary 3D surface (`Sphere`, `Cone`, `Cylinder`, \
`Prism`, `Cube`, or `Polyhedron`) with a restrained fill opacity. Draw only \
edges that belong to the solid, using its exact 3D vertex coordinates and \
`Line3D`; do not approximate edges with screen-space `Line`s, draw diameters \
through curved surfaces, or add duplicate outlines. Use dashed, muted lines \
only for edges that are actually hidden from the chosen camera. Keep labels \
outside the surface and use `add_fixed_in_frame_mobjects` for explanatory \
text that should remain readable over a 3D camera view.
- For a prism, define the bottom and top vertices once as 3D NumPy points, \
then reuse those same points for the faces, corresponding vertical edges, and \
vertex markers. For a cone, place the apex and base in a consistent coordinate \
system and do not extend the axis or generatrices beyond the apex/base. For a \
sphere, avoid drawing straight lines across its surface as if they were solid \
edges; use great-circle arcs or latitude/longitude curves instead.
- Use Text only for ordinary natural language. Prefer create_content(...) \
from pipeline.math_rendering for any on-screen label whose content may mix \
prose and math. Use MathTex / create_math only for pure equations. Never put \
raw LaTeX, scripts (^ or _), or mathematical source inside Text.
- Use the pipeline's `create_math` for TeX; it installs the supported template \
even if a scene tries Manim's `Tex` constructor.
- `create_math` accepts one or more positional TeX strings, like MathTex; \
use separate positional parts only when you need separate alignment/isolation.
- Do not index into a `create_math(...)` result unless it was constructed \\
with multiple positional TeX strings. For coloring individual symbols, use \\
`.set_color_by_tex(r"...", COLOR)` instead of assuming `mobject[1]` exists.
- Common scalar functions from Python's math module and NumPy (`np`) are \
available in generated scenes; use `np` for arrays and vectorized graph functions.
- For `Axes.get_riemann_rectangles(...)`, use the singular `color=` argument; \
`colors=` is not supported by Manim Community Edition.
- Transform methods such as `.scale()`, `.shift()`, `.rotate()`, and `.move_to()` \
do not accept styling arguments such as `color=`. Set colors on the mobject \
constructor or with `.set_color(...)` before applying transforms.
- Use Manim's defined color constants such as RED, BLUE, GREEN, YELLOW, \\
ORANGE, PURPLE, PINK, TEAL, GOLD, WHITE, and GREY. `BROWN` and `VIOLET` are \\
not defined in this Manim version; use `ManimColor("#8B4513")` or \\
`ManimColor("#8A2BE2")` for those hues.
- For a filled circular wedge, call `Sector(radius=..., start_angle=..., \
angle=...)`; use `AnnularSector(inner_radius=..., outer_radius=...)` for a ring wedge.
- For crossing line segments use `pipeline_line_intersection(line_a, line_b)`; \
Manim Mobjects do not provide a `get_intersection` method.
- For `Arc(..., arc_center=...)`, pass a numeric point/vector such as `center` \
or `dot.get_center()`, never a Mobject like `Dot` directly. Arc start angles \
are measured from the arc center; compute the angle to an endpoint with \
`np.arctan2(y, x)`.
- Use `Ellipse(width=..., height=...)` for a complete oval. `Arc` is circular \
by default; if an elliptical arc is needed, the pipeline supports \
`Arc(width=..., height=..., ...)` by resizing it after construction.
- Keep Mobjects and points distinct: use `mobject.get_center()` in coordinate \
arithmetic (`point + RIGHT`, `Dot(point)`, `arc_center=...`). `move_to` and \
`shift` accept either a Mobject or a point/vector, but arithmetic with a \
Mobject is invalid; use `mobject.get_center()` when you mean its position. \
`Circle.point_at_angle(angle)` also returns a point, whereas the circle itself \
is a Mobject.
- Write TeX-bearing Python strings as raw strings (for example, \
`r"\\frac{a}{b}"`) or escape every backslash. Ordinary strings interpret \
sequences such as `\\f` and `\\t` before MathTex receives them.
- Keep every on-screen element within the default frame (do not manually \
change camera/frame size). The pipeline runs compose_named_mobjects before \
the voiceover; still choose readable initial positions. Clean up each visual \
beat with FadeOut before introducing an unrelated one so the screen doesn't \
get cluttered. When a diagram, equation, and motion cannot fit, use sequential \
shots instead of shrinking everything.
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
- Validated visual-director plan (empty means keep a clear static composition):
{visual_plan}

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


def _find_voiceover_with(construct: ast.FunctionDef) -> ast.With | ast.AsyncWith | None:
    for node in construct.body:
        if isinstance(node, (ast.With, ast.AsyncWith)) and any(
            isinstance(item.context_expr, ast.Call)
            and isinstance(item.context_expr.func, ast.Attribute)
            and item.context_expr.func.attr == "voiceover"
            for item in node.items
        ):
            return node
    for node in ast.walk(construct):
        if isinstance(node, (ast.With, ast.AsyncWith)) and any(
            isinstance(item.context_expr, ast.Call)
            and isinstance(item.context_expr.func, ast.Attribute)
            and item.context_expr.func.attr == "voiceover"
            for item in node.items
        ):
            return node
    return None


def _is_camera_frame_root(node: ast.AST) -> bool:
    """True for the bare ``self.camera.frame`` attribute chain."""
    return (
        isinstance(node, ast.Attribute)
        and node.attr == "frame"
        and isinstance(node.value, ast.Attribute)
        and node.value.attr == "camera"
        and isinstance(node.value.value, ast.Name)
        and node.value.value.id == "self"
    )


def _is_camera_frame_expr(node: ast.AST) -> bool:
    """True for ``self.camera.frame`` and any ``.animate...`` / method chain on it."""
    if _is_camera_frame_root(node):
        return True
    if isinstance(node, ast.Attribute):
        return _is_camera_frame_expr(node.value)
    if isinstance(node, ast.Call):
        return _is_camera_frame_expr(node.func)
    return False


def _strip_conflicting_camera_ops(voiceover: ast.With | ast.AsyncWith) -> None:
    """Remove generated camera.frame.animate / UpdateFromFunc(frame) fights.

    Only CameraController may mutate the 2D frame during a planned shot.
    """
    cleaned: list[ast.stmt] = []
    for statement in voiceover.body:
        if isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Call):
            call = statement.value
            if (
                isinstance(call.func, ast.Attribute)
                and isinstance(call.func.value, ast.Name)
                and call.func.value.id == "self"
                and call.func.attr == "play"
            ):
                kept_args = []
                for arg in call.args:
                    if _is_camera_frame_expr(arg):
                        continue
                    if (
                        isinstance(arg, ast.Call)
                        and isinstance(arg.func, ast.Name)
                        and arg.func.id in {"UpdateFromFunc", "UpdateFromAlphaFunc"}
                        and arg.args
                        and _is_camera_frame_expr(arg.args[0])
                    ):
                        continue
                    kept_args.append(arg)
                if not kept_args:
                    continue
                call.args = kept_args
        cleaned.append(statement)
    voiceover.body = cleaned


def _resolve_declared_targets(construct: ast.FunctionDef, visual_plan: Any) -> dict[str, str]:
    declared_names = {
        node.id for node in ast.walk(construct)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)
    }
    by_normalized_name = {name.replace("_", "").lower(): name for name in declared_names}
    target_names: dict[str, str] = {}
    for obj in getattr(visual_plan, "objects", ()):
        object_id = obj.get("id", "")
        candidate = object_id if object_id in declared_names else by_normalized_name.get(object_id.replace("_", "").lower())
        if candidate:
            target_names[object_id] = candidate
    for object_id in (
        getattr(visual_plan, "main_subject", ""),
        *getattr(visual_plan, "secondary", ()),
        *getattr(visual_plan, "keep_readable", ()),
    ):
        if not object_id or object_id in target_names:
            continue
        candidate = object_id if object_id in declared_names else by_normalized_name.get(object_id.replace("_", "").lower())
        if candidate:
            target_names[object_id] = candidate
    for event in getattr(visual_plan, "camera", ()):
        object_id = getattr(event, "target", None)
        if not object_id or object_id in target_names:
            continue
        candidate = object_id if object_id in declared_names else by_normalized_name.get(object_id.replace("_", "").lower())
        if candidate:
            target_names[object_id] = candidate
    return target_names


def inject_camera_plan(code: str, visual_plan: Any) -> str:
    """Inject CameraController, layout composition, and strip conflicting frame ops.

    The package model often forgets camera helpers or writes competing
    ``camera.frame.animate`` calls. One controller owns the frame; layout runs
    after mobjects exist and before narration timing starts.
    """
    tree = ast.parse(_rawify_invalid_tex_strings(code))
    classes = [node for node in tree.body if isinstance(node, ast.ClassDef)]
    if len(classes) != 1:
        return code
    scene_class = classes[0]
    construct = next((node for node in scene_class.body if isinstance(node, ast.FunctionDef) and node.name == "construct"), None)
    if construct is None:
        return code
    voiceover = _find_voiceover_with(construct)
    if voiceover is None:
        return code

    _strip_conflicting_camera_ops(voiceover)

    target_names = _resolve_declared_targets(construct, visual_plan) if visual_plan else {}
    camera_events = list(getattr(visual_plan, "camera", ()) or ()) if visual_plan else []
    usable_events: list[Any] = []
    for event in camera_events:
        if event.command in {"push_in", "pan_to", "follow_object"} and event.target not in target_names:
            continue
        if event.command == "orbit_scene" and getattr(visual_plan, "mode", "2d") != "3d":
            continue
        usable_events.append(event)

    needs_moving_camera = bool(usable_events) and any(
        event.command in {"push_in", "pull_back", "pan_to", "follow_object", "transition_to_graph", "freeze_and_annotate"}
        for event in usable_events
    )
    moving_camera_bases = {"MovingCameraScene", "SyncedMovingCameraVoiceoverScene"}
    three_d_bases = {"ThreeDScene", "SyncedThreeDVoiceoverScene"}
    if needs_moving_camera and not any(isinstance(base, ast.Name) and base.id in moving_camera_bases for base in scene_class.bases):
        scene_class.bases.append(ast.Name(id="MovingCameraScene", ctx=ast.Load()))
    if visual_plan and getattr(visual_plan, "mode", "2d") == "3d" and any(event.command == "orbit_scene" for event in usable_events):
        if not any(isinstance(base, ast.Name) and base.id in three_d_bases for base in scene_class.bases):
            scene_class.bases.append(ast.Name(id="ThreeDScene", ctx=ast.Load()))

    # Composition pass: reposition independent labels before the voiceover clock.
    # Namespace keys are Python variable names, not visual-plan object ids.
    readable_vars = [target_names[i] for i in (getattr(visual_plan, "keep_readable", ()) or ()) if i in target_names] if visual_plan else []
    title_vars = [
        var for name, var in target_names.items()
        if name.lower() in {"title", "heading", "observation"} or var.lower() in {"title", "heading", "observation"}
    ]
    equation_vars = [
        var for name, var in target_names.items()
        if "equation" in name.lower() or "formula" in name.lower() or var.lower() in {"equation", "formula", "eq"}
    ]
    locked_vars = []
    if visual_plan:
        main = getattr(visual_plan, "main_subject", "") or ""
        secondary = set(getattr(visual_plan, "secondary", ()) or ())
        for object_id, var in target_names.items():
            if object_id == main or object_id in secondary:
                locked_vars.append(var)
    # Avoid double-injecting layout on rewrites.
    already_layout = any(
        isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "_pipeline_layout" for t in getattr(node, "targets", []))
        for node in construct.body
    )
    layout_source = (
        "_pipeline_layout = compose_named_mobjects(\n"
        "    self,\n"
        "    {name: value for name, value in locals().items() if not str(name).startswith('_')},\n"
        f"    locked_ids={locked_vars!r},\n"
        f"    readable_ids={readable_vars!r},\n"
        f"    title_ids={title_vars!r},\n"
        f"    equation_ids={equation_vars!r},\n"
        ")\n"
        "_pipeline_qc = validate_scene_composition(\n"
        "    self,\n"
        "    {name: value for name, value in locals().items() if not str(name).startswith('_')},\n"
        ")\n"
    )
    # Insert layout immediately before the voiceover block inside construct.
    voiceover_index = next(
        (index for index, node in enumerate(construct.body) if node is voiceover),
        None,
    )
    if voiceover_index is not None and not already_layout:
        construct.body[voiceover_index:voiceover_index] = ast.parse(layout_source).body
        # Re-find voiceover after insertion (same object still valid).

    if usable_events and needs_moving_camera:
        register_lines = ["_camera_controller = CameraController.ensure(self)"]
        for object_id, var_name in target_names.items():
            register_lines.append(f"_camera_controller.register({object_id!r}, {var_name})")
        plan_items = []
        for event in usable_events:
            plan_items.append(
                "{"
                f"'command': {event.command!r}, "
                f"'target': {event.target!r}, "
                f"'start': {float(event.start)!r}, "
                f"'duration': {float(event.duration)!r}, "
                f"'reason': {getattr(event, 'reason', '')!r}"
                "}"
            )
        register_lines.append(f"_camera_controller.set_plan([{', '.join(plan_items)}])")
        register_lines.append("_camera_controller.attach()")
        register_lines.append("_camera_controller.bind_tracker(tracker)")
        teardown = [
            "_camera_controller.finish()",
            "validate_scene_composition(self, {name: value for name, value in locals().items() if not str(name).startswith('_')}, controller=_camera_controller)",
        ]
        # Avoid double-injection when re-running inject on already rewritten code.
        already = any(
            isinstance(node, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "_camera_controller" for t in node.targets)
            for node in voiceover.body
            if isinstance(node, ast.Assign)
        )
        if not already:
            voiceover.body[0:0] = ast.parse("\n".join(register_lines)).body
            voiceover.body.extend(ast.parse("\n".join(teardown)).body)

    ast.fix_missing_locations(tree)
    return ast.unparse(tree) + "\n"


def _rawify_invalid_tex_strings(code: str) -> str:
    """Make invalid Python TeX escapes raw before Python parses the module.

    Most TeX escapes that Python consumes can be repaired from AST string
    values. Commands such as ``\\xi`` and ``\\underbrace`` are worse: Python
    treats ``\\x`` and ``\\u`` as incomplete hex/Unicode escapes and rejects
    the entire file before an AST exists. Only string tokens that fail Python
    literal evaluation are changed; valid Python escape sequences are kept.
    """
    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(code).readline))
    except (tokenize.TokenError, IndentationError):
        return code

    for index, token in enumerate(tokens):
        if token.type != tokenize.STRING:
            continue
        match = re.match(r"(?i)^([rubf]*)(\"\"\"|'''|\"|')", token.string)
        if not match:
            continue
        prefix, quote = match.groups()
        lowered = prefix.lower()
        if "r" in lowered or "b" in lowered:
            continue
        try:
            ast.literal_eval(token.string)
        except (SyntaxError, ValueError):
            if not re.search(r"\\[A-Za-z]+", token.string[len(prefix) + len(quote) :]):
                continue
            # `r` is valid on ordinary and formatted strings. Unicode is the
            # default for Python 3, so discard the obsolete `u` prefix.
            safe_prefix = "fr" if "f" in lowered else "r"
            tokens[index] = token._replace(string=safe_prefix + token.string[len(prefix) :])
    return tokenize.untokenize(tokens)


class _UniversalTextTransformer(ast.NodeTransformer):
    """Route generated Text/MathTex/VoiceoverScene through pipeline helpers."""

    _MOBJECT_CONSTRUCTORS = {
        "Angle", "AnnularSector", "Arrow", "Arc", "Brace", "Circle", "DashedLine",
        "Dot", "Ellipse", "Group", "Line", "NumberPlane", "Polygon", "Rectangle", "Sector",
        "Square", "SurroundingRectangle", "Text", "Triangle", "VGroup", "VMobject",
        "Vector", "axes", "create_content", "create_math", "graph",
    }

    def __init__(self) -> None:
        self.mobject_names: set[str] = set()
        self.mobject_kinds: dict[str, str] = {}

    def _is_mobject_name(self, node: ast.expr) -> bool:
        return isinstance(node, ast.Name) and node.id in self.mobject_names

    def _track_assignment(self, node: ast.Assign | ast.AnnAssign) -> None:
        value = node.value
        is_mobject = (
            isinstance(value, ast.Call)
            and isinstance(value.func, ast.Name)
            and value.func.id in self._MOBJECT_CONSTRUCTORS
        )
        kind = value.func.id if is_mobject else None
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        for target in targets:
            if isinstance(target, ast.Name):
                if is_mobject:
                    self.mobject_names.add(target.id)
                    self.mobject_kinds[target.id] = kind or ""
                else:
                    self.mobject_names.discard(target.id)
                    self.mobject_kinds.pop(target.id, None)

    def visit_Assign(self, node: ast.Assign) -> ast.AST:
        self.generic_visit(node)
        self._track_assignment(node)
        return node

    def visit_AnnAssign(self, node: ast.AnnAssign) -> ast.AST:
        self.generic_visit(node)
        self._track_assignment(node)
        return node

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
        outer_names = self.mobject_names
        outer_kinds = self.mobject_kinds
        self.mobject_names = set()
        self.mobject_kinds = {}
        self.generic_visit(node)
        self.mobject_names = outer_names
        self.mobject_kinds = outer_kinds
        return node

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> ast.AST:
        outer_names = self.mobject_names
        outer_kinds = self.mobject_kinds
        self.mobject_names = set()
        self.mobject_kinds = {}
        self.generic_visit(node)
        self.mobject_names = outer_names
        self.mobject_kinds = outer_kinds
        return node

    def visit_Call(self, node: ast.Call) -> ast.AST:
        self.generic_visit(node)
        func_name: str | None = None
        if isinstance(node.func, ast.Name):
            func_name = node.func.id
        elif isinstance(node.func, ast.Attribute):
            # Manim APIs such as axes.get_riemann_rectangles are called as
            # attributes on scene objects, not as module-level functions.
            func_name = node.func.attr

        if func_name == "Text":
            node.func = ast.Name(id="create_content", ctx=ast.Load())
        elif func_name in {"MathTex", "Tex", "SingleStringMathTex"}:
            node.func = ast.Name(id="create_math", ctx=ast.Load())
        elif func_name == "line_intersection":
            # Avoid collision with Manim's similarly named utility, which
            # expects point arrays rather than Line mobjects.
            node.func = ast.Name(id="pipeline_line_intersection", ctx=ast.Load())
        elif func_name == "Sector":
            # Manim's Sector uses `radius`; `outer_radius` belongs to
            # AnnularSector and otherwise gets passed twice by Sector.__init__.
            has_radius = any(keyword.arg == "radius" for keyword in node.keywords)
            if has_radius:
                node.keywords = [keyword for keyword in node.keywords if keyword.arg != "outer_radius"]
            else:
                for keyword in node.keywords:
                    if keyword.arg == "outer_radius":
                        keyword.arg = "radius"
        if func_name == "get_riemann_rectangles":
            # Manim Community Edition accepts `color`, not `colors`.
            for keyword in node.keywords:
                if keyword.arg == "colors":
                    keyword.arg = "color"
        if (
            isinstance(node.func, ast.Attribute)
            and node.func.attr in {"scale", "shift", "rotate", "move_to", "next_to", "to_edge", "to_corner"}
        ):
            # Styling belongs to the Mobject, not to its geometry/placement methods.
            color_keywords = [keyword for keyword in node.keywords if keyword.arg == "color"]
            node.keywords = [keyword for keyword in node.keywords if keyword.arg != "color"]
            if color_keywords and isinstance(node.func.value, ast.Name):
                receiver = node.func.value
                if receiver.id in self.mobject_names:
                    set_color = ast.Call(
                        func=ast.Attribute(value=receiver, attr="set_color", ctx=ast.Load()),
                        args=[color_keywords[-1].value],
                        keywords=[],
                    )
                    node.func.value = set_color
        elif func_name == "Arc":
            dimension_keywords = {keyword.arg for keyword in node.keywords}
            if dimension_keywords & {"width", "height"}:
                node.func = ast.Name(id="pipeline_arc", ctx=ast.Load())
            # Arc.arc_center must be a coordinate vector. Models often pass a
            # Dot or another Mobject, which makes Manim's np.asarray fail.
            for keyword in node.keywords:
                if keyword.arg == "arc_center" and self._is_mobject_name(keyword.value):
                    keyword.value = ast.Call(
                        func=ast.Attribute(
                            value=keyword.value,
                            attr="get_center",
                            ctx=ast.Load(),
                        ),
                        args=[],
                        keywords=[],
                    )
        elif func_name == "Dot":
            # Dot's point parameter is a coordinate, not a Mobject. Convert
            # simple variable arguments; leave vector expressions unchanged.
            if node.args and self._is_mobject_name(node.args[0]):
                node.args[0] = ast.Call(
                    func=ast.Attribute(value=node.args[0], attr="get_center", ctx=ast.Load()),
                    args=[],
                    keywords=[],
                )
            for keyword in node.keywords:
                if keyword.arg == "point" and self._is_mobject_name(keyword.value):
                    keyword.value = ast.Call(
                        func=ast.Attribute(value=keyword.value, attr="get_center", ctx=ast.Load()),
                        args=[],
                        keywords=[],
                    )
        elif func_name == "GrowArrow":
            # GrowArrow reads get_start() from its input and only supports a
            # single Arrow, not a VGroup/Group of Arrow objects.
            grouped_arrows = (
                bool(node.args)
                and (
                    (isinstance(node.args[0], ast.Name)
                     and self.mobject_kinds.get(node.args[0].id) in {"VGroup", "Group"})
                    or (isinstance(node.args[0], ast.Call)
                        and isinstance(node.args[0].func, ast.Name)
                        and node.args[0].func.id in {"VGroup", "Group"})
                )
            )
            if grouped_arrows:
                node.func = ast.Name(id="Create", ctx=ast.Load())
        elif (
            isinstance(node.func, ast.Attribute)
            and node.func.attr == "get_intersection"
            and len(node.args) == 1
            and not node.keywords
        ):
            first_line = node.func.value
            node.func = ast.Name(id="pipeline_line_intersection", ctx=ast.Load())
            node.args.insert(0, first_line)
        return node

    def visit_Name(self, node: ast.Name) -> ast.AST:
        # These color names are common in generated palettes but are not
        # exported by Manim CE 0.22. Keep the intended hues using hex colors.
        fallback_colors = {
            "BROWN": "#8B4513",
            "VIOLET": "#8A2BE2",
        }
        if isinstance(node.ctx, ast.Load) and node.id in fallback_colors:
            return ast.Call(
                func=ast.Name(id="ManimColor", ctx=ast.Load()),
                args=[ast.Constant(value=fallback_colors[node.id])],
                keywords=[],
            )
        return node

    def visit_Constant(self, node: ast.Constant) -> ast.AST:
        """Repair common TeX escapes consumed by ordinary Python strings.

        For example, ``"\\frac{x}{y}"`` in generated source becomes a
        form-feed followed by ``rac`` before AST rewriting.  Restore known
        TeX commands so the one-call package can be validated without another
        paid model request.
        """
        if isinstance(node.value, str):
            repairs = {
                "\x08ar": r"\bar",
                "\x08egin": r"\begin",
                "\x08eta": r"\beta",
                "\x0crac": r"\frac",
                "\nabla": r"\nabla",
                "\rangle": r"\rangle",
                "\rceil": r"\rceil",
                "\rfloor": r"\rfloor",
            }
            value = node.value
            # Repair control-character prefixes only when they form a known
            # TeX command; ordinary tabs/newlines in narration remain intact.
            for malformed, corrected in repairs.items():
                value = value.replace(malformed, corrected)
            for suffix in ("heta", "ext", "imes", "an", "o"):
                value = value.replace("\t" + suffix, "\\t" + suffix)
            node.value = value
        return node

    def visit_ClassDef(self, node: ast.ClassDef) -> ast.AST:
        self.generic_visit(node)
        new_bases: list[ast.expr] = []
        uses_3d_scene = any(
            isinstance(base, ast.Name) and base.id == "ThreeDScene"
            for base in node.bases
        )
        uses_moving_camera = any(
            isinstance(base, ast.Name) and base.id == "MovingCameraScene"
            for base in node.bases
        )
        for base in node.bases:
            if isinstance(base, ast.Name) and base.id in {"ThreeDScene", "MovingCameraScene"}:
                continue
            if isinstance(base, ast.Name) and base.id in {"VoiceoverScene", "SyncedVoiceoverScene"}:
                if uses_3d_scene:
                    new_bases.append(ast.Name(id="SyncedThreeDVoiceoverScene", ctx=ast.Load()))
                elif uses_moving_camera:
                    new_bases.append(ast.Name(id="SyncedMovingCameraVoiceoverScene", ctx=ast.Load()))
                else:
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
    code = _rawify_invalid_tex_strings(code)
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        raise ValueError(f"generated scene is not valid Python: {exc}") from exc
    tree = _UniversalTextTransformer().visit(tree)
    # Remove any ImportFrom nodes that the transformer deleted (returned None).
    tree.body = [node for node in tree.body if node is not None]
    ast.fix_missing_locations(tree)

    helpers = ast.parse(
        "from pipeline.math_rendering import create_content, create_math, ensure_pipeline_tex_template, line_intersection as pipeline_line_intersection\n"
        "from pipeline.voiceover_sync import SyncedVoiceoverScene\n"
        "from pipeline.voiceover_sync import SyncedThreeDVoiceoverScene, SyncedMovingCameraVoiceoverScene\n"
        "from pipeline.manim_compat import arc_with_dimensions as pipeline_arc\n"
        "from pipeline.cinematic import push_in, pull_back, pan_to, follow_object, orbit_scene, freeze_and_annotate, reveal_system, transition_to_graph, trajectory_trail, vector_arrow, acceleration_arrow, trajectory_graph, CameraController, animate_physical, MOTION_PROFILES\n"
        "from pipeline.layout import compose_named_mobjects\n"
        "from pipeline.visual_qc import validate_scene_composition\n"
        "from pipeline.physics import Projectile\n"
        # Make the standard scalar math vocabulary available even when a
        # generated scene forgets to import it explicitly. `np` is exported
        # by Manim's wildcard import; keep numpy as the preferred API in the
        # prompt, while these aliases cover common bare-name usage.
        "import math\n"
        "from math import (acos, acosh, asin, asinh, atan, atan2, atanh, ceil, comb, copysign, cos, cosh, degrees, dist, e, erf, erfc, exp, expm1, factorial, fabs, floor, fmod, frexp, fsum, gamma, gcd, hypot, inf, isclose, isfinite, isinf, isnan, isqrt, ldexp, lgamma, lcm, log, log10, log1p, log2, modf, perm, pi, pow, prod, radians, remainder, sin, sinh, sqrt, tan, tanh, tau, trunc)\n"
        "import numpy as np\n"
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
    env = {
        **os.environ,
        "PYTHONPATH": str(project_root) + os.pathsep + os.environ.get("PYTHONPATH", ""),
        "PATH": tex_path_environment(),
    }
    return env


def _as_text_output(value: str | bytes | None) -> str:
    """Normalize subprocess output, including TimeoutExpired's bytes variant."""
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def _try_render(file_path: Path, class_name: str, project_root: Path) -> tuple[bool, str]:
    """Low-quality, offline dry-run used only to validate generated Manim code."""
    cmd = [
        sys.executable,
        "-m",
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
        output = _as_text_output(exc.stdout) + "\n" + _as_text_output(exc.stderr)
        return False, f"Offline validation timed out after 300 seconds.\n{output[-3000:]}"
    if result.returncode == 0:
        return True, ""
    stdout = _as_text_output(result.stdout)
    stderr = _as_text_output(result.stderr)
    return False, stdout[-3000:] + "\n" + stderr[-3000:]


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
        visual_plan=storyboard.visual_plan or "none",
    )

    code = generate_text(user_prompt, system_instruction=CODEGEN_SYSTEM_PROMPT, temperature=0.6)
    code = inject_camera_plan(_strip_code_fences(code), storyboard.visual_plan)
    code = normalize_visible_content(code)

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
        code = inject_camera_plan(_strip_code_fences(code), storyboard.visual_plan)
        code = normalize_visible_content(code)
        file_path.write_text(code, encoding="utf-8")

    # Clean up the throwaway low-quality preview media either way.
    shutil.rmtree(project_root / "generated" / "_codegen_preview_media", ignore_errors=True)

    raise RuntimeError(
        f"Scene {storyboard.class_name} failed to render after "
        f"{config.pipeline.max_codegen_retries + 1} attempts.\nLast error:\n{last_error}"
    )
