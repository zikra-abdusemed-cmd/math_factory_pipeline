"""Topic-independent conversion of visible strings into Manim mobjects.

Generated scenes should use :func:`create_content` for any string whose type is
not known in advance.  It returns Text, MathTex, or a visually contiguous
VGroup containing both.  MathTex errors are intentionally never hidden by a
plain-text fallback: raw TeX in a video is worse than a failed validation.

This module also installs a pipeline-wide TeX template that works with BasicTeX
(no ``standalone.cls`` required).  Manim's default template needs ``standalone``,
which many minimal TeX installs lack.
"""

from __future__ import annotations

import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from manim import DOWN, RIGHT, MathTex, Text, TexTemplate, VGroup, config


Kind = Literal["text", "math"]

# Commands which create symbols or mathematical layout.  Unknown commands are
# also treated as math; the list makes command boundaries easier to recognize.
_TEX_COMMAND = re.compile(r"\\[A-Za-z]+|\\[{}_^%#$&]|\\[,;:! ]")
_EXPLICIT_MATH = re.compile(r"(\$\$.*?\$\$|\$[^$\n]+\$|\\\(.*?\\\)|\\\[.*?\\\])", re.S)
_MATH_SIGNAL = re.compile(
    r"(?:\\[A-Za-z]+|[_^]|[=<>≤≥≠≈∈∉⊂⊆∪∩∫∑∏√∞±×÷∂∇→↦]|"
    r"\b[A-Za-z][A-Za-z0-9]*\s*\([^\n()]*\)\s*=|"
    r"\b\d+(?:\.\d+)?\s*[+\-*/]\s*\d)"
)
_EQUATION = re.compile(
    r"(?<!\w)(?:\\[A-Za-z]+|[A-Za-z0-9]|[{}()[\],.|])+"
    r"(?:\s*(?:\\[A-Za-z]+|[_^=<>+\-*/]|[≤≥≠≈∈∉⊂⊆∪∩∫∑∏√∞±×÷∂∇→↦°πθ])"
    r"\s*(?:\\[A-Za-z]+|[A-Za-z0-9]|[{}()[\],.|])*)+"
)

# Common macOS / Linux TeX binary locations.  Subprocess Manim runs must see
# these even when the parent shell PATH is incomplete (e.g. GUI-launched uvicorn).
_TEX_BIN_CANDIDATES = (
    Path("/Library/TeX/texbin"),
    Path("/usr/local/texlive/2026basic/bin/universal-darwin"),
    Path("/usr/local/texlive/2025basic/bin/universal-darwin"),
    Path("/usr/local/texlive/2024basic/bin/universal-darwin"),
    Path("/usr/bin"),
)


def ensure_tex_bin_on_path() -> str | None:
    """Prepend a known TeX bin directory to PATH if latex is not already found."""
    if shutil.which("latex") and shutil.which("dvisvgm"):
        return str(Path(shutil.which("latex")).parent)
    for candidate in _TEX_BIN_CANDIDATES:
        if (candidate / "latex").exists():
            path = os.environ.get("PATH", "")
            prefix = str(candidate)
            if prefix not in path.split(os.pathsep):
                os.environ["PATH"] = prefix + os.pathsep + path
            return prefix
    return None


def pipeline_tex_template() -> TexTemplate:
    """
    TeX template that does not require ``standalone.cls``.

    BasicTeX (common on macOS) ships amsmath/amssymb but often omits standalone.
    Manim's default documentclass fails hard in that case; this template uses
    ``article`` + a tight empty page and compiles to DVI for dvisvgm.
    """
    template = TexTemplate()
    template.documentclass = r"\documentclass{article}"
    template.preamble = r"""
\usepackage[english]{babel}
\usepackage{amsmath}
\usepackage{amssymb}
\pagestyle{empty}
\setlength{\textwidth}{20cm}
\setlength{\textheight}{20cm}
\setlength{\oddsidemargin}{-1in}
\setlength{\topmargin}{-1in}
\setlength{\headheight}{0pt}
\setlength{\headsep}{0pt}
\setlength{\topskip}{0pt}
\setlength{\footskip}{0pt}
\setlength{\parindent}{0pt}
"""
    template.tex_compiler = "latex"
    template.output_format = ".dvi"
    return template


_PIPELINE_TEX_TEMPLATE: TexTemplate | None = None


def ensure_pipeline_tex_template() -> TexTemplate:
    """Install the pipeline TeX template as Manim's process-wide default."""
    global _PIPELINE_TEX_TEMPLATE
    ensure_tex_bin_on_path()
    if _PIPELINE_TEX_TEMPLATE is None:
        _PIPELINE_TEX_TEMPLATE = pipeline_tex_template()
    config.tex_template = _PIPELINE_TEX_TEMPLATE
    return _PIPELINE_TEX_TEMPLATE


@dataclass(frozen=True)
class ContentPart:
    kind: Kind
    value: str


def is_math_content(value: str) -> bool:
    """Return whether *value* contains mathematical source/notation."""
    return bool(_EXPLICIT_MATH.search(value) or _MATH_SIGNAL.search(value))


def _strip_math_delimiters(value: str) -> str:
    value = value.strip()
    for left, right in (("$$", "$$"), ("$", "$"), (r"\(", r"\)"), (r"\[", r"\]")):
        if value.startswith(left) and value.endswith(right):
            return value[len(left) : -len(right)].strip()
    return value


def _append(parts: list[ContentPart], kind: Kind, value: str) -> None:
    if not value:
        return
    if parts and parts[-1].kind == kind:
        parts[-1] = ContentPart(kind, parts[-1].value + value)
    else:
        parts.append(ContentPart(kind, value))


def split_mixed_content(value: str) -> list[ContentPart]:
    """Split natural language and math without relying on a school topic.

    Explicit TeX delimiters have priority.  Undelimited expressions are found
    through notation structure (operators, scripts, commands and relations),
    not vocabulary such as "sets" or "algebra".
    """
    if not isinstance(value, str):
        raise TypeError(f"visible content must be str, got {type(value).__name__}")
    if not value:
        return [ContentPart("text", value)]

    stripped = value.strip()
    # A string that begins with TeX or gets its first math signal in the first
    # whitespace token is a formula/label, not a prose sentence.  Keeping the
    # expression intact is important for constructs containing spaces inside
    # braces (matrices, \text{}, \sum x, vector components, and so on).
    first_signal = _MATH_SIGNAL.search(stripped)
    first_space = stripped.find(" ")
    if first_signal and (stripped.startswith("\\") or first_space < 0 or first_signal.start() <= first_space):
        return [ContentPart("math", _strip_math_delimiters(stripped))]

    parts: list[ContentPart] = []
    cursor = 0
    for explicit in _EXPLICIT_MATH.finditer(value):
        prefix = value[cursor : explicit.start()]
        parts.extend(_split_undelimited(prefix))
        _append(parts, "math", _strip_math_delimiters(explicit.group()))
        cursor = explicit.end()
    parts.extend(_split_undelimited(value[cursor:]))
    return _merge_formula_fragments(parts) or [ContentPart("text", value)]


def _merge_formula_fragments(parts: list[ContentPart]) -> list[ContentPart]:
    """Join math fragments separated only by formula whitespace."""
    merged: list[ContentPart] = []
    index = 0
    while index < len(parts):
        if (
            index + 2 < len(parts)
            and parts[index].kind == "math"
            and parts[index + 1].kind == "text"
            and parts[index + 1].value.isspace()
            and parts[index + 2].kind == "math"
        ):
            _append(merged, "math", parts[index].value + parts[index + 1].value + parts[index + 2].value)
            index += 3
        else:
            _append(merged, parts[index].kind, parts[index].value)
            index += 1
    return merged


def _split_undelimited(value: str) -> list[ContentPart]:
    if not is_math_content(value):
        return [ContentPart("text", value)] if value else []
    matches = list(_EQUATION.finditer(value))
    if not matches:
        # A standalone command such as \in or \alpha is unambiguously math.
        return [ContentPart("math", value.strip())] if _TEX_COMMAND.search(value) else [ContentPart("text", value)]
    result: list[ContentPart] = []
    cursor = 0
    for match in matches:
        start, end = match.span()
        # Sentence punctuation is prose unless it participates in the formula.
        while end > start and value[end - 1] in ".;:!?":
            end -= 1
        _append(result, "text", value[cursor:start])
        _append(result, "math", value[start:end].strip())
        cursor = end
    _append(result, "text", value[cursor:])
    return result


def _math_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    # Shared Manim styling arguments. Text-only options (font, weight, slant,
    # line_spacing, disable_ligatures) must not leak into MathTex.
    allowed = {"color", "font_size", "stroke_width", "fill_opacity", "stroke_opacity", "z_index"}
    return {key: val for key, val in kwargs.items() if key in allowed}


def create_text(value: str, **kwargs: Any):
    """Create guaranteed ordinary-language text."""
    if is_math_content(value):
        raise ValueError("create_text received mathematical notation; use create_content or create_math")
    return Text(value, **kwargs)


def create_math(value: str, **kwargs: Any):
    """Create math, propagating TeX errors instead of exposing source text."""
    template = ensure_pipeline_tex_template()
    try:
        return MathTex(
            _strip_math_delimiters(value),
            tex_template=template,
            **_math_kwargs(kwargs),
        )
    except Exception as exc:
        raise RuntimeError(
            "MathTex failed to render mathematical notation "
            f"{value!r}. The pipeline will not fall back to Text()/raw LaTeX. "
            f"Underlying error: {exc}"
        ) from exc


def create_mixed_content(value: str, **kwargs: Any):
    """Create a single line composed of correctly typed text/math parts."""
    parts = split_mixed_content(value)
    objects = [
        Text(part.value, **kwargs) if part.kind == "text" else create_math(part.value, **kwargs)
        for part in parts
        if part.value
    ]
    if len(objects) == 1:
        return objects[0]
    group = VGroup(*objects).arrange(RIGHT, buff=0.08, aligned_edge=DOWN)
    return group


def create_content(value: str, **kwargs: Any):
    """Universal entry point used by generated Manim scene code."""
    ensure_pipeline_tex_template()
    parts = split_mixed_content(value)
    if all(part.kind == "text" for part in parts):
        return Text(value, **kwargs)
    if len(parts) == 1 and parts[0].kind == "math":
        return create_math(parts[0].value, **kwargs)
    return create_mixed_content(value, **kwargs)


def ensure_math_renderer_available() -> None:
    """Fail early with a precise diagnosis when MathTex cannot compile."""
    ensure_tex_bin_on_path()
    missing = [name for name in ("latex", "dvisvgm") if shutil.which(name) is None]
    if missing:
        raise RuntimeError(
            "Mathematical rendering requires a LaTeX toolchain; missing: "
            + ", ".join(missing)
            + ". Install MacTeX/BasicTeX (macOS) or TeX Live with dvisvgm (Linux), "
            "ensure /Library/TeX/texbin is on PATH, then restart the pipeline. "
            "Plain Text fallback is intentionally disabled."
        )

    ensure_pipeline_tex_template()
    try:
        MathTex(r"x^2", tex_template=ensure_pipeline_tex_template())
    except Exception as exc:
        message = str(exc)
        hint = ""
        if "standalone.cls" in message:
            hint = (
                " Your TeX install is missing standalone.cls; the pipeline "
                "template should avoid that package — this is unexpected."
            )
        raise RuntimeError(
            "LaTeX is installed but MathTex smoke-test failed."
            + hint
            + f" Underlying error: {exc}"
        ) from exc
