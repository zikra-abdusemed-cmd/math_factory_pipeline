"""
Pre-generate narration audio into the shared manim-voiceover cache.

Final Manim renders used to call the TTS API inside every `self.voiceover()`
block. That put network latency on the critical path of each scene render and
could re-bill the same narration if codegen validation and final render diverged.

This module synthesizes each unique narration string once (into the shared
cache used by `get_speech_service()`), so Manim only does a cache hit.
"""

from __future__ import annotations

import ast
from pathlib import Path

from tts_services.active_service import get_speech_service, voiceover_cache_dir
from tts_services.wav_utils import ensure_sane_voiceover_wav, scrub_voiceover_cache, wav_duration_seconds


def extract_voiceover_texts(scene_path: Path) -> list[str]:
    """Return every literal `self.voiceover(text=...)` string in a scene file."""
    source = scene_path.read_text(encoding="utf-8")
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []

    texts: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not (isinstance(func, ast.Attribute) and func.attr == "voiceover"):
            continue

        for keyword in node.keywords:
            if keyword.arg == "text" and isinstance(keyword.value, ast.Constant) and isinstance(keyword.value.value, str):
                texts.append(keyword.value.value)

        if node.args and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str):
            texts.append(node.args[0].value)

    return texts


def _unique_preserve_order(texts: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for text in texts:
        normalized = " ".join(text.split())
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        ordered.append(text)
    return ordered


def collect_voiceover_texts(scene_paths: list[Path], fallback_narrations: list[str] | None = None) -> list[str]:
    """Prefer exact strings from generated scenes; fall back to script narrations."""
    texts: list[str] = []
    for path in scene_paths:
        texts.extend(extract_voiceover_texts(path))
    if not texts and fallback_narrations:
        texts.extend(fallback_narrations)
    return _unique_preserve_order(texts)


def pregenerate_voiceovers(texts: list[str]) -> int:
    """
    Synthesize each narration into the shared voiceover cache.

    Returns the number of unique strings processed (cache hits still count —
    they are cheap and confirm the cache is warm for Manim).
    """
    unique = _unique_preserve_order(texts)
    if not unique:
        return 0

    cache_dir = voiceover_cache_dir()
    scrub = scrub_voiceover_cache(cache_dir)
    if scrub["repaired"] or scrub["deleted"]:
        print(
            f"      -> cache scrub: repaired={scrub['repaired']} "
            f"deleted={scrub['deleted']} ok={scrub['ok']}"
        )

    service = get_speech_service(cache_dir=cache_dir)
    for i, text in enumerate(unique, start=1):
        print(f"      ({i}/{len(unique)}) TTS: {text[:72]}{'…' if len(text) > 72 else ''}")
        # `_wrap_generate_from_text` both writes the wav and appends cache.json,
        # which is what Manim's VoiceoverScene checks on the next render.
        result = service._wrap_generate_from_text(text)
        audio_path = Path(service.cache_dir) / str(result["original_audio"])
        duration = ensure_sane_voiceover_wav(audio_path, text)
        print(f"         -> {duration:.1f}s ({audio_path.name})")
    return len(unique)


def assert_cached_voiceovers_sane(texts: list[str]) -> None:
    """Final guard before Manim: every narration must resolve to a sane-duration WAV."""
    unique = _unique_preserve_order(texts)
    if not unique:
        return
    service = get_speech_service()
    for text in unique:
        result = service.generate_from_text(text)
        audio_path = Path(service.cache_dir) / str(result["original_audio"])
        duration = wav_duration_seconds(audio_path)
        ensure_sane_voiceover_wav(audio_path, text)
        print(f"      verified {audio_path.name}: {duration:.1f}s")
