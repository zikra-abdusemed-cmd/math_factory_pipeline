"""
Every generated Manim scene imports get_speech_service() from here instead
of importing a concrete provider directly. Change VOICE_PROVIDER in .env
and every already-generated scene picks up the new provider automatically.
"""

from __future__ import annotations

from pathlib import Path

from manim_voiceover.services.base import SpeechService

import config


def voiceover_cache_dir() -> Path:
    """
    Shared TTS cache so narrations are synthesized once across runs/renders.

    Offline validation uses a separate directory so silent placeholders never
    mix with real OpenAI WAVs in the production cache.
    """
    if config.pipeline.voice_provider == "offline":
        path = config.GENERATED_DIR / "_voiceover_cache_offline"
    else:
        path = config.GENERATED_DIR / "_voiceover_cache"
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_speech_service(cache_dir: Path | None = None) -> SpeechService:
    provider = config.pipeline.voice_provider
    kwargs: dict[str, object] = {"cache_dir": cache_dir or voiceover_cache_dir()}

    if provider == "openai":
        from tts_services.openai_tts_service import OpenAITTSService

        return OpenAITTSService(**kwargs)
    if provider == "offline":
        from tts_services.offline_silent_service import OfflineSilentService

        return OfflineSilentService(**kwargs)
    raise ValueError(f"Unknown VOICE_PROVIDER: {provider!r} (expected 'openai' or 'offline')")
