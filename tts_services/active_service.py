"""
Every generated Manim scene imports get_speech_service() from here instead
of importing a concrete provider directly. That's the whole trick for
"use Gemini now, swap to OmniVoice later": change VOICE_PROVIDER in .env
and every already-generated scene picks up the new provider automatically,
no regeneration or code edits needed.
"""

from __future__ import annotations

from manim_voiceover.services.base import SpeechService

import config


def get_speech_service() -> SpeechService:
    provider = config.pipeline.voice_provider
    if provider == "gemini":
        from tts_services.gemini_tts_service import GeminiTTSService

        return GeminiTTSService()
    if provider == "omnivoice":
        from tts_services.omnivoice_service import OmniVoiceService

        return OmniVoiceService()
    if provider == "offline":
        from tts_services.offline_silent_service import OfflineSilentService

        return OfflineSilentService()
    raise ValueError(f"Unknown VOICE_PROVIDER: {provider!r} (expected 'gemini', 'omnivoice', or 'offline')")
