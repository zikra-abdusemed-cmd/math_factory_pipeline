"""
Central configuration for the pipeline.

Everything that might change between "testing" and "production" (API keys,
model names, which voice provider is active) lives here and is read from
environment variables / a .env file. Swapping OmniVoice in later is meant to
be a one-line change: set VOICE_PROVIDER=omnivoice and fill in the OMNIVOICE_*
values -- no other code should need to change.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

# Load a .env file from the project root if present. Real secrets should live
# in .env (gitignored), never hardcoded here.
load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent
GENERATED_DIR = PROJECT_ROOT / "generated"


def _env_int(name: str, default: int) -> int:
    val = os.getenv(name)
    return int(val) if val else default


def _env_float(name: str, default: float) -> float:
    val = os.getenv(name)
    return float(val) if val else default


@dataclass
class GeminiConfig:
    api_key: str = field(default_factory=lambda: os.getenv("GEMINI_API_KEY", ""))
    # Text model used for script / storyboard / Manim-code generation.
    text_model: str = field(default_factory=lambda: os.getenv("GEMINI_TEXT_MODEL", "gemini-3-flash-preview"))
    # TTS model used ONLY as the "test" stand-in for OmniVoice.
    tts_model: str = field(default_factory=lambda: os.getenv("GEMINI_TTS_MODEL", "gemini-3.1-flash-tts-preview"))
    tts_voice: str = field(default_factory=lambda: os.getenv("GEMINI_TTS_VOICE", "Kore"))
    # 503s are temporary capacity failures.  Keep these separately tunable so
    # an interactive run can wait for capacity without retrying forever.
    retry_attempts: int = field(default_factory=lambda: _env_int("GEMINI_RETRY_ATTEMPTS", 6))
    retry_base_delay_seconds: float = field(
        default_factory=lambda: _env_float("GEMINI_RETRY_BASE_DELAY_SECONDS", 2.0)
    )
    retry_max_delay_seconds: float = field(
        default_factory=lambda: _env_float("GEMINI_RETRY_MAX_DELAY_SECONDS", 45.0)
    )


@dataclass
class OmniVoiceConfig:
    # "openai_compatible" -> a self-hosted omnivoice-server (OpenAI /v1/audio/speech shape)
    # "wavespeed"          -> the hosted WaveSpeed OmniVoice REST API
    backend: str = field(default_factory=lambda: os.getenv("OMNIVOICE_BACKEND", "openai_compatible"))
    base_url: str = field(default_factory=lambda: os.getenv("OMNIVOICE_BASE_URL", "http://127.0.0.1:8880"))
    api_key: str = field(default_factory=lambda: os.getenv("OMNIVOICE_API_KEY", ""))
    voice: str = field(default_factory=lambda: os.getenv("OMNIVOICE_VOICE", "alloy"))
    voice_description: str = field(
        default_factory=lambda: os.getenv("OMNIVOICE_VOICE_DESCRIPTION", "male, calm, moderate pitch")
    )


@dataclass
class PipelineConfig:
    # "gemini" (default, for testing) or "omnivoice" (swap in later)
    voice_provider: str = field(default_factory=lambda: os.getenv("VOICE_PROVIDER", "gemini"))
    # manim quality flag: ql (fast/low), qm (medium), qh (high), qk (4k)
    render_quality: str = field(default_factory=lambda: os.getenv("RENDER_QUALITY", "qh"))
    max_scenes: int = field(default_factory=lambda: _env_int("MAX_SCENES", 6))
    max_codegen_retries: int = field(default_factory=lambda: _env_int("MAX_CODEGEN_RETRIES", 3))
    target_seconds_per_beat: float = field(
        default_factory=lambda: _env_float("TARGET_SECONDS_PER_BEAT", 25.0)
    )


gemini = GeminiConfig()
omnivoice = OmniVoiceConfig()
pipeline = PipelineConfig()


def require_gemini_key() -> str:
    if not gemini.api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is not set. Put it in a .env file "
            "(see .env.example) or export it in your shell."
        )
    return gemini.api_key
