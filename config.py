"""
Central configuration for the pipeline.

Everything that might change between "testing" and "production" (API keys,
model names, which voice provider is active) lives here and is read from
environment variables / a .env file.
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
class OpenAIConfig:
    api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    # Text model used for script / storyboard / Manim-code generation.
    text_model: str = field(default_factory=lambda: os.getenv("OPENAI_TEXT_MODEL", "gpt-5.6-terra"))
    # TTS model used for narration.
    tts_model: str = field(default_factory=lambda: os.getenv("OPENAI_TTS_MODEL", "gpt-4o-mini-tts"))
    tts_voice: str = field(default_factory=lambda: os.getenv("OPENAI_TTS_VOICE", "alloy"))
    retry_attempts: int = field(default_factory=lambda: _env_int("OPENAI_RETRY_ATTEMPTS", 6))
    retry_base_delay_seconds: float = field(
        default_factory=lambda: _env_float("OPENAI_RETRY_BASE_DELAY_SECONDS", 2.0)
    )
    retry_max_delay_seconds: float = field(
        default_factory=lambda: _env_float("OPENAI_RETRY_MAX_DELAY_SECONDS", 45.0)
    )
    # Speech generation is bounded so a stuck request cannot consume the whole
    # scene-render timeout. Prefer pre-generating TTS via pipeline/tts_pregen.py
    # so Manim itself only hits the shared voiceover cache.
    tts_timeout_seconds: float = field(
        default_factory=lambda: _env_float("OPENAI_TTS_TIMEOUT_SECONDS", 60.0)
    )
    tts_retry_attempts: int = field(default_factory=lambda: _env_int("OPENAI_TTS_RETRY_ATTEMPTS", 3))


@dataclass
class PipelineConfig:
    # "openai" or "offline".
    # Keep a fast, non-network default for local iteration when no explicit
    # provider is requested and no OpenAI key is configured. A user can always
    # override this with VOICE_PROVIDER in the environment.
    voice_provider: str = field(
        default_factory=lambda: os.getenv("VOICE_PROVIDER")
        or ("offline" if not os.getenv("OPENAI_API_KEY") else "openai")
    )
    # manim quality flag: ql (fast/low), qm (balanced), qh (high), qk (4k)
    # Prefer ql for iterative/dev runs; explicit env or CLI overrides still win.
    render_quality: str = field(default_factory=lambda: os.getenv("RENDER_QUALITY", "ql"))
    max_scenes: int = field(default_factory=lambda: _env_int("MAX_SCENES", 6))
    max_codegen_retries: int = field(default_factory=lambda: _env_int("MAX_CODEGEN_RETRIES", 3))
    target_seconds_per_beat: float = field(
        default_factory=lambda: _env_float("TARGET_SECONDS_PER_BEAT", 25.0)
    )
    # High-quality Manim renders can legitimately take longer than 15 minutes
    # on a CPU-only machine. Keep a finite, user-configurable safety limit.
    render_timeout_seconds: int = field(
        default_factory=lambda: _env_int("RENDER_TIMEOUT_SECONDS", 3600)
    )


openai = OpenAIConfig()
pipeline = PipelineConfig()


def require_openai_key() -> str:
    if not openai.api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Put it in a .env file "
            "(see .env.example) or export it in your shell."
        )
    return openai.api_key
