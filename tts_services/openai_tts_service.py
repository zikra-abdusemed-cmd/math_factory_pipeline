"""manim-voiceover SpeechService backed by OpenAI's speech API."""

from __future__ import annotations

import random
import time
from pathlib import Path

from manim import logger
from manim_voiceover._typing import VoiceoverData
from manim_voiceover.helper import remove_bookmarks
from manim_voiceover.services.base import PathLike, SpeechService, initialize_speech_service, path_to_string

import config
from pipeline.openai_client import get_client
from tts_services.wav_utils import ensure_sane_voiceover_wav


class OpenAITTSService(SpeechService):
    """SpeechService using ``gpt-4o-mini-tts`` and a built-in OpenAI voice."""

    def __init__(self, voice_name: str | None = None, model: str | None = None, **kwargs: object) -> None:
        initialize_speech_service(self, kwargs)
        self.voice_name = voice_name or config.openai.tts_voice
        self.model = model or config.openai.tts_model

    def generate_from_text(
        self,
        text: str,
        cache_dir: PathLike | None = None,
        path: PathLike | None = None,
        **kwargs: object,
    ) -> VoiceoverData:
        if cache_dir is None:
            cache_dir = self.cache_dir
        input_text = remove_bookmarks(text)
        input_data = {
            "input_text": input_text,
            "service": "openai_tts",
            "voice_name": self.voice_name,
            "model": self.model,
        }

        cached_result = self.get_cached_result(input_data, cache_dir)
        if cached_result is not None:
            cached_audio = Path(cache_dir) / str(cached_result["original_audio"])
            if cached_audio.is_file():
                try:
                    duration = ensure_sane_voiceover_wav(cached_audio, input_text)
                    logger.info("OpenAITTSService: cache hit %s (%.2fs)", cached_audio.name, duration)
                    return cached_result
                except Exception as exc:
                    logger.warning(
                        "OpenAITTSService: rejecting corrupt cached wav %s (%s); regenerating",
                        cached_audio,
                        exc,
                    )
                    cached_audio.unlink(missing_ok=True)
            else:
                logger.warning(
                    "OpenAITTSService: cache entry missing on disk (%s); regenerating",
                    cached_audio,
                )

        audio_path = self.get_audio_basename(input_data) + ".wav" if path is None else path_to_string(path)
        full_path = Path(cache_dir) / audio_path
        full_path.parent.mkdir(parents=True, exist_ok=True)
        self._write_speech_audio(input_text, full_path)

        duration = ensure_sane_voiceover_wav(full_path, input_text)
        logger.info("OpenAITTSService: wrote %s (%.2fs)", full_path, duration)
        return {"input_text": text, "input_data": input_data, "original_audio": audio_path}

    def _write_speech_audio(self, input_text: str, full_path: Path) -> None:
        """Write narration with bounded, retryable OpenAI speech requests."""
        max_attempts = max(1, config.openai.tts_retry_attempts)
        for attempt in range(max_attempts):
            try:
                # Disable the SDK's nested automatic retries. This service owns
                # the retry budget, keeping a failed TTS request predictable.
                client = get_client().with_options(max_retries=0)
                with client.audio.speech.with_streaming_response.create(
                    model=self.model,
                    voice=self.voice_name,
                    input=input_text,
                    response_format="wav",
                    timeout=config.openai.tts_timeout_seconds,
                ) as response:
                    response.stream_to_file(full_path)
                # OpenAI streams WAVs with RIFF/data sizes set to 0xFFFFFFFF.
                # Patch headers immediately so Manim never sees a ~25h duration.
                ensure_sane_voiceover_wav(full_path, input_text)
                return
            except Exception as exc:
                full_path.unlink(missing_ok=True)
                message = str(exc).upper()
                retryable = any(
                    code in message for code in ("408", "409", "429", "500", "502", "503", "504", "TIMEOUT")
                ) or "OUTSIDE THE ALLOWED RANGE" in message
                if not retryable or attempt == max_attempts - 1:
                    raise RuntimeError(
                        f"OpenAI TTS failed after {attempt + 1}/{max_attempts} attempts. "
                        f"Each attempt is limited to {config.openai.tts_timeout_seconds:g} seconds: {exc}"
                    ) from exc
                maximum_delay = min(
                    config.openai.retry_max_delay_seconds,
                    config.openai.retry_base_delay_seconds * (2**attempt),
                )
                delay = random.uniform(maximum_delay / 2, maximum_delay)
                logger.warning(
                    "OpenAI TTS request failed (%s); retrying in %.1f seconds (%s/%s)",
                    exc,
                    delay,
                    attempt + 2,
                    max_attempts,
                )
                time.sleep(delay)
