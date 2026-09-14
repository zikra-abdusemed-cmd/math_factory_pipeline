"""
manim-voiceover SpeechService backed by Gemini's native text-to-speech
(generateContent with responseModalities=["AUDIO"]).

This exists purely as a *stand-in* for OmniVoice while the rest of the
pipeline is being built and tested. When you're ready to switch, implement
`tts_services/omnivoice_service.py::OmniVoiceService.generate_from_text`
the same way and flip VOICE_PROVIDER=omnivoice in .env -- nothing else in
the codebase has to change (see tts_services/active_service.py).

Gemini TTS returns raw 16-bit PCM audio at 24kHz, mono. We wrap it in a
proper .wav header with Python's stdlib `wave` module before handing the
path back to manim-voiceover.
"""

from __future__ import annotations

import wave
from pathlib import Path

from manim import logger
from manim_voiceover._typing import VoiceoverData
from manim_voiceover.helper import remove_bookmarks
from manim_voiceover.services.base import PathLike, SpeechService, initialize_speech_service, path_to_string

import config
from pipeline.gemini_client import generate_content_with_retry

PCM_SAMPLE_RATE = 24000
PCM_SAMPLE_WIDTH = 2  # 16-bit
PCM_CHANNELS = 1


class GeminiTTSService(SpeechService):
    """SpeechService that calls the Gemini API's TTS models."""

    def __init__(self, voice_name: str | None = None, model: str | None = None, **kwargs: object) -> None:
        initialize_speech_service(self, kwargs)
        self.voice_name = voice_name or config.gemini.tts_voice
        self.model = model or config.gemini.tts_model

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
            "service": "gemini_tts",
            "voice_name": self.voice_name,
            "model": self.model,
        }

        cached_result = self.get_cached_result(input_data, cache_dir)
        if cached_result is not None:
            return cached_result

        if path is None:
            audio_path = self.get_audio_basename(input_data) + ".wav"
        else:
            audio_path = path_to_string(path)

        full_path = Path(cache_dir) / audio_path
        pcm_bytes = self._call_gemini_tts(input_text)
        self._write_wav(full_path, pcm_bytes)

        logger.info(f"GeminiTTSService: wrote {full_path}")

        json_dict: VoiceoverData = {
            "input_text": text,
            "input_data": input_data,
            "original_audio": audio_path,
        }
        return json_dict

    # -- internals ---------------------------------------------------

    def _call_gemini_tts(self, text: str) -> bytes:
        from google.genai import types

        response = generate_content_with_retry(
            model=self.model,
            contents=text,
            config=types.GenerateContentConfig(
                response_modalities=["AUDIO"],
                speech_config=types.SpeechConfig(
                    voice_config=types.VoiceConfig(
                        prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=self.voice_name)
                    )
                ),
            ),
        )
        part = response.candidates[0].content.parts[0]
        data = part.inline_data.data
        if data is None:
            raise RuntimeError("Gemini TTS response did not contain audio data.")
        return data

    @staticmethod
    def _write_wav(path: Path, pcm_bytes: bytes) -> None:
        with wave.open(str(path), "wb") as wav_file:
            wav_file.setnchannels(PCM_CHANNELS)
            wav_file.setsampwidth(PCM_SAMPLE_WIDTH)
            wav_file.setframerate(PCM_SAMPLE_RATE)
            wav_file.writeframes(pcm_bytes)
