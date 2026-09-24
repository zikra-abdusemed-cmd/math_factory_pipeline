"""
A zero-dependency, zero-network SpeechService that generates silence sized
to roughly how long the text would take to speak. Useful for dry-running
the whole mechanical pipeline (storyboard -> Manim code -> render -> concat)
without burning OpenAI TTS calls, and for CI. Select it with
VOICE_PROVIDER=offline.
"""

from __future__ import annotations

import wave
from pathlib import Path

from manim_voiceover._typing import VoiceoverData
from manim_voiceover.helper import remove_bookmarks
from manim_voiceover.services.base import PathLike, SpeechService, initialize_speech_service, path_to_string

SAMPLE_RATE = 24000
WORDS_PER_SECOND = 2.5


class OfflineSilentService(SpeechService):
    def __init__(self, **kwargs: object) -> None:
        initialize_speech_service(self, kwargs)

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
        input_data = {"input_text": input_text, "service": "offline_silent"}

        cached_result = self.get_cached_result(input_data, cache_dir)
        if cached_result is not None:
            return cached_result

        audio_path = (self.get_audio_basename(input_data) + ".wav") if path is None else path_to_string(path)
        full_path = Path(cache_dir) / audio_path

        n_words = max(1, len(input_text.split()))
        duration_seconds = max(0.6, n_words / WORDS_PER_SECOND)
        n_frames = int(duration_seconds * SAMPLE_RATE)

        with wave.open(str(full_path), "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(SAMPLE_RATE)
            wav_file.writeframes(b"\x00\x00" * n_frames)

        return {"input_text": text, "input_data": input_data, "original_audio": audio_path}
