"""
manim-voiceover SpeechService backed by OmniVoice.

This is the "later we'll change it" half of the swap. Two backends are
implemented because OmniVoice shows up in two common shapes in the wild:

  1. "openai_compatible" - a self-hosted OmniVoice server (e.g. the
     `omnivoice-server` PyPI package) exposing an OpenAI-compatible
     POST /v1/audio/speech endpoint that returns raw audio bytes directly.
     Needs a GPU box running the server; set OMNIVOICE_BASE_URL to it.

  2. "wavespeed" - the hosted WaveSpeed OmniVoice REST API, which is an
     async submit-a-task-then-poll-for-a-result-url flow. Needs only an
     OMNIVOICE_API_KEY (no GPU / self-hosting required).

Pick whichever matches what you actually have access to via
OMNIVOICE_BACKEND in .env; both write out a normal audio file and return the
same VoiceoverData shape manim-voiceover expects, so the rest of the
pipeline never needs to know which one is active.

NOTE: this file is a real, runnable implementation, but it's untested
against a live OmniVoice endpoint (no credentials were available while
building this). Double check the request/response shape against whichever
OmniVoice deployment you point it at and adjust field names if needed.
"""

from __future__ import annotations

import time
from pathlib import Path

import requests
from manim import logger
from manim_voiceover._typing import VoiceoverData
from manim_voiceover.helper import remove_bookmarks
from manim_voiceover.services.base import PathLike, SpeechService, initialize_speech_service, path_to_string

import config


class OmniVoiceService(SpeechService):
    """SpeechService that calls an OmniVoice TTS backend."""

    def __init__(
        self,
        voice: str | None = None,
        voice_description: str | None = None,
        backend: str | None = None,
        **kwargs: object,
    ) -> None:
        initialize_speech_service(self, kwargs)
        self.backend = backend or config.omnivoice.backend
        self.voice = voice or config.omnivoice.voice
        self.voice_description = voice_description or config.omnivoice.voice_description

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
            "service": "omnivoice",
            "backend": self.backend,
            "voice": self.voice,
        }

        cached_result = self.get_cached_result(input_data, cache_dir)
        if cached_result is not None:
            return cached_result

        if path is None:
            audio_path = self.get_audio_basename(input_data) + ".wav"
        else:
            audio_path = path_to_string(path)

        full_path = Path(cache_dir) / audio_path

        if self.backend == "openai_compatible":
            audio_bytes = self._call_openai_compatible(input_text)
        elif self.backend == "wavespeed":
            audio_bytes = self._call_wavespeed(input_text)
        else:
            raise ValueError(f"Unknown OMNIVOICE_BACKEND: {self.backend!r}")

        full_path.write_bytes(audio_bytes)
        logger.info(f"OmniVoiceService ({self.backend}): wrote {full_path}")

        json_dict: VoiceoverData = {
            "input_text": text,
            "input_data": input_data,
            "original_audio": audio_path,
        }
        return json_dict

    # -- backend 1: self-hosted, OpenAI-compatible server -------------

    def _call_openai_compatible(self, text: str) -> bytes:
        url = f"{config.omnivoice.base_url.rstrip('/')}/v1/audio/speech"
        headers = {"Content-Type": "application/json"}
        if config.omnivoice.api_key:
            headers["Authorization"] = f"Bearer {config.omnivoice.api_key}"

        payload = {
            "model": "omnivoice",
            "input": text,
            "voice": self.voice,
            "instructions": self.voice_description,
            "response_format": "wav",
        }
        resp = requests.post(url, json=payload, headers=headers, timeout=120)
        resp.raise_for_status()
        return resp.content

    # -- backend 2: hosted WaveSpeed API (submit + poll) ---------------

    def _call_wavespeed(self, text: str, poll_interval: float = 2.0, timeout: float = 120.0) -> bytes:
        if not config.omnivoice.api_key:
            raise RuntimeError("OMNIVOICE_API_KEY is required for the wavespeed backend.")

        submit_url = "https://api.wavespeed.ai/api/v3/wavespeed-ai/omnivoice/text-to-speech"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {config.omnivoice.api_key}",
        }
        payload = {"text": text, "voice_description": self.voice_description}

        submit_resp = requests.post(submit_url, headers=headers, json=payload, timeout=30)
        submit_resp.raise_for_status()
        request_id = submit_resp.json()["data"]["id"]

        result_url = f"https://api.wavespeed.ai/api/v3/predictions/{request_id}/result"
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            poll_resp = requests.get(result_url, headers=headers, timeout=30)
            poll_resp.raise_for_status()
            result = poll_resp.json()["data"]
            status = result.get("status")
            if status == "completed":
                audio_url = result["outputs"][0]
                audio_resp = requests.get(audio_url, timeout=60)
                audio_resp.raise_for_status()
                return audio_resp.content
            if status == "failed":
                raise RuntimeError(f"WaveSpeed OmniVoice task failed: {result}")
            time.sleep(poll_interval)

        raise TimeoutError("Timed out waiting for WaveSpeed OmniVoice result.")
