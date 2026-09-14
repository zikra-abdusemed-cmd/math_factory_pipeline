"""
Thin wrapper around the `google-genai` SDK for the text-generation calls
(script writing, storyboarding, Manim code generation).

This is intentionally small and dependency-light so that if Google changes
the SDK surface again, there is exactly one place to fix it.
"""

from __future__ import annotations

import json
import logging
import random
import re
import time
from typing import Any

from google import genai
from google.genai import types

import config

_client: genai.Client | None = None
logger = logging.getLogger(__name__)


def _is_daily_quota_exhausted(message: str) -> bool:
    """Return whether Gemini has explicitly reported a per-day quota limit."""
    normalized = message.upper()
    return (
        "QUOTA EXCEEDED" in normalized
        and ("PERDAY" in normalized or "PER DAY" in normalized or "REQUESTSPERDAY" in normalized)
    )


def generate_content_with_retry(**kwargs: Any) -> Any:
    """Retry transient Gemini capacity and rate-limit errors with jitter."""
    max_attempts = max(1, config.gemini.retry_attempts)
    for attempt in range(max_attempts):
        try:
            return get_client().models.generate_content(**kwargs)
        except Exception as exc:
            message = str(exc).upper()
            if _is_daily_quota_exhausted(message):
                raise RuntimeError(
                    "Gemini's daily quota for this model is exhausted. Wait for the quota reset, "
                    "enable billing/use a model with available quota, or set VOICE_PROVIDER=offline "
                    "to complete a silent render without Gemini TTS."
                ) from exc
            retryable = any(
                code in message
                for code in ("408", "429", "500", "502", "503", "504", "UNAVAILABLE", "RESOURCE_EXHAUSTED")
            )
            if not retryable or attempt == max_attempts - 1:
                raise
            # Full jitter avoids a herd of callers retrying on the same
            # exponential schedule after a shared capacity spike.
            maximum_delay = min(
                config.gemini.retry_max_delay_seconds,
                config.gemini.retry_base_delay_seconds * (2 ** attempt),
            )
            delay = random.uniform(maximum_delay / 2, maximum_delay)
            print(
                f"Gemini temporarily unavailable; retrying in {delay:.1f}s "
                f"({attempt + 2}/{max_attempts})...",
                flush=True,
            )
            logger.debug("Gemini request failed transiently: %s", exc)
            time.sleep(delay)


def get_client() -> genai.Client:
    global _client
    if _client is None:
        _client = genai.Client(api_key=config.require_gemini_key())
    return _client


def generate_text(
    prompt: str,
    system_instruction: str | None = None,
    temperature: float = 0.8,
    model: str | None = None,
) -> str:
    """Plain free-text generation."""
    resp = generate_content_with_retry(
        model=model or config.gemini.text_model,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=temperature,
        ),
    )
    return resp.text or ""


def generate_json(
    prompt: str,
    system_instruction: str | None = None,
    temperature: float = 0.7,
    model: str | None = None,
) -> Any:
    """
    Generation constrained to JSON output. We ask the model for
    response_mime_type="application/json" AND still defensively strip
    markdown fences / leading-trailing junk before parsing, since that
    combination has proven the most robust across Gemini model versions.
    """
    resp = generate_content_with_retry(
        model=model or config.gemini.text_model,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=temperature,
            response_mime_type="application/json",
        ),
    )
    raw = resp.text or ""
    return _parse_json_loose(raw)


def _parse_json_loose(raw: str) -> Any:
    cleaned = raw.strip()
    cleaned = re.sub(r"^```(json)?", "", cleaned.strip())
    cleaned = re.sub(r"```$", "", cleaned.strip())
    cleaned = cleaned.strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        # Last resort: grab the largest {...} or [...] block in the text.
        match = re.search(r"(\{.*\}|\[.*\])", cleaned, re.DOTALL)
        if match:
            return json.loads(match.group(1))
        raise ValueError(f"Gemini did not return parseable JSON: {e}\n---\n{raw[:2000]}")
