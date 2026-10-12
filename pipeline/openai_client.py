"""Small OpenAI Responses API wrapper for the pipeline's text-generation calls."""

from __future__ import annotations

import json
import logging
import random
import re
import time
from typing import Any

from openai import OpenAI

import config

_client: OpenAI | None = None
logger = logging.getLogger(__name__)


def _is_retryable(message: str) -> bool:
    return any(code in message.upper() for code in ("408", "409", "429", "500", "502", "503", "504"))


def _create_response_with_retry(**kwargs: Any) -> Any:
    """Create a response, retrying transient rate-limit and server errors."""
    max_attempts = max(1, config.openai.retry_attempts)
    for attempt in range(max_attempts):
        try:
            return get_client().responses.create(**kwargs)
        except Exception as exc:
            if not _is_retryable(str(exc)) or attempt == max_attempts - 1:
                raise
            maximum_delay = min(config.openai.retry_max_delay_seconds, config.openai.retry_base_delay_seconds * (2 ** attempt))
            delay = random.uniform(maximum_delay / 2, maximum_delay)
            print(f"OpenAI temporarily unavailable; retrying in {delay:.1f}s ({attempt + 2}/{max_attempts})...", flush=True)
            logger.debug("OpenAI request failed transiently: %s", exc)
            time.sleep(delay)


def get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=config.require_openai_key())
    return _client


def generate_text(prompt: str, system_instruction: str | None = None, temperature: float = 0.8, model: str | None = None) -> str:
    """Plain free-text generation."""
    del temperature  # GPT-5.6 Responses handles sampling internally.
    response = _create_response_with_retry(model=model or config.openai.text_model, input=prompt, instructions=system_instruction)
    return response.output_text or ""


def generate_json(prompt: str, system_instruction: str | None = None, temperature: float = 0.7, model: str | None = None) -> Any:
    """Generate JSON, retaining defensive parsing for malformed model output."""
    del temperature
    response = _create_response_with_retry(
        model=model or config.openai.text_model,
        input=prompt,
        instructions=system_instruction,
        text={"format": {"type": "json_object"}},
    )
    return _parse_json_loose(response.output_text or "")


def generate_video_package(title: str, extra_instructions: str | None = None) -> Any:
    """Generate narration, storyboard, and all scene sources in one model request."""
    from pipeline.scene_codegen import CODEGEN_SYSTEM_PROMPT
    from pipeline.script_gen import SCRIPT_SYSTEM_PROMPT
    from pipeline.storyboard_gen import STORYBOARD_SYSTEM_PROMPT

    words_per_second = 2.5
    target_seconds = config.pipeline.target_seconds_per_beat
    min_words = int(target_seconds * words_per_second * 0.7)
    max_words = int(target_seconds * words_per_second * 1.3)
    script_rules = SCRIPT_SYSTEM_PROMPT.format(
        target_seconds=int(target_seconds), min_words=min_words, max_words=max_words
    )
    direction = f"\nAdditional creative direction: {extra_instructions.strip()}" if extra_instructions and extra_instructions.strip() else ""
    prompt = f"""Create a complete math explainer video package in one response.

Title: {title}
Produce 4 to {config.pipeline.max_scenes} narration beats, each around {target_seconds} seconds.{direction}

NARRATION RULES:
{script_rules}

STORYBOARD RULES:
{STORYBOARD_SYSTEM_PROMPT}

SCENE CODE RULES:
{CODEGEN_SYSTEM_PROMPT}

Return one JSON object with this exact shape:
{{
  "title": "...",
  "audience": "...",
  "beats": [{{"id":"b1","beat_title":"...","narration":"..."}}],
  "scenes": [{{"id":"b1","class_name":"...","visual_summary":"...","steps":["..."],"on_screen_text":[],"code":"complete Python source for this scene"}}]
}}

Return exactly one scene per beat in the same order. Each code value must be a complete standalone Python file satisfying all scene code rules. Escape newlines and quotes as valid JSON. No markdown or extra keys.
"""
    response = _create_response_with_retry(
        model=config.openai.text_model,
        input=prompt,
        text={"format": {"type": "json_object"}},
    )
    return _parse_json_loose(response.output_text or "")


def _parse_json_loose(raw: str) -> Any:
    cleaned = re.sub(r"```$", "", re.sub(r"^```(json)?", "", raw.strip()).strip()).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as exc:
        match = re.search(r"(\{.*\}|\[.*\])", cleaned, re.DOTALL)
        if match:
            return json.loads(match.group(1))
        raise ValueError(f"OpenAI did not return parseable JSON: {exc}\n---\n{raw[:2000]}") from exc
