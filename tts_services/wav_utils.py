"""Helpers for OpenAI/streaming WAV files used by manim-voiceover.

OpenAI's speech API writes WAV files with RIFF/data chunk sizes set to
``0xFFFFFFFF`` (streaming placeholder). Python's ``wave`` / mutagen then
report ~25 hours of audio, and Manim stretches every animation to match.
"""

from __future__ import annotations

import struct
import wave
from pathlib import Path

# Hard ceiling for a single beat. Narration targets ~20-30s; anything far
# beyond this is treated as corrupt / unusable for rendering.
MAX_VOICEOVER_SECONDS = 180.0
MIN_VOICEOVER_SECONDS = 0.25
WORDS_PER_SECOND = 2.5


def expected_speech_seconds(text: str) -> float:
    words = max(1, len(text.split()))
    return max(MIN_VOICEOVER_SECONDS, words / WORDS_PER_SECOND)


def max_allowed_seconds(text: str) -> float:
    """Allow generous slack over the word-count estimate, but never past the hard cap."""
    return min(MAX_VOICEOVER_SECONDS, max(30.0, expected_speech_seconds(text) * 4.0))


def fix_streaming_wav_headers(path: Path) -> bool:
    """
    Rewrite RIFF + ``data`` chunk sizes from the on-disk file length.

    Returns True when the file was modified.
    """
    raw = path.read_bytes()
    if len(raw) < 44 or raw[0:4] != b"RIFF" or raw[8:12] != b"WAVE":
        raise ValueError(f"{path} is not a RIFF/WAVE file")

    data = bytearray(raw)
    offset = 12
    data_chunk_offset: int | None = None
    while offset + 8 <= len(data):
        chunk_id = bytes(data[offset : offset + 4])
        chunk_size = struct.unpack_from("<I", data, offset + 4)[0]
        if chunk_id == b"data":
            data_chunk_offset = offset
            break
        if chunk_size in (0xFFFFFFFF, 0x7FFFFFFF):
            raise ValueError(f"{path} has an unbounded non-data chunk {chunk_id!r}")
        offset += 8 + chunk_size + (chunk_size % 2)

    if data_chunk_offset is None:
        raise ValueError(f"{path} has no data chunk")

    actual_data_size = len(data) - (data_chunk_offset + 8)
    if actual_data_size <= 0:
        raise ValueError(f"{path} has an empty data chunk")

    changed = False
    correct_riff_size = len(data) - 8
    if struct.unpack_from("<I", data, 4)[0] != correct_riff_size:
        struct.pack_into("<I", data, 4, correct_riff_size)
        changed = True

    if struct.unpack_from("<I", data, data_chunk_offset + 4)[0] != actual_data_size:
        struct.pack_into("<I", data, data_chunk_offset + 4, actual_data_size)
        changed = True

    if changed:
        path.write_bytes(data)
    return changed


def wav_duration_seconds(path: Path) -> float:
    with wave.open(str(path), "rb") as handle:
        rate = handle.getframerate()
        if rate <= 0:
            raise ValueError(f"{path} has an invalid sample rate")
        return handle.getnframes() / float(rate)


def ensure_sane_voiceover_wav(path: Path, text: str = "") -> float:
    """
    Fix streaming headers if needed and assert the duration is usable for Manim.

    Returns the corrected duration in seconds.
    """
    fix_streaming_wav_headers(path)
    duration = wav_duration_seconds(path)
    limit = max_allowed_seconds(text) if text.strip() else MAX_VOICEOVER_SECONDS
    if duration < MIN_VOICEOVER_SECONDS or duration > limit:
        raise RuntimeError(
            f"Voiceover WAV {path.name} has duration {duration:.2f}s, "
            f"outside the allowed range [{MIN_VOICEOVER_SECONDS:.2f}, {limit:.2f}]s. "
            "Refusing to hand a pathological duration to Manim."
        )
    return duration


def scrub_voiceover_cache(cache_dir: Path) -> dict[str, int]:
    """
    Repair or delete corrupt WAVs under a cache directory.

    OpenAI streaming WAVs are repaired in place (PCM payload is fine).
    Files that remain pathological after repair are deleted so the next TTS
    call regenerates them.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    repaired = 0
    deleted = 0
    ok = 0

    for wav_path in sorted(cache_dir.glob("*.wav")):
        try:
            changed = fix_streaming_wav_headers(wav_path)
            duration = wav_duration_seconds(wav_path)
            if duration < MIN_VOICEOVER_SECONDS or duration > MAX_VOICEOVER_SECONDS:
                wav_path.unlink(missing_ok=True)
                deleted += 1
                continue
            if changed:
                repaired += 1
            else:
                ok += 1
        except Exception:
            wav_path.unlink(missing_ok=True)
            deleted += 1

    return {"ok": ok, "repaired": repaired, "deleted": deleted}
