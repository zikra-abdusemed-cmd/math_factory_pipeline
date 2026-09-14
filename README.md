# Math Video Pipeline

Turn a title into a fully narrated math explainer video:

```
title -> narration script -> Manim storyboard -> Manim scene code (per beat)
      -> render each scene (audio + animation already synced/embedded)
      -> stitch into one final .mp4
```

Gemini is used for **all** the text generation (script, storyboard, Manim
code). For the **voice**, you can start with Gemini's own TTS (zero extra
setup, good enough to sanity-check the whole pipeline) and switch to
**OmniVoice** later by changing one line in `.env` — no code changes needed
anywhere else.

## How it fits together

- **manim-voiceover** does the heavy lifting for audio/video sync: each
  scene wraps its narration in `with self.voiceover(text=...) as tracker:`
  and animates with `run_time=tracker.duration`, so the animation always
  exactly fills the spoken audio. manim then embeds that audio directly into
  the rendered `.mp4` — there's no separate "add audio" step, it's already
  in the file coming out of each scene render.
- **Self-repair loop**: LLM-written Manim code is usually right, not always.
  After generating a scene, the pipeline test-renders it at low quality; if
  it errors, the traceback goes straight back to Gemini with a "fix this"
  prompt (up to `MAX_CODEGEN_RETRIES` times) before moving on.
- **Voice provider is a single switch point**: every generated scene imports
  `get_speech_service()` from `tts_services/active_service.py` instead of a
  concrete provider. That function reads `VOICE_PROVIDER` from `.env` and
  returns the right one. Regenerating scenes is never required to swap voices.

## Project layout

```
config.py                       All settings, read from .env
main.py                         CLI entry point
pipeline/
  gemini_client.py              Thin wrapper around google-genai
  script_gen.py                 title -> narration script (beats)
  storyboard_gen.py             script -> Manim visual plan per beat
  scene_codegen.py              storyboard+narration -> Manim code, with self-repair
  render.py                     runs `manim` to render each scene to .mp4
  concat.py                     stitches scene clips into the final video
  orchestrator.py               wires the above into one run
tts_services/
  active_service.py             the single switch point (VOICE_PROVIDER)
  gemini_tts_service.py         test voice: Gemini's native TTS
  omnivoice_service.py          production voice: OmniVoice (see below)
  offline_silent_service.py     free, no-API dry run of the mechanics
generated/<title-slug>/         all output for one run (see below)
```

After a run, `generated/<title-slug>/` contains:

- `script.json`, `storyboard.json` — inspectable intermediate artifacts
- `scenes/scene_<id>.py` — the actual generated Manim source, per beat
- `media/` — manim's normal render output (frames, per-scene .mp4s, cached audio)
- `<title-slug>.mp4` — the final stitched video

Nothing here is deleted between runs, so if step 5 (concat) fails for some
reason, your rendered scene clips are still sitting in `media/` and you can
concat them by hand or re-run.

## Setup

**1. System dependencies** (Manim needs these to render at all):

```bash
# Ubuntu/Debian
sudo apt-get install -y libcairo2-dev libpango1.0-dev ffmpeg pkg-config python3-dev

# macOS
brew install cairo pango ffmpeg
```

SoX is optional (only used if you ever set a non-1.0 `global_speed` on a
voice); safe to skip.

**2. Python environment**

```bash
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

**3. Configure**

```bash
cp .env.example .env
```

Open `.env` and set `GEMINI_API_KEY` to a real key from
[Google AI Studio](https://aistudio.google.com/). Leave `VOICE_PROVIDER=gemini`
for now.

## Usage

```bash
python main.py "The Pythagorean Theorem"
```

Useful flags/env vars:

```bash
# Faster, lower-quality draft while you're iterating on prompts
python main.py "Why 0.999... equals 1" --quality ql

# Sanity-check the mechanics (storyboard -> code -> render -> concat)
# with zero API calls and silent placeholder audio
VOICE_PROVIDER=offline python main.py "Test Topic" --quality ql

# Cap how many scenes/beats a video is allowed to have
MAX_SCENES=4 python main.py "A Quick Fact About Primes"
```

The final path is printed at the end, e.g.:
`generated/the-pythagorean-theorem/the-pythagorean-theorem.mp4`

A single run makes a handful of Gemini text calls (script, storyboard, one
per scene for code — plus one more per retry if a scene needs fixing) and
one Gemini TTS call per `self.voiceover(...)` block. For a 5-beat video
with 2 voiceover blocks per beat, that's roughly 5-15 text calls and
~10 TTS calls — keep an eye on your API usage/billing, especially at
`--quality qh`/`qk` where each render also takes real render time.

## Switching to OmniVoice

`tts_services/omnivoice_service.py` has two backends already implemented —
pick whichever matches what you actually have:

- **`OMNIVOICE_BACKEND=openai_compatible`** — a self-hosted OmniVoice server
  (e.g. the `omnivoice-server` PyPI package) exposing an OpenAI-compatible
  `POST /v1/audio/speech`. Needs a GPU box; set `OMNIVOICE_BASE_URL` to it.
- **`OMNIVOICE_BACKEND=wavespeed`** — the hosted WaveSpeed OmniVoice REST API
  (submit-a-task-then-poll flow). Just needs `OMNIVOICE_API_KEY`, no GPU.

Whichever you use, set `VOICE_PROVIDER=omnivoice` in `.env` and everything
downstream picks it up automatically. **This half was written against the
public docs for each backend but not run against a live OmniVoice endpoint**
(no credentials were available while building this) — double-check the
request/response field names against your actual deployment and adjust
`_call_openai_compatible` / `_call_wavespeed` if it doesn't match exactly.

## Troubleshooting

- **"pangocairo not found" during `pip install`** → you skipped the system
  dependencies step above; install `libcairo2-dev`/`libpango1.0-dev` first.
- **A scene keeps failing all `MAX_CODEGEN_RETRIES` attempts** → open
  `generated/<slug>/scenes/scene_<id>.py` and look at the last error printed;
  the storyboard step occasionally asks for something too elaborate for one
  beat. Simplifying `TARGET_SECONDS_PER_BEAT` (shorter beats, simpler asks)
  usually helps, as does raising `MAX_CODEGEN_RETRIES`.
- **ffmpeg concat fails** → this only happens if scene renders somehow used
  inconsistent encoder settings; the pipeline already falls back to a
  re-encoding concat automatically, so if both attempts fail check
  `generated/<slug>/media/videos/*/**.mp4` play individually to isolate
  which clip is corrupt.
- **Gemini model name errors (404 / not found)** → Google renames/retires
  preview model strings periodically; check the current names at
  https://ai.google.dev/gemini-api/docs/models and update `GEMINI_TEXT_MODEL`
  / `GEMINI_TTS_MODEL` in `.env`.
- **Gemini `503 UNAVAILABLE` / “high demand”** → this is a temporary service
  capacity error, not a bad prompt or API key. The pipeline retries transient
  `429`/`5xx` responses with exponential backoff and jitter; tune
  `GEMINI_RETRY_ATTEMPTS`, `GEMINI_RETRY_BASE_DELAY_SECONDS`, and
  `GEMINI_RETRY_MAX_DELAY_SECONDS` in `.env` if needed. If it still exhausts
  retries, wait and rerun, or switch `GEMINI_TEXT_MODEL` from the preview
  model to a currently supported stable Flash model shown in Gemini's model
  list.
- **Gemini TTS `429 RESOURCE_EXHAUSTED` with `GenerateRequestsPerDay`** → the
  free-tier daily request quota for the TTS model is spent. Retrying cannot
  fix this: wait for its reset, enable billing/use a model with available
  quota, or set `VOICE_PROVIDER=offline` to render a silent video while you
  validate the animation pipeline.
