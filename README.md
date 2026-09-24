# Math Video Pipeline

Turn a title into a fully narrated math explainer video:

```
title -> narration script -> Manim storyboard -> Manim scene code (per beat)
      -> pre-generate TTS into a shared cache
      -> render each scene (audio + animation already synced/embedded)
      -> stitch into one final .mp4
```

OpenAI is used for **all** text generation (script, storyboard, Manim code)
with `gpt-5.6-terra`, and narration uses `gpt-4o-mini-tts`.

## How it fits together

- **manim-voiceover** muxes narration into each scene. Generated scenes are
  rewritten to subclass `SyncedVoiceoverScene`, which caps every `play`/`wait`
  to the remaining spoken duration so animations cannot run past the voice.
  Clips are then duration-normalized before concat so tiny frame-rounding
  mismatches do not accumulate across beats.
- **TTS is pre-generated**: before any final Manim render, the pipeline
  synthesizes each unique narration once into `generated/_voiceover_cache/`.
  Final renders hit that cache instead of calling the TTS API mid-render.
  Re-running the same (or overlapping) narration text also reuses the cache.
- **OpenAI WAV headers are patched**: OpenAI streams WAVs with RIFF size
  `0xFFFFFFFF`, which made Manim treat ~15s of audio as ~25 hours. The pipeline
  rewrites those headers and refuses any voiceover longer than a sane beat.
- **Self-repair loop**: LLM-written Manim code is usually right, not always.
  After generating a scene, the pipeline test-renders it at low quality with
  silent offline audio; if it errors, the traceback goes straight back to the
  model (up to `MAX_CODEGEN_RETRIES` times) before moving on.
- **Voice provider is a single switch point**: every generated scene imports
  `get_speech_service()` from `tts_services/active_service.py`. That function
  reads `VOICE_PROVIDER` from `.env` (`openai` or `offline`).

## Project layout

```
config.py                       All settings, read from .env
main.py                         CLI entry point
pipeline/
  openai_client.py              Thin wrapper around the OpenAI Responses API
  script_gen.py                 title -> narration script (beats)
  storyboard_gen.py             script -> Manim visual plan per beat
  scene_codegen.py              storyboard+narration -> Manim code, with self-repair
  tts_pregen.py                 pre-warm shared TTS cache from scene narrations
  render.py                     runs `manim` to render each scene to .mp4
  concat.py                     stitches scene clips into the final video
  orchestrator.py               wires the above into one run
tts_services/
  active_service.py             the single switch point (VOICE_PROVIDER)
  openai_tts_service.py         narration: OpenAI gpt-4o-mini-tts
  offline_silent_service.py     free, no-API dry run of the mechanics
webapp/                         local FastAPI UI
generated/<title-slug>/         all output for one run (see below)
generated/_voiceover_cache/     shared TTS wav + cache.json across runs
```

After a run, `generated/<title-slug>/` contains:

- `script.json`, `storyboard.json` — inspectable intermediate artifacts
- `scenes/scene_<id>.py` — the actual generated Manim source, per beat
- `media/` — manim's normal render output (frames, per-scene .mp4s)
- `<title-slug>.mp4` — the final stitched video

Nothing here is deleted between runs, so if concat fails for some reason,
your rendered scene clips are still sitting in `media/`.

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

Open `.env` and set `OPENAI_API_KEY` to a real OpenAI API key. Leave
`VOICE_PROVIDER=openai` for production narration.

## Usage

```bash
python main.py "The Pythagorean Theorem"
```

Useful flags/env vars:

```bash
# Faster, lower-quality draft while you're iterating on prompts
python main.py "Why 0.999... equals 1" --quality ql

# High-quality 1080p/60fps final export (substantially slower on CPU-only Macs)
python main.py "Why 0.999... equals 1" --quality qh

# Add a creative brief for the narration and visual style
python main.py "Introduction to Sets" --quality qm \
  --description "Use a friendly classroom example and explain Venn diagrams."

# Sanity-check the mechanics (storyboard -> code -> render -> concat)
# with zero API calls and silent placeholder audio
VOICE_PROVIDER=offline python main.py "Test Topic" --quality ql

# Cap how many scenes/beats a video is allowed to have
MAX_SCENES=4 python main.py "A Quick Fact About Primes"
```

The final path is printed at the end, e.g.:
`generated/the-pythagorean-theorem/the-pythagorean-theorem.mp4`

## Local web UI

After completing setup, start the local interface with:

```bash
uvicorn webapp.server:app
```

Then open [http://127.0.0.1:8000](http://127.0.0.1:8000). The UI uses the
same `.env` and configuration as the CLI, lets you pick render quality, permits
one active generation at a time, and keeps the final video under
`generated/<title-slug>/` as usual. Do not use `--reload` or multiple Uvicorn
workers while generating a video — either would replace the in-memory job
queue and make a still-running browser poll return 404.

A single run makes a handful of OpenAI text calls (script, storyboard, one
per scene for code — plus one more per retry if a scene needs fixing) and
**one OpenAI TTS call per unique narration string** (cached thereafter). Keep
an eye on billing at `--quality qh`/`qk` where each render also takes real
render time.

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
- **OpenAI `429` or `5xx` response** → this is commonly a temporary rate or
  capacity issue. The pipeline retries with exponential backoff and jitter;
  tune `OPENAI_RETRY_ATTEMPTS`, `OPENAI_RETRY_BASE_DELAY_SECONDS`, and
  `OPENAI_RETRY_MAX_DELAY_SECONDS` in `.env` if needed.
- **OpenAI model name error (404 / not found)** → confirm your project can
  access `gpt-5.6-terra` and `gpt-4o-mini-tts`, then check the values in
  `OPENAI_TEXT_MODEL` and `OPENAI_TTS_MODEL`.
- **TTS still seems slow during render** → check that
  `generated/_voiceover_cache/cache.json` contains entries for your narration;
  the pre-generate step should print one line per unique string before rendering.
- **A single draft scene takes tens of minutes / hours** → almost always a
  corrupt streaming WAV (`RIFF` size `0xFFFFFFFF`). Delete
  `generated/_voiceover_cache/` and re-run, or let the next pre-generate step
  scrub/repair it automatically. Confirm each pre-generate line prints a
  sane duration like `-> 17.2s`.
# System dependency for mathematical rendering

Generated scenes use Manim `MathTex` for notation via `pipeline/math_rendering.py`.
You need a working LaTeX toolchain with `latex` and `dvisvgm` on `PATH`
(BasicTeX or MacTeX on macOS; TeX Live + dvisvgm on Linux).

The pipeline does **not** require the `standalone` LaTeX package: it installs a
custom TeX template based on `article` + `amsmath`/`amssymb`, which BasicTeX
already includes. A smoke-test runs before codegen; if MathTex cannot compile,
generation stops with a clear error instead of showing raw LaTeX through `Text`.

On macOS, ensure TeX binaries are visible:

```bash
export PATH="/Library/TeX/texbin:$PATH"
```

Optional (only if you want Manim's default standalone template elsewhere):

```bash
sudo tlmgr install standalone preview
```
