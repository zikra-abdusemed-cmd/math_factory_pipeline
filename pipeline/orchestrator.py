"""
Ties every step together: title -> script -> storyboard -> per-beat scene
code -> TTS pre-generation -> per-beat render -> final concatenated video.

Every intermediate artifact (script.json, storyboard.json, generated scene
.py files, per-scene .mp4s) is kept on disk under generated/<slug>/ so a
run can be inspected, debugged, or resumed by hand if something goes wrong
partway through.
"""

from __future__ import annotations

import dataclasses
import json
import re
from pathlib import Path
from collections.abc import Callable

from slugify import slugify

import config
from pipeline.render import render_scene
from pipeline.concat import concat_videos
from pipeline.scene_codegen import GeneratedScene, normalize_visible_content, _strip_code_fences, _try_render
from pipeline.script_gen import Script, ScriptBeat
from pipeline.storyboard_gen import StoryboardScene
from pipeline.openai_client import generate_video_package, generate_missing_video_scenes
from pipeline.tts_pregen import assert_cached_voiceovers_sane, collect_voiceover_texts, pregenerate_voiceovers
from pipeline.math_rendering import ensure_math_renderer_available


ProgressCallback = Callable[[str, dict], None]


def _match_scenes_to_beats(beat_ids: list[str], scenes: list[dict]) -> dict[str, dict]:
    """Match scenes by exact IDs, falling back to promised order when complete."""
    if any(not isinstance(scene, dict) or not isinstance(scene.get("id"), str) for scene in scenes):
        raise ValueError("Every generated scene must be an object with a string 'id' field.")

    scene_ids = [scene["id"] for scene in scenes]
    scenes_by_id = {scene["id"]: scene for scene in scenes}
    if len(scenes) == len(beat_ids) and len(scenes_by_id) == len(beat_ids) and set(scene_ids) == set(beat_ids):
        return scenes_by_id

    if len(scenes) == len(beat_ids):
        # The prompt requires scenes in beat order. When the model only gets
        # identifiers wrong, pairing by that order salvages the complete reply.
        print(
            "Warning: scene IDs did not match beat IDs; pairing scenes by their "
            f"requested order. beat_ids={beat_ids!r}, scene_ids={scene_ids!r}",
            flush=True,
        )
        return {beat_id: scene for beat_id, scene in zip(beat_ids, scenes)}

    duplicate_ids = sorted({scene_id for scene_id in scene_ids if scene_ids.count(scene_id) > 1})
    missing_ids = [beat_id for beat_id in beat_ids if beat_id not in scenes_by_id]
    extra_ids = [scene_id for scene_id in scene_ids if scene_id not in beat_ids]
    raise ValueError(
        "Single-call package must contain exactly one scene for every narration beat. "
        f"Expected {len(beat_ids)} scenes; received {len(scenes)}. "
        f"Missing beat IDs: {missing_ids!r}; duplicate scene IDs: {duplicate_ids!r}; "
        f"unexpected scene IDs: {extra_ids!r}."
    )


def _prepare_missing_scene_request(
    beats: list[ScriptBeat], scenes: list[object]
) -> tuple[list[dict], list[dict[str, str]]]:
    """Keep one valid scene per expected ID and describe omitted beats."""
    expected_ids = {beat.id for beat in beats}
    retained: list[dict] = []
    seen: set[str] = set()
    for scene in scenes:
        if not isinstance(scene, dict):
            continue
        scene_id = scene.get("id")
        if not isinstance(scene_id, str) or scene_id not in expected_ids or scene_id in seen:
            continue
        retained.append(scene)
        seen.add(scene_id)

    missing = []
    for beat in beats:
        if beat.id in seen:
            continue
        suffix = re.sub(r"[^A-Za-z0-9]+", " ", beat.id).title().replace(" ", "") or "Beat"
        if not suffix[0].isalpha():
            suffix = f"Beat{suffix}"
        missing.append({
            "id": beat.id,
            "beat_title": beat.beat_title,
            "narration": beat.narration,
            "class_name": f"SupplementalScene{suffix}",
        })
    return retained, missing


def _parse_script_beats(raw_beats: list[object]) -> list[ScriptBeat]:
    """Validate model-provided beats before indexing or writing artifacts."""
    beats: list[ScriptBeat] = []
    seen_ids: set[str] = set()
    for index, raw in enumerate(raw_beats, start=1):
        if not isinstance(raw, dict):
            raise ValueError(f"Narration beat {index} must be an object.")
        beat_id = raw.get("id")
        narration = raw.get("narration")
        if not isinstance(beat_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", beat_id.strip()):
            raise ValueError(
                f"Narration beat {index} must have a non-empty 'id' containing only letters, numbers, '_' or '-'."
            )
        beat_id = beat_id.strip()
        if beat_id in seen_ids:
            raise ValueError(f"Narration beat IDs must be unique; duplicate ID: {beat_id!r}.")
        if not isinstance(narration, str) or not narration.strip():
            raise ValueError(f"Narration beat {beat_id!r} must have non-empty string narration.")
        beat_title = raw.get("beat_title", beat_id)
        if not isinstance(beat_title, str):
            beat_title = beat_id
        beats.append(ScriptBeat(id=beat_id, beat_title=beat_title, narration=narration.strip()))
        seen_ids.add(beat_id)
    if len(beats) < 1:
        raise ValueError("OpenAI package must include at least one narration beat.")
    scene_limit = max(4, config.pipeline.max_scenes)
    if len(beats) > scene_limit:
        raise ValueError(
            f"Video package exceeds the effective scene limit of {scene_limit}; received {len(beats)} beats. "
            "The package generator requires at least 4 beats, so MAX_SCENES values below 4 are treated as 4."
        )
    return beats


def _build_storyboard(script: Script, scenes_by_id: dict[str, dict]) -> list[StoryboardScene]:
    """Validate scene metadata so malformed JSON fails before artifact writes."""
    storyboard: list[StoryboardScene] = []
    seen_class_names: set[str] = set()
    for beat in script.beats:
        scene = scenes_by_id[beat.id]
        class_name = scene.get("class_name")
        code = scene.get("code")
        if not isinstance(class_name, str) or not re.fullmatch(r"[A-Za-z_]\w*", class_name):
            raise ValueError(f"Scene for beat {beat.id!r} must have a valid Python 'class_name'.")
        if class_name in seen_class_names:
            raise ValueError(f"Generated scene class names must be unique; duplicate: {class_name!r}.")
        if not isinstance(code, str) or not code.strip():
            raise ValueError(f"Scene for beat {beat.id!r} must include non-empty Python 'code'.")
        steps = scene.get("steps", [])
        on_screen_text = scene.get("on_screen_text", [])
        if not isinstance(steps, list) or not all(isinstance(item, str) for item in steps):
            steps = []
        if not isinstance(on_screen_text, list) or not all(isinstance(item, str) for item in on_screen_text):
            on_screen_text = []
        visual_summary = scene.get("visual_summary", "")
        if not isinstance(visual_summary, str):
            visual_summary = ""
        storyboard.append(StoryboardScene(
            id=beat.id,
            class_name=class_name,
            visual_summary=visual_summary,
            steps=steps,
            on_screen_text=on_screen_text,
        ))
        seen_class_names.add(class_name)
    return storyboard


def run_pipeline(
    title: str,
    quality: str | None = None,
    extra_instructions: str | None = None,
    on_progress: ProgressCallback | None = None,
) -> Path:
    def report(stage: str, **detail: object) -> None:
        if on_progress:
            on_progress(stage, detail)

    quality = quality or config.pipeline.render_quality
    slug = slugify(title, max_length=60) or "untitled"
    run_dir = config.GENERATED_DIR / slug
    run_dir.mkdir(parents=True, exist_ok=True)
    scenes_dir = run_dir / "scenes"
    media_dir = run_dir / "media"

    print(f"[1/4] Generating complete video package in one text API call: {title!r}")
    report("generating_script", title=title)
    # Check local TeX before consuming the package-generation API request.
    ensure_math_renderer_available()
    package = generate_video_package(title, extra_instructions=extra_instructions)
    if not isinstance(package, dict):
        raise ValueError(f"OpenAI package must be a JSON object, got {type(package).__name__}.")
    raw_beats = package.get("beats")
    raw_scenes = package.get("scenes")
    if not isinstance(raw_beats, list) or not isinstance(raw_scenes, list):
        raise ValueError(
            "OpenAI package must include 'beats' and 'scenes' arrays; "
            f"got beats={type(raw_beats).__name__}, scenes={type(raw_scenes).__name__}."
        )
    package_title = package.get("title", title)
    package_audience = package.get("audience", "")
    script = Script(
        title=package_title if isinstance(package_title, str) and package_title.strip() else title,
        audience=package_audience if isinstance(package_audience, str) else "",
        beats=_parse_script_beats(raw_beats),
    )
    beat_ids = [beat.id for beat in script.beats]
    if len(raw_scenes) != len(beat_ids):
        retained_scenes, missing_beats = _prepare_missing_scene_request(script.beats, raw_scenes)
        print(
            f"Package omitted {len(missing_beats)} scene(s); requesting only those beat scenes.",
            flush=True,
        )
        report(
            "completing_missing_scenes",
            scene_total=len(beat_ids),
            missing_scene_ids=[beat["id"] for beat in missing_beats],
        )
        supplemental_scenes = generate_missing_video_scenes(
            title=script.title,
            audience=script.audience,
            missing_beats=missing_beats,
            existing_scenes=retained_scenes,
            extra_instructions=extra_instructions,
        )
        raw_scenes = retained_scenes + supplemental_scenes
    scenes_by_id = _match_scenes_to_beats(beat_ids, raw_scenes)
    storyboard = _build_storyboard(script, scenes_by_id)
    (run_dir / "script.json").write_text(
        json.dumps(dataclasses.asdict(script), indent=2), encoding="utf-8"
    )
    print(f"      -> {len(script.beats)} beats")
    report("script_generated", scene_total=len(script.beats))

    print("[2/4] Saving storyboard and scene files from package")
    report("generating_storyboard", scene_total=len(script.beats))
    (run_dir / "storyboard.json").write_text(
        json.dumps([dataclasses.asdict(s) for s in storyboard], indent=2), encoding="utf-8"
    )
    report("storyboard_generated", scene_total=len(storyboard))

    generated_scenes = []

    scenes_dir.mkdir(parents=True, exist_ok=True)
    for i, scene_plan in enumerate(storyboard, start=1):
        print(f"[2/4] ({i}/{len(storyboard)}) Validating generated scene '{scene_plan.class_name}'")
        report(
            "generating_scene_code",
            scene_index=i,
            scene_total=len(storyboard),
            class_name=scene_plan.class_name,
        )
        code = normalize_visible_content(_strip_code_fences(scenes_by_id[scene_plan.id]["code"]))
        module_stem = f"scene_{scene_plan.id}"
        file_path = scenes_dir / f"{module_stem}.py"
        file_path.write_text(code, encoding="utf-8")
        ok, error = _try_render(file_path, scene_plan.class_name, config.PROJECT_ROOT)
        if not ok:
            raise RuntimeError(
                f"Generated scene {scene_plan.class_name} failed local validation.\n{error}"
            )
        generated = GeneratedScene(class_name=scene_plan.class_name, file_path=file_path, module_stem=module_stem)
        generated_scenes.append(generated)
        report(
            "scene_code_generated",
            scene_index=i,
            scene_total=len(storyboard),
            class_name=generated.class_name,
        )

    print("[3/4] Pre-generating narration audio (separate TTS API; shared cache)")
    report("pregenerating_voiceovers", scene_total=len(generated_scenes))
    voice_texts = collect_voiceover_texts(
        [scene.file_path for scene in generated_scenes],
        fallback_narrations=[beat.narration for beat in script.beats],
    )
    warmed = pregenerate_voiceovers(voice_texts)
    print(f"      -> {warmed} unique narration(s) ready")
    assert_cached_voiceovers_sane(voice_texts)
    report("voiceovers_ready", count=warmed, scene_total=len(generated_scenes))

    clip_paths: list[Path] = []
    for i, generated in enumerate(generated_scenes, start=1):
        print(f"[3/4] ({i}/{len(generated_scenes)}) Rendering '{generated.class_name}' at quality={quality}")
        report(
            "rendering_scene",
            scene_index=i,
            scene_total=len(generated_scenes),
            class_name=generated.class_name,
        )
        clip_path = render_scene(
            generated,
            media_dir=media_dir,
            project_root=config.PROJECT_ROOT,
            quality=quality,
        )
        clip_paths.append(clip_path)
        print(f"      -> {clip_path}")
        report(
            "scene_rendered",
            scene_index=i,
            scene_total=len(generated_scenes),
            class_name=generated.class_name,
        )

    print(f"[4/4] Stitching {len(clip_paths)} clips into the final video")
    report("stitching", scene_total=len(clip_paths))
    output_path = run_dir / f"{slug}.mp4"
    final_path = concat_videos(clip_paths, output_path=output_path, tmp_dir=run_dir / "_concat_tmp")

    print(f"\nDone: {final_path}")
    report("final_video_stitched", path=str(final_path), scene_total=len(clip_paths))
    return final_path
