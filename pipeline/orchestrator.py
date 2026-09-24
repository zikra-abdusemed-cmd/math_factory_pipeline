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
from pathlib import Path
from collections.abc import Callable

from slugify import slugify

import config
from pipeline.render import render_scene
from pipeline.concat import concat_videos
from pipeline.scene_codegen import generate_scene_code
from pipeline.script_gen import generate_script
from pipeline.storyboard_gen import generate_storyboard
from pipeline.tts_pregen import assert_cached_voiceovers_sane, collect_voiceover_texts, pregenerate_voiceovers


ProgressCallback = Callable[[str, dict], None]


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

    print(f"[1/6] Generating narration script for: {title!r}")
    report("generating_script", title=title)
    script = generate_script(title, extra_instructions=extra_instructions)
    (run_dir / "script.json").write_text(
        json.dumps(dataclasses.asdict(script), indent=2), encoding="utf-8"
    )
    print(f"      -> {len(script.beats)} beats")
    report("script_generated", scene_total=len(script.beats))

    print("[2/6] Generating Manim storyboard")
    report("generating_storyboard", scene_total=len(script.beats))
    storyboard = generate_storyboard(script)
    (run_dir / "storyboard.json").write_text(
        json.dumps([dataclasses.asdict(s) for s in storyboard], indent=2), encoding="utf-8"
    )
    report("storyboard_generated", scene_total=len(storyboard))

    beats_by_id = {b.id: b for b in script.beats}
    generated_scenes = []

    for i, scene_plan in enumerate(storyboard, start=1):
        beat = beats_by_id[scene_plan.id]
        print(f"[3/6] ({i}/{len(storyboard)}) Generating + validating Manim code for '{scene_plan.class_name}'")
        report(
            "generating_scene_code",
            scene_index=i,
            scene_total=len(storyboard),
            class_name=scene_plan.class_name,
        )
        generated = generate_scene_code(
            beat=beat,
            storyboard=scene_plan,
            scenes_dir=scenes_dir,
            project_root=config.PROJECT_ROOT,
        )
        generated_scenes.append(generated)
        report(
            "scene_code_generated",
            scene_index=i,
            scene_total=len(storyboard),
            class_name=generated.class_name,
        )

    print("[4/6] Pre-generating narration audio (shared TTS cache)")
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
        print(f"[5/6] ({i}/{len(generated_scenes)}) Rendering '{generated.class_name}' at quality={quality}")
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

    print(f"[6/6] Stitching {len(clip_paths)} clips into the final video")
    report("stitching", scene_total=len(clip_paths))
    output_path = run_dir / f"{slug}.mp4"
    final_path = concat_videos(clip_paths, output_path=output_path, tmp_dir=run_dir / "_concat_tmp")

    print(f"\nDone: {final_path}")
    report("final_video_stitched", path=str(final_path), scene_total=len(clip_paths))
    return final_path
