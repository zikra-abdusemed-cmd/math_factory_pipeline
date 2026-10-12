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
from pipeline.scene_codegen import GeneratedScene, normalize_visible_content, _strip_code_fences, _try_render
from pipeline.script_gen import Script, ScriptBeat
from pipeline.storyboard_gen import StoryboardScene
from pipeline.openai_client import generate_video_package
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

    print(f"[1/4] Generating complete video package in one text API call: {title!r}")
    report("generating_script", title=title)
    package = generate_video_package(title, extra_instructions=extra_instructions)
    script = Script(
        title=package.get("title", title),
        audience=package.get("audience", ""),
        beats=[ScriptBeat(id=b["id"], beat_title=b.get("beat_title", b["id"]), narration=b["narration"].strip()) for b in package["beats"]],
    )
    raw_scenes = package["scenes"]
    scenes_by_id = {s["id"]: s for s in raw_scenes}
    if len(scenes_by_id) != len(script.beats) or any(b.id not in scenes_by_id for b in script.beats):
        raise ValueError("Single-call package must contain exactly one scene for every narration beat.")
    storyboard = [
        StoryboardScene(
            id=b.id,
            class_name=scenes_by_id[b.id]["class_name"],
            visual_summary=scenes_by_id[b.id].get("visual_summary", ""),
            steps=list(scenes_by_id[b.id].get("steps", [])),
            on_screen_text=list(scenes_by_id[b.id].get("on_screen_text", [])),
        )
        for b in script.beats
    ]
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

    beats_by_id = {b.id: b for b in script.beats}
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
                f"Single-call generated scene {scene_plan.class_name} failed local validation. "
                f"No repair API call was made, to preserve the one-call limit.\n{error}"
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
