"""
Ties every step together: title -> script -> storyboard -> per-beat scene
code -> per-beat render -> final concatenated, narrated video.

Every intermediate artifact (script.json, storyboard.json, generated scene
.py files, per-scene .mp4s) is kept on disk under generated/<slug>/ so a
run can be inspected, debugged, or resumed by hand if something goes wrong
partway through.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

from slugify import slugify

import config
from pipeline.render import render_scene
from pipeline.concat import concat_videos
from pipeline.scene_codegen import generate_scene_code
from pipeline.script_gen import generate_script
from pipeline.storyboard_gen import generate_storyboard


def run_pipeline(title: str, quality: str | None = None) -> Path:
    quality = quality or config.pipeline.render_quality
    slug = slugify(title, max_length=60) or "untitled"
    run_dir = config.GENERATED_DIR / slug
    run_dir.mkdir(parents=True, exist_ok=True)
    scenes_dir = run_dir / "scenes"
    media_dir = run_dir / "media"

    print(f"[1/5] Generating narration script for: {title!r}")
    script = generate_script(title)
    (run_dir / "script.json").write_text(
        json.dumps(dataclasses.asdict(script), indent=2), encoding="utf-8"
    )
    print(f"      -> {len(script.beats)} beats")

    print("[2/5] Generating Manim storyboard")
    storyboard = generate_storyboard(script)
    (run_dir / "storyboard.json").write_text(
        json.dumps([dataclasses.asdict(s) for s in storyboard], indent=2), encoding="utf-8"
    )

    beats_by_id = {b.id: b for b in script.beats}
    clip_paths: list[Path] = []

    for i, scene_plan in enumerate(storyboard, start=1):
        beat = beats_by_id[scene_plan.id]
        print(f"[3/5] ({i}/{len(storyboard)}) Generating + validating Manim code for '{scene_plan.class_name}'")
        generated = generate_scene_code(
            beat=beat,
            storyboard=scene_plan,
            scenes_dir=scenes_dir,
            project_root=config.PROJECT_ROOT,
        )

        print(f"[4/5] ({i}/{len(storyboard)}) Rendering '{generated.class_name}' at quality={quality}")
        clip_path = render_scene(generated, media_dir=media_dir, project_root=config.PROJECT_ROOT, quality=quality)
        clip_paths.append(clip_path)
        print(f"      -> {clip_path}")

    print(f"[5/5] Stitching {len(clip_paths)} clips into the final video")
    output_path = run_dir / f"{slug}.mp4"
    final_path = concat_videos(clip_paths, output_path=output_path, tmp_dir=run_dir / "_concat_tmp")

    print(f"\nDone: {final_path}")
    return final_path
