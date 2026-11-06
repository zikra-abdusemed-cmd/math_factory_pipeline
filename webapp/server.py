"""FastAPI server for running one local video-generation job at a time."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
import json
import base64
from pathlib import Path
import re
import shutil
from threading import Lock
from typing import Any, Literal
from uuid import uuid4
import zipfile

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from slugify import slugify

import config
from pipeline.orchestrator import run_pipeline
from pipeline.olx import Course, UnitSelection, course_manifest, load_course, load_course_folder, selected_unit_components

STATIC_DIR = Path(__file__).resolve().parent / "static"
Quality = Literal["ql", "qm", "qh", "qk"]


class GenerateRequest(BaseModel):
    title: str = Field(default="Educational video", min_length=1, max_length=160)
    prompt: str | None = Field(default=None, max_length=2_000)
    quality: Quality | None = None


class ComponentSelection(BaseModel):
    section_id: str
    subsection_id: str
    unit_id: str
    component_indices: list[int] = Field(min_length=1, max_length=2_000)


class CourseGenerateRequest(BaseModel):
    import_id: str
    selections: list[ComponentSelection] = Field(min_length=1, max_length=5_000)
    prompt: str | None = Field(default=None, max_length=2_000)
    quality: Quality | None = None


class OLXFolderUpload(BaseModel):
    files: list[dict[str, str]] = Field(min_length=1, max_length=20_000)


@dataclass
class Job:
    status: str = "running"
    stage: str = "starting"
    detail: dict[str, Any] = field(default_factory=dict)
    video_path: Path | None = None
    download_name: str | None = None
    error: str | None = None


app = FastAPI(title="Math Video Pipeline")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

_jobs: dict[str, Job] = {}
_imports: dict[str, tuple[Course, Path]] = {}
_lock = Lock()
_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="math-video")


def _job_response(job_id: str, job: Job) -> dict[str, Any]:
    return {
        "status": job.status,
        "stage": job.stage,
        "detail": job.detail,
        "video_url": f"/api/jobs/{job_id}/video" if job.status == "done" else None,
        "partial_url": f"/api/jobs/{job_id}/video" if job.video_path and job.video_path.is_file() else None,
        "download_name": job.download_name,
        "error": job.error,
    }


def _run_job(job_id: str, title: str, prompt: str | None, quality: str | None) -> None:
    def progress(stage: str, detail: dict) -> None:
        with _lock:
            job = _jobs[job_id]
            job.stage = stage
            job.detail = detail

    try:
        video_path = run_pipeline(
            title,
            quality=quality,
            extra_instructions=prompt,
            on_progress=progress,
        )
    except Exception as exc:  # Keep errors available to the local UI.
        with _lock:
            job = _jobs[job_id]
            job.status = "error"
            job.stage = "error"
            job.error = str(exc) or exc.__class__.__name__
        return

    with _lock:
        job = _jobs[job_id]
        job.status = "done"
        job.stage = "done"
        job.video_path = video_path


@app.get("/api/config")
def get_config() -> dict[str, Any]:
    """Defaults the UI can show without exposing secrets."""
    return {
        "default_quality": config.pipeline.render_quality,
        "voice_provider": config.pipeline.voice_provider,
        "max_scenes": max(4, config.pipeline.max_scenes),
        "qualities": [
            {"id": "ql", "label": "Draft", "hint": "Fast, low resolution"},
            {"id": "qm", "label": "Balanced", "hint": "Good for most runs"},
            {"id": "qh", "label": "High", "hint": "1080p — slower"},
            {"id": "qk", "label": "4K", "hint": "Slowest, largest files"},
        ],
    }


@app.post("/api/generate")
def generate_video(request: GenerateRequest) -> dict[str, str]:
    title = request.title.strip() or "Educational video"

    with _lock:
        if any(job.status == "running" for job in _jobs.values()):
            raise HTTPException(
                status_code=409,
                detail="A video is already being generated. Please wait for it to finish.",
            )
        job_id = uuid4().hex
        _jobs[job_id] = Job(
            detail={
                "title": title,
                "quality": request.quality or config.pipeline.render_quality,
            }
        )

    _executor.submit(_run_job, job_id, title, request.prompt, request.quality)
    return {"job_id": job_id}


@app.post("/api/olx/preview")
async def preview_olx(request: Request, filename: str) -> dict[str, Any]:
    try:
        course, temporary = load_course(filename, await request.body(), temp_parent=config.GENERATED_DIR)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Could not read OLX archive: {exc}") from exc
    import_id = uuid4().hex
    persistent_dir = config.GENERATED_DIR / "_olx_imports" / import_id
    persistent_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(temporary.name, persistent_dir)
    temporary.cleanup()
    with _lock:
        _imports[import_id] = (course, persistent_dir)
    return {"import_id": import_id, "manifest": course_manifest(course)}


@app.post("/api/olx/preview-folder")
def preview_olx_folder(request: OLXFolderUpload) -> dict[str, Any]:
    try:
        course, temporary = load_course_folder(request.files, temp_parent=config.GENERATED_DIR)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Could not read OLX folder: {exc}") from exc
    import_id = uuid4().hex
    persistent_dir = config.GENERATED_DIR / "_olx_imports" / import_id
    persistent_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(temporary.name, persistent_dir)
    temporary.cleanup()
    with _lock:
        _imports[import_id] = (course, persistent_dir)
    return {"import_id": import_id, "manifest": course_manifest(course)}


@app.post("/api/olx/generate")
def generate_olx(request: CourseGenerateRequest) -> dict[str, str]:
    with _lock:
        imported = _imports.get(request.import_id)
        if any(job.status == "running" for job in _jobs.values()):
            raise HTTPException(status_code=409, detail="A video job is already running. Please wait for it to finish.")
    if not imported:
        raise HTTPException(status_code=404, detail="OLX preview expired. Upload the course again.")
    course, _source_dir = imported
    try:
        units = selected_unit_components(course, [item.model_dump() for item in request.selections])
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    job_id = uuid4().hex
    output_dir = config.GENERATED_DIR / "olx_courses" / job_id
    output_dir.mkdir(parents=True, exist_ok=True)
    with _lock:
        _jobs[job_id] = Job(detail={"title": course.title, "scope": "selected_subsections", "unit_total": len(units), "quality": request.quality or config.pipeline.render_quality})
    _executor.submit(_run_course_job, job_id, course, units, output_dir, request.prompt, request.quality)
    return {"job_id": job_id}


def _run_course_job(job_id: str, course: Course, units: list[UnitSelection], output_dir: Path, prompt: str | None, quality: str | None) -> None:
    videos: list[Path] = []
    table: list[dict[str, str]] = []
    try:
        for index, selection in enumerate(units, start=1):
            section_title = selection.section_title
            subsection_title = selection.subsection_title
            unit = selection.unit
            def progress(stage: str, detail: dict[str, Any]) -> None:
                with _lock:
                    job = _jobs[job_id]
                    job.stage = stage
                    job.detail = {"title": course.title, "unit_title": unit.title, "unit_index": index, "unit_total": len(units), **detail}
            chosen_components = [unit.components[component_index] for component_index in selection.component_indices]
            source = "\n\n".join(f"[{component.kind}: {component.title}]\n{component.text}" for component in chosen_components)
            content_title = ", ".join(component.title for component in chosen_components)
            video_reference = f"{section_title} — {subsection_title} — {unit.title}"
            short_labels = ("introduction", "overview", "key vocabulary", "vocabulary", "objectives", "learning objectives")
            selected_labels = " ".join([section_title, subsection_title, unit.title, *(component.title for component in chosen_components)]).lower()
            is_explicit_main = "main content" in selected_labels or "main lesson" in selected_labels
            is_brief = not is_explicit_main and any(label in selected_labels for label in short_labels)
            if is_brief:
                min_scenes = max_scenes = 4
                target_seconds_per_beat = 22
                duration_guidance = (
                    "This is a short course orientation subsection (introduction, overview, key vocabulary, or objectives). "
                    "Make exactly 4 concise narration beats for a total spoken length of about 1–2 minutes. "
                    "Cover only the selected material; do not pad or add unrelated teaching."
                )
            else:
                source_words = len(re.findall(r"\b[\w'-]+\b", source))
                max_scenes = min(18, max(1, (source_words + 49) // 50))
                min_scenes = 1
                target_seconds_per_beat = 25
                duration_guidance = (
                    f"This is main course content with about {source_words} source words. Choose the number of narration beats "
                    "to explain all substantive selected material clearly, scaling with its amount and complexity without padding. "
                    "Each beat is about 25 seconds; keep total spoken video length at or below 10 minutes. "
                    f"Use no more than {max_scenes} beats (the absolute limit is 18, keeping the video below 10 minutes)."
                )
            video = run_pipeline(
                video_reference,
                quality=quality,
                extra_instructions=prompt,
                on_progress=progress,
                source_content=source,
                min_scenes=min_scenes,
                max_scenes=max_scenes,
                duration_guidance=duration_guidance,
                target_seconds_per_beat=target_seconds_per_beat,
            )
            section_slug = slugify(section_title, max_length=45) or "section"
            subsection_slug = slugify(subsection_title, max_length=45) or "subsection"
            unit_slug = slugify(unit.title, max_length=55) or f"unit-{index:03d}"
            relative_video = Path(section_slug) / subsection_slug / unit_slug / video.name
            destination = output_dir / relative_video
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(video, destination)
            videos.append(destination)
            table.append({"section": section_title, "subsection": subsection_title, "unit": unit.title, "content": content_title, "video": relative_video.as_posix()})
        manifest_path = output_dir / "course_manifest.json"
        manifest_path.write_text(json.dumps({"course": course.title, "units": table}, ensure_ascii=False, indent=2), encoding="utf-8")
        archive = output_dir / "course-videos.zip"
        with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
            bundle.write(manifest_path, manifest_path.name)
            for path in videos:
                bundle.write(path, path.relative_to(output_dir).as_posix())
        with _lock:
            job = _jobs[job_id]
            job.status = "done"
            job.stage = "done"
            job.video_path = archive
            job.download_name = archive.name
            job.detail = {"title": course.title, "unit_total": len(units), "videos": table}
    except Exception as exc:
        with _lock:
            job = _jobs[job_id]
            job.status = "error"
            job.stage = "error"
            job.error = str(exc) or exc.__class__.__name__
            manifest_path = output_dir / "course_manifest.json"
            manifest_path.write_text(json.dumps({"course": course.title, "units": table, "failed": job.detail.get("unit_title")}, ensure_ascii=False, indent=2), encoding="utf-8")
            archive = output_dir / "partial-course-videos.zip"
            with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
                bundle.write(manifest_path, manifest_path.name)
                for path in videos:
                    bundle.write(path, path.relative_to(output_dir).as_posix())
            job.video_path = archive
            job.download_name = archive.name
            job.detail = {"title": course.title, "completed_units": table, "unit_total": len(units)}


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str) -> dict[str, Any]:
    with _lock:
        job = _jobs.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found.")
        return _job_response(job_id, job)


@app.get("/api/jobs/{job_id}/video")
def get_video(job_id: str) -> FileResponse:
    with _lock:
        job = _jobs.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found.")
        video_path = job.video_path

    if job.status not in {"done", "error"} or not video_path or not video_path.is_file():
        raise HTTPException(status_code=404, detail="The finished video is not available yet.")
    return FileResponse(
        video_path,
        media_type="application/zip" if video_path.suffix == ".zip" else "video/mp4",
        filename=job.download_name or video_path.name,
        content_disposition_type="attachment" if video_path.suffix == ".zip" else "inline",
    )


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/favicon.ico", include_in_schema=False)
def favicon() -> FileResponse:
    """Serve the app icon for browsers that request the conventional path."""
    return FileResponse(STATIC_DIR / "favicon.svg", media_type="image/svg+xml")
