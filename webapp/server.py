"""FastAPI server for running one local video-generation job at a time."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import Any, Literal
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

import config
from pipeline.orchestrator import run_pipeline

STATIC_DIR = Path(__file__).resolve().parent / "static"
Quality = Literal["ql", "qm", "qh", "qk"]


class GenerateRequest(BaseModel):
    title: str = Field(min_length=1, max_length=160)
    prompt: str | None = Field(default=None, max_length=2_000)
    quality: Quality | None = None


@dataclass
class Job:
    status: str = "running"
    stage: str = "starting"
    detail: dict[str, Any] = field(default_factory=dict)
    video_path: Path | None = None
    error: str | None = None


app = FastAPI(title="Math Video Pipeline")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

_jobs: dict[str, Job] = {}
_lock = Lock()
_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="math-video")


def _job_response(job_id: str, job: Job) -> dict[str, Any]:
    return {
        "status": job.status,
        "stage": job.stage,
        "detail": job.detail,
        "video_url": f"/api/jobs/{job_id}/video" if job.status == "done" else None,
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
        "max_scenes": config.pipeline.max_scenes,
        "qualities": [
            {"id": "ql", "label": "Draft", "hint": "Fast, low resolution"},
            {"id": "qm", "label": "Balanced", "hint": "Good for most runs"},
            {"id": "qh", "label": "High", "hint": "1080p — slower"},
            {"id": "qk", "label": "4K", "hint": "Slowest, largest files"},
        ],
    }


@app.post("/api/generate")
def generate_video(request: GenerateRequest) -> dict[str, str]:
    title = request.title.strip()
    if not title:
        raise HTTPException(status_code=422, detail="Title cannot be blank.")

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

    if job.status != "done" or not video_path or not video_path.is_file():
        raise HTTPException(status_code=404, detail="The finished video is not available yet.")
    return FileResponse(
        video_path,
        media_type="video/mp4",
        filename=video_path.name,
        content_disposition_type="inline",
    )


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/favicon.ico", include_in_schema=False)
def favicon() -> FileResponse:
    """Serve the app icon for browsers that request the conventional path."""
    return FileResponse(STATIC_DIR / "favicon.svg", media_type="image/svg+xml")
