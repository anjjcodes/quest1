"""HTTP routes. Validation happens up-front so bad input fails synchronously with 422."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException, Request, status
from fastapi.responses import FileResponse

from dialogue_locator import __version__
from dialogue_locator.api.jobs import Job, JobManager
from dialogue_locator.api.schemas import (
    HealthResponse,
    JobCreate,
    JobListResponse,
    JobResponse,
    JobStatus,
    ProgressSchema,
)
from dialogue_locator.config import Settings
from dialogue_locator.exceptions import InvalidURLError
from dialogue_locator.matching.matcher import TargetDialogue
from dialogue_locator.media.downloader import is_url, validate_url
from dialogue_locator.pipeline import PipelineRequest

router = APIRouter(prefix="/api", tags=["jobs"])


def _manager(request: Request) -> JobManager:
    return request.app.state.jobs


def _settings(request: Request) -> Settings:
    return request.app.state.settings


def job_to_response(job: Job) -> JobResponse:
    def prog(item):
        event, at = item
        return ProgressSchema(stage=event.stage.value, message=event.message, fraction=event.fraction, details=event.details, at=at)

    return JobResponse(
        job_id=job.id,
        status=job.status,
        source=job.request.source,
        dialogue=job.request.dialogue,
        created_at=job.created_at,
        started_at=job.started_at,
        finished_at=job.finished_at,
        progress=prog(job.progress) if job.progress else None,
        progress_log=[prog(p) for p in job.progress_log],
        result=job.result.to_dict() if job.result else None,
        error=job.error,
        frame_url=f"/api/jobs/{job.id}/frame" if job.frame_path else None,
    )


@router.get("/health", response_model=HealthResponse)
def health(request: Request) -> HealthResponse:
    s = _settings(request)
    return HealthResponse(
        version=__version__,
        fast_model=s.whisper.fast_model,
        verify_model=s.whisper.verify_model,
        verification_enabled=s.verification.enabled,
        match_threshold=s.matching.match_threshold,
        active_jobs=_manager(request).active_count,
    )


@router.post("/jobs", response_model=JobResponse, status_code=status.HTTP_202_ACCEPTED)
def create_job(body: JobCreate, request: Request) -> JobResponse:
    settings = _settings(request)
    # Fail fast on bad input (raises InvalidDialogueError / InvalidURLError -> 422 via app handler).
    TargetDialogue.parse(body.dialogue, settings.matching)
    source = body.source.strip()
    if "://" in source or is_url(source):
        validate_url(source)  # any URL-looking input must be a valid http(s) URL
    elif not Path(source).expanduser().is_file():
        raise InvalidURLError(
            f"'{source}' is neither an http(s) URL nor an existing file.", details={"source": source}
        )
    req = PipelineRequest(source=body.source.strip(), dialogue=body.dialogue.strip(), reuse_cached_media=body.reuse_cached_media)
    job = _manager(request).submit(req)
    return job_to_response(job)


@router.get("/jobs", response_model=JobListResponse)
def list_jobs(request: Request) -> JobListResponse:
    return JobListResponse(jobs=[job_to_response(j) for j in _manager(request).list()])


@router.get("/jobs/{job_id}", response_model=JobResponse)
def get_job(job_id: str, request: Request) -> JobResponse:
    job = _manager(request).get(job_id)
    if job is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"Job {job_id} not found")
    return job_to_response(job)


@router.get("/jobs/{job_id}/frame")
def get_frame(job_id: str, request: Request) -> FileResponse:
    job = _manager(request).get(job_id)
    if job is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"Job {job_id} not found")
    path = job.frame_path
    if path is None or not path.is_file():
        raise HTTPException(status.HTTP_404_NOT_FOUND, "No frame available for this job")
    media_type = "image/png" if path.suffix == ".png" else "image/jpeg"
    return FileResponse(path, media_type=media_type, filename=f"{job_id}_frame{path.suffix}")


@router.delete("/jobs/{job_id}", response_model=JobResponse)
def cancel_or_delete_job(job_id: str, request: Request) -> JobResponse:
    """Cancel a queued/running job; a finished job is removed from memory."""
    manager = _manager(request)
    job = manager.get(job_id)
    if job is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"Job {job_id} not found")
    if job.status.finished:
        manager.remove(job_id)
        return job_to_response(job)
    return job_to_response(manager.cancel(job_id) or job)


__all__ = ["router", "job_to_response", "JobStatus"]
