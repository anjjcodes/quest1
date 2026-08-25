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
    SettingsView,
    StageToggles,
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
        settings=job.settings_view,
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


def settings_to_view(s: Settings) -> SettingsView:
    return SettingsView(
        stages=StageToggles(
            verification=s.verification.enabled,
            face_detection=s.face_detection.enabled,
            mouth_movement=s.mouth_movement.enabled,
        ),
        match_threshold=s.matching.match_threshold,
        face_min_confidence=s.face_detection.min_detection_confidence,
        mouth_movement_threshold=s.mouth_movement.movement_threshold,
        mouth_min_face_frames=s.mouth_movement.min_face_frames,
        mouth_max_window_seconds=s.mouth_movement.max_window_seconds,
        max_video_height=s.download.max_height,
    )


def apply_setting_overrides(
    defaults: Settings, view: SettingsView
) -> tuple[Settings, SettingsView]:
    """Return ``(effective settings, normalised view)`` for one job.

    Stage dependencies cascade downstream: verification off also turns off the
    face check, and the face check off also turns off the mouth check - the
    visual stages build on what came before them, never the other way round.
    """
    verification = view.stages.verification
    face = view.stages.face_detection and verification
    mouth = view.stages.mouth_movement and face

    effective = defaults.model_copy(
        update={
            "verification": defaults.verification.model_copy(update={"enabled": verification}),
            "matching": defaults.matching.model_copy(
                update={"match_threshold": view.match_threshold}
            ),
            "face_detection": defaults.face_detection.model_copy(
                update={"enabled": face, "min_detection_confidence": view.face_min_confidence}
            ),
            "mouth_movement": defaults.mouth_movement.model_copy(
                update={
                    "enabled": mouth,
                    "movement_threshold": view.mouth_movement_threshold,
                    "min_face_frames": view.mouth_min_face_frames,
                    "max_window_seconds": view.mouth_max_window_seconds,
                }
            ),
            "download": defaults.download.model_copy(update={"max_height": view.max_video_height}),
        }
    )
    return effective, settings_to_view(effective)


@router.get("/settings", response_model=SettingsView)
def get_default_settings(request: Request) -> SettingsView:
    """The server-default settings. Jobs may override them per job via
    ``JobCreate.settings``; nothing is ever stored server-side."""
    return settings_to_view(_settings(request))


@router.post("/jobs", response_model=JobResponse, status_code=status.HTTP_202_ACCEPTED)
def create_job(body: JobCreate, request: Request) -> JobResponse:
    defaults = _settings(request)
    # Per-job settings overrides: normalise the stage cascade and build a
    # pipeline for this job only (cheap - model caches are process-wide).
    # Nothing is stored server-side, so clients reset to defaults naturally.
    if body.settings is not None:
        effective, view = apply_setting_overrides(defaults, body.settings)
        pipeline = request.app.state.pipeline_factory(effective)
    else:
        effective, view = defaults, settings_to_view(defaults)
        pipeline = None  # the shared default pipeline

    # Fail fast on bad input (raises InvalidDialogueError / InvalidURLError -> 422 via app handler).
    TargetDialogue.parse(body.dialogue, effective.matching)
    source = body.source.strip()
    if "://" in source or is_url(source):
        validate_url(source)  # any URL-looking input must be a valid http(s) URL
    elif not Path(source).expanduser().is_file():
        raise InvalidURLError(
            f"'{source}' is neither an http(s) URL nor an existing file.", details={"source": source}
        )
    req = PipelineRequest(source=body.source.strip(), dialogue=body.dialogue.strip(), reuse_cached_media=body.reuse_cached_media)
    job = _manager(request).submit(req, pipeline=pipeline, settings_view=view)
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
