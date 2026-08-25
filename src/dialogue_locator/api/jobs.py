"""In-memory job manager.

Runs pipeline jobs on a small thread pool (Whisper is CPU/GPU-bound, so
``max_concurrent_jobs`` defaults to 1), records progress events, supports
cooperative cancellation and expires finished jobs after a retention period.

State is process-local by design for V1; swapping in Redis/DB later only
touches this module.
"""

from __future__ import annotations

import logging
import threading
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from dialogue_locator.api.schemas import JobStatus
from dialogue_locator.exceptions import DialogueLocatorError, PipelineCancelledError
from dialogue_locator.models import LocalizationResult, ProgressEvent
from dialogue_locator.pipeline import DialoguePipeline, PipelineRequest, save_result
from dialogue_locator.pipeline.pipeline import RESULT_FILENAME

logger = logging.getLogger(__name__)

PROGRESS_LOG_SIZE = 50


def _now() -> datetime:
    return datetime.now(UTC)


@dataclass
class Job:
    request: PipelineRequest
    pipeline: DialoguePipeline | None = None  # per-job override; None = manager default
    settings_view: Any | None = None  # normalised SettingsView echoed in responses
    status: JobStatus = JobStatus.QUEUED
    created_at: datetime = field(default_factory=_now)
    started_at: datetime | None = None
    finished_at: datetime | None = None
    progress: tuple[ProgressEvent, datetime] | None = None
    progress_log: deque[tuple[ProgressEvent, datetime]] = field(default_factory=lambda: deque(maxlen=PROGRESS_LOG_SIZE))
    result: LocalizationResult | None = None
    error: dict[str, Any] | None = None
    cancel_event: threading.Event = field(default_factory=threading.Event)

    @property
    def id(self) -> str:
        return self.request.job_id

    @property
    def frame_path(self) -> Path | None:
        if self.result and self.result.frame:
            return self.result.frame.image_path
        return None


class JobManager:
    def __init__(self, pipeline: DialoguePipeline, max_concurrent: int = 1, retention_seconds: int = 3600) -> None:
        self._pipeline = pipeline
        self._retention = retention_seconds
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()
        self._executor = ThreadPoolExecutor(max_workers=max_concurrent, thread_name_prefix="job")

    # ------------------------------------------------------------------ #
    def submit(
        self,
        request: PipelineRequest,
        pipeline: DialoguePipeline | None = None,
        settings_view: Any | None = None,
    ) -> Job:
        job = Job(request=request, pipeline=pipeline, settings_view=settings_view)
        with self._lock:
            self._expire_locked()
            self._jobs[job.id] = job
        logger.info("Job %s queued (%r / %r)", job.id, request.source, request.dialogue)
        self._executor.submit(self._run, job)
        return job

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            self._expire_locked()
            return self._jobs.get(job_id)

    def list(self) -> list[Job]:
        with self._lock:
            self._expire_locked()
            return sorted(self._jobs.values(), key=lambda j: j.created_at, reverse=True)

    def cancel(self, job_id: str) -> Job | None:
        job = self.get(job_id)
        if job is None:
            return None
        if job.status.finished:
            return job
        job.cancel_event.set()
        if job.status is JobStatus.QUEUED:
            # Not started yet: mark immediately; _run will notice the event and exit.
            self._finish(job, JobStatus.CANCELLED)
        logger.info("Job %s cancellation requested", job_id)
        return job

    def remove(self, job_id: str) -> bool:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None or not job.status.finished:
                return False
            del self._jobs[job_id]
            return True

    @property
    def active_count(self) -> int:
        with self._lock:
            return sum(1 for j in self._jobs.values() if not j.status.finished)

    def shutdown(self) -> None:
        for job in self.list():
            job.cancel_event.set()
        self._executor.shutdown(wait=False, cancel_futures=True)

    # ------------------------------------------------------------------ #
    def _run(self, job: Job) -> None:
        if job.cancel_event.is_set():
            if not job.status.finished:
                self._finish(job, JobStatus.CANCELLED)
            return
        job.status = JobStatus.RUNNING
        job.started_at = _now()

        def on_progress(event: ProgressEvent) -> None:
            stamped = (event, _now())
            job.progress = stamped
            job.progress_log.append(stamped)

        pipeline = job.pipeline or self._pipeline
        try:
            result = pipeline.run(job.request, progress=on_progress, should_cancel=job.cancel_event.is_set)
        except PipelineCancelledError as exc:
            job.error = exc.to_dict()
            self._finish(job, JobStatus.CANCELLED)
            return
        except DialogueLocatorError as exc:
            job.error = exc.to_dict()
            self._finish(job, JobStatus.FAILED)
            return
        except Exception as exc:  # noqa: BLE001 - never let a worker thread die silently
            logger.exception("Job %s crashed outside the pipeline boundary", job.id)
            job.error = {"error": type(exc).__name__, "stage": "pipeline", "message": str(exc), "details": {}}
            self._finish(job, JobStatus.FAILED)
            return

        job.result = result
        try:
            out_dir = pipeline.settings.storage.output_dir / job.id
            save_result(result, out_dir / RESULT_FILENAME)
        except OSError as exc:  # pragma: no cover
            logger.warning("Could not persist result for job %s: %s", job.id, exc)
        self._finish(job, JobStatus.DONE)

    @staticmethod
    def _finish(job: Job, status: JobStatus) -> None:
        job.status = status
        job.finished_at = _now()
        logger.info("Job %s -> %s", job.id, status.value)

    def _expire_locked(self) -> None:
        if self._retention <= 0:
            return
        now = _now()
        expired = [
            jid
            for jid, j in self._jobs.items()
            if j.status.finished and j.finished_at and (now - j.finished_at).total_seconds() > self._retention
        ]
        for jid in expired:
            del self._jobs[jid]
            logger.debug("Job %s expired from memory", jid)
