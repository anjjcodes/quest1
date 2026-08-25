"""Pydantic request/response models for the HTTP API.

Result payloads are produced from ``LocalizationResult.to_dict()`` so the CLI
JSON and the API JSON are identical; the schemas below type the important
fields for OpenAPI docs while allowing the full record through.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class JobCreate(BaseModel):
    source: str = Field(..., description="Video URL (YouTube, ok.ru, ...) or a local file path", examples=["https://ok.ru/video/248244667877"])
    dialogue: str = Field(..., description="The spoken dialogue to locate", examples=["My mind rebels at stagnation"])
    reuse_cached_media: bool = Field(True, description="Reuse a previous download/extraction of the same source")


class JobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    CANCELLED = "cancelled"

    @property
    def finished(self) -> bool:
        return self in (JobStatus.DONE, JobStatus.FAILED, JobStatus.CANCELLED)


class ProgressSchema(BaseModel):
    stage: str
    message: str
    fraction: float | None = None
    details: dict[str, Any] = Field(default_factory=dict)
    at: datetime


class CandidateSchema(BaseModel):
    model_config = ConfigDict(extra="allow")
    score: float
    start: float
    end: float
    timestamp: str
    matched_text: str


class VerificationSchema(BaseModel):
    model_config = ConfigDict(extra="allow")
    verifier: str
    status: str
    score: float | None = None
    refined: CandidateSchema | None = None
    message: str | None = None


class FrameSchema(BaseModel):
    model_config = ConfigDict(extra="allow")
    frame_number: int
    timestamp: float
    timestamp_str: str
    fps: float
    width: int | None = None
    height: int | None = None


class FaceBoxSchema(BaseModel):
    model_config = ConfigDict(extra="allow")
    x: int
    y: int
    width: int
    height: int
    confidence: float


class FaceDetectionSchema(BaseModel):
    """V2: faces found in the extracted frame."""

    model_config = ConfigDict(extra="allow")
    face_present: bool
    face_count: int
    faces: list[FaceBoxSchema] = Field(default_factory=list)


class MouthMovementSchema(BaseModel):
    """V3: lip activity over the matched dialogue window."""

    model_config = ConfigDict(extra="allow")
    moving: bool | None = None
    movement_score: float | None = None
    threshold: float
    frames_analyzed: int
    frames_with_face: int


class ResultSchema(BaseModel):
    """Mirrors ``LocalizationResult.to_dict()``."""

    model_config = ConfigDict(extra="allow")
    status: str
    dialogue: str
    source_url: str
    timestamp: str | None = None
    frame_number: int | None = None
    matched_text: str | None = None
    match_score: float | None = None
    match: CandidateSchema | None = None
    first_pass: CandidateSchema | None = None
    verifications: list[VerificationSchema] = Field(default_factory=list)
    frame: FrameSchema | None = None
    face_present: bool | None = Field(
        None, description="V2: set once the face check ran on the frame; None if it did not run"
    )
    face_detection: FaceDetectionSchema | None = None
    mouth_moving: bool | None = Field(
        None, description="V3: set once the mouth check reached a verdict; None otherwise"
    )
    mouth_movement: MouthMovementSchema | None = None
    near_misses: list[CandidateSchema] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    stage_timings: dict[str, float] = Field(default_factory=dict)
    transcribed_seconds: float | None = None


class ErrorSchema(BaseModel):
    error: str
    stage: str
    message: str
    details: dict[str, Any] = Field(default_factory=dict)
    near_misses: list[dict[str, Any]] | None = None


class JobResponse(BaseModel):
    job_id: str
    status: JobStatus
    source: str
    dialogue: str
    created_at: datetime
    started_at: datetime | None = None
    finished_at: datetime | None = None
    progress: ProgressSchema | None = Field(None, description="Most recent progress event")
    progress_log: list[ProgressSchema] = Field(default_factory=list, description="Recent progress events, oldest first")
    result: ResultSchema | None = None
    error: ErrorSchema | None = None
    frame_url: str | None = Field(None, description="URL of the extracted frame image, when available")


class JobListResponse(BaseModel):
    jobs: list[JobResponse]


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str
    fast_model: str
    verify_model: str
    verification_enabled: bool
    match_threshold: float
    active_jobs: int
