"""Domain models shared by all pipeline stages.

These are plain dataclasses (no pydantic, no FastAPI) so the core pipeline
stays framework-independent. The API layer converts them into response
schemas; the CLI prints them directly.

Data flow between stages::

    Downloader        -> VideoInfo
    AudioExtractor    -> AudioInfo
    Transcriber       -> Iterator[Word]                (streamed)
    Matcher           -> MatchCandidate | near-miss list
    Verifier          -> VerificationOutcome           (0..n, one per verifier)
    FrameExtractor    -> FrameInfo
    Pipeline          -> LocalizationResult            (aggregates all of the above)
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def format_timestamp(seconds: float) -> str:
    """Render seconds as ``HH:MM:SS.mmm`` (the format requested by the PS)."""
    if seconds < 0:
        seconds = 0.0
    whole = int(seconds)
    millis = int(round((seconds - whole) * 1000))
    if millis == 1000:  # rounding overflow, e.g. 1.9996 -> 2.000
        whole += 1
        millis = 0
    hours, rem = divmod(whole, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"


class PipelineStage(str, Enum):
    """Names of the pipeline stages, used in progress events, logs and errors."""

    INPUT = "input"
    DOWNLOAD = "download"
    AUDIO = "audio"
    TRANSCRIPTION = "transcription"
    MATCHING = "matching"
    VERIFICATION = "verification"
    FRAME = "frame"
    DONE = "done"


# --------------------------------------------------------------------------- #
# Media
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class VideoInfo:
    """A downloaded (or local) video file and its probed properties."""

    path: Path
    source_url: str
    title: str | None = None
    duration: float | None = None  # seconds
    fps: float | None = None
    width: int | None = None
    height: int | None = None
    frame_count: int | None = None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["path"] = str(self.path)
        return data


@dataclass(frozen=True)
class AudioInfo:
    """An extracted audio track ready for ASR."""

    path: Path
    sample_rate: int
    channels: int
    duration: float | None = None  # seconds

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["path"] = str(self.path)
        return data


# --------------------------------------------------------------------------- #
# Transcription
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Word:
    """A single transcribed word with absolute timestamps (seconds into the video).

    ``text`` is the raw ASR output (may carry punctuation / leading space);
    normalisation is the matcher's job, not the transcriber's.
    """

    text: str
    start: float
    end: float
    probability: float | None = None
    segment_index: int | None = None  # which ASR segment produced it (debugging)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# --------------------------------------------------------------------------- #
# Matching
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class MatchCandidate:
    """A window of transcript words scored against the target dialogue.

    Used both for accepted matches (score >= threshold) and for near-misses
    reported when nothing crosses the threshold.
    """

    score: float  # RapidFuzz score, 0-100
    start: float  # start time of the first word in the window
    end: float  # end time of the last word in the window
    matched_text: str  # the transcript words in the window, joined
    words: tuple[Word, ...] = field(default=(), repr=False)
    word_index_start: int | None = None  # index into the full word stream
    word_index_end: int | None = None  # exclusive

    @property
    def timestamp(self) -> str:
        return format_timestamp(self.start)

    def to_dict(self, include_words: bool = False) -> dict[str, Any]:
        data: dict[str, Any] = {
            "score": round(self.score, 2),
            "start": round(self.start, 3),
            "end": round(self.end, 3),
            "timestamp": self.timestamp,
            "matched_text": self.matched_text,
            "word_index_start": self.word_index_start,
            "word_index_end": self.word_index_end,
        }
        if include_words:
            data["words"] = [w.to_dict() for w in self.words]
        return data


# --------------------------------------------------------------------------- #
# Verification
# --------------------------------------------------------------------------- #
class VerificationStatus(str, Enum):
    CONFIRMED = "confirmed"  # verifier agrees; its (possibly refined) timing is used
    REJECTED = "rejected"  # verifier disagrees; original timing kept, warning raised
    SKIPPED = "skipped"  # verifier disabled or not applicable
    FAILED = "failed"  # verifier crashed; original timing kept, warning raised


@dataclass(frozen=True)
class VerificationOutcome:
    """Result of one verifier run on a candidate.

    Deliberately generic so that V2's visual "speaker on camera" verifier can
    return the same shape (e.g. ``verifier="visual_speaker"``, ``score`` = face
    / lip-activity confidence, ``refined`` = the first frame where the speaker
    is actually visible).
    """

    verifier: str  # e.g. "asr_large_model", "visual_speaker"
    status: VerificationStatus
    score: float | None = None
    refined: MatchCandidate | None = None  # verifier's own localisation, if any
    message: str | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "verifier": self.verifier,
            "status": self.status.value,
            "score": None if self.score is None else round(self.score, 2),
            "refined": self.refined.to_dict() if self.refined else None,
            "message": self.message,
            "details": self.details,
        }


# --------------------------------------------------------------------------- #
# Frame
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class FrameInfo:
    """The extracted frame image and how it was derived."""

    frame_number: int
    timestamp: float  # seconds; the exact time of the extracted frame
    fps: float
    image_path: Path
    width: int | None = None
    height: int | None = None

    @property
    def timestamp_str(self) -> str:
        return format_timestamp(self.timestamp)

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame_number": self.frame_number,
            "timestamp": round(self.timestamp, 3),
            "timestamp_str": self.timestamp_str,
            "fps": round(self.fps, 3),
            "image_path": str(self.image_path),
            "width": self.width,
            "height": self.height,
        }


# --------------------------------------------------------------------------- #
# Final result
# --------------------------------------------------------------------------- #
class ResultStatus(str, Enum):
    FOUND = "found"
    NOT_FOUND = "not_found"


@dataclass
class LocalizationResult:
    """Everything the pipeline knows after running on one (url, dialogue) pair."""

    status: ResultStatus
    dialogue: str
    source_url: str
    video: VideoInfo | None = None

    # Populated when status == FOUND
    match: MatchCandidate | None = None  # the final (possibly refined) localisation
    first_pass: MatchCandidate | None = None  # the fast-model candidate before verification
    verifications: list[VerificationOutcome] = field(default_factory=list)
    frame: FrameInfo | None = None

    # Populated when status == NOT_FOUND
    near_misses: list[MatchCandidate] = field(default_factory=list)

    # Always
    warnings: list[str] = field(default_factory=list)
    stage_timings: dict[str, float] = field(default_factory=dict)  # stage -> seconds
    transcribed_seconds: float | None = None  # how much audio the fast pass consumed

    @property
    def found(self) -> bool:
        return self.status is ResultStatus.FOUND

    @property
    def timestamp(self) -> str | None:
        """``HH:MM:SS.mmm`` of the resolved frame (or of the match if no frame)."""
        if self.frame is not None:
            return self.frame.timestamp_str
        if self.match is not None:
            return self.match.timestamp
        return None

    @property
    def frame_number(self) -> int | None:
        return self.frame.frame_number if self.frame else None

    @property
    def matched_text(self) -> str | None:
        return self.match.matched_text if self.match else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "dialogue": self.dialogue,
            "source_url": self.source_url,
            "timestamp": self.timestamp,
            "frame_number": self.frame_number,
            "matched_text": self.matched_text,
            "match_score": round(self.match.score, 2) if self.match else None,
            "video": self.video.to_dict() if self.video else None,
            "match": self.match.to_dict() if self.match else None,
            "first_pass": self.first_pass.to_dict() if self.first_pass else None,
            "verifications": [v.to_dict() for v in self.verifications],
            "frame": self.frame.to_dict() if self.frame else None,
            "near_misses": [m.to_dict() for m in self.near_misses],
            "warnings": list(self.warnings),
            "stage_timings": {k: round(v, 3) for k, v in self.stage_timings.items()},
            "transcribed_seconds": (
                None if self.transcribed_seconds is None else round(self.transcribed_seconds, 2)
            ),
        }


# --------------------------------------------------------------------------- #
# Progress reporting
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ProgressEvent:
    """Emitted by the pipeline as it advances. Consumed by CLI (print) or API (job status)."""

    stage: PipelineStage
    message: str
    fraction: float | None = None  # 0.0-1.0 within the stage, if known
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage.value,
            "message": self.message,
            "fraction": self.fraction,
            "details": self.details,
        }


#: Signature of the callback the pipeline calls to report progress.
ProgressCallback = Callable[[ProgressEvent], None]
