"""Domain models shared by all pipeline stages.

These are plain dataclasses (no pydantic, no FastAPI) so the core pipeline
stays framework-independent. The API layer converts them into response
schemas; the CLI prints them directly.

Data flow between stages::

    Downloader        -> MediaInfo (search fetch) / VideoInfo (full-quality fetch)
    AudioExtractor    -> AudioInfo
    Transcriber       -> Iterator[Word]                (streamed)
    Matcher           -> MatchCandidate | near-miss list
    Verifier          -> VerificationOutcome           (0..n, one per verifier)
    FrameExtractor    -> FrameInfo
    FaceDetector      -> FaceDetectionResult           (V2: faces in a frame)
    MouthMovementAnalyzer -> MouthMovementResult       (V3: lip activity in a window)
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
    DOWNLOAD = "download"  # cheap search fetch (audio-only if possible)
    AUDIO = "audio"
    TRANSCRIPTION = "transcription"
    MATCHING = "matching"
    DOWNLOAD_VIDEO = "download_video"  # full-quality fetch, only after a match
    VERIFICATION = "verification"
    FRAME = "frame"
    FACE_DETECTION = "face_detection"  # V2: is a human face visible in the frame?
    MOUTH_MOVEMENT = "mouth_movement"  # V3: is the mouth moving during the line?
    DONE = "done"


# --------------------------------------------------------------------------- #
# Media
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class VideoInfo:
    """A downloaded (or local) video file and its probed properties.

    May be a *clip* of the source rather than the whole video: ``clip_start``
    is how many seconds into the source this file begins (0.0 for a full
    video). Absolute source timestamps map to ``t - clip_start`` in the file;
    ``duration`` / ``frame_count`` describe the file, not the source.
    """

    path: Path
    source_url: str
    title: str | None = None
    duration: float | None = None  # seconds
    fps: float | None = None
    width: int | None = None
    height: int | None = None
    frame_count: int | None = None
    clip_start: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["path"] = str(self.path)
        return data


@dataclass(frozen=True)
class MediaInfo:
    """The cheap search-pass download: audio-only when the host offers it.

    Feeds audio extraction and transcription only. The full-quality
    :class:`VideoInfo` is fetched separately, after a match is confirmed,
    for frame extraction.
    """

    path: Path
    source_url: str
    title: str | None = None
    duration: float | None = None  # seconds
    has_video: bool = False  # False for audio-only downloads

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
# Face detection (V2)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class FaceBox:
    """One detected face: pixel bounding box (top-left origin) + confidence."""

    x: int
    y: int
    width: int
    height: int
    confidence: float  # detector score, 0-1

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["confidence"] = round(self.confidence, 3)
        return data


@dataclass(frozen=True)
class FaceDetectionResult:
    """Faces found in a single frame, best-scoring first."""

    faces: tuple[FaceBox, ...]
    image_width: int
    image_height: int

    @property
    def face_present(self) -> bool:
        return len(self.faces) > 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "face_present": self.face_present,
            "face_count": len(self.faces),
            "faces": [f.to_dict() for f in self.faces],
            "image_width": self.image_width,
            "image_height": self.image_height,
        }


# --------------------------------------------------------------------------- #
# Mouth movement (V3)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class MouthMovementResult:
    """Lip activity over the video frames of a dialogue window.

    ``moving`` is ``None`` when the analyser saw too few frames with a face to
    judge (indeterminate), otherwise whether the movement score - the standard
    deviation of the per-frame mouth-openness signal - reached the threshold.
    """

    moving: bool | None
    movement_score: float | None  # None when no frame had a face
    threshold: float
    frames_analyzed: int
    frames_with_face: int
    window_start: float  # absolute seconds into the source, as analysed
    window_end: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "moving": self.moving,
            "movement_score": None
            if self.movement_score is None
            else round(self.movement_score, 4),
            "threshold": self.threshold,
            "frames_analyzed": self.frames_analyzed,
            "frames_with_face": self.frames_with_face,
            "window_start": round(self.window_start, 3),
            "window_end": round(self.window_end, 3),
        }


# --------------------------------------------------------------------------- #
# Final result
# --------------------------------------------------------------------------- #
class ResultStatus(str, Enum):
    FOUND = "found"
    NOT_FOUND = "not_found"
    # V2: the dialogue was heard at the timestamp, but no human face is visible
    # in the frame - it is not an *onscreen* dialogue. Match/frame details are
    # still populated so the caller can see what was found.
    NOT_ONSCREEN = "not_onscreen"


@dataclass
class LocalizationResult:
    """Everything the pipeline knows after running on one (url, dialogue) pair."""

    status: ResultStatus
    dialogue: str
    source_url: str
    video: VideoInfo | None = None

    # Populated when the dialogue was localised (status FOUND or NOT_ONSCREEN)
    match: MatchCandidate | None = None  # the final (possibly refined) localisation
    first_pass: MatchCandidate | None = None  # the fast-model candidate before verification
    verifications: list[VerificationOutcome] = field(default_factory=list)
    frame: FrameInfo | None = None
    face_detection: FaceDetectionResult | None = None  # V2; None = check not run
    mouth_movement: MouthMovementResult | None = None  # V3; None = check not run

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

    @property
    def face_present(self) -> bool | None:
        """Whether a face is visible in the extracted frame (None = check not run)."""
        return self.face_detection.face_present if self.face_detection else None

    @property
    def mouth_moving(self) -> bool | None:
        """Whether the mouth moves during the line (None = not run or indeterminate)."""
        return self.mouth_movement.moving if self.mouth_movement else None

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
            "face_present": self.face_present,
            "face_detection": self.face_detection.to_dict() if self.face_detection else None,
            "mouth_moving": self.mouth_moving,
            "mouth_movement": self.mouth_movement.to_dict() if self.mouth_movement else None,
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
