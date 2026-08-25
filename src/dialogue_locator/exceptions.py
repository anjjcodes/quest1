"""Exception hierarchy for the dialogue locator.

Every failure raised by the pipeline derives from :class:`DialogueLocatorError`
and carries a ``stage`` name, so callers (CLI, API) can report *where* the
failure happened without inspecting exception types individually.

Guidelines
----------
* Stage modules raise the most specific exception that applies.
* Third-party exceptions (yt-dlp, ffmpeg, ctranslate2, cv2) are caught at the
  stage boundary and re-raised as one of these, chained with ``from``.
* "Dialogue not found" is *not* an error in the pipeline's own terms - the
  pipeline returns a "not found" result with near-misses instead.
"""

from __future__ import annotations

from typing import Any


class DialogueLocatorError(Exception):
    """Base class for all application errors."""

    #: Pipeline stage this error belongs to. Overridden by subclasses.
    stage: str = "pipeline"

    def __init__(self, message: str, *, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details: dict[str, Any] = details or {}

    def to_dict(self) -> dict[str, Any]:
        """Serialisable representation for API responses and logs."""
        return {
            "error": type(self).__name__,
            "stage": self.stage,
            "message": self.message,
            "details": self.details,
        }

    def __str__(self) -> str:  # pragma: no cover - trivial
        return f"[{self.stage}] {self.message}"


# --------------------------------------------------------------------------- #
# Input validation
# --------------------------------------------------------------------------- #
class InvalidInputError(DialogueLocatorError):
    stage = "input"


class InvalidURLError(InvalidInputError):
    """The provided video URL is malformed or not supported."""


class InvalidDialogueError(InvalidInputError):
    """The target dialogue is empty, too short, or otherwise unusable."""


class ConfigurationError(DialogueLocatorError):
    """A required external tool or setting is missing/misconfigured."""

    stage = "config"


# --------------------------------------------------------------------------- #
# Media acquisition
# --------------------------------------------------------------------------- #
class DownloadError(DialogueLocatorError):
    """Fetching the video failed (network, geo-block, removed video, ...)."""

    stage = "download"


class UnsupportedVideoError(DownloadError):
    """The URL points to a site/format the downloader cannot handle."""


class AudioExtractionError(DialogueLocatorError):
    """FFmpeg could not produce an audio track (no audio stream, corrupt file...)."""

    stage = "audio"


# --------------------------------------------------------------------------- #
# Speech processing
# --------------------------------------------------------------------------- #
class TranscriptionError(DialogueLocatorError):
    """The ASR model failed to load or crashed while decoding."""

    stage = "transcription"


class VerificationError(DialogueLocatorError):
    """The second-pass verifier failed. Usually downgraded to a warning."""

    stage = "verification"


class PipelineCancelledError(DialogueLocatorError):
    """The caller asked the pipeline to stop (e.g. user cancelled the job)."""

    stage = "pipeline"


# --------------------------------------------------------------------------- #
# Output
# --------------------------------------------------------------------------- #
class FrameExtractionError(DialogueLocatorError):
    """The frame at the resolved timestamp could not be read or written."""

    stage = "frame"


class FaceDetectionError(DialogueLocatorError):
    """The face detector could not run (model missing/undownloadable, bad image)."""

    stage = "face_detection"


class MouthMovementError(DialogueLocatorError):
    """The mouth-movement analyser could not run (model or video unreadable)."""

    stage = "mouth_movement"
