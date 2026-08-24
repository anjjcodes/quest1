"""Thin wrapper around ``ffprobe`` to read container/stream metadata.

Used by the downloader (to fill :class:`VideoInfo`), the audio extractor (to
check an audio stream exists) and the frame extractor (to cross-check fps).
Kept separate so each of those stages stays small and so probing can be
mocked in unit tests.
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

from dialogue_locator.exceptions import ConfigurationError, UnsupportedVideoError

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MediaProbe:
    """Subset of ffprobe output the pipeline cares about."""

    duration: float | None
    has_video: bool
    has_audio: bool
    fps: float | None = None
    width: int | None = None
    height: int | None = None
    frame_count: int | None = None
    video_codec: str | None = None
    audio_codec: str | None = None


def ensure_binary(name: str) -> str:
    """Return the resolved path of an executable or raise ``ConfigurationError``."""
    resolved = shutil.which(name)
    if resolved is None:
        logger.error("Required binary '%s' not found on PATH", name)
        raise ConfigurationError(
            f"Required binary '{name}' not found on PATH. Install FFmpeg "
            "(https://ffmpeg.org) and make sure ffmpeg/ffprobe are executable.",
            details={"binary": name},
        )
    return resolved


def _parse_rate(value: str | None) -> float | None:
    """ffprobe reports frame rates as fractions like ``30000/1001``."""
    if not value or value in ("0/0", "0"):
        return None
    try:
        return float(Fraction(value))
    except (ValueError, ZeroDivisionError):
        return None


def _to_float(value: str | float | None) -> float | None:
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


def _to_int(value: str | int | None) -> int | None:
    try:
        return None if value is None else int(value)
    except (TypeError, ValueError):
        return None


def probe_media(path: Path, ffprobe_binary: str = "ffprobe", timeout: int = 60) -> MediaProbe:
    """Probe ``path`` and return its key properties.

    Raises:
        ConfigurationError: ffprobe is not installed.
        UnsupportedVideoError: the file is missing, unreadable, or has no streams.
    """
    binary = ensure_binary(ffprobe_binary)
    if not path.is_file():
        logger.error("Media file does not exist: %s", path)
        raise UnsupportedVideoError(f"Media file does not exist: {path}", details={"path": str(path)})

    cmd = [
        binary,
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        str(path),
    ]
    logger.debug("Probing media: %s", " ".join(cmd))
    try:
        completed = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)
    except subprocess.TimeoutExpired as exc:
        raise UnsupportedVideoError(
            f"ffprobe timed out after {timeout}s on {path.name}", details={"path": str(path)}
        ) from exc

    if completed.returncode != 0:
        logger.error("ffprobe failed on %s (exit %d): %s", path, completed.returncode, completed.stderr.strip())
        raise UnsupportedVideoError(
            f"ffprobe could not read {path.name}: {completed.stderr.strip() or 'unknown error'}",
            details={"path": str(path), "stderr": completed.stderr.strip()},
        )

    try:
        data = json.loads(completed.stdout or "{}")
    except json.JSONDecodeError as exc:
        raise UnsupportedVideoError(
            f"ffprobe returned invalid JSON for {path.name}", details={"path": str(path)}
        ) from exc

    streams = data.get("streams", [])
    fmt = data.get("format", {})
    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    audio = next((s for s in streams if s.get("codec_type") == "audio"), None)

    if video is None and audio is None:
        logger.error("No audio or video streams in %s", path)
        raise UnsupportedVideoError(
            f"No audio or video streams found in {path.name}", details={"path": str(path)}
        )

    duration = _to_float(fmt.get("duration"))
    if duration is None and video is not None:
        duration = _to_float(video.get("duration"))

    fps = width = height = frame_count = None
    if video is not None:
        # avg_frame_rate is the true average; r_frame_rate is the container "tick" rate
        # (can be inflated for VFR). Prefer avg, fall back to r.
        fps = _parse_rate(video.get("avg_frame_rate")) or _parse_rate(video.get("r_frame_rate"))
        width = _to_int(video.get("width"))
        height = _to_int(video.get("height"))
        frame_count = _to_int(video.get("nb_frames"))
        if frame_count is None and fps and duration:
            # MP4 from yt-dlp usually has nb_frames; webm/mkv often do not.
            frame_count = int(round(duration * fps))

    probe = MediaProbe(
        duration=duration,
        has_video=video is not None,
        has_audio=audio is not None,
        fps=fps,
        width=width,
        height=height,
        frame_count=frame_count,
        video_codec=video.get("codec_name") if video else None,
        audio_codec=audio.get("codec_name") if audio else None,
    )
    logger.debug("Probe result for %s: %s", path.name, probe)
    return probe
