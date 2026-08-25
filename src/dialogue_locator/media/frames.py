"""Frame extraction: timestamp -> frame number -> image file.

Frame numbering
---------------
Frame ``k`` is displayed during ``[k / fps, (k + 1) / fps)``. The frame on
screen at time ``t`` is therefore ``floor(t * fps)`` (with a small epsilon so
``t`` exactly on a boundary maps to the frame that starts there). The reported
``FrameInfo.timestamp`` is the frame's own presentation time ``k / fps``, which
may differ from the requested ``t`` by less than one frame duration.

Seeking
-------
OpenCV's ``CAP_PROP_POS_FRAMES`` seek is fast but not always exact on every
container (it depends on the backend and keyframe layout). We verify the
position OpenCV reports after seeking; if it is off, or decoding fails, we
fall back to ``ffmpeg -ss <t> -i <video> -frames:v 1`` which is slower but
frame-accurate.

Besides :meth:`FrameExtractor.extract` (one frame to disk), :meth:`read_frame`
returns a frame as a numpy array so a future visual verifier can scan a range
of frames without re-opening the video for each one.
"""

from __future__ import annotations

import logging
import math
import subprocess
from pathlib import Path

import cv2
import numpy as np

from dialogue_locator.config import FrameConfig
from dialogue_locator.exceptions import FrameExtractionError
from dialogue_locator.media.probe import ensure_binary
from dialogue_locator.models import FrameInfo, VideoInfo, format_timestamp

logger = logging.getLogger(__name__)

_EPS = 1e-6


def timestamp_to_frame(timestamp: float, fps: float) -> int:
    """Index of the frame on screen at ``timestamp`` seconds."""
    if fps <= 0:
        raise FrameExtractionError(f"Invalid fps: {fps}")
    return max(0, int(math.floor(timestamp * fps + _EPS)))


def frame_to_timestamp(frame_number: int, fps: float) -> float:
    return frame_number / fps


class FrameExtractor:
    """Extract single frames from a video file."""

    def __init__(
        self, config: FrameConfig, ffmpeg_binary: str = "ffmpeg", ffprobe_binary: str = "ffprobe"
    ) -> None:
        self._config = config
        self._ffmpeg = ffmpeg_binary
        self._ffprobe = ffprobe_binary

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def extract(self, video: VideoInfo, timestamp: float, dest_path: Path) -> FrameInfo:
        """Save the frame on screen at ``timestamp`` to ``dest_path`` and describe it.

        ``timestamp`` is absolute (seconds into the *source*). ``video`` may be
        a clip (``clip_start > 0``): seeking happens clip-relative, while the
        reported frame number / timestamp stay absolute. ``dest_path``'s suffix
        is replaced by the configured image format.
        """
        fps = self._resolve_fps(video)
        local_frame = timestamp_to_frame(max(0.0, timestamp - video.clip_start), fps)
        local_frame = self._clamp_to_video(local_frame, video, fps)
        frame_time = video.clip_start + frame_to_timestamp(local_frame, fps)
        frame_number = timestamp_to_frame(frame_time, fps)

        dest_path = dest_path.with_suffix(f".{self._config.image_format}")
        try:
            dest_path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            logger.error("Cannot create output directory %s: %s", dest_path.parent, exc)
            raise FrameExtractionError(
                f"Cannot write frame: output directory unavailable ({exc})", details={"path": str(dest_path)}
            ) from exc

        logger.info(
            "Extracting frame %d (%s, requested %s, %.3f fps%s) from %s",
            frame_number,
            format_timestamp(frame_time),
            format_timestamp(timestamp),
            fps,
            f", clip offset {video.clip_start:.3f}s" if video.clip_start else "",
            video.path.name,
        )
        image = self.read_frame(video, local_frame, fps)
        self._write_image(image, dest_path)

        height, width = image.shape[:2]
        info = FrameInfo(
            frame_number=frame_number,
            timestamp=frame_time,
            fps=fps,
            image_path=dest_path,
            width=width,
            height=height,
        )
        logger.info("Frame saved: %s (%dx%d)", dest_path, width, height)
        return info

    def read_frame(self, video: VideoInfo, frame_number: int, fps: float | None = None) -> np.ndarray:
        """Return frame ``frame_number`` as a BGR array (OpenCV, ffmpeg fallback)."""
        fps = fps or self._resolve_fps(video)
        image = self._read_with_opencv(video.path, frame_number)
        if image is None:
            logger.warning(
                "OpenCV could not seek/decode frame %d of %s accurately; using ffmpeg", frame_number, video.path.name
            )
            image = self._read_with_ffmpeg(video.path, frame_to_timestamp(frame_number, fps))
        return image

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    def _resolve_fps(self, video: VideoInfo) -> float:
        if video.fps and video.fps > 0:
            return float(video.fps)
        cap = cv2.VideoCapture(str(video.path))
        try:
            fps = cap.get(cv2.CAP_PROP_FPS) if cap.isOpened() else 0.0
        finally:
            cap.release()
        if not fps or fps <= 0:
            logger.error("Cannot determine fps of %s", video.path)
            raise FrameExtractionError(
                f"Cannot determine frame rate of {video.path.name}", details={"path": str(video.path)}
            )
        logger.debug("fps from OpenCV: %.3f", fps)
        return float(fps)

    @staticmethod
    def _clamp_to_video(frame_number: int, video: VideoInfo, fps: float) -> int:
        last: int | None = None
        if video.frame_count and video.frame_count > 0:
            last = video.frame_count - 1
        elif video.duration:
            last = max(0, int(math.floor(video.duration * fps + _EPS)) - 1)
        if last is not None and frame_number > last:
            logger.warning(
                "Requested frame %d is beyond the last frame %d of %s; clamping", frame_number, last, video.path.name
            )
            return last
        return frame_number

    @staticmethod
    def _read_with_opencv(path: Path, frame_number: int) -> np.ndarray | None:
        cap = cv2.VideoCapture(str(path))
        try:
            if not cap.isOpened():
                logger.error("OpenCV cannot open %s", path)
                raise FrameExtractionError(f"Cannot open video: {path.name}", details={"path": str(path)})
            if not cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number):
                return None
            landed = cap.get(cv2.CAP_PROP_POS_FRAMES)
            if abs(landed - frame_number) > 0.5:
                logger.debug("OpenCV seek landed on %.0f instead of %d", landed, frame_number)
                return None
            ok, image = cap.read()
            if not ok or image is None or image.size == 0:
                return None
            return image
        finally:
            cap.release()

    def _read_with_ffmpeg(self, path: Path, timestamp: float) -> np.ndarray:
        binary = ensure_binary(self._ffmpeg)
        # Output raw BGR to stdout: no temp file, no re-encode loss.
        cmd = [
            binary, "-hide_banner", "-nostdin", "-v", "error",
            "-ss", f"{timestamp:.6f}", "-i", str(path),
            "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "bgr24", "pipe:1",
        ]  # fmt: skip
        probe_cmd = [
            ensure_binary(self._ffprobe),
            "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height",
            "-of", "csv=p=0", str(path),
        ]  # fmt: skip
        try:
            dims = subprocess.run(probe_cmd, capture_output=True, text=True, check=True, timeout=60).stdout
            width, height = (int(x) for x in dims.strip().split(",")[:2])
            out = subprocess.run(cmd, capture_output=True, check=True, timeout=120).stdout
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, ValueError) as exc:
            stderr = getattr(exc, "stderr", b"") or b""
            logger.error("ffmpeg frame fallback failed at %.3fs: %s", timestamp, stderr[-300:] if isinstance(stderr, bytes) else stderr)
            raise FrameExtractionError(
                f"Could not decode a frame at {format_timestamp(timestamp)} from {path.name}",
                details={"path": str(path), "timestamp": timestamp},
            ) from exc
        expected = width * height * 3
        if len(out) < expected:
            logger.error("ffmpeg returned %d bytes, expected %d (frame beyond end?)", len(out), expected)
            raise FrameExtractionError(
                f"No frame available at {format_timestamp(timestamp)} in {path.name}",
                details={"path": str(path), "timestamp": timestamp},
            )
        return np.frombuffer(out[:expected], dtype=np.uint8).reshape(height, width, 3).copy()

    def _write_image(self, image: np.ndarray, dest_path: Path) -> None:
        params: list[int] = []
        if self._config.image_format == "jpg":
            params = [cv2.IMWRITE_JPEG_QUALITY, self._config.jpeg_quality]
        ok = cv2.imwrite(str(dest_path), image, params)
        if not ok or not dest_path.is_file():
            logger.error("Failed to write frame image to %s", dest_path)
            raise FrameExtractionError(f"Failed to write image: {dest_path}", details={"path": str(dest_path)})
