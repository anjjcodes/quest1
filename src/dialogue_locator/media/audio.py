"""Audio extraction with FFmpeg.

Produces 16 kHz mono 16-bit PCM WAV, which is exactly what Whisper consumes,
so the transcriber never has to resample. Two operations:

* :meth:`AudioExtractor.extract`      - the whole track, for the streaming pass
* :meth:`AudioExtractor.extract_clip` - a ``[start, end]`` window, for the
  verification pass (large model on +/- N seconds around a candidate).

FFmpeg is driven through ``subprocess`` rather than a Python binding to keep
the dependency surface small and the failure modes transparent (stderr is
captured and surfaced in the raised error).
"""

from __future__ import annotations

import logging
import subprocess
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from dialogue_locator.config import AudioConfig
from dialogue_locator.exceptions import AudioExtractionError
from dialogue_locator.media.probe import ensure_binary, probe_media
from dialogue_locator.models import (
    AudioInfo,
    PipelineStage,
    ProgressCallback,
    ProgressEvent,
    VideoInfo,
)

logger = logging.getLogger(__name__)

_AUDIO_FILENAME = "audio.wav"


class AudioExtractor:
    """Extract PCM audio from a video (or audio) file using FFmpeg."""

    def __init__(self, config: AudioConfig) -> None:
        self._config = config

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def extract(
        self,
        video: VideoInfo,
        dest_dir: Path,
        progress: ProgressCallback | None = None,
        reuse_existing: bool = True,
    ) -> AudioInfo:
        """Extract the full audio track of ``video`` into ``dest_dir/audio.wav``."""
        ensure_binary(self._config.ffmpeg_binary)  # fail fast, before any work or logging
        dest_dir.mkdir(parents=True, exist_ok=True)
        out_path = dest_dir / _AUDIO_FILENAME

        if reuse_existing and self._is_fresh(out_path, video.path):
            logger.info("Reusing previously extracted audio: %s", out_path)
            return self._describe(out_path)

        self._require_audio_stream(video.path)
        logger.info(
            "Extracting audio from %s -> %s (%d Hz, %d ch)",
            video.path.name,
            out_path,
            self._config.sample_rate,
            self._config.channels,
        )
        self._emit(progress, "Extracting audio", 0.0, {"path": str(out_path)})

        cmd = self._base_command(video.path, out_path)
        self._run_ffmpeg(cmd, out_path, total_duration=video.duration, progress=progress)

        info = self._describe(out_path)
        logger.info("Audio ready: %s (%.1fs)", out_path.name, info.duration or 0.0)
        self._emit(progress, "Audio extracted", 1.0, {"duration": info.duration})
        return info

    def extract_clip(
        self,
        source: Path,
        start: float,
        end: float,
        dest_path: Path,
    ) -> AudioInfo:
        """Extract ``[start, end]`` seconds of ``source`` into ``dest_path``.

        ``source`` may be the original video or an already extracted WAV.
        Timestamps in the resulting clip are relative to ``start``; callers
        must add ``start`` back to map them to the video timeline.
        """
        ensure_binary(self._config.ffmpeg_binary)
        start = max(0.0, float(start))
        end = float(end)
        if end <= start:
            logger.error("Invalid clip range %.3f-%.3f", start, end)
            raise AudioExtractionError(
                f"Invalid clip range: start={start:.3f} must be < end={end:.3f}",
                details={"start": start, "end": end},
            )
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        duration = end - start
        logger.debug("Extracting clip %.3f-%.3fs from %s -> %s", start, end, source.name, dest_path)

        # -ss before -i: fast seek to the nearest keyframe, then decode up to the exact
        # time. Accurate for transcoding since ffmpeg 2.1; no drift for WAV input.
        cmd = self._base_command(source, dest_path, pre_input=["-ss", f"{start:.3f}", "-t", f"{duration:.3f}"])
        self._run_ffmpeg(cmd, dest_path, total_duration=None, progress=None)
        return self._describe(dest_path)

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    def _base_command(
        self, source: Path, out_path: Path, pre_input: Iterable[str] = ()
    ) -> list[str]:
        binary = ensure_binary(self._config.ffmpeg_binary)
        return [
            binary,
            "-hide_banner",
            "-nostdin",
            "-v",
            "error",
            "-y",
            *pre_input,
            "-i",
            str(source),
            "-vn",  # drop video
            "-sn",  # drop subtitles
            "-dn",  # drop data streams
            "-ac",
            str(self._config.channels),
            "-ar",
            str(self._config.sample_rate),
            "-c:a",
            "pcm_s16le",
            "-progress",
            "pipe:1",  # machine-readable progress on stdout
            "-nostats",
            str(out_path),
        ]

    def _run_ffmpeg(
        self,
        cmd: list[str],
        out_path: Path,
        total_duration: float | None,
        progress: ProgressCallback | None,
    ) -> None:
        logger.debug("Running: %s", " ".join(cmd))
        try:
            proc = subprocess.Popen(  # noqa: S603 - args are built internally, not from user text
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
        except OSError as exc:
            raise AudioExtractionError(f"Could not start ffmpeg: {exc}") from exc

        last_bucket = -1
        try:
            assert proc.stdout is not None
            for line in proc.stdout:
                # Lines look like: out_time_us=12345678 / progress=continue|end
                if progress is None or total_duration is None or total_duration <= 0:
                    continue
                if line.startswith("out_time_us=") or line.startswith("out_time_ms="):
                    try:
                        # Both keys are microseconds in practice (ffmpeg quirk).
                        done = int(line.split("=", 1)[1]) / 1_000_000
                    except ValueError:
                        continue
                    fraction = max(0.0, min(done / total_duration, 1.0))
                    bucket = int(fraction * 20)
                    if bucket != last_bucket:
                        last_bucket = bucket
                        self._emit(progress, f"Extracting audio {fraction:.0%}", fraction, {})
            _, stderr = proc.communicate(timeout=self._config.timeout_seconds)
        except subprocess.TimeoutExpired as exc:
            proc.kill()
            proc.communicate()
            logger.error("ffmpeg timed out after %ds, killed (output %s)", self._config.timeout_seconds, out_path)
            raise AudioExtractionError(
                f"ffmpeg timed out after {self._config.timeout_seconds}s",
                details={"output": str(out_path)},
            ) from exc

        if proc.returncode != 0:
            message = (stderr or "").strip().splitlines()
            tail = message[-1] if message else "unknown ffmpeg error"
            logger.error("ffmpeg failed (exit %d): %s", proc.returncode, tail)
            raise AudioExtractionError(
                f"ffmpeg failed (exit {proc.returncode}): {tail}",
                details={"output": str(out_path), "stderr": "\n".join(message[-10:])},
            )

        if not out_path.is_file():
            logger.error("ffmpeg reported success but %s does not exist", out_path)
            raise AudioExtractionError(
                "ffmpeg reported success but produced no file", details={"output": str(out_path)}
            )

    def _require_audio_stream(self, path: Path) -> None:
        probe = probe_media(path, self._config.ffprobe_binary)
        if not probe.has_audio:
            logger.error("No audio stream in %s; cannot transcribe", path)
            raise AudioExtractionError(
                f"'{path.name}' has no audio stream, so the dialogue cannot be transcribed.",
                details={"path": str(path)},
            )

    def _describe(self, path: Path) -> AudioInfo:
        probe = probe_media(path, self._config.ffprobe_binary)
        if not probe.has_audio or not probe.duration or probe.duration <= 0:
            logger.error("Extracted file %s has no audio data (duration=%s)", path, probe.duration)
            raise AudioExtractionError(
                f"'{path.name}' contains no audio data (requested range beyond end of media, "
                "or source has no decodable audio).",
                details={"path": str(path), "duration": probe.duration},
            )
        return AudioInfo(
            path=path,
            sample_rate=self._config.sample_rate,
            channels=self._config.channels,
            duration=probe.duration,
        )

    @staticmethod
    def _is_fresh(out_path: Path, source: Path) -> bool:
        """True if ``out_path`` exists, is non-trivial and newer than ``source``."""
        try:
            return (
                out_path.is_file()
                and out_path.stat().st_size > 0
                and out_path.stat().st_mtime >= source.stat().st_mtime
            )
        except OSError:
            return False

    @staticmethod
    def _emit(
        progress: ProgressCallback | None, message: str, fraction: float, details: dict[str, Any]
    ) -> None:
        if progress is not None:
            progress(ProgressEvent(PipelineStage.AUDIO, message, fraction, details))
