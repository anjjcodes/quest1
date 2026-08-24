"""Transcriber interface and shared audio helpers.

The pipeline depends only on :class:`Transcriber`; the concrete Faster-Whisper
implementation lives in :mod:`dialogue_locator.transcription.faster_whisper`.
This keeps ASR swappable (e.g. whisper.cpp, a cloud API, or a fake for tests)
without touching matching / verification code.

Contract
--------
``transcribe()`` is a *lazy generator*. Words are yielded as soon as the
model finishes each segment, in chronological order, with absolute timestamps
(``offset`` + model time). The consumer may stop iterating at any point; the
transcriber must not do work beyond what was consumed. This is what makes
"stop at the first match" cost only the audio up to the match.
"""

from __future__ import annotations

import logging
import wave
from abc import ABC, abstractmethod
from collections.abc import Iterator
from pathlib import Path

import numpy as np

from dialogue_locator.exceptions import TranscriptionError
from dialogue_locator.models import ProgressCallback, Word

logger = logging.getLogger(__name__)

#: Sample rate every transcriber in this package expects.
TARGET_SAMPLE_RATE = 16_000


def load_pcm(path: Path, expected_rate: int = TARGET_SAMPLE_RATE) -> np.ndarray:
    """Load a 16-bit PCM WAV as float32 samples in ``[-1, 1]``.

    We decode the WAV ourselves (stdlib ``wave`` + numpy) instead of letting
    the ASR library decode via PyAV: it is trivial for PCM, avoids a second
    FFmpeg binding at runtime, and gives us an in-memory array that the
    verifier can slice for its search window without touching disk.
    """
    try:
        with wave.open(str(path), "rb") as wav:
            rate, channels, width, n = (
                wav.getframerate(),
                wav.getnchannels(),
                wav.getsampwidth(),
                wav.getnframes(),
            )
            raw = wav.readframes(n)
    except (OSError, wave.Error, EOFError) as exc:
        logger.error("Cannot read WAV %s: %s", path, exc)
        raise TranscriptionError(f"Cannot read WAV file {path}: {exc}", details={"path": str(path)}) from exc

    if width != 2:
        raise TranscriptionError(
            f"Expected 16-bit PCM WAV, got {width * 8}-bit: {path}", details={"path": str(path)}
        )
    if rate != expected_rate:
        logger.error("WAV %s has sample rate %d, expected %d", path, rate, expected_rate)
        raise TranscriptionError(
            f"Expected {expected_rate} Hz audio, got {rate} Hz: {path}. "
            "Extract audio with AudioExtractor (AudioConfig.sample_rate) first.",
            details={"path": str(path), "sample_rate": rate},
        )

    samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if channels > 1:  # downmix defensively; the extractor normally emits mono
        samples = samples.reshape(-1, channels).mean(axis=1)
    if samples.size == 0:
        logger.error("WAV %s contains no samples", path)
        raise TranscriptionError(f"WAV file contains no samples: {path}", details={"path": str(path)})
    logger.debug("Loaded %s: %d samples (%.2fs) at %d Hz", path.name, samples.size, samples.size / rate, rate)
    return samples


def pcm_duration(samples: np.ndarray, rate: int = TARGET_SAMPLE_RATE) -> float:
    return float(samples.shape[0]) / rate


class Transcriber(ABC):
    """Streams timestamped words from 16 kHz mono audio."""

    #: Human-readable identifier, e.g. ``"faster_whisper:small"``. Used in logs/results.
    name: str = "transcriber"

    def warm_up(self) -> None:
        """Load models / allocate resources ahead of the first call. Optional."""

    @abstractmethod
    def transcribe(
        self,
        audio: np.ndarray | Path,
        *,
        offset: float = 0.0,
        progress: ProgressCallback | None = None,
    ) -> Iterator[Word]:
        """Yield :class:`Word` objects in chronological order.

        Args:
            audio: float32 samples at 16 kHz, or a path to such a WAV file.
            offset: seconds to add to every timestamp (used when ``audio`` is a
                clip cut from a longer track).
            progress: optional callback; implementations emit
                ``PipelineStage.TRANSCRIPTION`` events as decoding advances.
        """

    def transcribe_all(self, audio: np.ndarray | Path, *, offset: float = 0.0) -> list[Word]:
        """Convenience: exhaust the stream into a list (for clips / tests)."""
        return list(self.transcribe(audio, offset=offset))

    @staticmethod
    def as_samples(audio: np.ndarray | Path) -> np.ndarray:
        if isinstance(audio, np.ndarray):
            if audio.dtype != np.float32:
                audio = audio.astype(np.float32)
            return audio
        return load_pcm(Path(audio))
