"""Faster-Whisper implementation of :class:`Transcriber`.

Design notes
------------
* Models are loaded lazily and cached process-wide (keyed by name/device/
  compute type) so the fast and verify models are each loaded once and shared
  between jobs. Loading ``medium`` can take tens of seconds.
* ``transcribe()`` wraps faster-whisper's own lazy segment generator, so
  decoding only proceeds as far as the consumer pulls words.
* Word timestamps come from Whisper's cross-attention alignment. If a segment
  has none (rare: alignment failure), the segment text is split into words
  spread evenly across the segment so downstream matching still works, and a
  warning is logged.
"""

from __future__ import annotations

import logging
import os
import threading
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from dialogue_locator.config import WhisperConfig
from dialogue_locator.exceptions import TranscriptionError
from dialogue_locator.models import (
    PipelineStage,
    ProgressCallback,
    ProgressEvent,
    Word,
    format_timestamp,
)
from dialogue_locator.transcription.base import Transcriber, pcm_duration

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Model cache
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class _ModelKey:
    name: str
    device: str
    compute_type: str
    cpu_threads: int
    download_root: str | None


class WhisperModelCache:
    """Process-wide cache of loaded ``WhisperModel`` instances (thread-safe)."""

    def __init__(self) -> None:
        self._models: dict[_ModelKey, Any] = {}
        self._lock = threading.Lock()

    def get(self, model_name: str, config: WhisperConfig) -> Any:
        # 0 = use every core: ctranslate2's own default is 4 intra-op threads,
        # which leaves half an 8-core machine idle. Resolved here so the cache
        # key holds the effective value.
        cpu_threads = config.cpu_threads or (os.cpu_count() or 4)
        key = _ModelKey(
            model_name,
            config.device,
            config.compute_type,
            cpu_threads,
            str(config.download_root) if config.download_root else None,
        )
        with self._lock:
            model = self._models.get(key)
            if model is None:
                model = self._load(key)
                self._models[key] = model
            else:
                logger.debug("Whisper model '%s' served from cache", key.name)
            return model

    def clear(self) -> None:
        with self._lock:
            self._models.clear()

    @staticmethod
    def _load(key: _ModelKey) -> Any:
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:  # pragma: no cover - environment problem
            raise TranscriptionError(
                "faster-whisper is not installed (pip install faster-whisper)."
            ) from exc

        logger.info(
            "Loading Whisper model '%s' (device=%s, compute_type=%s, cpu_threads=%d) ...",
            key.name,
            key.device,
            key.compute_type,
            key.cpu_threads,
        )
        try:
            model = WhisperModel(
                key.name,
                device=key.device,
                compute_type=key.compute_type,
                cpu_threads=key.cpu_threads,
                download_root=key.download_root,
            )
        except Exception as exc:  # noqa: BLE001 - boundary: HF download / ct2 errors
            logger.error("Failed to load Whisper model '%s': %s", key.name, exc)
            raise TranscriptionError(
                f"Failed to load Whisper model '{key.name}': {exc}",
                details={"model": key.name, "device": key.device, "compute_type": key.compute_type},
            ) from exc
        logger.info("Whisper model '%s' loaded.", key.name)
        return model


#: Default shared cache. Tests may construct their own.
default_model_cache = WhisperModelCache()


# --------------------------------------------------------------------------- #
# Transcriber
# --------------------------------------------------------------------------- #
class FasterWhisperTranscriber(Transcriber):
    """Stream word-level transcription using Faster-Whisper.

    Args:
        model_name: Whisper size or path/HF id (``tiny``, ``small``, ``medium``,
            ``large-v3``, ``distil-large-v3`` ...).
        config: decoding settings shared by all models.
        model_cache: where loaded models live; defaults to the process-wide cache.
    """

    def __init__(
        self,
        model_name: str,
        config: WhisperConfig,
        model_cache: WhisperModelCache | None = None,
    ) -> None:
        self.model_name = model_name
        self._config = config
        self._cache = model_cache or default_model_cache
        self.name = f"faster_whisper:{model_name}"

    # ------------------------------------------------------------------ #
    def warm_up(self) -> None:
        self._model()

    def _model(self) -> Any:
        return self._cache.get(self.model_name, self._config)

    def _decode_options(self) -> dict[str, Any]:
        opts: dict[str, Any] = {
            "word_timestamps": True,
            "beam_size": self._config.beam_size,
            "vad_filter": self._config.vad_filter,
            "condition_on_previous_text": self._config.condition_on_previous_text,
        }
        if self._config.language:
            opts["language"] = self._config.language
        if self._config.vad_filter:
            opts["vad_parameters"] = {"min_silence_duration_ms": self._config.vad_min_silence_ms}
        return opts

    # ------------------------------------------------------------------ #
    def transcribe(
        self,
        audio: np.ndarray | Path,
        *,
        offset: float = 0.0,
        progress: ProgressCallback | None = None,
    ) -> Iterator[Word]:
        samples = self.as_samples(audio)
        duration = pcm_duration(samples)
        model = self._model()

        logger.info(
            "[%s] transcribing %.1fs of audio (offset %.2fs, lang=%s, beam=%d, vad=%s)",
            self.name,
            duration,
            offset,
            self._config.language or "auto",
            self._config.beam_size,
            self._config.vad_filter,
        )

        try:
            segments, info = model.transcribe(samples, **self._decode_options())
        except Exception as exc:  # noqa: BLE001
            logger.error("[%s] failed to start decoding: %s", self.name, exc)
            raise TranscriptionError(
                f"Whisper failed to start decoding: {exc}", details={"model": self.model_name}
            ) from exc

        if info is not None and getattr(info, "language", None):
            logger.debug(
                "[%s] language=%s (p=%.2f)",
                self.name,
                info.language,
                getattr(info, "language_probability", 0.0) or 0.0,
            )

        last_bucket = -1
        n_words = 0
        try:
            for seg_index, segment in enumerate(segments):
                words = self._segment_words(segment, seg_index, offset)
                n_words += len(words)
                logger.debug(
                    "[%s] seg %d [%s - %s] %r",
                    self.name,
                    seg_index,
                    format_timestamp(segment.start + offset),
                    format_timestamp(segment.end + offset),
                    segment.text.strip(),
                )
                yield from words

                if progress is not None and duration > 0:
                    fraction = min(segment.end / duration, 1.0)
                    bucket = int(fraction * 50)  # every 2 %
                    if bucket != last_bucket:
                        last_bucket = bucket
                        progress(
                            ProgressEvent(
                                PipelineStage.TRANSCRIPTION,
                                f"Transcribed {format_timestamp(segment.end + offset)} "
                                f"/ {format_timestamp(duration + offset)}",
                                fraction,
                                {"words": n_words, "position": segment.end + offset},
                            )
                        )
        except TranscriptionError:
            raise
        except GeneratorExit:
            logger.info("[%s] transcription stopped early by consumer after %d words", self.name, n_words)
            raise
        except Exception as exc:  # noqa: BLE001
            logger.error("[%s] failed while decoding after %d words: %s", self.name, n_words, exc)
            raise TranscriptionError(
                f"Whisper failed while decoding: {exc}", details={"model": self.model_name}
            ) from exc

        logger.info("[%s] finished: %d words over %.1fs of audio", self.name, n_words, duration)

    # ------------------------------------------------------------------ #
    @staticmethod
    def _segment_words(segment: Any, seg_index: int, offset: float) -> list[Word]:
        raw_words = getattr(segment, "words", None)
        out: list[Word] = []
        if raw_words:
            for w in raw_words:
                text = (w.word or "").strip()
                if not text:
                    continue
                out.append(
                    Word(
                        text=text,
                        start=float(w.start) + offset,
                        end=float(w.end) + offset,
                        probability=float(w.probability) if w.probability is not None else None,
                        segment_index=seg_index,
                    )
                )
            return out

        # Fallback: no word alignment for this segment. Spread words evenly.
        tokens = (segment.text or "").split()
        if not tokens:
            return out
        logger.warning(
            "Segment %d has no word timestamps; distributing %d words evenly over %.2fs",
            seg_index,
            len(tokens),
            segment.end - segment.start,
        )
        step = (segment.end - segment.start) / len(tokens)
        for i, tok in enumerate(tokens):
            start = segment.start + i * step
            out.append(
                Word(
                    text=tok,
                    start=start + offset,
                    end=start + step + offset,
                    probability=None,
                    segment_index=seg_index,
                )
            )
        return out
