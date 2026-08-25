"""Verifier interface.

A verifier takes the first-pass :class:`MatchCandidate` plus the audio and
dialogue the pipeline knows about and returns a :class:`VerificationOutcome`.
Verifiers are chained by the pipeline; each one sees the candidate as refined
by the previous ones. They exist to answer one question: *is this really the
line, and exactly when does it start?*

V1 ships one verifier (:class:`~dialogue_locator.verification.asr_verifier.AsrVerifier`,
a larger Whisper model over a short window); another audio-based check (e.g.
a second ASR engine as an independent judge) would plug in here. The visual
checks (V2 face presence, V3 mouth movement) are *not* verifiers: they judge
onscreen-ness after the frame exists, so they run as their own pipeline
stages (see ``vision/`` and the pipeline docstring).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from dialogue_locator.models import MatchCandidate, VerificationOutcome


@dataclass
class VerificationContext:
    """Everything a verifier may need. Built once per job by the pipeline."""

    dialogue: str
    audio_samples: np.ndarray  # float32, 16 kHz mono, full track
    audio_path: Path
    sample_rate: int = 16_000

    @property
    def audio_duration(self) -> float:
        return float(self.audio_samples.shape[0]) / self.sample_rate

    def slice_audio(self, start: float, end: float) -> tuple[np.ndarray, float, float]:
        """Return ``(samples, actual_start, actual_end)`` for ``[start, end]`` clamped to the track."""
        start = max(0.0, float(start))
        end = min(self.audio_duration, float(end))
        if end <= start:
            return self.audio_samples[0:0], start, start
        a = int(round(start * self.sample_rate))
        b = int(round(end * self.sample_rate))
        return self.audio_samples[a:b], start, end


class Verifier(ABC):
    """Confirms, refines or rejects a candidate match."""

    #: Identifier recorded in :attr:`VerificationOutcome.verifier`.
    name: str = "verifier"

    def warm_up(self) -> None:
        """Load models / allocate resources ahead of the first job. Optional."""

    @abstractmethod
    def verify(self, candidate: MatchCandidate, context: VerificationContext) -> VerificationOutcome:
        """Return an outcome. Must not raise for expected failures - return
        ``VerificationStatus.FAILED`` with a message instead so the pipeline can
        fall back to the first-pass result with a warning."""
