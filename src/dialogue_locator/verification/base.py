"""Verifier interface.

A verifier takes the first-pass :class:`MatchCandidate` plus everything the
pipeline knows (audio, video, dialogue) and returns a
:class:`VerificationOutcome`. Verifiers are chained by the pipeline; each
one sees the candidate as refined by the previous ones.

Version 1 ships one verifier (:class:`~dialogue_locator.verification.asr_verifier.AsrVerifier`,
a larger Whisper model over a short window). Version 2's visual "is the speaker
on camera" check plugs in here as another :class:`Verifier` subclass - it
gets ``context.video.path`` and the candidate's time range, and can return a
``refined`` candidate whose ``start`` is the first frame the speaker is visible.
Nothing upstream (download, transcription, matching) needs to change.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from dialogue_locator.models import MatchCandidate, VerificationOutcome, VideoInfo


@dataclass
class VerificationContext:
    """Everything a verifier may need. Built once per job by the pipeline."""

    dialogue: str
    audio_samples: np.ndarray  # float32, 16 kHz mono, full track
    audio_path: Path
    sample_rate: int = 16_000
    video: VideoInfo | None = None  # lets V2 verifiers open the video file
    extra: dict[str, Any] = field(default_factory=dict)  # free-form, for future verifiers

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

    @abstractmethod
    def verify(self, candidate: MatchCandidate, context: VerificationContext) -> VerificationOutcome:
        """Return an outcome. Must not raise for expected failures - return
        ``VerificationStatus.FAILED`` with a message instead so the pipeline can
        fall back to the first-pass result with a warning."""
