"""Tests for the verification stage (fake transcriber; real models only opt-in)."""

from __future__ import annotations

import logging
import os
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest

from dialogue_locator.config import MatchingConfig, VerificationConfig, WhisperConfig
from dialogue_locator.exceptions import TranscriptionError
from dialogue_locator.models import MatchCandidate, VerificationStatus, Word
from dialogue_locator.transcription.base import Transcriber
from dialogue_locator.verification.asr_verifier import AsrVerifier
from dialogue_locator.verification.base import VerificationContext, Verifier
from tests.conftest import requires_ffmpeg

logger = logging.getLogger("tests")
DIALOGUE = "My mind rebels at stagnation"
SR = 16_000


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def words_at(text: str, start: float, step: float = 0.4) -> list[Word]:
    return [Word(t, start + i * step, start + i * step + step * 0.8) for i, t in enumerate(text.split())]


def candidate(text: str, start: float, score: float) -> MatchCandidate:
    ws = words_at(text, start)
    return MatchCandidate(score=score, start=ws[0].start, end=ws[-1].end, matched_text=text, words=tuple(ws))


class FakeTranscriber(Transcriber):
    """Returns preset words; records the clip and offset it was given."""

    name = "fake:large"

    def __init__(self, words: list[Word] | None = None, error: Exception | None = None):
        self._words = words or []
        self._error = error
        self.calls: list[tuple[int, float]] = []

    def transcribe(self, audio, *, offset=0.0, progress=None) -> Iterator[Word]:
        self.calls.append((int(audio.shape[0]), offset))
        if self._error:
            raise self._error
        # words are given in *clip-relative* time; apply the offset like a real transcriber
        for w in self._words:
            yield Word(w.text, w.start + offset, w.end + offset, w.probability)


def context(duration: float = 120.0) -> VerificationContext:
    return VerificationContext(
        dialogue=DIALOGUE,
        audio_samples=np.zeros(int(duration * SR), dtype=np.float32),
        audio_path=Path("audio.wav"),
    )


def verifier(tr: Transcriber, **cfg) -> AsrVerifier:
    return AsrVerifier(tr, MatchingConfig(), VerificationConfig(**cfg))


# --------------------------------------------------------------------------- #
# VerificationContext
# --------------------------------------------------------------------------- #
def test_slice_audio_clamps_to_track():
    ctx = context(duration=10.0)
    clip, s, e = ctx.slice_audio(-5.0, 3.0)
    assert (s, e) == (0.0, 3.0) and clip.shape[0] == 3 * SR
    clip, s, e = ctx.slice_audio(8.0, 25.0)
    assert (s, e) == (8.0, 10.0) and clip.shape[0] == 2 * SR
    clip, s, e = ctx.slice_audio(50.0, 60.0)  # entirely beyond the end
    assert clip.shape[0] == 0
    assert ctx.audio_duration == 10.0


def test_verifier_is_abstract():
    with pytest.raises(TypeError):
        Verifier()  # type: ignore[abstract]


# --------------------------------------------------------------------------- #
# AsrVerifier outcomes
# --------------------------------------------------------------------------- #
def test_confirmed_uses_large_model_timestamps():
    first = candidate("my mine rebels at stagnation", start=60.0, score=96.4)
    # In the clip (starting at 40 s), the large model hears the exact line at 20.35 s -> 60.35 s absolute
    tr = FakeTranscriber(words_at("well then my mind rebels at stagnation indeed", start=19.55))
    out = verifier(tr, search_window_seconds=20.0).verify(first, context())

    assert out.status is VerificationStatus.CONFIRMED
    assert out.verifier == "asr_large_model"
    assert out.score == 100.0
    assert out.refined is not None
    assert out.refined.matched_text == "my mind rebels at stagnation"
    assert out.refined.start == pytest.approx(40.0 + 19.55 + 2 * 0.4)  # absolute time, from offset
    assert out.details["shift_seconds"] == pytest.approx(out.refined.start - 60.0, abs=1e-3)
    # clip was +/- 20 s around the candidate: [40.0, 81.92 + 20] and offset == clip start
    n_samples, offset = tr.calls[0]
    assert offset == 40.0
    assert n_samples == pytest.approx((first.end + 20.0 - 40.0) * SR, abs=2)


def test_rejected_when_score_drops_too_far():
    first = candidate("my mind rebels at stagnation", start=60.0, score=100.0)
    tr = FakeTranscriber(words_at("my mind revolts against stagnant water", start=20.0))  # ~70
    out = verifier(tr, max_score_drop=5.0).verify(first, context())

    assert out.status is VerificationStatus.REJECTED
    assert out.refined is not None and out.score == out.refined.score < 95
    assert "keeping first-pass timestamp" in out.message
    assert out.details["first_pass_score"] == 100.0


def test_small_drop_within_tolerance_is_confirmed():
    first = candidate("my mind rebels at stagnation", start=60.0, score=100.0)
    tr = FakeTranscriber(words_at("my mind rebels at stag nation", start=20.0))  # 98.2
    out = verifier(tr, max_score_drop=5.0).verify(first, context())
    assert out.status is VerificationStatus.CONFIRMED
    assert out.score == pytest.approx(98.2, abs=0.1)


def test_higher_score_than_first_pass_is_confirmed():
    first = candidate("my mind reveals a stagnation", start=60.0, score=92.9)
    tr = FakeTranscriber(words_at("my mind rebels at stagnation", start=20.0))
    out = verifier(tr).verify(first, context())
    assert out.status is VerificationStatus.CONFIRMED and out.score == 100.0


def test_transcription_failure_returns_failed_not_raise():
    first = candidate("my mind rebels at stagnation", start=60.0, score=100.0)
    tr = FakeTranscriber(error=TranscriptionError("model exploded"))
    logger.info(">>> expecting an ERROR log: verifier transcription failure must be downgraded to FAILED")
    out = verifier(tr).verify(first, context())
    assert out.status is VerificationStatus.FAILED
    assert out.refined is None
    assert "model exploded" in out.message
    assert out.details["error"]["stage"] == "transcription"
    logger.info("<<< got FAILED outcome as expected")


def test_no_words_in_window_is_rejected():
    first = candidate("my mind rebels at stagnation", start=60.0, score=100.0)
    logger.info(">>> expecting a WARNING: large model produced no words")
    out = verifier(FakeTranscriber([])).verify(first, context())
    assert out.status is VerificationStatus.REJECTED and out.score == 0.0
    logger.info("<<< got REJECTED as expected")


def test_disabled_verifier_is_skipped():
    first = candidate("my mind rebels at stagnation", start=60.0, score=100.0)
    tr = FakeTranscriber(words_at("x", 0))
    out = verifier(tr, enabled=False).verify(first, context())
    assert out.status is VerificationStatus.SKIPPED
    assert tr.calls == []  # model never invoked


def test_window_clamped_at_start_of_track():
    first = candidate("my mind rebels at stagnation", start=3.0, score=100.0)
    tr = FakeTranscriber(words_at("my mind rebels at stagnation", start=3.0))
    out = verifier(tr, search_window_seconds=20.0).verify(first, context(duration=60.0))
    _, offset = tr.calls[0]
    assert offset == 0.0
    assert out.details["clip_start"] == 0.0
    assert out.refined.start == pytest.approx(3.0)


def test_window_clamped_at_end_of_track():
    first = candidate("my mind rebels at stagnation", start=58.0, score=100.0)
    tr = FakeTranscriber(words_at("my mind rebels at stagnation", start=20.0))
    out = verifier(tr, search_window_seconds=20.0).verify(first, context(duration=60.0))
    assert out.details["clip_end"] == 60.0
    n_samples, offset = tr.calls[0]
    assert offset == 38.0 and n_samples == 22 * SR


def test_candidate_beyond_audio_is_failed():
    first = candidate("my mind rebels at stagnation", start=500.0, score=100.0)
    logger.info(">>> expecting an ERROR log: empty audio window")
    out = verifier(FakeTranscriber(), search_window_seconds=1.0).verify(first, context(duration=60.0))
    assert out.status is VerificationStatus.FAILED and "Empty audio window" in out.message
    logger.info("<<< got FAILED as expected")


def test_outcome_serialises():
    first = candidate("my mind rebels at stagnation", start=60.0, score=100.0)
    tr = FakeTranscriber(words_at("my mind rebels at stagnation", start=20.0))
    d = verifier(tr).verify(first, context()).to_dict()
    assert d["verifier"] == "asr_large_model" and d["status"] == "confirmed"
    assert d["refined"]["timestamp"] == "00:01:00.000"
    assert set(d["details"]) >= {"model", "clip_start", "clip_end", "first_pass_score", "seconds", "clip_words", "shift_seconds"}


# --------------------------------------------------------------------------- #
# Real models (opt-in): tiny first pass, base verification
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(os.environ.get("DL_RUN_MODEL_TESTS") != "1", reason="set DL_RUN_MODEL_TESTS=1")
@requires_ffmpeg
def test_real_tiny_then_base(tmp_path: Path):
    import shutil
    import subprocess

    from dialogue_locator.matching.matcher import StreamingMatcher
    from dialogue_locator.transcription.base import load_pcm
    from dialogue_locator.transcription.faster_whisper import FasterWhisperTranscriber

    if not shutil.which("say"):
        pytest.skip("macOS 'say' not available")
    aiff, wav = tmp_path / "s.aiff", tmp_path / "s.wav"
    subprocess.run(["say", "-o", str(aiff), "I need work. My mind rebels at stagnation. Give me problems."], check=True)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(aiff), "-ac", "1", "-ar", "16000", str(wav)], check=True)

    cfg = WhisperConfig(vad_filter=False)
    samples = load_pcm(wav)
    first = StreamingMatcher(DIALOGUE, MatchingConfig()).feed_many(FasterWhisperTranscriber("tiny", cfg).transcribe(samples))
    assert first is not None
    ctx = VerificationContext(dialogue=DIALOGUE, audio_samples=samples, audio_path=wav)
    out = AsrVerifier(FasterWhisperTranscriber("base", cfg), MatchingConfig(), VerificationConfig(search_window_seconds=5)).verify(first, ctx)
    assert out.status is VerificationStatus.CONFIRMED
    assert abs(out.refined.start - first.start) < 1.0
