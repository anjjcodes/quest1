"""Tests for the transcription layer.

Unit tests use a fake Whisper model so they run in milliseconds without
downloading weights. ``test_real_tiny_model`` exercises the real library and
is opt-in via ``DL_RUN_MODEL_TESTS=1`` (downloads ~75 MB on first run).
"""

from __future__ import annotations

import logging
import os
import wave
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from dialogue_locator.config import WhisperConfig
from dialogue_locator.exceptions import TranscriptionError
from dialogue_locator.models import PipelineStage, ProgressEvent, Word
from dialogue_locator.transcription.base import Transcriber, load_pcm, pcm_duration
from dialogue_locator.transcription.faster_whisper import (
    FasterWhisperTranscriber,
    WhisperModelCache,
)
from tests.conftest import expect_error, requires_ffmpeg

logger = logging.getLogger("tests")


# --------------------------------------------------------------------------- #
# WAV loading
# --------------------------------------------------------------------------- #
def _write_wav(path: Path, rate: int, channels: int, seconds: float, width: int = 2) -> None:
    n = int(rate * seconds)
    t = np.arange(n) / rate
    sig = (np.sin(2 * np.pi * 440 * t) * 0.5 * 32767).astype(np.int16)
    if channels == 2:
        sig = np.stack([sig, sig], axis=1)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(width)
        w.setframerate(rate)
        w.writeframes(sig.tobytes())


def test_load_pcm_mono(tmp_path: Path):
    p = tmp_path / "a.wav"
    _write_wav(p, 16_000, 1, 1.5)
    s = load_pcm(p)
    assert s.dtype == np.float32
    assert s.shape == (24_000,)
    assert np.abs(s).max() <= 1.0
    assert pcm_duration(s) == pytest.approx(1.5)


def test_load_pcm_downmixes_stereo(tmp_path: Path):
    p = tmp_path / "s.wav"
    _write_wav(p, 16_000, 2, 1.0)
    assert load_pcm(p).shape == (16_000,)


def test_load_pcm_rejects_wrong_rate(tmp_path: Path):
    p = tmp_path / "r.wav"
    _write_wav(p, 44_100, 1, 1.0)
    with expect_error(TranscriptionError, match="16000 Hz"):
        load_pcm(p)


def test_load_pcm_rejects_non_wav(tmp_path: Path):
    p = tmp_path / "x.wav"
    p.write_bytes(b"not a wav")
    with expect_error(TranscriptionError):
        load_pcm(p)


@requires_ffmpeg
def test_load_pcm_from_extractor_output(sample_wav: Path):
    s = load_pcm(sample_wav)
    assert pcm_duration(s) == pytest.approx(4.0, abs=0.01)


# --------------------------------------------------------------------------- #
# Fake Whisper model
# --------------------------------------------------------------------------- #
def _w(word: str, start: float, end: float, p: float = 0.9):
    return SimpleNamespace(word=word, start=start, end=end, probability=p)


def _seg(start: float, end: float, text: str, words):
    return SimpleNamespace(start=start, end=end, text=text, words=words)


class FakeWhisperModel:
    """Mimics faster_whisper.WhisperModel.transcribe() lazily."""

    def __init__(self, segments, fail_at: int | None = None):
        self._segments = segments
        self._fail_at = fail_at
        self.calls: list[dict] = []
        self.segments_pulled = 0

    def transcribe(self, audio, **options):
        self.calls.append(options)
        info = SimpleNamespace(language="en", language_probability=0.99, duration=len(audio) / 16_000)
        return self._gen(), info

    def _gen(self):
        for i, s in enumerate(self._segments):
            if self._fail_at is not None and i == self._fail_at:
                raise RuntimeError("decoder exploded")
            self.segments_pulled += 1
            yield s


SEGMENTS = [
    _seg(0.5, 2.0, " Hello there, friend.", [_w(" Hello", 0.5, 0.9), _w(" there,", 0.9, 1.3), _w(" friend.", 1.5, 2.0)]),
    _seg(2.5, 4.0, " My mind rebels", [_w(" My", 2.5, 2.7), _w(" mind", 2.7, 3.0), _w(" rebels", 3.0, 4.0, 0.4)]),
    _seg(4.0, 5.0, " at stagnation", [_w(" at", 4.0, 4.2), _w(" stagnation", 4.2, 5.0)]),
    _seg(5.0, 5.5, "", [_w("   ", 5.0, 5.5)]),  # whitespace-only word must be dropped
]
NO_ALIGNMENT_SEGMENT = _seg(4.0, 5.0, " at stagnation", None)  # words=None -> fallback path


@pytest.fixture
def fake_cache():
    class Cache(WhisperModelCache):
        def __init__(self, model):
            super().__init__()
            self.model = model
            self.loads = 0

        def get(self, model_name, config):
            self.loads += 1
            return self.model

    return Cache


def _transcriber(cache) -> FasterWhisperTranscriber:
    return FasterWhisperTranscriber("fake-small", WhisperConfig(), model_cache=cache)


def test_words_are_stripped_offset_and_indexed(fake_cache):
    model = FakeWhisperModel(SEGMENTS)
    tr = _transcriber(fake_cache(model))
    words = tr.transcribe_all(np.zeros(16_000 * 6, dtype=np.float32), offset=100.0)

    assert [w.text for w in words] == ["Hello", "there,", "friend.", "My", "mind", "rebels", "at", "stagnation"]
    assert words[0].start == pytest.approx(100.5) and words[0].end == pytest.approx(100.9)
    assert words[0].segment_index == 0 and words[3].segment_index == 1
    assert words[5].probability == pytest.approx(0.4)
    assert words[6].start == pytest.approx(104.0) and words[7].end == pytest.approx(105.0)
    assert tr.name == "faster_whisper:fake-small"


def test_fallback_when_segment_has_no_word_timestamps(fake_cache, caplog):
    """A segment without alignment must still yield words, spread evenly, with a WARNING."""
    logger.info(">>> expecting a WARNING: segment without word timestamps triggers the even-spread fallback")
    model = FakeWhisperModel([NO_ALIGNMENT_SEGMENT])
    with caplog.at_level(logging.WARNING, logger="dialogue_locator.transcription"):
        words = _transcriber(fake_cache(model)).transcribe_all(np.zeros(16_000 * 6, np.float32), offset=100.0)

    assert [w.text for w in words] == ["at", "stagnation"]
    assert words[0].start == pytest.approx(104.0) and words[0].end == pytest.approx(104.5)
    assert words[1].start == pytest.approx(104.5) and words[1].end == pytest.approx(105.0)
    assert all(w.probability is None for w in words)
    assert any("no word timestamps" in r.message for r in caplog.records if r.levelno == logging.WARNING)
    logger.info("<<< fallback WARNING observed as expected")


def test_decode_options_follow_config(fake_cache):
    model = FakeWhisperModel(SEGMENTS)
    cfg = WhisperConfig(language=None, beam_size=2, vad_filter=False, condition_on_previous_text=True)
    FasterWhisperTranscriber("m", cfg, model_cache=fake_cache(model)).transcribe_all(np.zeros(16_000, np.float32))
    opts = model.calls[0]
    assert opts["word_timestamps"] is True
    assert opts["beam_size"] == 2 and opts["vad_filter"] is False
    assert opts["condition_on_previous_text"] is True
    assert "language" not in opts and "vad_parameters" not in opts

    model2 = FakeWhisperModel(SEGMENTS)
    FasterWhisperTranscriber("m", WhisperConfig(vad_min_silence_ms=750), model_cache=fake_cache(model2)).transcribe_all(
        np.zeros(16_000, np.float32)
    )
    assert model2.calls[0]["language"] == "en"
    assert model2.calls[0]["vad_parameters"] == {"min_silence_duration_ms": 750}


def test_streaming_is_lazy_and_stops_early(fake_cache):
    model = FakeWhisperModel(SEGMENTS)
    tr = _transcriber(fake_cache(model))
    stream = tr.transcribe(np.zeros(16_000 * 6, np.float32))
    first = next(stream)
    assert first.text == "Hello"
    assert model.segments_pulled == 1  # only one segment decoded so far
    stream.close()  # consumer stops: no further segments are pulled
    assert model.segments_pulled == 1


def test_progress_events(fake_cache):
    model = FakeWhisperModel(SEGMENTS)
    events: list[ProgressEvent] = []
    _transcriber(fake_cache(model)).transcribe_all(np.zeros(16_000 * 10, np.float32), offset=0.0)
    list(_transcriber(fake_cache(model)).transcribe(np.zeros(16_000 * 10, np.float32), progress=events.append))
    assert events and all(e.stage is PipelineStage.TRANSCRIPTION for e in events)
    fractions = [e.fraction for e in events]
    assert fractions == sorted(fractions)
    assert fractions[-1] == pytest.approx(0.55)  # last segment ends at 5.5 of 10 s
    assert events[-1].details["words"] == 8


def test_decoder_error_is_wrapped(fake_cache):
    model = FakeWhisperModel(SEGMENTS, fail_at=1)
    tr = _transcriber(fake_cache(model))
    stream = tr.transcribe(np.zeros(16_000, np.float32))
    assert next(stream).text == "Hello"
    with expect_error(TranscriptionError, match="while decoding") as exc:
        list(stream)
    assert exc.value.stage == "transcription"


def test_transcribe_start_error_is_wrapped(fake_cache):
    class Broken:
        def transcribe(self, *a, **k):
            raise ValueError("bad options")

    tr = _transcriber(fake_cache(Broken()))
    with expect_error(TranscriptionError, match="start decoding"):
        list(tr.transcribe(np.zeros(16_000, np.float32)))


def test_accepts_path_input(fake_cache, tmp_path: Path):
    p = tmp_path / "in.wav"
    _write_wav(p, 16_000, 1, 2.0)
    model = FakeWhisperModel(SEGMENTS[:1])
    words = _transcriber(fake_cache(model)).transcribe_all(p)
    assert len(words) == 3
    assert len(model.calls[0]) > 0


def test_model_cache_loads_once(monkeypatch):
    import dialogue_locator.transcription.faster_whisper as fw

    loads: list[str] = []

    def fake_load(key):
        loads.append(key.name)
        return object()

    monkeypatch.setattr(fw.WhisperModelCache, "_load", staticmethod(fake_load))
    cache = WhisperModelCache()
    cfg = WhisperConfig()
    a = cache.get("small", cfg)
    b = cache.get("small", cfg)
    c = cache.get("medium", cfg)
    d = cache.get("small", WhisperConfig(compute_type="float32"))
    assert a is b and a is not c and a is not d
    assert loads == ["small", "medium", "small"]


def test_model_load_failure_is_wrapped(monkeypatch):
    import dialogue_locator.transcription.faster_whisper as fw

    class ExplodingModel:
        def __init__(self, *a, **k):
            raise OSError("no network")

    monkeypatch.setattr("faster_whisper.WhisperModel", ExplodingModel)
    with expect_error(TranscriptionError, match="Failed to load"):
        WhisperModelCache().get("tiny", WhisperConfig())
    assert fw  # module imported


def test_transcriber_is_abstract():
    with pytest.raises(TypeError):
        Transcriber()  # type: ignore[abstract]


# --------------------------------------------------------------------------- #
# Real model (opt-in)
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(os.environ.get("DL_RUN_MODEL_TESTS") != "1", reason="set DL_RUN_MODEL_TESTS=1")
@requires_ffmpeg
def test_real_tiny_model_on_speech(tmp_path: Path):
    """Synthesise speech with macOS `say` if available, else skip."""
    import shutil
    import subprocess

    if not shutil.which("say"):
        pytest.skip("macOS 'say' not available for speech synthesis")
    aiff = tmp_path / "speech.aiff"
    wav = tmp_path / "speech.wav"
    subprocess.run(["say", "-o", str(aiff), "my mind rebels at stagnation"], check=True)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(aiff), "-ac", "1", "-ar", "16000", str(wav)], check=True)

    tr = FasterWhisperTranscriber("tiny", WhisperConfig(vad_filter=False))
    words: list[Word] = tr.transcribe_all(wav)
    text = " ".join(w.text.lower().strip(".,") for w in words)
    assert "mind" in text and "stagnation" in text
    assert all(w.start <= w.end for w in words)


# --------------------------------------------------------------------------- #
# Thread sizing
# --------------------------------------------------------------------------- #
def test_zero_threads_resolves_to_performance_cores(monkeypatch):
    """cpu_threads=0 must size to the performance cores, not every core.

    On Apple Silicon the efficiency cores make decoding ~17x slower, so the
    resolved value has to be the P-core count and it has to reach the model.
    """
    import dialogue_locator.transcription.faster_whisper as fw

    monkeypatch.setattr(fw, "performance_cores", lambda: 4)
    seen: list[int] = []

    def fake_load(key):
        seen.append(key.cpu_threads)
        return object()

    monkeypatch.setattr(fw.WhisperModelCache, "_load", staticmethod(fake_load))
    fw.WhisperModelCache().get("tiny", WhisperConfig(cpu_threads=0))
    assert seen == [4]


def test_explicit_thread_count_is_respected(monkeypatch):
    import dialogue_locator.transcription.faster_whisper as fw

    monkeypatch.setattr(fw, "performance_cores", lambda: 4)
    seen: list[int] = []
    monkeypatch.setattr(
        fw.WhisperModelCache,
        "_load",
        staticmethod(lambda key: seen.append(key.cpu_threads) or object()),
    )
    fw.WhisperModelCache().get("tiny", WhisperConfig(cpu_threads=2))
    assert seen == [2]


def test_performance_cores_uses_sysctl_when_available(monkeypatch):
    import subprocess

    import dialogue_locator.transcription.faster_whisper as fw

    fw.performance_cores.cache_clear()
    monkeypatch.setattr(
        fw.subprocess,
        "run",
        lambda *a, **k: subprocess.CompletedProcess(a[0], 0, stdout="4\n", stderr=""),
    )
    try:
        assert fw.performance_cores() == 4
    finally:
        fw.performance_cores.cache_clear()


@pytest.mark.parametrize(
    "outcome",
    [OSError("no sysctl"), ValueError("bad int")],
    ids=["missing-sysctl", "unparseable"],
)
def test_performance_cores_falls_back_to_cpu_count(monkeypatch, outcome):
    """Off macOS there is no sysctl; fall back rather than crash the load."""
    import dialogue_locator.transcription.faster_whisper as fw

    fw.performance_cores.cache_clear()

    def boom(*a, **k):
        raise outcome

    monkeypatch.setattr(fw.subprocess, "run", boom)
    monkeypatch.setattr(fw.os, "cpu_count", lambda: 16)
    try:
        assert fw.performance_cores() == 16
    finally:
        fw.performance_cores.cache_clear()


def test_performance_cores_rejects_nonsense_core_count(monkeypatch):
    import subprocess

    import dialogue_locator.transcription.faster_whisper as fw

    fw.performance_cores.cache_clear()
    monkeypatch.setattr(
        fw.subprocess,
        "run",
        lambda *a, **k: subprocess.CompletedProcess(a[0], 0, stdout="0\n", stderr=""),
    )
    monkeypatch.setattr(fw.os, "cpu_count", lambda: 8)
    try:
        assert fw.performance_cores() == 8
    finally:
        fw.performance_cores.cache_clear()
