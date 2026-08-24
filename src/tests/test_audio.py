import subprocess
import wave
from pathlib import Path

import pytest

from dialogue_locator.config import AudioConfig
from dialogue_locator.exceptions import AudioExtractionError, ConfigurationError
from dialogue_locator.media.audio import AudioExtractor
from dialogue_locator.models import PipelineStage, ProgressEvent, VideoInfo
from tests.conftest import requires_ffmpeg

pytestmark = requires_ffmpeg


def _video_info(path: Path, duration: float = 3.0) -> VideoInfo:
    return VideoInfo(path=path, source_url=str(path), duration=duration, fps=25.0)


def _wav_params(path: Path) -> tuple[int, int, int, float]:
    with wave.open(str(path)) as w:
        return w.getframerate(), w.getnchannels(), w.getsampwidth(), w.getnframes() / w.getframerate()


@pytest.fixture
def extractor() -> AudioExtractor:
    return AudioExtractor(AudioConfig())


def test_extract_produces_16k_mono_pcm(extractor: AudioExtractor, sample_video: Path, tmp_path: Path):
    events: list[ProgressEvent] = []
    info = extractor.extract(_video_info(sample_video), tmp_path, progress=events.append)

    assert info.path == tmp_path / "audio.wav"
    assert (info.sample_rate, info.channels) == (16_000, 1)
    assert info.duration == pytest.approx(3.0, abs=0.1)
    rate, ch, width, dur = _wav_params(info.path)
    assert (rate, ch, width) == (16_000, 1, 2)
    assert dur == pytest.approx(3.0, abs=0.1)
    assert events and all(e.stage is PipelineStage.AUDIO for e in events)
    assert events[0].fraction == 0.0 and events[-1].fraction == 1.0


def test_extract_respects_config(sample_video: Path, tmp_path: Path):
    info = AudioExtractor(AudioConfig(sample_rate=8000, channels=2)).extract(_video_info(sample_video), tmp_path)
    rate, ch, _, _ = _wav_params(info.path)
    assert (rate, ch) == (8000, 2)


def test_extract_reuses_existing(extractor: AudioExtractor, sample_video: Path, tmp_path: Path, monkeypatch):
    extractor.extract(_video_info(sample_video), tmp_path)

    def boom(*a, **k):
        raise AssertionError("ffmpeg must not run again")

    monkeypatch.setattr(AudioExtractor, "_run_ffmpeg", boom)
    info = extractor.extract(_video_info(sample_video), tmp_path)
    assert info.duration == pytest.approx(3.0, abs=0.1)

    # reuse_existing=False must go back to ffmpeg
    with pytest.raises(AssertionError, match="must not run again"):
        extractor.extract(_video_info(sample_video), tmp_path, reuse_existing=False)


def test_extract_no_audio_stream(extractor: AudioExtractor, silent_video: Path, tmp_path: Path):
    with pytest.raises(AudioExtractionError) as exc:
        extractor.extract(_video_info(silent_video, 2.0), tmp_path)
    assert "no audio stream" in exc.value.message
    assert exc.value.stage == "audio"


def test_extract_missing_ffmpeg(sample_video: Path, tmp_path: Path):
    with pytest.raises(ConfigurationError):
        AudioExtractor(AudioConfig(ffmpeg_binary="ffmpeg-does-not-exist")).extract(
            _video_info(sample_video), tmp_path
        )


def test_extract_clip_from_wav(extractor: AudioExtractor, sample_wav: Path, tmp_path: Path):
    clip = extractor.extract_clip(sample_wav, 1.0, 2.5, tmp_path / "clip.wav")
    assert clip.duration == pytest.approx(1.5, abs=0.02)
    _, _, _, dur = _wav_params(clip.path)
    assert dur == pytest.approx(1.5, abs=0.02)


def test_extract_clip_from_video(extractor: AudioExtractor, sample_video: Path, tmp_path: Path):
    clip = extractor.extract_clip(sample_video, 0.5, 2.0, tmp_path / "nested" / "clip.wav")
    assert clip.path.is_file()
    assert clip.duration == pytest.approx(1.5, abs=0.05)


def test_extract_clip_clamps_negative_start(extractor: AudioExtractor, sample_wav: Path, tmp_path: Path):
    clip = extractor.extract_clip(sample_wav, -5.0, 1.0, tmp_path / "clip.wav")
    assert clip.duration == pytest.approx(1.0, abs=0.02)


def test_extract_clip_invalid_range(extractor: AudioExtractor, sample_wav: Path, tmp_path: Path):
    with pytest.raises(AudioExtractionError):
        extractor.extract_clip(sample_wav, 3.0, 1.0, tmp_path / "clip.wav")


def test_extract_clip_beyond_end(extractor: AudioExtractor, sample_wav: Path, tmp_path: Path):
    with pytest.raises(AudioExtractionError) as exc:
        extractor.extract_clip(sample_wav, 100.0, 110.0, tmp_path / "clip.wav")
    assert "no audio data" in exc.value.message


def test_extract_timeout(sample_video: Path, tmp_path: Path, monkeypatch):
    """Simulate ffmpeg hanging: communicate() raises TimeoutExpired."""
    import dialogue_locator.media.audio as audio_module
    from dialogue_locator.media.probe import MediaProbe

    class HangingProc:
        returncode = None
        stdout = iter(())

        def communicate(self, timeout=None):
            if timeout is not None:  # first (timed) call hangs; the post-kill drain returns
                raise subprocess.TimeoutExpired(cmd="ffmpeg", timeout=timeout)
            return "", ""

        def kill(self):
            pass

    # Skip the real ffprobe pre-check so the only subprocess is our fake ffmpeg.
    monkeypatch.setattr(
        audio_module, "probe_media", lambda *a, **k: MediaProbe(duration=3.0, has_video=True, has_audio=True)
    )
    monkeypatch.setattr(audio_module.subprocess, "Popen", lambda *a, **k: HangingProc())
    with pytest.raises(AudioExtractionError) as exc:
        AudioExtractor(AudioConfig(timeout_seconds=1)).extract(_video_info(sample_video), tmp_path)
    assert "timed out" in exc.value.message
