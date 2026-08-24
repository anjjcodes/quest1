from pathlib import Path

import pytest

from dialogue_locator.exceptions import ConfigurationError, UnsupportedVideoError
from dialogue_locator.media.probe import _parse_rate, ensure_binary, probe_media
from tests.conftest import expect_error, requires_ffmpeg


@pytest.mark.parametrize(
    "value, expected",
    [("25/1", 25.0), ("30000/1001", pytest.approx(29.97, abs=0.01)), ("0/0", None), (None, None), ("abc", None)],
)
def test_parse_rate(value, expected):
    assert _parse_rate(value) == expected


def test_ensure_binary_missing():
    with expect_error(ConfigurationError) as exc:
        ensure_binary("definitely-not-a-binary-xyz")
    assert exc.value.stage == "config"


@requires_ffmpeg
def test_probe_video_with_audio(sample_video: Path):
    p = probe_media(sample_video)
    assert p.has_video and p.has_audio
    assert p.fps == 25.0
    assert (p.width, p.height) == (320, 240)
    assert p.duration == pytest.approx(3.0, abs=0.2)
    assert p.frame_count == 75
    assert p.video_codec == "h264" and p.audio_codec == "aac"


@requires_ffmpeg
def test_probe_silent_video(silent_video: Path):
    p = probe_media(silent_video)
    assert p.has_video and not p.has_audio


@requires_ffmpeg
def test_probe_audio_only(sample_wav: Path):
    p = probe_media(sample_wav)
    assert p.has_audio and not p.has_video
    assert p.fps is None and p.frame_count is None
    assert p.duration == pytest.approx(4.0, abs=0.05)


@requires_ffmpeg
def test_probe_missing_file(tmp_path: Path):
    with expect_error(UnsupportedVideoError):
        probe_media(tmp_path / "nope.mp4")


@requires_ffmpeg
def test_probe_non_media(not_media: Path):
    with expect_error(UnsupportedVideoError):
        probe_media(not_media)
