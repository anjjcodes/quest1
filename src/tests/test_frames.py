import logging
from pathlib import Path

import cv2
import numpy as np
import pytest

from dialogue_locator.config import FrameConfig
from dialogue_locator.exceptions import FrameExtractionError
from dialogue_locator.media.frames import FrameExtractor, frame_to_timestamp, timestamp_to_frame
from dialogue_locator.models import VideoInfo
from tests.conftest import expect_error, requires_ffmpeg

logger = logging.getLogger("tests")
pytestmark = requires_ffmpeg


@pytest.mark.parametrize(
    "t, fps, expected",
    [(0.0, 25, 0), (0.039, 25, 0), (0.04, 25, 1), (1.0, 25, 25), (2.999, 25, 74), (10.0, 29.97, 299), (-1.0, 25, 0)],
)
def test_timestamp_to_frame(t, fps, expected):
    assert timestamp_to_frame(t, fps) == expected


def test_frame_to_timestamp_roundtrip():
    for n in (0, 1, 74, 1000):
        assert timestamp_to_frame(frame_to_timestamp(n, 25), 25) == n


def test_invalid_fps():
    with expect_error(FrameExtractionError):
        timestamp_to_frame(1.0, 0)


def _video(path: Path, **overrides) -> VideoInfo:
    base = dict(path=path, source_url=str(path), fps=25.0, duration=3.0, frame_count=75, width=320, height=240)
    base.update(overrides)
    return VideoInfo(**base)


def test_extract_frame_jpg(sample_video: Path, tmp_path: Path):
    info = FrameExtractor(FrameConfig()).extract(_video(sample_video), 1.5, tmp_path / "frame.xyz")
    assert info.frame_number == 37  # floor(1.5 * 25)
    assert info.timestamp == pytest.approx(37 / 25)
    assert info.timestamp_str == "00:00:01.480"
    assert info.fps == 25.0
    assert info.image_path == tmp_path / "frame.jpg"
    assert (info.width, info.height) == (320, 240)
    img = cv2.imread(str(info.image_path))
    assert img is not None and img.shape == (240, 320, 3)


def test_extract_frame_png(sample_video: Path, tmp_path: Path):
    info = FrameExtractor(FrameConfig(image_format="png")).extract(_video(sample_video), 0.0, tmp_path / "f.jpg")
    assert info.image_path.suffix == ".png" and info.frame_number == 0
    assert cv2.imread(str(info.image_path)).shape == (240, 320, 3)


def test_different_timestamps_give_different_frames(sample_video: Path, tmp_path: Path):
    ex = FrameExtractor(FrameConfig(image_format="png"))
    a = ex.extract(_video(sample_video), 0.0, tmp_path / "a")
    b = ex.extract(_video(sample_video), 2.0, tmp_path / "b")
    ia, ib = cv2.imread(str(a.image_path)), cv2.imread(str(b.image_path))
    assert np.abs(ia.astype(int) - ib.astype(int)).mean() > 5  # testsrc pattern moves over time


def test_opencv_and_ffmpeg_paths_agree(sample_video: Path):
    """Seek accuracy: both decoders must return (nearly) the same pixels for the same frame."""
    ex = FrameExtractor(FrameConfig())
    v = _video(sample_video)
    n = 50
    via_cv = ex._read_with_opencv(sample_video, n)
    via_ff = ex._read_with_ffmpeg(sample_video, frame_to_timestamp(n, 25.0))
    assert via_cv is not None and via_cv.shape == via_ff.shape
    diff_same = np.abs(via_cv.astype(int) - via_ff.astype(int)).mean()
    diff_other = np.abs(via_cv.astype(int) - ex._read_with_ffmpeg(sample_video, frame_to_timestamp(n + 10, 25.0)).astype(int)).mean()
    logger.info("mean abs diff: same frame %.2f, frame+10 %.2f", diff_same, diff_other)
    assert diff_same < 3.0  # codec noise only
    assert diff_other > diff_same * 3  # clearly a different frame


def test_ffmpeg_fallback_used_when_opencv_fails(sample_video: Path, tmp_path: Path, monkeypatch, caplog):
    ex = FrameExtractor(FrameConfig())
    monkeypatch.setattr(FrameExtractor, "_read_with_opencv", staticmethod(lambda path, n: None))
    logger.info(">>> expecting a WARNING: OpenCV path disabled, ffmpeg fallback must kick in")
    with caplog.at_level(logging.WARNING, logger="dialogue_locator.media.frames"):
        info = ex.extract(_video(sample_video), 1.0, tmp_path / "f")
    assert info.frame_number == 25 and info.image_path.is_file()
    assert any("using ffmpeg" in r.message for r in caplog.records)
    logger.info("<<< fallback observed as expected")


def test_fps_resolved_from_video_when_missing(sample_video: Path, tmp_path: Path):
    info = FrameExtractor(FrameConfig()).extract(_video(sample_video, fps=None), 1.0, tmp_path / "f")
    assert info.fps == 25.0 and info.frame_number == 25


def test_beyond_end_clamps_to_last_frame(sample_video: Path, tmp_path: Path, caplog):
    logger.info(">>> expecting a WARNING: timestamp beyond the end is clamped to the last frame")
    with caplog.at_level(logging.WARNING, logger="dialogue_locator.media.frames"):
        info = FrameExtractor(FrameConfig()).extract(_video(sample_video), 99.0, tmp_path / "f")
    assert info.frame_number == 74
    assert any("clamping" in r.message for r in caplog.records)
    logger.info("<<< clamped as expected")


def test_negative_timestamp_is_frame_zero(sample_video: Path, tmp_path: Path):
    assert FrameExtractor(FrameConfig()).extract(_video(sample_video), -3.0, tmp_path / "f").frame_number == 0


def test_unreadable_video(tmp_path: Path, not_media: Path):
    with expect_error(FrameExtractionError) as exc:
        FrameExtractor(FrameConfig()).extract(_video(not_media), 0.0, tmp_path / "f")
    assert exc.value.stage == "frame"


def test_missing_video(tmp_path: Path):
    with expect_error(FrameExtractionError):
        FrameExtractor(FrameConfig()).extract(_video(tmp_path / "nope.mp4"), 0.0, tmp_path / "f")


def test_write_failure(sample_video: Path, tmp_path: Path):
    bad_dir = tmp_path / "file_not_dir"
    bad_dir.write_text("x")
    with expect_error(FrameExtractionError, match="write"):
        FrameExtractor(FrameConfig()).extract(_video(sample_video), 0.0, bad_dir / "sub" / "f")


def test_read_frame_returns_array(sample_video: Path):
    img = FrameExtractor(FrameConfig()).read_frame(_video(sample_video), 10)
    assert isinstance(img, np.ndarray) and img.shape == (240, 320, 3) and img.dtype == np.uint8
