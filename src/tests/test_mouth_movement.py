"""Tests for dialogue_locator.vision.mouth_movement.

Offline tests inject a fake landmarker (same convention as the fake MediaPipe
detector in test_face_detector.py): it yields scripted per-frame mouth-openness
values in MediaPipe's landmark shape, so the windowing, clip-offset mapping,
signal maths and verdict logic run for real over real video frames
(the ffmpeg-generated ``sample_video`` fixture).

``test_real_model_*`` downloads the actual Face Landmarker model and is marked
``network``.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from dialogue_locator.config import MouthMovementConfig
from dialogue_locator.exceptions import MouthMovementError
from dialogue_locator.models import MouthMovementResult, VideoInfo
from dialogue_locator.vision import MouthMovementAnalyzer
from tests.conftest import expect_error, requires_ffmpeg

pytestmark = requires_ffmpeg

NO_FACE = None  # marker in a scripted series


def landmarks(openness: float) -> list[SimpleNamespace]:
    """Landmarks whose inner-lip gap / mouth width equals ``openness``.

    Mouth corners (61, 291) are 0.2 apart; the inner lips (13, 14) are placed
    ``openness * 0.2`` apart vertically. Other indices are never touched.
    """
    pts = [SimpleNamespace(x=0.0, y=0.0)] * 478
    gap = openness * 0.2
    pts[61] = SimpleNamespace(x=0.4, y=0.5)
    pts[291] = SimpleNamespace(x=0.6, y=0.5)
    pts[13] = SimpleNamespace(x=0.5, y=0.5 - gap / 2)
    pts[14] = SimpleNamespace(x=0.5, y=0.5 + gap / 2)
    return pts


class FakeLandmarker:
    """Yields a scripted openness series, one value per detect call (cycled)."""

    def __init__(self, series: list[float | None]):
        self.series = series
        self.calls = 0
        self.timestamps: list[int] = []
        self.closed = False

    def detect_for_video(self, image, timestamp_ms: int) -> SimpleNamespace:
        value = self.series[self.calls % len(self.series)]
        self.calls += 1
        self.timestamps.append(timestamp_ms)
        faces = [] if value is NO_FACE else [landmarks(value)]
        return SimpleNamespace(face_landmarks=faces)

    def close(self) -> None:
        self.closed = True


def analyzer(series: list[float | None], **config) -> MouthMovementAnalyzer:
    """Analyzer whose _load() hands out fresh FakeLandmarkers (recorded on .fakes)."""
    cfg = MouthMovementConfig(model_path="unused/model.task", **config)
    a = MouthMovementAnalyzer(cfg)
    a.fakes = []

    def load() -> FakeLandmarker:  # bypass the real _load: no download, no mp task
        fake = FakeLandmarker(series)
        a.fakes.append(fake)
        return fake

    a._load = load
    return a


def video(path: Path, clip_start: float = 0.0) -> VideoInfo:
    return VideoInfo(path=path, source_url="t://v", fps=25.0, duration=3.0, clip_start=clip_start)


# --------------------------------------------------------------------------- #
# verdicts
# --------------------------------------------------------------------------- #
def test_talking_mouth_is_moving(sample_video):
    a = analyzer([0.0, 0.3])  # open/close every frame, like speech
    result = a.analyze(video(sample_video), 0.0, 1.0)
    assert result.moving is True
    assert result.movement_score == pytest.approx(0.15, abs=0.01)
    assert result.frames_analyzed == 26  # frames 0..25 at 25 fps
    assert result.frames_with_face == 26


def test_still_mouth_is_not_moving(sample_video):
    a = analyzer([0.05])  # constant openness: a face, but no lip activity
    result = a.analyze(video(sample_video), 0.0, 1.0)
    assert result.moving is False
    assert result.movement_score == pytest.approx(0.0, abs=1e-9)


def test_too_few_face_frames_is_indeterminate(sample_video):
    # Face visible in 3 of 26 frames; min_face_frames=5 -> no verdict.
    series = [0.0, 0.3, 0.1] + [NO_FACE] * 23
    a = analyzer(series, min_face_frames=5)
    result = a.analyze(video(sample_video), 0.0, 1.0)
    assert result.moving is None
    assert result.frames_with_face == 3
    assert result.movement_score is not None  # what little was seen is still reported


def test_no_face_at_all(sample_video):
    a = analyzer([NO_FACE])
    result = a.analyze(video(sample_video), 0.0, 1.0)
    assert result.moving is None and result.movement_score is None
    assert result.frames_with_face == 0 and result.frames_analyzed == 26


# --------------------------------------------------------------------------- #
# windowing
# --------------------------------------------------------------------------- #
def test_clip_offset_maps_absolute_window(sample_video):
    # Absolute 1.0..2.0 in a clip starting at 1.0 -> local frames 0..25.
    a = analyzer([0.0, 0.3])
    result = a.analyze(video(sample_video, clip_start=1.0), 1.0, 2.0)
    assert result.frames_analyzed == 26
    assert result.window_start == pytest.approx(1.0)
    assert result.window_end == pytest.approx(2.0)
    # VIDEO-mode timestamps are monotonically increasing
    ts = a.fakes[0].timestamps
    assert ts == sorted(ts) and len(set(ts)) == len(ts)


def test_window_capped_to_max_seconds(sample_video):
    a = analyzer([0.0, 0.3], max_window_seconds=1.0)
    result = a.analyze(video(sample_video), 0.0, 60.0)
    assert result.frames_analyzed == 26  # 1 s at 25 fps, not 60 s
    assert result.window_end - result.window_start == pytest.approx(1.0)


def test_window_past_end_of_clip_stops_at_last_frame(sample_video):
    # sample_video is 3 s; a 2..5 s window reads only what exists.
    a = analyzer([0.0, 0.3])
    result = a.analyze(video(sample_video), 2.0, 5.0)
    assert 0 < result.frames_analyzed <= 26
    assert result.window_end <= 3.0 + 1 / 25.0


def test_invalid_window_rejected(sample_video):
    with expect_error(MouthMovementError, match="Invalid analysis window"):
        analyzer([0.1]).analyze(video(sample_video), 2.0, 2.0)


def test_unreadable_video(tmp_path):
    with expect_error(MouthMovementError, match="Cannot open") as exc:
        analyzer([0.1]).analyze(video(tmp_path / "missing.mp4"), 0.0, 1.0)
    assert exc.value.stage == "mouth_movement"


# --------------------------------------------------------------------------- #
# model download / lifecycle
# --------------------------------------------------------------------------- #
def test_model_download_failure(tmp_path, sample_video):
    cfg = MouthMovementConfig(
        model_path=tmp_path / "model.task", model_url="file:///nonexistent/model.task"
    )
    a = MouthMovementAnalyzer(cfg)
    with expect_error(MouthMovementError, match="Cannot download"):
        a.analyze(video(sample_video), 0.0, 1.0)
    assert not cfg.model_path.exists()


def test_fresh_landmarker_per_analyze(sample_video):
    # Regression: MediaPipe's VIDEO mode requires monotonically increasing
    # timestamps per landmarker instance. A reused landmarker crashed the
    # second job of a server session ("Input timestamp must be monotonically
    # increasing") because its window started before the first job's ended.
    # Every analyze() must therefore get its own landmarker and close it.
    a = analyzer([0.1])
    a.analyze(video(sample_video), 2.0, 2.5)  # late window first...
    a.analyze(video(sample_video), 0.0, 0.5)  # ...then an earlier one
    assert len(a.fakes) == 2
    assert all(fake.closed for fake in a.fakes)
    assert a.fakes[1].timestamps[0] < a.fakes[0].timestamps[0]  # rewound safely


def test_unexpected_landmarker_error_is_wrapped(sample_video):
    # The stage's contract is to raise only MouthMovementError, so the
    # pipeline's fail-open guard catches MediaPipe runtime crashes too.
    a = analyzer([0.1])

    class ExplodingLandmarker(FakeLandmarker):
        def detect_for_video(self, image, timestamp_ms):
            raise RuntimeError("Input timestamp must be monotonically increasing.")

    a._load = lambda: ExplodingLandmarker([])
    with expect_error(MouthMovementError, match="monotonically"):
        a.analyze(video(sample_video), 0.0, 1.0)


# --------------------------------------------------------------------------- #
# result model
# --------------------------------------------------------------------------- #
def test_result_to_dict():
    result = MouthMovementResult(
        moving=True,
        movement_score=0.09876,
        threshold=0.02,
        frames_analyzed=26,
        frames_with_face=25,
        window_start=7.2601,
        window_end=8.26,
    )
    assert result.to_dict() == {
        "moving": True,
        "movement_score": 0.0988,
        "threshold": 0.02,
        "frames_analyzed": 26,
        "frames_with_face": 25,
        "window_start": 7.26,
        "window_end": 8.26,
    }


# --------------------------------------------------------------------------- #
# calibration
# --------------------------------------------------------------------------- #
def test_default_face_confidence_matches_detector_calibration():
    # The landmarker's internal face detector needs the same lowered threshold
    # as FaceDetectionConfig, or V3 goes indeterminate on the very mid/wide
    # shots V2 was calibrated to accept.
    assert MouthMovementConfig().min_face_confidence == 0.3


# --------------------------------------------------------------------------- #
# real model (network)
# --------------------------------------------------------------------------- #
@pytest.mark.network
def test_real_model_on_faceless_video(tmp_path, sample_video):
    # The ffmpeg test pattern has no face: the real landmarker must come back
    # indeterminate, never "not moving".
    cfg = MouthMovementConfig(model_path=tmp_path / "face_landmarker.task")
    a = MouthMovementAnalyzer(cfg)
    result = a.analyze(video(sample_video), 0.0, 1.0)
    assert result.frames_with_face == 0
    assert result.moving is None and result.movement_score is None
    assert cfg.model_path.is_file()  # model was downloaded and cached
