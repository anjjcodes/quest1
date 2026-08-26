"""Tests for dialogue_locator.vision.mouth_movement.

Offline tests inject a fake landmarker and a fake face detector (same
convention as the fake MediaPipe detector in test_face_detector.py): the
landmarker yields scripted per-frame mouth-openness values in MediaPipe's
landmark shape, so the windowing, clip-offset mapping, cropping, scoring and
verdict logic run for real over real video frames (the ffmpeg-generated
``sample_video`` fixture).

``test_real_model_*`` downloads the actual Face Landmarker model and is marked
``network``.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from dialogue_locator.config import FaceDetectionConfig, MouthMovementConfig
from dialogue_locator.exceptions import FaceDetectionError, MouthMovementError
from dialogue_locator.models import FaceBox, FaceDetectionResult, MouthMovementResult, VideoInfo
from dialogue_locator.vision import FaceDetector, MouthMovementAnalyzer
from tests.conftest import expect_error, requires_ffmpeg

pytestmark = requires_ffmpeg

NO_FACE = None  # marker in a scripted series

# sample_video is 320x240 @ 25 fps; a box this size lands well inside it.
BOX = FaceBox(x=100, y=60, width=80, height=80, confidence=0.9)


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
        self.shapes: list[tuple[int, int]] = []  # (height, width) of each input
        self.closed = False

    def detect_for_video(self, image, timestamp_ms: int) -> SimpleNamespace:
        value = self.series[self.calls % len(self.series)]
        self.calls += 1
        self.timestamps.append(timestamp_ms)
        array = image.numpy_view()
        self.shapes.append((array.shape[0], array.shape[1]))
        faces = [] if value is NO_FACE else [landmarks(value)]
        return SimpleNamespace(face_landmarks=faces)

    def close(self) -> None:
        self.closed = True


class FakeFaceDetector:
    """Stands in for the V2 BlazeFace detector: scripted boxes, no model.

    ``boxes`` is one box per call (cycled), so a test can move the box the way
    a camera cut would; the single-box form covers the common case.
    """

    def __init__(
        self,
        box: FaceBox | None = BOX,
        error: Exception | None = None,
        boxes: list[FaceBox | None] | None = None,
    ):
        self.boxes = boxes if boxes is not None else [box]
        self.error = error
        self.calls = 0
        self.log_flags: list[bool] = []

    def detect(self, image: np.ndarray, log_result: bool = True) -> FaceDetectionResult:
        box = self.boxes[self.calls % len(self.boxes)]
        self.calls += 1
        self.log_flags.append(log_result)
        if self.error is not None:
            raise self.error
        height, width = image.shape[:2]
        faces = () if box is None else (box,)
        return FaceDetectionResult(faces=faces, image_width=width, image_height=height)


def analyzer(
    series: list[float | None],
    detector: FakeFaceDetector | None = None,
    **config,
) -> MouthMovementAnalyzer:
    """Analyzer whose _load() hands out fresh FakeLandmarkers (recorded on .fakes)."""
    cfg = MouthMovementConfig(model_path="unused/model.task", **config)
    a = MouthMovementAnalyzer(cfg, face_detector=detector or FakeFaceDetector())
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
    assert result.movement_start is None  # nothing moved, nothing to point at


def test_too_few_face_frames_is_indeterminate(sample_video):
    # Face visible in 3 of 26 frames; min_face_frames=5 -> no verdict.
    series = [0.0, 0.3, 0.1] + [NO_FACE] * 23
    a = analyzer(series, min_face_frames=5)
    result = a.analyze(video(sample_video), 0.0, 1.0)
    assert result.moving is None
    assert result.frames_with_face == 3
    assert result.movement_score is not None  # what little was seen is still reported
    assert result.movement_start is None


def test_no_face_at_all(sample_video):
    a = analyzer([NO_FACE])
    result = a.analyze(video(sample_video), 0.0, 1.0)
    assert result.moving is None and result.movement_score is None
    assert result.frames_with_face == 0 and result.frames_analyzed == 26


# --------------------------------------------------------------------------- #
# scoring: sliding window, not one average over the whole line
# --------------------------------------------------------------------------- #
# 73 frames (a 0..2.9 s window at 25 fps): a listener holding a resting mouth,
# then the camera cuts to the speaker for the last 12 frames. The levels are
# the ones measured on the Spider-Man trailer clip this scoring was built for
# (~0.017 over the whole line, ~0.04 over the speaking half-second), so the
# verdict flips on exactly the margin the real case turns on.
LISTENER = [0.12] * 61
SPEAKER = [0.09, 0.17] * 6
LATE_SPEECH = LISTENER + SPEAKER


def test_speech_at_the_end_of_the_line_still_counts(sample_video):
    # The case this scoring exists for: the camera is on a listener for most of
    # the line and cuts to the speaker for its last half-second. Averaged over
    # the whole window the burst disappears; the best 0.5 s window finds it.
    result = analyzer(LATE_SPEECH).analyze(video(sample_video), 0.0, 2.9)
    assert result.frames_with_face == 73  # series covers the window, no cycling
    assert result.moving is True
    assert result.movement_score == pytest.approx(0.04, abs=0.005)
    assert result.movement_start >= len(LISTENER) / 25.0 - 1e-9


def test_score_is_the_best_window_not_the_whole_window(sample_video):
    # Same series, judged with a window long enough to swallow the burst: the
    # score drops under the threshold and the verdict flips. This is the old
    # behaviour, and the regression guard on score_window_seconds.
    diluted = analyzer(LATE_SPEECH, score_window_seconds=10.0).analyze(
        video(sample_video), 0.0, 2.9
    )
    focused = analyzer(LATE_SPEECH).analyze(video(sample_video), 0.0, 2.9)
    assert diluted.movement_score < MouthMovementConfig().movement_threshold
    assert diluted.moving is False and focused.moving is True


def test_cut_between_two_still_faces_is_not_movement(sample_video):
    # Two shots, each a still mouth, at different openness, with the face
    # detected right across the cut. The step is not a mouth opening: a window
    # must not span it, or every shot/reverse-shot reads as speech.
    elsewhere = FaceBox(x=20, y=140, width=70, height=70, confidence=0.9)
    detector = FakeFaceDetector(boxes=[BOX] * 13 + [elsewhere] * 13)
    a = analyzer([0.05] * 13 + [0.25] * 13, detector=detector)
    result = a.analyze(video(sample_video), 0.0, 1.0)
    assert result.moving is False
    assert result.movement_score == pytest.approx(0.0, abs=1e-9)


def test_a_drifting_mouth_is_not_a_speaking_mouth(sample_video):
    # Regression: cropping to the face raised the resolution of the signal, and
    # with it the slide in openness produced by a listener simply turning their
    # head - enough raw spread (~0.04 measured) to clear the threshold without
    # a word being said. Subtracting the window's trend leaves nothing.
    drift = [0.25 - 0.01 * i for i in range(13)] + [0.13] * 13
    result = analyzer(drift).analyze(video(sample_video), 0.0, 1.0)
    # Raw spread over the sliding part of that window is ~0.035, comfortably
    # over the threshold; the residual left by the fit is a tenth of it (all of
    # it from the kink where the drift stops, not from any mouth activity).
    assert result.movement_score < 0.01
    assert result.moving is False


def test_oscillation_survives_detrending(sample_video):
    # The other half of the same guard: a mouth that opens and closes around a
    # drifting baseline must keep its score. Same drift as above, plus speech.
    speech = [0.25 - 0.01 * i + (0.04 if i % 2 else -0.04) for i in range(26)]
    result = analyzer(speech).analyze(video(sample_video), 0.0, 1.0)
    assert result.movement_score == pytest.approx(0.04, abs=0.005)
    assert result.moving is True


def test_a_face_moving_within_a_shot_is_not_a_cut(sample_video):
    # The cut check must not fire on ordinary motion, or long takes get chopped
    # into runs too short to score. A box drifting a few pixels a frame stays
    # one shot.
    drifting = [FaceBox(x=100 + i, y=60, width=80, height=80, confidence=0.9) for i in range(26)]
    a = analyzer([0.09, 0.17] * 13, detector=FakeFaceDetector(boxes=drifting))
    result = a.analyze(video(sample_video), 0.0, 1.0)
    assert result.moving is True
    assert result.movement_score == pytest.approx(0.04, abs=0.005)


def test_windows_do_not_span_a_gap_in_the_face(sample_video):
    # Losing the face is the other way a cut shows up. Openness either side of
    # the gap must not be compared: each run here is still, so the verdict is
    # "not moving" even though the two levels are far apart.
    a = analyzer([0.05] * 9 + [NO_FACE] * 4 + [0.30] * 13, min_face_frames=5)
    result = a.analyze(video(sample_video), 0.0, 1.0)
    assert result.frames_with_face == 22
    assert result.moving is False
    assert result.movement_score == pytest.approx(0.0, abs=1e-9)


def test_short_runs_are_never_scored(sample_video):
    # A two-frame run can have any spread by chance; it is not evidence of
    # speech, so it must not produce a verdict on its own.
    a = analyzer([0.0, 0.6] + [NO_FACE] * 6, min_face_frames=5)
    result = a.analyze(video(sample_video), 0.0, 1.0)
    assert result.moving is None
    assert result.movement_start is None


def test_movement_start_points_at_the_active_window(sample_video):
    # Silence, then speech from frame 12 (0.48 s) in a clip starting at 10 s.
    a = analyzer([0.12] * 12 + [0.09, 0.17] * 7)
    result = a.analyze(video(sample_video, clip_start=10.0), 10.0, 11.0)
    assert result.moving is True
    assert result.movement_start == pytest.approx(10.0 + 12 / 25.0, abs=1 / 25.0)
    assert result.window_start <= result.movement_start <= result.window_end


# --------------------------------------------------------------------------- #
# face crop: what the landmarker is actually fed
# --------------------------------------------------------------------------- #
def test_landmarker_sees_an_upscaled_crop_not_the_frame(sample_video):
    # Regression: the landmarker's own detector misses faces that are small in
    # the frame (a 290 px face in 1920x1080 wide shot), so a speaker plainly on
    # camera contributed no frames. Every frame must reach it as a padded crop,
    # upscaled to min_crop_size.
    detector = FakeFaceDetector()
    a = analyzer([0.1], detector=detector, min_crop_size=256)
    a.analyze(video(sample_video), 0.0, 1.0)
    assert detector.calls == 26  # every frame, not just the first
    # 80 px box + 0.6 padding each side = 176 px, upscaled to 256.
    assert set(a.fakes[0].shapes) == {(256, 256)}
    assert not any(detector.log_flags)  # per-frame detections stay out of the log


def test_crop_is_not_upscaled_when_already_large_enough(sample_video):
    a = analyzer([0.1], min_crop_size=64)
    a.analyze(video(sample_video), 0.0, 1.0)
    assert set(a.fakes[0].shapes) == {(176, 176)}  # padded box, left alone


def test_last_box_is_carried_over_frames_detection_misses(sample_video):
    # Regression: BlazeFace flickers on faces that are small in the frame - a
    # 290px face in 1920x1080 was found in 4 of 26 frames - leaving no run long
    # enough to score. The last box is reused so the landmarker can judge those
    # frames, which is where the reliable signal is.
    detector = FakeFaceDetector(boxes=[BOX, None, None, None])
    a = analyzer([0.09, 0.17], detector=detector, box_carry_seconds=0.4)
    result = a.analyze(video(sample_video), 0.0, 1.0)
    assert set(a.fakes[0].shapes) == {(256, 256)}  # every frame cropped, none whole
    assert result.frames_with_face == 26
    assert result.moving is True


def test_the_carry_expires(sample_video):
    # Bounded, so a cut cannot go on feeding a stale box to the landmarker and
    # attributing a new shot's face to the previous run. 0.4 s at 25 fps = 10
    # frames, then the frame is landmarked whole again.
    detector = FakeFaceDetector(boxes=[BOX] + [None] * 25)
    a = analyzer([0.1], detector=detector, box_carry_seconds=0.4)
    a.analyze(video(sample_video), 0.0, 1.0)
    shapes = a.fakes[0].shapes
    assert shapes[:11] == [(256, 256)] * 11  # the detection plus 10 carried frames
    assert set(shapes[11:]) == {(240, 320)}  # carry expired: whole frames again


def test_carry_disabled_leaves_gaps_alone(sample_video):
    detector = FakeFaceDetector(boxes=[BOX, None])
    a = analyzer([0.1], detector=detector, box_carry_seconds=0.0)
    a.analyze(video(sample_video), 0.0, 1.0)
    assert set(a.fakes[0].shapes) == {(256, 256), (240, 320)}  # alternating, no carry


def test_a_cut_is_caught_when_detection_resumes_after_a_carry(sample_video):
    # The carried box is identical to the one before it, so no shot change can
    # be seen while carrying; the check must still fire on the frame a real
    # detection comes back somewhere else.
    elsewhere = FaceBox(x=20, y=140, width=70, height=70, confidence=0.9)
    detector = FakeFaceDetector(boxes=[BOX] * 8 + [None] * 2 + [elsewhere] * 16)
    a = analyzer([0.05] * 10 + [0.25] * 16, detector=detector, min_face_frames=5)
    result = a.analyze(video(sample_video), 0.0, 1.0)
    assert result.moving is False  # two still shots, not one moving mouth
    assert result.movement_score == pytest.approx(0.0, abs=1e-9)


def test_falls_back_to_the_whole_frame_without_a_box(sample_video):
    # No face box for a frame -> landmark it whole (the pre-crop behaviour),
    # never skip it: the landmarker may still find something.
    a = analyzer([0.1], detector=FakeFaceDetector(box=None))
    result = a.analyze(video(sample_video), 0.0, 1.0)
    assert set(a.fakes[0].shapes) == {(240, 320)}  # the sample_video frame size
    assert result.frames_with_face == 26


def test_failing_detector_falls_back_and_warns_once(sample_video, caplog):
    # A broken face detector must degrade the stage, not fail it - and must not
    # log once per frame while doing so.
    detector = FakeFaceDetector(error=FaceDetectionError("model gone"))
    a = analyzer([0.1], detector=detector)
    with caplog.at_level("WARNING"):
        result = a.analyze(video(sample_video), 0.0, 1.0)
    assert result.moving is False  # still judged, on whole frames
    assert set(a.fakes[0].shapes) == {(240, 320)}
    assert sum("landmarking whole frames" in r.message for r in caplog.records) == 1


def test_crop_warning_resets_between_calls(sample_video, caplog):
    detector = FakeFaceDetector(error=FaceDetectionError("model gone"))
    a = analyzer([0.1], detector=detector)
    with caplog.at_level("WARNING"):
        a.analyze(video(sample_video), 0.0, 0.5)
        a.analyze(video(sample_video), 1.0, 1.5)
    assert sum("landmarking whole frames" in r.message for r in caplog.records) == 2


def test_default_detector_is_created_lazily():
    # Constructing the analyzer must not build (or download) a face detector;
    # the pipeline injects its own, and importing stays cheap.
    a = MouthMovementAnalyzer(MouthMovementConfig(model_path="unused/model.task"))
    assert a._face_detector is None
    assert isinstance(a._detector(), FaceDetector)


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
    a = MouthMovementAnalyzer(cfg, face_detector=FakeFaceDetector())
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
        movement_start=7.9004,
        face_start=7.3,
    )
    assert result.to_dict() == {
        "moving": True,
        "movement_score": 0.0988,
        "threshold": 0.02,
        "frames_analyzed": 26,
        "frames_with_face": 25,
        "window_start": 7.26,
        "window_end": 8.26,
        "movement_start": 7.9,
        "face_start": 7.3,
    }


# --------------------------------------------------------------------------- #
# calibration
# --------------------------------------------------------------------------- #
def test_landmarker_is_stricter_than_the_detector_feeding_it():
    # V2 runs on whole frames and must accept faint mid/wide-shot faces, so its
    # threshold is lowered. The landmarker sees a tight crop and is the second
    # opinion on whether that crop is a face at all, so it is stricter:
    # BlazeFace fires on blurred rubble at 0.45-0.62, above what it scores some
    # real faces, and cannot police itself. Strict enough to reject rubble,
    # loose enough to hold a real speaker in a soft, upscaled TV master.
    assert FaceDetectionConfig().min_detection_confidence == 0.3
    assert MouthMovementConfig().min_face_confidence == 0.4


def test_default_threshold_sits_between_the_measured_clusters():
    # Measured over the cached corpus of real matches, with the face crop and
    # box carry in place: faces present but silent (still faces, a listener
    # turning their head, a voice-over read with the mouth shut) score
    # 0.000-0.021; faces speaking on camera score 0.046-0.117. Moving the
    # default outside that gap silently changes every verdict, so it is pinned.
    threshold = MouthMovementConfig().movement_threshold
    assert 0.021 < threshold < 0.046


# --------------------------------------------------------------------------- #
# real model (network)
# --------------------------------------------------------------------------- #
@pytest.mark.network
def test_real_model_on_faceless_video(tmp_path, sample_video):
    # The ffmpeg test pattern has no face: the real landmarker must come back
    # indeterminate, never "not moving".
    cfg = MouthMovementConfig(model_path=tmp_path / "face_landmarker.task")
    face_cfg = FaceDetectionConfig(model_path=tmp_path / "blaze_face.tflite")
    a = MouthMovementAnalyzer(cfg, face_detector=FaceDetector(face_cfg))
    result = a.analyze(video(sample_video), 0.0, 1.0)
    assert result.frames_with_face == 0
    assert result.moving is None and result.movement_score is None
    assert cfg.model_path.is_file()  # model was downloaded and cached
