"""Tests for dialogue_locator.vision.face_detector.

Offline tests fake the MediaPipe task at the ``FaceDetector._load`` boundary
(same convention as the fake Whisper model in test_transcription.py): the fake
returns detections in MediaPipe's shape (``bounding_box`` + ``categories``),
so everything around the model - validation, box clamping/sorting, file
reading, result serialisation - is exercised for real.

``test_real_model_*`` download the actual BlazeFace model (and a real face
photo) and are marked ``network``.
"""

from __future__ import annotations

import urllib.request
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from dialogue_locator.config import FaceDetectionConfig
from dialogue_locator.exceptions import FaceDetectionError
from dialogue_locator.models import FaceBox, FaceDetectionResult
from dialogue_locator.vision import FaceDetector
from tests.conftest import expect_error


def config(tmp_path: Path, **overrides) -> FaceDetectionConfig:
    overrides.setdefault("model_path", tmp_path / "models" / "blaze.tflite")
    return FaceDetectionConfig(**overrides)


def mp_detection(x: int, y: int, w: int, h: int, score: float) -> SimpleNamespace:
    """A detection in the shape MediaPipe's Tasks API returns."""
    return SimpleNamespace(
        bounding_box=SimpleNamespace(origin_x=x, origin_y=y, width=w, height=h),
        categories=[SimpleNamespace(score=score)],
    )


class FakeMpDetector:
    def __init__(self, detections: list[SimpleNamespace]) -> None:
        self._detections = detections
        self.closed = False

    def detect(self, mp_image) -> SimpleNamespace:
        return SimpleNamespace(detections=self._detections)

    def close(self) -> None:
        self.closed = True


def fake_loaded(detector: FaceDetector, detections: list[SimpleNamespace]) -> FakeMpDetector:
    fake = FakeMpDetector(detections)
    detector._detector = fake  # bypass _load(): no model download, no mediapipe task
    return fake


def bgr_image(width: int = 640, height: int = 360) -> np.ndarray:
    return np.zeros((height, width, 3), dtype=np.uint8)


# --------------------------------------------------------------------------- #
# detect(): result mapping
# --------------------------------------------------------------------------- #
def test_face_present(tmp_path):
    det = FaceDetector(config(tmp_path))
    fake_loaded(det, [mp_detection(100, 50, 80, 80, 0.9)])
    result = det.detect(bgr_image())
    assert result.face_present is True
    assert result.faces == (FaceBox(x=100, y=50, width=80, height=80, confidence=0.9),)
    assert (result.image_width, result.image_height) == (640, 360)


def test_no_face(tmp_path):
    det = FaceDetector(config(tmp_path))
    fake_loaded(det, [])
    result = det.detect(bgr_image())
    assert result.face_present is False
    assert result.faces == ()


def test_faces_sorted_by_confidence(tmp_path):
    det = FaceDetector(config(tmp_path))
    fake_loaded(det, [mp_detection(0, 0, 10, 10, 0.6), mp_detection(20, 20, 10, 10, 0.95)])
    result = det.detect(bgr_image())
    assert [f.confidence for f in result.faces] == [0.95, 0.6]


def test_boxes_clamped_to_image(tmp_path):
    det = FaceDetector(config(tmp_path))
    # BlazeFace can report boxes that start before 0 or run past the edge.
    fake_loaded(det, [mp_detection(-10, -5, 50, 50, 0.8), mp_detection(600, 340, 100, 100, 0.7)])
    result = det.detect(bgr_image(width=640, height=360))
    assert result.faces[0] == FaceBox(x=0, y=0, width=40, height=45, confidence=0.8)
    assert result.faces[1] == FaceBox(x=600, y=340, width=40, height=20, confidence=0.7)


def test_box_fully_outside_image_is_dropped(tmp_path):
    det = FaceDetector(config(tmp_path))
    fake_loaded(det, [mp_detection(700, 400, 50, 50, 0.9)])
    assert det.detect(bgr_image(width=640, height=360)).face_present is False


# --------------------------------------------------------------------------- #
# detect(): input validation
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "bad_image",
    [
        np.zeros((360, 640), dtype=np.uint8),  # grayscale
        np.zeros((360, 640, 4), dtype=np.uint8),  # BGRA
        np.zeros((360, 640, 3), dtype=np.float32),  # wrong dtype
        np.zeros((0, 0, 3), dtype=np.uint8),  # empty
        "not an array",
    ],
    ids=["grayscale", "bgra", "float32", "empty", "not-array"],
)
def test_detect_rejects_bad_input(tmp_path, bad_image):
    det = FaceDetector(config(tmp_path))
    fake_loaded(det, [])
    with expect_error(FaceDetectionError):
        det.detect(bad_image)


# --------------------------------------------------------------------------- #
# detect_file()
# --------------------------------------------------------------------------- #
def test_detect_file(tmp_path):
    image_path = tmp_path / "frame.jpg"
    cv2.imwrite(str(image_path), bgr_image(width=320, height=240))
    det = FaceDetector(config(tmp_path))
    fake_loaded(det, [mp_detection(10, 10, 30, 30, 0.75)])
    result = det.detect_file(image_path)
    assert result.face_present is True
    assert (result.image_width, result.image_height) == (320, 240)


def test_detect_file_unreadable(tmp_path):
    missing = tmp_path / "nope.jpg"
    det = FaceDetector(config(tmp_path))
    with expect_error(FaceDetectionError, match="Cannot read image") as exc:
        det.detect_file(missing)
    assert exc.value.stage == "face_detection"


# --------------------------------------------------------------------------- #
# Model download
# --------------------------------------------------------------------------- #
def test_model_download_failure(tmp_path):
    cfg = config(tmp_path, model_url="file:///nonexistent/blaze.tflite")
    det = FaceDetector(cfg)
    with expect_error(FaceDetectionError, match="Cannot download"):
        det.detect(bgr_image())
    assert not cfg.model_path.exists()


def test_model_download_too_small_is_rejected(tmp_path, monkeypatch):
    served = tmp_path / "served.bin"
    served.write_bytes(b"<Error>NoSuchKey</Error>")
    cfg = config(tmp_path, model_url=served.as_uri())
    det = FaceDetector(cfg)
    with expect_error(FaceDetectionError, match="bytes"):
        det.detect(bgr_image())
    assert not cfg.model_path.exists()


def test_existing_model_is_not_redownloaded(tmp_path, monkeypatch):
    cfg = config(tmp_path)
    cfg.model_path.parent.mkdir(parents=True)
    cfg.model_path.write_bytes(b"cached")

    def explode(*args, **kwargs):  # any network touch fails the test
        raise AssertionError("urlopen called despite cached model")

    monkeypatch.setattr(urllib.request, "urlopen", explode)
    assert FaceDetector(cfg)._ensure_model() == cfg.model_path


# --------------------------------------------------------------------------- #
# Lifecycle
# --------------------------------------------------------------------------- #
def test_close_releases_detector(tmp_path):
    det = FaceDetector(config(tmp_path))
    fake = fake_loaded(det, [])
    with det:
        det.detect(bgr_image())
    assert fake.closed is True
    assert det._detector is None


# --------------------------------------------------------------------------- #
# Result model
# --------------------------------------------------------------------------- #
def test_result_to_dict():
    result = FaceDetectionResult(
        faces=(FaceBox(x=1, y=2, width=3, height=4, confidence=0.87654),),
        image_width=640,
        image_height=360,
    )
    assert result.to_dict() == {
        "face_present": True,
        "face_count": 1,
        "faces": [{"x": 1, "y": 2, "width": 3, "height": 4, "confidence": 0.877}],
        "image_width": 640,
        "image_height": 360,
    }


# --------------------------------------------------------------------------- #
# Calibration
# --------------------------------------------------------------------------- #
def test_default_confidence_calibrated_for_wide_shots():
    # Faces in cinematic mid/wide shots score ~0.34 with BlazeFace while junk
    # detections stay <=~0.2 (measured on real pipeline frames). MediaPipe's
    # usual 0.5 default misses those faces - keep the calibrated 0.3.
    assert FaceDetectionConfig().min_detection_confidence == 0.3


# --------------------------------------------------------------------------- #
# Real model (network)
# --------------------------------------------------------------------------- #
@pytest.mark.network
def test_real_model_detects_a_face(tmp_path):
    face_url = "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/lena.jpg"
    face_path = tmp_path / "face.jpg"
    face_path.write_bytes(urllib.request.urlopen(face_url, timeout=60).read())

    with FaceDetector(config(tmp_path)) as det:
        result = det.detect_file(face_path)
        assert result.face_present is True
        assert result.faces[0].confidence >= 0.5
        # And a blank frame through the same real model finds nothing.
        assert det.detect(bgr_image()).face_present is False
