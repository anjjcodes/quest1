"""Face detection: is at least one human face visible in a frame?

V2 works on the frame already localised by V1's dialogue match (see
``FrameExtractor.read_frame``), so this module deliberately takes a single
image, not a video: callers hand it a BGR array (or an image file) and get
back a :class:`~dialogue_locator.models.FaceDetectionResult` with every face
found and its bounding box / confidence.

Backend
-------
MediaPipe's BlazeFace short-range detector via the Tasks API
(``mediapipe.tasks.python.vision.FaceDetector``). The ``.tflite`` model file
(~230 KB) is not bundled with the pip package; like the Whisper weights it is
downloaded once on first use and cached at ``FaceDetectionConfig.model_path``.

``mediapipe`` is imported lazily inside :meth:`FaceDetector._load` so that
importing this module (e.g. from the CLI) stays cheap, and so tests can fake
the backend without the real library in the loop.

Note: mediapipe >= 1.0 aborts the process on macOS when creating a
``FaceDetector`` (Metal helper init fails even with the CPU delegate), hence
the ``<1.0`` pin in requirements.txt.

A ``FaceDetector`` instance is not thread-safe (the underlying MediaPipe task
is stateful); create one per worker, or reuse a single instance from the one
pipeline job the server runs at a time.
"""

from __future__ import annotations

import logging
import urllib.error
import urllib.request
from pathlib import Path
from types import TracebackType
from typing import Any

import cv2
import numpy as np

from dialogue_locator.config import FaceDetectionConfig
from dialogue_locator.exceptions import FaceDetectionError
from dialogue_locator.models import FaceBox, FaceDetectionResult

logger = logging.getLogger(__name__)


class FaceDetector:
    """Detect human faces in a single image (BGR array or file)."""

    def __init__(self, config: FaceDetectionConfig) -> None:
        self._config = config
        self._detector: Any = None  # created lazily on first detect()

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def detect(self, image: np.ndarray) -> FaceDetectionResult:
        """Return the faces visible in ``image`` (BGR, as produced by OpenCV)."""
        self._validate_image(image)
        detector = self._ensure_loaded()

        import mediapipe as mp

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        try:
            mp_result = detector.detect(mp_image)
        except Exception as exc:
            logger.error("MediaPipe face detection failed: %s", exc)
            raise FaceDetectionError(f"Face detection failed: {exc}") from exc

        height, width = image.shape[:2]
        result = self._to_result(mp_result, width, height)
        logger.info(
            "Face detection: %d face(s) in %dx%d frame%s",
            len(result.faces),
            width,
            height,
            f", best confidence {result.faces[0].confidence:.3f}" if result.faces else "",
        )
        return result

    def detect_file(self, path: Path) -> FaceDetectionResult:
        """Convenience wrapper: run :meth:`detect` on an image file (e.g. the V1 output frame)."""
        image = cv2.imread(str(path))
        if image is None:
            logger.error("Cannot read image file %s", path)
            raise FaceDetectionError(
                f"Cannot read image: {Path(path).name}", details={"path": str(path)}
            )
        return self.detect(image)

    def close(self) -> None:
        """Release the underlying MediaPipe task. The instance can be re-used; a
        later detect() reloads it."""
        if self._detector is not None:
            self._detector.close()
            self._detector = None

    def __enter__(self) -> FaceDetector:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    @staticmethod
    def _validate_image(image: np.ndarray) -> None:
        if not isinstance(image, np.ndarray) or image.ndim != 3 or image.shape[2] != 3:
            raise FaceDetectionError(
                "Face detection needs a HxWx3 BGR image array, got "
                f"{type(image).__name__} with shape {getattr(image, 'shape', None)}"
            )
        if image.dtype != np.uint8:
            raise FaceDetectionError(f"Face detection needs a uint8 image, got dtype {image.dtype}")
        if image.shape[0] == 0 or image.shape[1] == 0:
            raise FaceDetectionError("Face detection got an empty image")

    def _ensure_loaded(self) -> Any:
        if self._detector is None:
            self._detector = self._load()
        return self._detector

    def _load(self) -> Any:
        """Create the MediaPipe task (downloading the model file if needed)."""
        model_path = self._ensure_model()
        try:
            from mediapipe.tasks.python import BaseOptions
            from mediapipe.tasks.python import vision as mp_vision
        except ImportError as exc:
            raise FaceDetectionError(
                "mediapipe is not installed; install requirements.txt into the venv"
            ) from exc

        logger.info(
            "Loading BlazeFace detector (model=%s, min_confidence=%.2f)",
            model_path,
            self._config.min_detection_confidence,
        )
        options = mp_vision.FaceDetectorOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            min_detection_confidence=self._config.min_detection_confidence,
        )
        try:
            return mp_vision.FaceDetector.create_from_options(options)
        except Exception as exc:
            logger.error("Cannot create MediaPipe face detector: %s", exc)
            raise FaceDetectionError(
                f"Cannot load face detection model: {exc}", details={"model_path": str(model_path)}
            ) from exc

    def _ensure_model(self) -> Path:
        """Return the model file path, downloading it first if it is missing."""
        model_path = self._config.model_path
        if model_path.is_file():
            return model_path

        url = self._config.model_url
        logger.info("Face model missing; downloading %s -> %s", url, model_path)
        try:
            model_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = model_path.with_suffix(model_path.suffix + ".part")
            with urllib.request.urlopen(url, timeout=self._config.download_timeout_seconds) as resp:
                data = resp.read()
            if len(data) < 1024:  # a storage error page, not a model
                raise FaceDetectionError(
                    f"Face model download returned only {len(data)} bytes", details={"url": url}
                )
            tmp_path.write_bytes(data)
            tmp_path.replace(model_path)  # atomic: never leave a half-written model
        except FaceDetectionError:
            raise
        except (urllib.error.URLError, OSError, ValueError) as exc:
            logger.error("Face model download failed: %s", exc)
            raise FaceDetectionError(
                f"Cannot download face detection model: {exc}",
                details={"url": url, "model_path": str(model_path)},
            ) from exc
        logger.info("Face model saved: %s (%d bytes)", model_path, len(data))
        return model_path

    @staticmethod
    def _to_result(mp_result: Any, width: int, height: int) -> FaceDetectionResult:
        """Convert a MediaPipe result to our model, clamping boxes to the image."""
        faces = []
        for detection in mp_result.detections:
            box = detection.bounding_box
            x = max(0, int(box.origin_x))
            y = max(0, int(box.origin_y))
            w = min(int(box.origin_x) + int(box.width), width) - x
            h = min(int(box.origin_y) + int(box.height), height) - y
            if w <= 0 or h <= 0:
                continue  # box entirely outside the frame
            score = float(detection.categories[0].score) if detection.categories else 0.0
            faces.append(FaceBox(x=x, y=y, width=w, height=h, confidence=score))
        faces.sort(key=lambda f: f.confidence, reverse=True)
        return FaceDetectionResult(faces=tuple(faces), image_width=width, image_height=height)
