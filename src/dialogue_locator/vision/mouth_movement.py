"""Mouth-movement detection: are the lips moving while the dialogue is spoken?

V3 builds on V2's face check. Given the video and the matched dialogue window,
this module tracks facial landmarks across the window's frames with MediaPipe's
Face Landmarker (VIDEO running mode, which smooths landmarks temporally) and
reduces the lips to one number per frame:

    openness = inner-lip gap (landmarks 13-14) / mouth width (landmarks 61-291)

Dividing by the mouth width makes the signal invariant to face size, camera
distance and resolution. The *movement score* is the standard deviation of the
openness series over the window; speech produces rapid open/close cycles
(measured ~0.09 on a talking face) while a closed or still mouth stays near the
landmark jitter floor (~0.001), so a threshold between the two separates them
cleanly. The verdict is indeterminate (``moving=None``) when too few frames
contain a face - absence of a face is not evidence of a still mouth.

Frames are read sequentially with one seek (cheap: the pipeline's clip is only
a few seconds long). Like the V2 detector, the model file is downloaded once on
first use, ``mediapipe`` is imported lazily, and instances are not thread-safe.

A fresh MediaPipe landmarker is created for every :meth:`analyze` call and
closed afterwards. VIDEO mode requires timestamps to increase monotonically for
the *lifetime of the landmarker*, so a reused instance crashes on the second
job (its window starts before the previous job's ended) - and would carry
tracking state from one video into another. Creation costs a fraction of a
second, irrelevant next to the pipeline stages around it.
"""

from __future__ import annotations

import logging
import math
from typing import Any

import cv2

from dialogue_locator.config import MouthMovementConfig
from dialogue_locator.exceptions import MouthMovementError
from dialogue_locator.models import MouthMovementResult, VideoInfo, format_timestamp
from dialogue_locator.vision.model_files import ensure_model_file

logger = logging.getLogger(__name__)

# FaceMesh landmark indices (see MediaPipe's canonical face model).
_UPPER_LIP_INNER = 13
_LOWER_LIP_INNER = 14
_MOUTH_CORNER_LEFT = 61
_MOUTH_CORNER_RIGHT = 291


class MouthMovementAnalyzer:
    """Decide whether a face's mouth moves during a window of a video."""

    def __init__(self, config: MouthMovementConfig) -> None:
        self._config = config

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def analyze(self, video: VideoInfo, start: float, end: float) -> MouthMovementResult:
        """Analyse mouth movement in ``[start, end]`` (absolute source seconds).

        ``video`` may be a clip (``clip_start > 0``), like in
        :class:`~dialogue_locator.media.frames.FrameExtractor`: seeking happens
        clip-relative while the reported window stays absolute.
        """
        if end <= start:
            raise MouthMovementError(f"Invalid analysis window: {start:.3f}s .. {end:.3f}s")
        if end - start > self._config.max_window_seconds:
            end = start + self._config.max_window_seconds

        landmarker = self._load()
        try:
            openness, frames_total, actual_start, actual_end = self._openness_series(
                landmarker, video, start, end
            )
        except MouthMovementError:
            raise
        except Exception as exc:
            # Contract: this stage only ever raises MouthMovementError, so the
            # pipeline's fail-open guard catches everything (e.g. MediaPipe
            # runtime errors) instead of failing the whole job.
            logger.error("Mouth analysis crashed: %s", exc)
            raise MouthMovementError(f"Mouth analysis failed: {exc}") from exc
        finally:
            landmarker.close()

        score: float | None = None
        moving: bool | None = None
        if openness:
            mean = sum(openness) / len(openness)
            score = math.sqrt(sum((x - mean) ** 2 for x in openness) / len(openness))
        if len(openness) >= self._config.min_face_frames:
            moving = score >= self._config.movement_threshold

        result = MouthMovementResult(
            moving=moving,
            movement_score=score,
            threshold=self._config.movement_threshold,
            frames_analyzed=frames_total,
            frames_with_face=len(openness),
            window_start=actual_start,
            window_end=actual_end,
        )
        logger.info(
            "Mouth movement %s .. %s: %d/%d frames with a face, score %s (threshold %.3f) -> %s",
            format_timestamp(actual_start),
            format_timestamp(actual_end),
            len(openness),
            frames_total,
            "-" if score is None else f"{score:.4f}",
            self._config.movement_threshold,
            {True: "moving", False: "not moving", None: "indeterminate"}[moving],
        )
        return result

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    def _openness_series(
        self, landmarker: Any, video: VideoInfo, start: float, end: float
    ) -> tuple[list[float], int, float, float]:
        """Return ``(openness per face frame, frames read, actual_start, actual_end)``."""
        import mediapipe as mp

        cap = cv2.VideoCapture(str(video.path))
        try:
            if not cap.isOpened():
                logger.error("OpenCV cannot open %s", video.path)
                raise MouthMovementError(
                    f"Cannot open video: {video.path.name}", details={"path": str(video.path)}
                )
            fps = video.fps or cap.get(cv2.CAP_PROP_FPS)
            if not fps or fps <= 0:
                raise MouthMovementError(
                    f"Cannot determine frame rate of {video.path.name}",
                    details={"path": str(video.path)},
                )

            first_frame = max(0, int((start - video.clip_start) * fps))
            last_frame = max(first_frame, int((end - video.clip_start) * fps))
            cap.set(cv2.CAP_PROP_POS_FRAMES, first_frame)

            openness: list[float] = []
            frames_total = 0
            for index in range(first_frame, last_frame + 1):
                ok, frame = cap.read()
                if not ok or frame is None:
                    break  # window runs past the end of the clip
                frames_total += 1
                image = mp.Image(
                    image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                )
                # VIDEO mode needs monotonically increasing timestamps in ms.
                detected = landmarker.detect_for_video(image, int(index / fps * 1000))
                if detected.face_landmarks:
                    openness.append(self._openness(detected.face_landmarks[0]))
            actual_start = video.clip_start + first_frame / fps
            actual_end = video.clip_start + (first_frame + max(frames_total - 1, 0)) / fps
            return openness, frames_total, actual_start, actual_end
        finally:
            cap.release()

    @staticmethod
    def _openness(landmarks: Any) -> float:
        """Inner-lip gap normalised by mouth width for one frame's landmarks."""

        def dist(a: int, b: int) -> float:
            return math.dist((landmarks[a].x, landmarks[a].y), (landmarks[b].x, landmarks[b].y))

        gap = dist(_UPPER_LIP_INNER, _LOWER_LIP_INNER)
        width = dist(_MOUTH_CORNER_LEFT, _MOUTH_CORNER_RIGHT)
        return gap / max(width, 1e-6)

    def _load(self) -> Any:
        """Create the MediaPipe Face Landmarker (downloading the model if needed)."""
        model_path = ensure_model_file(
            self._config.model_path,
            self._config.model_url,
            self._config.download_timeout_seconds,
            MouthMovementError,
        )
        try:
            from mediapipe.tasks.python import BaseOptions
            from mediapipe.tasks.python import vision as mp_vision
            from mediapipe.tasks.python.vision.core.vision_task_running_mode import (
                VisionTaskRunningMode,
            )
        except ImportError as exc:
            raise MouthMovementError(
                "mediapipe is not installed; install requirements.txt into the venv"
            ) from exc

        logger.info(
            "Loading Face Landmarker (model=%s, min_face_confidence=%.2f)",
            model_path,
            self._config.min_face_confidence,
        )
        options = mp_vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=VisionTaskRunningMode.VIDEO,
            num_faces=1,  # judge the most prominent face
            min_face_detection_confidence=self._config.min_face_confidence,
            min_face_presence_confidence=self._config.min_face_confidence,
        )
        try:
            return mp_vision.FaceLandmarker.create_from_options(options)
        except Exception as exc:
            logger.error("Cannot create MediaPipe face landmarker: %s", exc)
            raise MouthMovementError(
                f"Cannot load face landmarker model: {exc}", details={"model_path": str(model_path)}
            ) from exc
