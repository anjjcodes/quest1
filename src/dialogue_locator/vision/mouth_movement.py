"""Mouth-movement detection: are the lips moving while the dialogue is spoken?

V3 builds on V2's face check. Given the video and the matched dialogue window,
this module reduces each frame's lips to one number with MediaPipe's Face
Landmarker (VIDEO running mode, which smooths landmarks temporally):

    openness = inner-lip gap (landmarks 13-14) / mouth width (landmarks 61-291)

Dividing by the mouth width makes the signal invariant to face size, camera
distance and resolution.

Feeding the landmarker
----------------------
Frames are *not* handed to the landmarker whole. The landmarker runs its own
face detector on whatever image it is given, and that detector only finds faces
that fill a decent part of the frame - in a cinematic wide shot a face 290 px
across in a 1920x1080 frame is simply not found, so a speaker who is plainly on
camera contributes no frames at all. V2's BlazeFace detector does find those
faces (it is calibrated for exactly this in ``FaceDetectionConfig``), so each
frame goes through it first, and the landmarker sees a padded, upscaled crop
around the best-scoring box. Frames where BlazeFace finds nothing fall back to
landmarking the whole frame, which is never worse than the old behaviour.

The crop is re-centred every frame, so the face stays roughly centred in the
landmarker's input - which is also what its VIDEO-mode tracking prefers.

BlazeFace itself flickers on faces that are small in the frame: measured on a
mid-shot of a man talking straight to camera, a 290-416px face scored 0.30-0.39
against a 0.30 threshold and was found in only 4 of 26 frames, which is not
enough consecutive samples to score. The detector letterboxes the whole frame
down to 192x192, so a face filling 15% of the width arrives ~29px wide however
large the source is - upscaling the frame cannot help. The last known box is
therefore carried forward onto frames where detection finds nothing, and the
landmarker decides: on the same clip that lifts 4 frames to 22, turning a
scattered dust of samples into a clean speech oscillation. The carry is bounded
by ``box_carry_seconds`` so a cut cannot attribute a new shot's face to the
previous run, and it costs no extra model calls.

Scoring
-------
The verdict is the *detrended* standard deviation of the openness series over
the most active sliding window of ``score_window_seconds``, not one number over
the whole dialogue window.

Detrended because a face that is merely present, not speaking, does not hold a
fixed mouth opening: it turns, tilts and settles, and openness slides smoothly
with it. Measured on a listener in a reaction shot, that slide alone reaches
0.042 - twice what a still mouth was calibrated at - while the same window's
residual after subtracting a straight line is 0.016. A speaking mouth, which
oscillates rather than slides, keeps almost all of its spread (0.052 raw, 0.047
detrended). Fitting a line per window and measuring what is left over separates
a mouth that is *moving* from a head that is moving; plain spread does not.

Measured over the cached corpus: faces present but silent - still faces, the
listener above, a voice-over read with the mouth shut - score 0.000-0.021, while
faces speaking on camera score 0.046-0.117. The default threshold sits in the
gap.

Sliding, because speech produces rapid open/close cycles that a single window
over the whole line averages away. Taking the maximum over short windows means:

* a line that is off camera (reaction shot, cutaway) for most of its length and
  on camera for the last second still reads as *moving*, which is what the
  brief asks for - the character is on camera saying the dialogue;
* a hard cut between two *still* faces cannot be read as movement, because a
  window never spans a shot change: the step between two faces' resting mouths
  is not a mouth opening. Cuts are spotted either by losing the face entirely
  or by the face box jumping (``shot_change_shift`` / ``shot_change_scale``).

The window that wins is reported as ``movement_start``: the first moment the
speaker is verifiably on camera saying the line, and so the frame worth
reporting. Windows are only scored over runs of consecutive same-shot face
frames, and only when a run offers at least ``min_face_frames`` samples -
absence of a face is not evidence of a still mouth, and a stray two-frame run
is not evidence of speech.

A fresh MediaPipe landmarker is created for every :meth:`analyze` call and
closed afterwards. VIDEO mode requires timestamps to increase monotonically for
the *lifetime of the landmarker*, so a reused instance crashes on the second
job (its window starts before the previous job's ended) - and would carry
tracking state from one video into another. Creation costs a fraction of a
second, irrelevant next to the pipeline stages around it.

The model file is downloaded once on first use, ``mediapipe`` is imported
lazily, and instances are not thread-safe.

Known limitation: when several faces share a frame, the highest-scoring box
wins, so the tracked face can change identity mid-window. The short scoring
windows contain the damage (a window is half a second of one shot) but do not
remove it.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from dialogue_locator.config import FaceDetectionConfig, MouthMovementConfig
from dialogue_locator.exceptions import FaceDetectionError, MouthMovementError
from dialogue_locator.models import FaceBox, MouthMovementResult, VideoInfo, format_timestamp
from dialogue_locator.vision.face_detector import FaceDetector
from dialogue_locator.vision.model_files import ensure_model_file

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Sample:
    """One landmarked frame: where it is, which shot it belongs to, how open."""

    index: int  # frame index within the clip
    shot: int  # bumped whenever the face box jumps; see _is_shot_change
    openness: float


# FaceMesh landmark indices (see MediaPipe's canonical face model).
_UPPER_LIP_INNER = 13
_LOWER_LIP_INNER = 14
_MOUTH_CORNER_LEFT = 61
_MOUTH_CORNER_RIGHT = 291


class MouthMovementAnalyzer:
    """Decide whether a face's mouth moves during a window of a video."""

    def __init__(
        self, config: MouthMovementConfig, face_detector: FaceDetector | None = None
    ) -> None:
        self._config = config
        # The pipeline injects the detector it already built for the V2 stage;
        # standalone callers (CLI experiments, tests) get a default one lazily,
        # so importing this module still costs nothing.
        self._face_detector = face_detector
        self._crop_warned = False

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

        self._crop_warned = False
        landmarker = self._load()
        try:
            samples, frames_total, fps, actual_start, actual_end = self._openness_series(
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

        score, moving, movement_start = self._score(samples, fps, video.clip_start)
        face_start = None if not samples else video.clip_start + samples[0].index / fps

        result = MouthMovementResult(
            moving=moving,
            movement_score=score,
            threshold=self._config.movement_threshold,
            frames_analyzed=frames_total,
            frames_with_face=len(samples),
            window_start=actual_start,
            window_end=actual_end,
            movement_start=movement_start,
            face_start=face_start,
        )
        logger.info(
            "Mouth movement %s .. %s: %d/%d frames with a face, best-window score %s "
            "(threshold %.3f) -> %s%s",
            format_timestamp(actual_start),
            format_timestamp(actual_end),
            len(samples),
            frames_total,
            "-" if score is None else f"{score:.4f}",
            self._config.movement_threshold,
            {True: "moving", False: "not moving", None: "indeterminate"}[moving],
            "" if movement_start is None else f" from {format_timestamp(movement_start)}",
        )
        return result

    # ------------------------------------------------------------------ #
    # Scoring
    # ------------------------------------------------------------------ #
    def _score(
        self, samples: list[Sample], fps: float, clip_start: float
    ) -> tuple[float | None, bool | None, float | None]:
        """Reduce the openness series to ``(score, moving, movement_start)``.

        The score is the largest detrended standard deviation of any sliding
        window of ``score_window_seconds`` that fits inside a run of consecutive
        face frames from one shot. With no scorable window the score falls back
        to the spread of whatever was seen (informational only) and the verdict
        stays indeterminate.

        The window is counted in *samples*, so a run with bridged gaps can span
        slightly more wall-clock than ``score_window_seconds``. Measured over the
        cached corpus the worst case is a 5-sample window spanning 0.38 s - still
        inside the nominal 0.5 s - because bridging only ever fires on runs that
        are short to begin with.
        """
        if not samples:
            return None, None, None

        window_frames = max(2, round(self._config.score_window_seconds * fps))
        min_samples = max(2, self._config.min_face_frames)

        best_score: float | None = None
        best_frame: int | None = None
        max_gap = round(self._config.max_gap_seconds * fps)
        for run in self._consecutive_runs(samples, max_gap):
            if len(run) < min_samples:
                continue  # too little to call, whatever its spread
            size = min(window_frames, len(run))
            for offset in range(len(run) - size + 1):
                window = run[offset : offset + size]
                score = self._movement([sample.openness for sample in window])
                if best_score is None or score > best_score:
                    best_score, best_frame = score, window[0].index

        if best_score is None:
            # Nothing scorable: report the overall spread so the caller can see
            # what little was measured, but do not turn it into a verdict.
            return self._movement([sample.openness for sample in samples]), None, None

        moving = best_score >= self._config.movement_threshold
        movement_start = clip_start + (best_frame or 0) / fps
        return best_score, moving, movement_start if moving else None

    @staticmethod
    def _consecutive_runs(samples: list[Sample], max_gap: int) -> list[list[Sample]]:
        """Split samples into runs of near-adjacent frames from the same shot.

        A long gap means the face was lost; a change of shot index means the
        face box jumped. Either way the frames on both sides belong to different
        shots - often to different people - and comparing their mouths would
        measure the cut, not a mouth.

        Gaps of up to ``max_gap`` frames are bridged: the landmarker drops the
        odd frame mid-shot, and treating each dropout as a cut leaves runs too
        short to hold a syllable - a half-open and a half-close, each of which
        the trend fit in :meth:`_movement` reads as drift and flattens away.
        """
        runs: list[list[Sample]] = []
        for sample in samples:
            previous = runs[-1][-1] if runs else None
            bridgeable = (
                previous is not None
                and sample.shot == previous.shot
                and 0 < sample.index - previous.index <= max_gap + 1
            )
            if bridgeable:
                runs[-1].append(sample)
            else:
                runs.append([sample])
        return runs

    @classmethod
    def _movement(cls, values: list[float]) -> float:
        """Spread of ``values`` around their best-fit line.

        The line absorbs the smooth drift of a head turning or settling, which
        is not the mouth opening; what is left is the frame-to-frame variation
        a mouth forming words produces. See the module docstring for the
        measurements this rests on.
        """
        count = len(values)
        mean_x = (count - 1) / 2
        mean_y = sum(values) / count
        variance_x = sum((i - mean_x) ** 2 for i in range(count))
        covariance = sum((i - mean_x) * (values[i] - mean_y) for i in range(count))
        slope = covariance / variance_x if variance_x else 0.0
        residuals = [values[i] - (mean_y + slope * (i - mean_x)) for i in range(count)]
        return cls._stdev(residuals)

    @staticmethod
    def _stdev(values: list[float]) -> float:
        mean = sum(values) / len(values)
        return math.sqrt(sum((x - mean) ** 2 for x in values) / len(values))

    # ------------------------------------------------------------------ #
    # Frame reading / landmarking
    # ------------------------------------------------------------------ #
    def _openness_series(
        self, landmarker: Any, video: VideoInfo, start: float, end: float
    ) -> tuple[list[Sample], int, float, float, float]:
        """Read the window and landmark it.

        Returns ``(samples, frames read, fps, actual_start, actual_end)``, with
        one :class:`Sample` per frame a face was landmarked in. Frames are read
        sequentially after one seek (cheap: the pipeline's clip is only a few
        seconds long).
        """
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

            samples: list[Sample] = []
            frames_total = 0
            shot = 0
            previous_box: FaceBox | None = None
            carried: FaceBox | None = None  # last detected box, reused while it lasts
            carried_frames = 0
            max_carry = round(self._config.box_carry_seconds * fps)
            for index in range(first_frame, last_frame + 1):
                ok, frame = cap.read()
                if not ok or frame is None:
                    break  # window runs past the end of the clip
                frames_total += 1
                box = self._detect_face(frame)
                if box is not None:
                    carried, carried_frames = box, 0
                elif carried is not None and carried_frames < max_carry:
                    box, carried_frames = carried, carried_frames + 1
                else:
                    carried, carried_frames = None, 0
                if self._is_shot_change(previous_box, box):
                    shot += 1
                previous_box = box
                image = frame if box is None else self._crop_to(frame, box)
                openness = self._frame_openness(landmarker, image, index, fps)
                if openness is not None:
                    samples.append(Sample(index=index, shot=shot, openness=openness))
            actual_start = video.clip_start + first_frame / fps
            actual_end = video.clip_start + (first_frame + max(frames_total - 1, 0)) / fps
            return samples, frames_total, fps, actual_start, actual_end
        finally:
            cap.release()

    def _frame_openness(
        self, landmarker: Any, image: np.ndarray, index: int, fps: float
    ) -> float | None:
        """Mouth openness for one image, or ``None`` if no face was landmarked."""
        import mediapipe as mp

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        )
        # VIDEO mode needs monotonically increasing timestamps in ms.
        detected = landmarker.detect_for_video(mp_image, int(index / fps * 1000))
        if not detected.face_landmarks:
            return None
        return self._openness(detected.face_landmarks[0])

    def _detect_face(self, frame: np.ndarray) -> FaceBox | None:
        """The best-scoring face box in ``frame``, or ``None`` if there is none.

        A failing detector must not fail the stage: it degrades to landmarking
        whole frames (the pre-crop behaviour), warned about once per call.
        """
        try:
            detection = self._detector().detect(frame, log_result=False)
        except FaceDetectionError as exc:
            if not self._crop_warned:
                logger.warning("Face crop unavailable, landmarking whole frames: %s", exc)
                self._crop_warned = True
            return None
        return detection.faces[0] if detection.faces else None  # FaceDetector sorts them

    def _crop_to(self, frame: np.ndarray, box: FaceBox) -> np.ndarray:
        """Padded, upscaled crop of ``frame`` around ``box``.

        Returns the frame unchanged if the box lies outside it - a carried box
        can, once the shot has moved on.
        """
        height, width = frame.shape[:2]
        pad = int(round(self._config.crop_padding * max(box.width, box.height)))
        x0, y0 = max(0, box.x - pad), max(0, box.y - pad)
        x1, y1 = min(width, box.x + box.width + pad), min(height, box.y + box.height + pad)
        crop = frame[y0:y1, x0:x1]
        if crop.size == 0:
            return frame

        # Upscale small crops: the landmarker downsamples its input, so a face
        # that is small in the source frame is lost unless it is enlarged.
        shortest = min(crop.shape[0], crop.shape[1])
        if shortest < self._config.min_crop_size:
            scale = self._config.min_crop_size / shortest
            crop = cv2.resize(
                crop,
                (max(1, round(crop.shape[1] * scale)), max(1, round(crop.shape[0] * scale))),
                interpolation=cv2.INTER_CUBIC,
            )
        return crop

    def _is_shot_change(self, previous: FaceBox | None, current: FaceBox | None) -> bool:
        """Did the camera cut between these two adjacent frames?

        Judged on the face box alone: within a shot a face drifts a few pixels
        per frame, while a cut drops a different face somewhere else in the
        frame or at a different size. Only called with two real boxes - with
        one missing there is no evidence either way, and splitting on a
        flickering detector would leave runs too short to score. A carried box
        is identical to the one before it, so a cut is caught on the frame
        detection resumes.
        """
        if previous is None or current is None:
            return False
        moved = math.dist(
            (previous.x + previous.width / 2, previous.y + previous.height / 2),
            (current.x + current.width / 2, current.y + current.height / 2),
        )
        if moved > self._config.shot_change_shift * max(previous.width, current.width):
            return True
        ratio = current.width / max(previous.width, 1)
        return ratio > self._config.shot_change_scale or ratio < 1 / self._config.shot_change_scale

    def _detector(self) -> FaceDetector:
        if self._face_detector is None:
            self._face_detector = FaceDetector(FaceDetectionConfig())
        return self._face_detector

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
