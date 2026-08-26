"""The dialogue localisation pipeline.

Stages, in order (each maps to a :class:`PipelineStage`)::

    input          validate dialogue + source           (fails before any download)
    download       VideoDownloader.fetch_search_media -> MediaInfo (audio-only if possible)
    audio          AudioExtractor    -> AudioInfo, PCM samples in memory
    transcription  fast Transcriber  -> Word stream   \\  consumed together: the matcher
    matching       StreamingMatcher  -> MatchCandidate /  stops the stream at the first match
    verification   Verifier chain    -> VerificationOutcome per verifier (refines or warns)
    download_video VideoDownloader.fetch_video_clip -> VideoInfo (a few seconds of
                   full-quality video around the *verified* timestamp)
    frame          FrameExtractor    -> FrameInfo (image on disk)
    face_detection FaceDetector     -> FaceDetectionResult (V2: is a human face
                   visible in the extracted frame? No face -> status becomes
                   NOT_ONSCREEN; a failed check fails open with a warning)
    mouth_movement MouthMovementAnalyzer -> MouthMovementResult (V3, only after
                   V2 confirms a face: do the lips move during the matched
                   window? Not moving -> NOT_ONSCREEN; indeterminate or failed
                   fails open with a warning)
    done

The search stages run on a cheap audio-first download; full-quality video is
downloaded only as a short clip around the verified match (so a NOT_FOUND job
never pays for a video download, and a FOUND job pays only for a few seconds).
Verification is audio-only, so it runs before any video exists; the visual
checks (V2/V3) run afterwards, on the clip and the extracted frame.

The pipeline knows nothing about FastAPI or HTML: it takes a
:class:`PipelineRequest`, reports through a :data:`ProgressCallback`, and
returns a :class:`LocalizationResult`. Every stage component can be injected
(constructor keyword arguments) so the orchestration is testable with fakes.

Extension points:

* Another *audio* confirmation check (e.g. a second ASR engine): pass an extra
  :class:`Verifier` in ``verifiers`` or append to ``build_default_verifiers``.
* Another *visual* validation stage (V4+): follow the V2/V3 pattern - a small
  analyser class in ``vision/``, a ``_stage_*`` method in :class:`_Run` that
  gates the verdict and fails open with a warning, and a config toggle.
"""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from dialogue_locator.config import Settings
from dialogue_locator.exceptions import (
    DialogueLocatorError,
    FaceDetectionError,
    FrameExtractionError,
    MouthMovementError,
    PipelineCancelledError,
)
from dialogue_locator.matching.matcher import StreamingMatcher
from dialogue_locator.media.audio import AudioExtractor
from dialogue_locator.media.downloader import VideoDownloader, is_url, validate_url
from dialogue_locator.media.frames import FrameExtractor, timestamp_to_frame
from dialogue_locator.models import (
    AudioInfo,
    FaceDetectionResult,
    LocalizationResult,
    MatchCandidate,
    MediaInfo,
    PipelineStage,
    ProgressCallback,
    ProgressEvent,
    ResultStatus,
    VerificationStatus,
    VideoInfo,
    format_timestamp,
)
from dialogue_locator.transcription.base import Transcriber, load_pcm
from dialogue_locator.transcription.faster_whisper import FasterWhisperTranscriber
from dialogue_locator.verification.asr_verifier import AsrVerifier
from dialogue_locator.verification.base import VerificationContext, Verifier
from dialogue_locator.vision.face_detector import FaceDetector
from dialogue_locator.vision.mouth_movement import MouthMovementAnalyzer

logger = logging.getLogger(__name__)

FRAME_FILENAME = "frame"
RESULT_FILENAME = "result.json"


@dataclass
class PipelineRequest:
    """One localisation job."""

    source: str  # http(s) URL or local path
    dialogue: str
    job_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    reuse_cached_media: bool = True  # reuse download/audio from an earlier run on the same source

    @property
    def source_key(self) -> str:
        """Stable directory name for media cached per source."""
        return hashlib.sha1(self.source.strip().encode("utf-8")).hexdigest()[:16]


def build_default_verifiers(settings: Settings, verify_transcriber: Transcriber | None = None) -> list[Verifier]:
    """The default verifier chain: one large-model ASR check of the candidate."""
    if not settings.verification.enabled:
        return []
    # The verifier re-hears a short window already known to contain speech, so the
    # VAD has nothing to save there - but under loud music it can delete the very
    # line being verified (it cost a perfect match a rejection on Infinity War).
    no_vad = settings.whisper.model_copy(update={"vad_filter": False})
    transcriber = verify_transcriber or FasterWhisperTranscriber(settings.whisper.verify_model, no_vad)
    return [AsrVerifier(transcriber, settings.matching, settings.verification)]


class DialoguePipeline:
    """Runs the full localisation pipeline for one request at a time."""

    def __init__(
        self,
        settings: Settings,
        *,
        downloader: VideoDownloader | None = None,
        audio_extractor: AudioExtractor | None = None,
        fast_transcriber: Transcriber | None = None,
        verifiers: list[Verifier] | None = None,
        retry_transcriber: Transcriber | None = None,
        frame_extractor: FrameExtractor | None = None,
        face_detector: FaceDetector | None = None,
        mouth_analyzer: MouthMovementAnalyzer | None = None,
    ) -> None:
        self.settings = settings
        self.downloader = downloader or VideoDownloader(settings.download, settings.audio.ffprobe_binary)
        self.audio_extractor = audio_extractor or AudioExtractor(settings.audio)
        # The streaming pass decodes greedily (fast_beam_size); the verify
        # transcriber built in build_default_verifiers keeps the full beam.
        fast_config = settings.whisper.model_copy(
            update={"beam_size": settings.whisper.fast_beam_size}
        )
        self.fast_transcriber = fast_transcriber or FasterWhisperTranscriber(
            settings.whisper.fast_model, fast_config
        )
        if retry_transcriber is not None:
            self.retry_transcriber: Transcriber | None = retry_transcriber
        elif fast_transcriber is None and settings.whisper.vad_filter and settings.whisper.retry_without_vad:
            # Same weights as fast_transcriber via WhisperModelCache (the VAD flag is
            # not part of the cache key), so this costs no extra memory or load time.
            no_vad = fast_config.model_copy(update={"vad_filter": False})
            self.retry_transcriber = FasterWhisperTranscriber(settings.whisper.fast_model, no_vad)
        else:
            self.retry_transcriber = None
        self.verifiers = build_default_verifiers(settings) if verifiers is None else verifiers
        self.frame_extractor = frame_extractor or FrameExtractor(
            settings.frame, settings.audio.ffmpeg_binary, settings.audio.ffprobe_binary
        )
        self.face_detector = face_detector or FaceDetector(settings.face_detection)
        # V3 crops each frame to the face before landmarking it, so it needs a
        # detector too - the same one, so a job's face_detection overrides apply
        # to both stages and only one model is loaded.
        self.mouth_analyzer = mouth_analyzer or MouthMovementAnalyzer(
            settings.mouth_movement, self.face_detector
        )

    # ------------------------------------------------------------------ #
    def warm_up(self) -> None:
        """Load ASR models up-front so job timings do not include model loading."""
        t0 = time.perf_counter()
        self.fast_transcriber.warm_up()
        for verifier in self.verifiers:
            verifier.warm_up()
        logger.info("Models ready in %.1fs", time.perf_counter() - t0)

    # ------------------------------------------------------------------ #
    def run(
        self,
        request: PipelineRequest,
        progress: ProgressCallback | None = None,
        should_cancel: Callable[[], bool] | None = None,
    ) -> LocalizationResult:
        """Execute all stages. Raises ``DialogueLocatorError`` subclasses on failure."""
        run = _Run(self, request, progress, should_cancel)
        return run.execute()


class _Run:
    """State for a single pipeline execution (keeps ``DialoguePipeline`` stateless)."""

    def __init__(
        self,
        pipeline: DialoguePipeline,
        request: PipelineRequest,
        progress: ProgressCallback | None,
        should_cancel: Callable[[], bool] | None,
    ) -> None:
        self.p = pipeline
        self.req = request
        self.progress = progress
        self.should_cancel = should_cancel or (lambda: False)
        self.settings = pipeline.settings
        self.timings: dict[str, float] = {}
        self.warnings: list[str] = []
        self.transcribed_seconds = 0.0  # how far into the audio the search got
        self.stage = PipelineStage.INPUT
        self.work_dir = self.settings.storage.work_dir / request.source_key
        self.output_dir = self.settings.storage.output_dir / request.job_id

    # ------------------------------------------------------------------ #
    def execute(self) -> LocalizationResult:
        t_start = time.perf_counter()
        logger.info("Job %s: source=%r dialogue=%r", self.req.job_id, self.req.source, self.req.dialogue)
        try:
            result = self._execute()
        except PipelineCancelledError as exc:
            exc.stage = self.stage.value
            logger.warning("Job %s cancelled in stage '%s'", self.req.job_id, exc.stage)
            raise
        except DialogueLocatorError as exc:
            logger.error("Job %s failed in stage '%s': %s", self.req.job_id, exc.stage, exc.message)
            raise
        except Exception as exc:  # noqa: BLE001 - last-resort boundary
            logger.exception("Job %s crashed in stage '%s'", self.req.job_id, self.stage.value)
            err = DialogueLocatorError(f"Unexpected error during {self.stage.value}: {exc}")
            err.stage = self.stage.value
            raise err from exc
        finally:
            self._cleanup()
        result.stage_timings["total"] = time.perf_counter() - t_start
        self._emit(PipelineStage.DONE, "Done", 1.0, {"status": result.status.value})
        logger.info(
            "Job %s finished: %s in %.1fs (%s)",
            self.req.job_id,
            result.status.value,
            result.stage_timings["total"],
            ", ".join(f"{k}={v:.1f}s" for k, v in result.stage_timings.items() if k != "total"),
        )
        return result

    def _execute(self) -> LocalizationResult:
        """One stage per line; each ``_stage_*`` method owns its timing and events."""
        matcher = self._stage_input()
        media = self._stage_download(self.req.source)
        audio, samples = self._stage_audio(media)
        match, matcher = self._stage_search(samples, matcher)

        result = LocalizationResult(
            status=ResultStatus.FOUND if match else ResultStatus.NOT_FOUND,
            dialogue=matcher.target.raw,
            source_url=self.req.source,
            video=None,  # full-quality video is fetched only after a match
            first_pass=match,
            match=match,
            near_misses=matcher.near_misses,
            warnings=self.warnings,
            stage_timings=self.timings,
            transcribed_seconds=self.transcribed_seconds,
        )
        if match is None:
            self._emit(
                PipelineStage.MATCHING,
                f"Dialogue not found (best score {matcher.best_score:.1f})",
                1.0,
                {"best_score": matcher.best_score, "near_misses": len(matcher.near_misses)},
            )
            return result
        self._emit(
            PipelineStage.MATCHING,
            f"Match at {match.timestamp} (score {match.score:.1f})",
            1.0,
            {"timestamp": match.timestamp, "score": match.score},
        )
        self._check_cancel()

        self._stage_verification(result, samples, audio)
        self._check_cancel()

        # A local audio-only source can be searched but has no frames to offer;
        # a URL always gets the clip (its search media may be audio-only even
        # though the source has video).
        if is_url(self.req.source) or media.has_video:
            video = self._stage_download_video(result)
            self._check_cancel()
            self._stage_frame(result, video)
            self._check_cancel()
            # The visual checks build on each other: no face check, no mouth check.
            if not self.settings.face_detection.enabled:
                logger.info("Face detection disabled; skipping visual checks")
                return result
            self._stage_face_detection(result)
            # V3 runs even when that one frame held no face. A line can open on
            # a title card or a cutaway and cut to the speaker a frame later;
            # only a scan of the whole window can tell that apart from a line
            # nobody is on camera for. V3 therefore owns the final verdict.
            #
            # A face check that *errored* is different from one that found
            # nothing: V3 crops to the boxes that detector produces, so with it
            # broken V3 would judge whole frames again - the very thing that
            # loses faces in wide shots - and could turn an infrastructure fault
            # into a NOT_ONSCREEN. It stays skipped, and the job fails open.
            if self.settings.mouth_movement.enabled and result.face_detection is not None:
                self._check_cancel()
                self._stage_mouth_movement(result, video)
        else:
            warning = "Source has no video stream; returning the timestamp without a frame."
            self.warnings.append(warning)
            logger.warning(warning)
        return result

    # ------------------------------------------------------------------ #
    # Stages (in pipeline order)
    # ------------------------------------------------------------------ #
    def _stage_input(self) -> StreamingMatcher:
        """Validate dialogue and URL; fails before any I/O."""
        self._begin(PipelineStage.INPUT, "Validating input")
        matcher = StreamingMatcher(self.req.dialogue, self.settings.matching)  # InvalidDialogueError
        if is_url(self.req.source):
            validate_url(self.req.source)  # InvalidURLError
        self._end(PipelineStage.INPUT)
        return matcher

    def _stage_download(self, source: str) -> MediaInfo:
        """Fetch the cheap search media (audio-only when the host offers it)."""
        self._begin(PipelineStage.DOWNLOAD, "Fetching audio for search")
        media = self.p.downloader.fetch_search_media(
            source, self.work_dir, progress=self.progress, reuse_existing=self.req.reuse_cached_media
        )
        self._end(PipelineStage.DOWNLOAD)
        self._check_cancel()
        return media

    def _stage_audio(self, media: MediaInfo) -> tuple[AudioInfo, np.ndarray]:
        """Extract 16 kHz mono PCM and load it into memory (shared with the verifier)."""
        self._begin(PipelineStage.AUDIO, "Extracting audio")
        audio = self.p.audio_extractor.extract(
            media, self.work_dir, progress=self.progress, reuse_existing=self.req.reuse_cached_media
        )
        samples = load_pcm(audio.path, self.settings.audio.sample_rate)
        self._end(PipelineStage.AUDIO)
        self._check_cancel()
        return audio, samples

    def _stage_search(
        self, samples: np.ndarray, matcher: StreamingMatcher
    ) -> tuple[MatchCandidate | None, StreamingMatcher]:
        """Stream the fast transcriber through the matcher; VAD-off retry on a miss."""
        self._begin(PipelineStage.TRANSCRIPTION, "Loading speech model")
        self.p.fast_transcriber.warm_up()
        self._emit(PipelineStage.TRANSCRIPTION, "Listening for the dialogue", 0.0, {})
        match, last_word_end = self._scan(self.p.fast_transcriber, samples, matcher)
        if match is None and self.p.retry_transcriber is not None:
            # The VAD can discard real speech buried under loud music/effects, so a
            # clean miss is re-checked once with the VAD off (decision #24) before
            # the pipeline reports not_found.
            logger.warning(
                "No match with VAD on (best score %.1f); retrying without voice-activity filter",
                matcher.best_score,
            )
            self._emit(
                PipelineStage.TRANSCRIPTION,
                "No match yet - listening again without the voice-activity filter",
                0.0,
                {"best_score": matcher.best_score},
            )
            retry_matcher = StreamingMatcher(self.req.dialogue, self.settings.matching)
            match, retry_end = self._scan(self.p.retry_transcriber, samples, retry_matcher)
            last_word_end = max(last_word_end, retry_end)
            if match is not None:
                self.warnings.append(
                    "Found only with the voice-activity filter disabled; "
                    "the line sits under loud background audio."
                )
            if retry_matcher.best_score >= matcher.best_score:
                matcher = retry_matcher  # report near misses from the better pass
        self._end(PipelineStage.TRANSCRIPTION)
        self.timings[PipelineStage.MATCHING.value] = 0.0  # folded into transcription
        self.transcribed_seconds = last_word_end
        return match, matcher

    def _stage_verification(
        self, result: LocalizationResult, samples: np.ndarray, audio: AudioInfo
    ) -> None:
        """Run the verifier chain (audio-only; the video clip does not exist yet).

        Each verifier sees the candidate as refined by the previous ones; a
        rejection or failure keeps the current timestamp and adds a warning.
        Sets ``result.match`` to the final (possibly refined) candidate.
        """
        self._begin(PipelineStage.VERIFICATION, "Verifying with larger model")
        context = VerificationContext(
            dialogue=result.dialogue,
            audio_samples=samples,
            audio_path=audio.path,
            sample_rate=self.settings.audio.sample_rate,
        )
        assert result.match is not None
        current = result.match
        for verifier in self.p.verifiers:
            outcome = verifier.verify(current, context)
            result.verifications.append(outcome)
            if outcome.status is VerificationStatus.CONFIRMED and outcome.refined is not None:
                if abs(outcome.refined.start - current.start) > 1e-3:
                    logger.info(
                        "%s refined timestamp %s -> %s",
                        outcome.verifier,
                        format_timestamp(current.start),
                        format_timestamp(outcome.refined.start),
                    )
                current = outcome.refined
            elif outcome.status in (VerificationStatus.REJECTED, VerificationStatus.FAILED):
                warning = f"{outcome.verifier}: {outcome.status.value} - {outcome.message}"
                self.warnings.append(warning)
                logger.warning("Verification warning: %s", warning)
            self._check_cancel()
        result.match = current
        self._end(PipelineStage.VERIFICATION)

    def _stage_download_video(self, result: LocalizationResult) -> VideoInfo:
        """Fetch a few seconds of full-quality video around the verified match.

        Only now is the timestamp final, so a NOT_FOUND job never pays for a
        video download and a FOUND job pays only for a short clip.
        """
        assert result.match is not None
        self._begin(PipelineStage.DOWNLOAD_VIDEO, "Fetching video clip")
        video = self.p.downloader.fetch_video_clip(
            self.req.source,
            result.match.start,
            result.match.end,
            self.work_dir,
            progress=self._retag(PipelineStage.DOWNLOAD_VIDEO),
            reuse_existing=self.req.reuse_cached_media,
        )
        self._end(PipelineStage.DOWNLOAD_VIDEO)
        result.video = video
        return video

    def _stage_frame(self, result: LocalizationResult, video: VideoInfo) -> None:
        """Save the frame at the verified start time into the job's output dir."""
        assert result.match is not None
        self._begin(PipelineStage.FRAME, "Extracting frame")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        result.frame = self.p.frame_extractor.extract(
            video, result.match.start, self.output_dir / FRAME_FILENAME
        )
        self._end(PipelineStage.FRAME)

    def _stage_face_detection(self, result: LocalizationResult) -> None:
        """V2: check the saved frame for a face and gate the verdict.

        A match with no face on camera is reported as NOT_ONSCREEN (details are
        kept so the caller can inspect what was found). Fails open: if the
        check itself cannot run, the localisation stands, with a warning.

        The verdict is only provisional when the V3 mouth check follows: that
        stage scans the whole dialogue window and can find the speaker in a
        later shot, overturning this one frame's answer.
        """
        assert result.frame is not None
        self._begin(PipelineStage.FACE_DETECTION, "Checking for a face in the frame")
        try:
            result.face_detection = self.p.face_detector.detect_file(result.frame.image_path)
        except FaceDetectionError as exc:
            warning = f"Face check failed: {exc.message}"
            self.warnings.append(warning)
            logger.warning("Face detection warning: %s", warning)
            self._emit(PipelineStage.FACE_DETECTION, warning, 1.0, {"error": exc.message})
        else:
            detected = result.face_detection
            if detected.face_present:
                message = f"{len(detected.faces)} face(s) visible in the frame"
            else:
                result.status = ResultStatus.NOT_ONSCREEN
                message = (
                    "No face in this frame - checking the rest of the line"
                    if self.settings.mouth_movement.enabled
                    else "No face visible - not an onscreen dialogue"
                )
                logger.info("No face in frame %s; verdict: not_onscreen", result.frame.image_path.name)
            self._emit(PipelineStage.FACE_DETECTION, message, 1.0, detected.to_dict())
        self._end(PipelineStage.FACE_DETECTION)

    def _stage_mouth_movement(self, result: LocalizationResult, video: VideoInfo) -> None:
        """V3: track the lips across the clip's frames and settle the verdict.

        This stage, not V2, decides whether the line is onscreen dialogue,
        because it is the only one that sees the whole window rather than a
        single frame:

        * a mouth moving anywhere during the line -> FOUND, and the answer frame
          moves to that moment (:meth:`_move_frame_to`). This holds even
          when the line's first frame had no face - it may open on a title card
          or a cutaway and cut to the speaker afterwards.
        * faces on camera but none of them moving -> NOT_ONSCREEN. The line is
          narration, dubbing or a reaction shot. The answer frame moves to that
          face if the line itself opened off camera, so the result shows what it
          is talking about.
        * barely any frame of the line landmarked -> NOT_ONSCREEN. V2 works on
          whole frames and fires on things that are not faces; a box no mesh
          fits is not a face, whatever its detection score.
        * no face in the reported frame and none to judge in the window ->
          NOT_ONSCREEN, the verdict V2 already reached.

        Fails open, like the face check, in the one case that is genuinely
        unmeasured: a face present throughout whose frames never line up into a
        scorable run. A crashed analyser fails open too.
        """
        assert result.match is not None
        self._begin(PipelineStage.MOUTH_MOVEMENT, "Checking for mouth movement")
        try:
            result.mouth_movement = self.p.mouth_analyzer.analyze(
                video, result.match.start, result.match.end
            )
        except MouthMovementError as exc:
            warning = f"Mouth check failed: {exc.message}"
            self.warnings.append(warning)
            logger.warning("Mouth movement warning: %s", warning)
            self._emit(PipelineStage.MOUTH_MOVEMENT, warning, 1.0, {"error": exc.message})
        else:
            movement = result.mouth_movement
            had_face = result.face_detection is not None and result.face_detection.face_present
            if movement.moving is True:
                message = f"Mouth movement detected (score {movement.movement_score:.3f})"
                moved = self._move_frame_to(result, video, movement.movement_start)
                if moved:
                    message += (
                        f"; frame moved to {result.frame.timestamp_str}, where the speaker "
                        "is on camera"
                    )
                if moved or had_face:
                    # Overturns a NOT_ONSCREEN from the face check when the line
                    # opened off camera: the speaker is on screen saying it, and
                    # the reported frame now shows them.
                    result.status = ResultStatus.FOUND
                elif result.status is ResultStatus.NOT_ONSCREEN:
                    # The speaker is on camera during the line but that frame
                    # could not be produced (_move_frame_to warned), so
                    # the frame in hand still shows no face. Do not claim
                    # onscreen over a frame that does not show it.
                    logger.warning(
                        "Mouth moving but the speaker's frame is unavailable; keeping not_onscreen"
                    )
            elif movement.moving is False:
                result.status = ResultStatus.NOT_ONSCREEN
                message = "Face visible but mouth not moving - not an onscreen dialogue"
                if not had_face and self._move_frame_to(result, video, movement.face_start):
                    # The line opened off camera - over a title card, or on a
                    # cutaway. Reporting that frame next to "mouth not moving"
                    # explains nothing; show the face the verdict is about.
                    message += f"; frame moved to {result.frame.timestamp_str}, where that face is"
                logger.info(
                    "Mouth still during %s-%s (score %.4f); verdict: not_onscreen",
                    format_timestamp(movement.window_start),
                    format_timestamp(movement.window_end),
                    movement.movement_score or 0.0,
                )
            elif movement.frames_with_face < self.settings.mouth_movement.min_face_frames:
                # A handful of landmarked frames across a whole line is evidence
                # *against* a face, not a missing measurement. BlazeFace fires on
                # blurred rubble at 0.45-0.62 - higher than it scores some real
                # faces - so a box it likes that the landmarker cannot land a
                # mesh on must not fail open into "found".
                result.status = ResultStatus.NOT_ONSCREEN
                message = "No face visible during the line - not an onscreen dialogue"
                logger.info(
                    "Only %d/%d frames landmarked in %s-%s; verdict: not_onscreen",
                    movement.frames_with_face,
                    movement.frames_analyzed,
                    format_timestamp(movement.window_start),
                    format_timestamp(movement.window_end),
                )
            elif not had_face:
                # Enough of a face to be real, but too fragmented to score, and
                # the reported frame shows none: V2's verdict stands.
                message = "No face visible during the line - not an onscreen dialogue"
                logger.info(
                    "No face to judge in %s-%s (%d/%d frames); verdict: not_onscreen",
                    format_timestamp(movement.window_start),
                    format_timestamp(movement.window_end),
                    movement.frames_with_face,
                    movement.frames_analyzed,
                )
            else:
                # A face was there throughout but its frames never lined up into
                # a scorable run. That is a real "cannot tell", so fail open.
                warning = (
                    "Mouth movement could not be judged: face landmarks in only "
                    f"{movement.frames_with_face} of {movement.frames_analyzed} frames."
                )
                self.warnings.append(warning)
                logger.warning("Mouth movement warning: %s", warning)
                message = warning
            self._emit(PipelineStage.MOUTH_MOVEMENT, message, 1.0, movement.to_dict())
        self._end(PipelineStage.MOUTH_MOVEMENT)

    def _move_frame_to(
        self, result: LocalizationResult, video: VideoInfo, target: float | None
    ) -> bool:
        """Re-extract the answer frame at ``target``, and re-run the face check.

        A line's first frame is often not the frame worth showing. The brief
        asks for the first frame where the character is *on camera* saying the
        line, so on a positive verdict ``target`` is where the mouth was found
        moving. On a negative one it is where the face that was judged silent
        first appears - a line can open over a title card and cut to the speaker
        immediately, and showing the title card next to "mouth not moving"
        explains nothing.

        Either way the V2 face check is redone there, so every visual field
        describes the same frame. The localisation itself is untouched:
        ``match.start`` still reports where the line begins in the audio, and
        the two can be compared.

        Returns whether the frame actually moved (it does not when the target is
        the frame already reported). Fails open like the checks around it: on an
        extraction or detection error the original frame stands, with a warning.
        """
        assert result.frame is not None
        fps = video.fps or result.frame.fps
        if target is None or not fps:
            return False
        if timestamp_to_frame(target, fps) == result.frame.frame_number:
            return False  # already the frame being reported

        try:
            target, detection = self._confirm_face(video, target, fps)
            frame = self.p.frame_extractor.extract(
                video, target, self.output_dir / FRAME_FILENAME
            )
        except (FrameExtractionError, FaceDetectionError) as exc:
            warning = (
                f"A face is on camera at {format_timestamp(target)}, but that frame could "
                f"not be extracted ({exc.message}); showing the line's first frame."
            )
            self.warnings.append(warning)
            logger.warning("Frame move failed: %s", warning)
            return False

        logger.info(
            "Face on camera from %s; answer frame moved %s -> %s (frame %d -> %d)",
            format_timestamp(target),
            result.frame.timestamp_str,
            frame.timestamp_str,
            result.frame.frame_number,
            frame.frame_number,
        )
        result.frame = frame
        result.face_detection = detection
        return True

    def _confirm_face(
        self, video: VideoInfo, target: float, fps: float
    ) -> tuple[float, FaceDetectionResult]:
        """Pick the frame to report, starting from ``target``.

        ``target`` is where the relevant window begins, but the detector is
        marginal on exactly the faces this feature exists for - one scored 0.30
        against a 0.30 threshold, so it flickered off again when the frame was
        re-encoded as JPEG. Reporting that frame leaves the result saying "mouth
        moving" and "no face" at once. So the first frames of the window are
        probed and the first one the detector confirms is reported; its
        detection is the one recorded, taken on the decoded frame rather than
        the re-encoded image so the two agree by construction.

        Falls back to ``target`` and its own detection when no frame in the
        window is confirmed - the window scan is still the better evidence, and
        the caller keeps the verdict.
        """
        first_frame = timestamp_to_frame(target - video.clip_start, fps)
        probes = max(1, round(self.settings.mouth_movement.score_window_seconds * fps))
        fallback: FaceDetectionResult | None = None
        for offset in range(probes):
            image = self.p.frame_extractor.read_frame(video, first_frame + offset, fps)
            detection = self.p.face_detector.detect(image, log_result=False)
            if detection.face_present:
                return video.clip_start + (first_frame + offset) / fps, detection
            if fallback is None:
                fallback = detection
        assert fallback is not None
        logger.info(
            "No frame in %s+%.1fs confirms a face; reporting the window's first frame",
            format_timestamp(target),
            probes / fps,
        )
        return target, fallback

    # ------------------------------------------------------------------ #
    def _scan(
        self, transcriber: Transcriber, samples: np.ndarray, matcher: StreamingMatcher
    ) -> tuple[MatchCandidate | None, float]:
        """Stream ``transcriber`` over ``samples`` through ``matcher``.

        Returns the first match (or ``None``) and the end time of the last word
        pulled from the stream; the stream is closed as soon as a match settles.
        """
        match: MatchCandidate | None = None
        last_word_end = 0.0
        stream = transcriber.transcribe(samples, offset=0.0, progress=self.progress)
        try:
            for word in stream:
                last_word_end = word.end
                match = matcher.feed(word)
                if match is not None:
                    break
                if self.should_cancel():
                    raise PipelineCancelledError("Cancelled during transcription")
        finally:
            stream.close()  # stops the ASR decoder immediately
        if match is None:
            match = matcher.finish()
        return match, last_word_end

    # ------------------------------------------------------------------ #
    def _begin(self, stage: PipelineStage, message: str) -> None:
        self.stage = stage
        self._stage_t0 = time.perf_counter()
        logger.info("---- [%s] %s", stage.value, message)
        self._emit(stage, message, 0.0, {})

    def _end(self, stage: PipelineStage) -> None:
        self.timings[stage.value] = time.perf_counter() - self._stage_t0
        logger.info("---- [%s] done in %.2fs", stage.value, self.timings[stage.value])

    def _retag(self, stage: PipelineStage) -> ProgressCallback | None:
        """Progress callback that re-labels events with ``stage``.

        The downloader always emits DOWNLOAD events; the second (full-quality)
        fetch should surface as DOWNLOAD_VIDEO to progress consumers.
        """
        if self.progress is None:
            return None

        def relay(event: ProgressEvent) -> None:
            self.progress(ProgressEvent(stage, event.message, event.fraction, event.details))

        return relay

    def _emit(self, stage: PipelineStage, message: str, fraction: float | None, details: dict[str, Any]) -> None:
        if self.progress is not None:
            self.progress(ProgressEvent(stage, message, fraction, details))

    def _check_cancel(self) -> None:
        if self.should_cancel():
            logger.warning("Job %s cancelled during %s", self.req.job_id, self.stage.value)
            raise PipelineCancelledError(f"Cancelled during {self.stage.value}")

    def _cleanup(self) -> None:
        if self.settings.storage.keep_intermediate:
            return
        try:
            shutil.rmtree(self.work_dir, ignore_errors=True)
            logger.debug("Removed work dir %s", self.work_dir)
        except OSError as exc:  # pragma: no cover
            logger.warning("Could not remove work dir %s: %s", self.work_dir, exc)


def save_result(result: LocalizationResult, path: Path) -> Path:
    """Write ``result.to_dict()`` as JSON next to the frame image."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
    logger.info("Result written to %s", path)
    return path
