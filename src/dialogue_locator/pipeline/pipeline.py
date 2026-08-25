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
    done

The search stages run on a cheap audio-first download; full-quality video is
downloaded only as a short clip around the verified match (so a NOT_FOUND job
never pays for a video download, and a FOUND job pays only for a few seconds).
Verification is audio-only in V1, so it runs before any video exists; a V2
visual verifier should fetch its own clip on demand (``fetch_video_clip`` is
cheap) rather than expect ``context.video``.

The pipeline knows nothing about FastAPI or HTML: it takes a
:class:`PipelineRequest`, reports through a :data:`ProgressCallback`, and
returns a :class:`LocalizationResult`. Every stage component can be injected
(constructor keyword arguments) so the orchestration is testable with fakes.

Extension point for V2: pass an extra :class:`Verifier` in ``verifiers`` (or
append to ``build_default_verifiers``); it receives the candidate as refined by
the ASR verifier and may refine it further (e.g. to the first on-camera frame).
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

from dialogue_locator.config import Settings
from dialogue_locator.exceptions import (
    DialogueLocatorError,
    FaceDetectionError,
    PipelineCancelledError,
)
from dialogue_locator.matching.matcher import StreamingMatcher, TargetDialogue
from dialogue_locator.media.audio import AudioExtractor
from dialogue_locator.media.downloader import VideoDownloader, is_url, validate_url
from dialogue_locator.media.frames import FrameExtractor
from dialogue_locator.models import (
    LocalizationResult,
    MatchCandidate,
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
    """The V1 verifier chain. V2 appends its visual verifier here."""
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
    ) -> None:
        self.settings = settings
        self.downloader = downloader or VideoDownloader(settings.download, settings.audio.ffprobe_binary)
        self.audio_extractor = audio_extractor or AudioExtractor(settings.audio)
        self.fast_transcriber = fast_transcriber or FasterWhisperTranscriber(
            settings.whisper.fast_model, settings.whisper
        )
        if retry_transcriber is not None:
            self.retry_transcriber: Transcriber | None = retry_transcriber
        elif fast_transcriber is None and settings.whisper.vad_filter and settings.whisper.retry_without_vad:
            # Same weights as fast_transcriber via WhisperModelCache (the VAD flag is
            # not part of the cache key), so this costs no extra memory or load time.
            no_vad = settings.whisper.model_copy(update={"vad_filter": False})
            self.retry_transcriber = FasterWhisperTranscriber(settings.whisper.fast_model, no_vad)
        else:
            self.retry_transcriber = None
        self.verifiers = build_default_verifiers(settings) if verifiers is None else verifiers
        self.frame_extractor = frame_extractor or FrameExtractor(settings.frame, settings.audio.ffmpeg_binary)
        self.face_detector = face_detector or FaceDetector(settings.face_detection)

    # ------------------------------------------------------------------ #
    def warm_up(self) -> None:
        """Load ASR models up-front so job timings do not include model loading."""
        t0 = time.perf_counter()
        self.fast_transcriber.warm_up()
        for verifier in self.verifiers:
            transcriber = getattr(verifier, "_transcriber", None)
            if isinstance(transcriber, Transcriber):
                transcriber.warm_up()
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
        # ---- input --------------------------------------------------------
        self._begin(PipelineStage.INPUT, "Validating input")
        matcher = StreamingMatcher(self.req.dialogue, self.settings.matching)  # InvalidDialogueError
        if is_url(self.req.source):
            validate_url(self.req.source)  # InvalidURLError
        target: TargetDialogue = matcher.target
        self._end(PipelineStage.INPUT)

        # ---- download (cheap search media) --------------------------------
        self._begin(PipelineStage.DOWNLOAD, "Fetching audio for search")
        media = self.p.downloader.fetch_search_media(
            self.req.source, self.work_dir, progress=self.progress, reuse_existing=self.req.reuse_cached_media
        )
        self._end(PipelineStage.DOWNLOAD)
        self._check_cancel()

        # ---- audio --------------------------------------------------------
        self._begin(PipelineStage.AUDIO, "Extracting audio")
        audio = self.p.audio_extractor.extract(
            media, self.work_dir, progress=self.progress, reuse_existing=self.req.reuse_cached_media
        )
        samples = load_pcm(audio.path, self.settings.audio.sample_rate)
        self._end(PipelineStage.AUDIO)
        self._check_cancel()

        # ---- transcription + matching (streamed) --------------------------
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

        result = LocalizationResult(
            status=ResultStatus.FOUND if match else ResultStatus.NOT_FOUND,
            dialogue=target.raw,
            source_url=self.req.source,
            video=None,  # full-quality video is fetched only after a match
            first_pass=match,
            match=match,
            near_misses=matcher.near_misses,
            warnings=self.warnings,
            stage_timings=self.timings,
            transcribed_seconds=last_word_end,
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

        # ---- verification (audio-only; the video clip does not exist yet) --
        self._begin(PipelineStage.VERIFICATION, "Verifying with larger model")
        context = VerificationContext(
            dialogue=target.raw,
            audio_samples=samples,
            audio_path=audio.path,
            sample_rate=self.settings.audio.sample_rate,
            video=None,
        )
        current = match
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
        self._check_cancel()

        # ---- download (video clip around the verified match) --------------
        # Only now is the timestamp final, so only these few seconds of full-
        # quality video are downloaded. A local audio-only source can be
        # searched but has no frames to offer; a URL always gets the clip (its
        # search media may be audio-only even though the source has video).
        if is_url(self.req.source) or media.has_video:
            self._begin(PipelineStage.DOWNLOAD_VIDEO, "Fetching video clip")
            video: VideoInfo = self.p.downloader.fetch_video_clip(
                self.req.source,
                current.start,
                current.end,
                self.work_dir,
                progress=self._retag(PipelineStage.DOWNLOAD_VIDEO),
                reuse_existing=self.req.reuse_cached_media,
            )
            self._end(PipelineStage.DOWNLOAD_VIDEO)
            result.video = video
            self._check_cancel()

            # ---- frame ----------------------------------------------------
            self._begin(PipelineStage.FRAME, "Extracting frame")
            self.output_dir.mkdir(parents=True, exist_ok=True)
            result.frame = self.p.frame_extractor.extract(video, current.start, self.output_dir / FRAME_FILENAME)
            self._end(PipelineStage.FRAME)
            self._check_cancel()

            # ---- face detection (V2) --------------------------------------
            # Runs on the saved output frame and gates the verdict: a match with
            # no face on camera is reported as NOT_ONSCREEN (details are kept so
            # the caller can inspect what was found). Fails open: if the check
            # itself cannot run, the localisation stands, with a warning.
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
                    message = "No face visible - not an onscreen dialogue"
                    logger.info("No face in frame %s; verdict: not_onscreen", result.frame.image_path.name)
                self._emit(PipelineStage.FACE_DETECTION, message, 1.0, detected.to_dict())
            self._end(PipelineStage.FACE_DETECTION)
        else:
            warning = "Source has no video stream; returning the timestamp without a frame."
            self.warnings.append(warning)
            logger.warning(warning)
        return result

    # ------------------------------------------------------------------ #
    def _scan(
        self, transcriber: Transcriber, samples: Any, matcher: StreamingMatcher
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
