"""Pipeline orchestration tests with fake stages (real ffmpeg/OpenCV for audio + frame)."""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from pathlib import Path

import pytest

from dialogue_locator.config import Settings
from dialogue_locator.exceptions import (
    DialogueLocatorError,
    DownloadError,
    FaceDetectionError,
    FrameExtractionError,
    InvalidDialogueError,
    InvalidURLError,
    MouthMovementError,
    PipelineCancelledError,
)
from dialogue_locator.models import (
    FaceBox,
    FaceDetectionResult,
    MatchCandidate,
    MediaInfo,
    MouthMovementResult,
    PipelineStage,
    ProgressEvent,
    ResultStatus,
    VerificationOutcome,
    VerificationStatus,
    VideoInfo,
    Word,
)
from dialogue_locator.pipeline import DialoguePipeline, PipelineRequest, save_result
from dialogue_locator.transcription.base import Transcriber
from dialogue_locator.verification.base import Verifier
from tests.conftest import expect_error, requires_ffmpeg

logger = logging.getLogger("tests")
pytestmark = requires_ffmpeg
DIALOGUE = "My mind rebels at stagnation"


# --------------------------------------------------------------------------- #
# fakes
# --------------------------------------------------------------------------- #
class FakeDownloader:
    def __init__(self, video_path: Path, error: Exception | None = None, has_video: bool = True):
        self.video_path = video_path
        self.error = error
        self.has_video = has_video  # of the cheap search media
        self.calls = 0  # search fetches
        self.video_calls = 0  # clip fetches
        self.clip_ranges: list[tuple[float, float]] = []

    def fetch_search_media(self, source, dest_dir, progress=None, reuse_existing=True):
        self.calls += 1
        if self.error:
            raise self.error
        if progress:
            progress(ProgressEvent(PipelineStage.DOWNLOAD, "fake search download", 1.0))
        return MediaInfo(path=self.video_path, source_url=source, duration=3.0, has_video=self.has_video)

    def fetch_video_clip(self, source, start, end, dest_dir, progress=None, reuse_existing=True):
        self.video_calls += 1
        self.clip_ranges.append((start, end))
        if self.error:
            raise self.error
        if progress:
            progress(ProgressEvent(PipelineStage.DOWNLOAD, "fake clip download", 1.0))
        # Serves the whole sample file as the "clip" (clip_start 0), like a local source.
        return VideoInfo(path=self.video_path, source_url=source, fps=25.0, duration=3.0, frame_count=75)


class ScriptedTranscriber(Transcriber):
    """Yields a scripted transcript; counts how many words were actually pulled."""

    name = "fake:fast"

    def __init__(self, text: str, step: float = 0.1):
        self.text, self.step = text, step
        self.pulled = 0
        self.warmed = 0

    def warm_up(self):
        self.warmed += 1

    def transcribe(self, audio, *, offset=0.0, progress=None) -> Iterator[Word]:
        for i, tok in enumerate(self.text.split()):
            self.pulled += 1
            t = offset + i * self.step
            yield Word(tok, t, t + self.step * 0.8)


class ScriptedVerifier(Verifier):
    name = "scripted"

    def __init__(self, status: VerificationStatus, shift: float = 0.0, score: float = 100.0):
        self.status, self.shift, self.score = status, shift, score
        self.seen: list[MatchCandidate] = []
        self.contexts = []

    def verify(self, candidate, context):
        self.seen.append(candidate)
        self.contexts.append(context)
        refined = None
        if self.status in (VerificationStatus.CONFIRMED, VerificationStatus.REJECTED):
            refined = MatchCandidate(
                score=self.score,
                start=candidate.start + self.shift,
                end=candidate.end + self.shift,
                matched_text=candidate.matched_text + " (refined)",
            )
        return VerificationOutcome(self.name, self.status, score=self.score, refined=refined, message="scripted")


ONE_FACE = (FaceBox(x=10, y=10, width=50, height=50, confidence=0.9),)


class FakeFaceDetector:
    """Answers both entry points the pipeline uses: ``detect_file`` for the V2
    stage's saved frame, ``detect`` for the frames the mouth stage probes when
    moving the answer frame to the speaker.

    ``per_call`` scripts one answer per call across both, clamping to the last
    entry, so a test can say "nothing on the line's first frame, a face on the
    next one probed".
    """

    def __init__(
        self,
        faces: tuple[FaceBox, ...] = ONE_FACE,
        error: Exception | None = None,
        per_call: list[tuple[FaceBox, ...] | Exception] | None = None,
    ):
        self.faces = faces
        self.error = error
        self.per_call = per_call
        self.paths: list[Path] = []  # detect_file calls only
        self.calls = 0  # both entry points

    def detect_file(self, path: Path) -> FaceDetectionResult:
        self.paths.append(path)
        return self._answer()

    def detect(self, image, log_result: bool = True) -> FaceDetectionResult:
        return self._answer()

    def _answer(self) -> FaceDetectionResult:
        self.calls += 1
        faces = self.faces
        if self.per_call:
            faces = self.per_call[min(self.calls, len(self.per_call)) - 1]
        if isinstance(faces, Exception):
            raise faces
        if self.error:
            raise self.error
        return FaceDetectionResult(faces=faces, image_width=320, image_height=240)


def mouth_result(
    moving: bool | None,
    score: float | None = 0.09,
    movement_start: float | None = None,
    face_start: float | None = None,
    frames_with_face: int | None = None,
) -> MouthMovementResult:
    return MouthMovementResult(
        moving=moving, movement_score=score, threshold=0.02,
        frames_analyzed=26,
        frames_with_face=(26 if moving is not None else 2)
        if frames_with_face is None
        else frames_with_face,
        window_start=0.6, window_end=1.6, movement_start=movement_start,
        face_start=face_start,
    )


class FakeMouthAnalyzer:
    def __init__(self, result: MouthMovementResult | None = None, error: Exception | None = None):
        self.result = result or mouth_result(moving=True)
        self.error = error
        self.calls: list[tuple[VideoInfo, float, float]] = []

    def analyze(self, video: VideoInfo, start: float, end: float) -> MouthMovementResult:
        self.calls.append((video, start, end))
        if self.error:
            raise self.error
        return self.result


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(storage={"work_dir": tmp_path / "work", "output_dir": tmp_path / "out"})


def make_pipeline(
    settings: Settings,
    sample_video: Path,
    transcript: str,
    verifiers=None,
    downloader=None,
    retry_transcriber=None,
    face_detector=None,
    mouth_analyzer=None,
) -> tuple[DialoguePipeline, ScriptedTranscriber]:
    tr = ScriptedTranscriber(transcript)
    p = DialoguePipeline(
        settings,
        downloader=downloader or FakeDownloader(sample_video),
        fast_transcriber=tr,
        verifiers=[] if verifiers is None else verifiers,
        retry_transcriber=retry_transcriber,
        face_detector=face_detector or FakeFaceDetector(),
        mouth_analyzer=mouth_analyzer or FakeMouthAnalyzer(),
    )
    return p, tr


TRANSCRIPT = "I need work give me problems My mind rebels at stagnation give me work and so on and on"


# --------------------------------------------------------------------------- #
# happy path
# --------------------------------------------------------------------------- #
def test_found_end_to_end(settings, sample_video, tmp_path):
    verifier = ScriptedVerifier(VerificationStatus.CONFIRMED, shift=-0.02)
    dl = FakeDownloader(sample_video)
    p, tr = make_pipeline(settings, sample_video, TRANSCRIPT, verifiers=[verifier], downloader=dl)
    events: list[ProgressEvent] = []
    req = PipelineRequest(source="https://example.com/v", dialogue=DIALOGUE, job_id="job1")
    result = p.run(req, progress=events.append)

    assert result.status is ResultStatus.FOUND and result.found
    assert result.first_pass.matched_text == "My mind rebels at stagnation"
    assert result.first_pass.start == pytest.approx(0.6)  # word 6 at 0.1 s steps
    # verifier refined by -0.02 s and its result is the final match
    assert result.match.start == pytest.approx(0.58)
    assert result.match.matched_text.endswith("(refined)")
    assert result.verifications[0].status is VerificationStatus.CONFIRMED
    assert result.warnings == []
    # frame at refined time: floor(0.58 * 25) = 14
    assert result.frame is not None and result.frame.frame_number == 14
    assert result.frame.image_path.is_file()
    assert result.frame.image_path.parent == settings.storage.output_dir / "job1"
    assert result.timestamp == "00:00:00.560"  # frame 14 / 25 fps
    # early stop: match at word index 10, settled over 2 more -> 11 words + 2 = 13 pulled, not all 20
    assert tr.pulled < len(TRANSCRIPT.split())
    assert result.transcribed_seconds == pytest.approx((tr.pulled - 1) * 0.1 + 0.08, abs=1e-6)
    assert tr.warmed == 1
    # timings + progress ("download" = search fetch, "download_video" = full-quality fetch)
    for stage in ("input", "download", "download_video", "audio", "transcription", "verification", "frame",
                  "face_detection", "mouth_movement", "total"):
        assert stage in result.stage_timings
    stages = [e.stage for e in events]
    assert stages[0] is PipelineStage.INPUT and stages[-1] is PipelineStage.DONE
    assert PipelineStage.MATCHING in stages and PipelineStage.FRAME in stages
    assert result.video.source_url == "https://example.com/v"
    # verification is audio-only and runs before any video is downloaded ...
    assert stages.index(PipelineStage.VERIFICATION) < stages.index(PipelineStage.DOWNLOAD_VIDEO)
    # ... so the clip is requested around the *verified* (refined) timestamps
    assert dl.video_calls == 1
    assert dl.clip_ranges[0][0] == pytest.approx(0.58)
    assert dl.clip_ranges[0] == (result.match.start, result.match.end)


def test_save_result_json(settings, sample_video, tmp_path):
    p, _ = make_pipeline(settings, sample_video, TRANSCRIPT)
    result = p.run(PipelineRequest(source=str(sample_video), dialogue=DIALOGUE))
    out = save_result(result, tmp_path / "r" / "result.json")
    data = json.loads(out.read_text())
    assert data["status"] == "found" and data["frame_number"] == result.frame_number
    assert data["stage_timings"]["total"] > 0


# --------------------------------------------------------------------------- #
# not found
# --------------------------------------------------------------------------- #
def test_not_found_reports_near_misses(settings, sample_video):
    dl = FakeDownloader(sample_video)
    p, tr = make_pipeline(settings, sample_video, "the weather is nice and my mind wanders a lot today", downloader=dl)
    result = p.run(PipelineRequest(source=str(sample_video), dialogue=DIALOGUE))
    assert result.status is ResultStatus.NOT_FOUND and not result.found
    assert result.match is None and result.frame is None
    assert result.near_misses and all(c.score < 80 for c in result.near_misses)
    assert tr.pulled == 11  # whole stream consumed
    assert "verification" not in result.stage_timings and "frame" not in result.stage_timings
    assert result.timestamp is None
    # the expensive full-quality download never happened
    assert dl.video_calls == 0 and result.video is None
    assert "download_video" not in result.stage_timings
    d = result.to_dict()
    assert d["near_misses"][0]["score"] == round(result.near_misses[0].score, 2)


# --------------------------------------------------------------------------- #
# VAD-off retry pass
# --------------------------------------------------------------------------- #
def test_vad_retry_finds_match_and_warns(settings, sample_video):
    # First pass "hears" only what the VAD let through; the retry hears the line.
    retry = ScriptedTranscriber(TRANSCRIPT)
    p, fast = make_pipeline(
        settings, sample_video, "loud music and explosions only", retry_transcriber=retry
    )
    result = p.run(PipelineRequest(source=str(sample_video), dialogue=DIALOGUE))
    assert result.status is ResultStatus.FOUND
    assert fast.pulled == 5  # first pass consumed fully, found nothing
    assert retry.pulled < len(TRANSCRIPT.split())  # retry still stops early
    assert result.match.start == pytest.approx(0.6)
    assert any("voice-activity" in w for w in result.warnings)


def test_vad_retry_miss_reports_near_misses_from_better_pass(settings, sample_video):
    retry = ScriptedTranscriber("my mind is elsewhere during stagnation season")
    p, fast = make_pipeline(settings, sample_video, "completely unrelated words here", retry_transcriber=retry)
    result = p.run(PipelineRequest(source=str(sample_video), dialogue=DIALOGUE))
    assert result.status is ResultStatus.NOT_FOUND
    assert fast.pulled == 4 and retry.pulled == 7  # both passes consumed fully
    # near misses come from the retry pass, which scored higher
    assert result.near_misses and "stagnation" in result.near_misses[0].matched_text
    assert not result.warnings  # the "found only without VAD" warning is for hits only


def test_no_vad_retry_when_not_configured(settings, sample_video):
    p, tr = make_pipeline(settings, sample_video, "completely unrelated words here")
    assert p.retry_transcriber is None
    result = p.run(PipelineRequest(source=str(sample_video), dialogue=DIALOGUE))
    assert result.status is ResultStatus.NOT_FOUND
    assert tr.pulled == 4  # single pass only


def test_default_pipeline_builds_no_vad_retry_transcriber(settings, sample_video):
    from dialogue_locator.transcription.faster_whisper import FasterWhisperTranscriber

    # Default build (no injected transcriber): retry twin exists, same model, VAD off.
    p = DialoguePipeline(settings, downloader=FakeDownloader(sample_video), verifiers=[])
    assert isinstance(p.retry_transcriber, FasterWhisperTranscriber)
    assert p.retry_transcriber._config.vad_filter is False
    assert p.fast_transcriber._config.vad_filter is True
    # both search transcribers decode greedily; the verify pass keeps the full beam
    assert p.fast_transcriber._config.beam_size == settings.whisper.fast_beam_size == 1
    assert p.retry_transcriber._config.beam_size == 1
    from dialogue_locator.pipeline.pipeline import build_default_verifiers
    assert build_default_verifiers(settings)[0]._transcriber._config.beam_size == 5

    # Disabled via config, or pointless because the VAD is already off: no retry twin.
    s_off = settings.model_copy(update={"whisper": settings.whisper.model_copy(update={"retry_without_vad": False})})
    assert DialoguePipeline(s_off, downloader=FakeDownloader(sample_video), verifiers=[]).retry_transcriber is None
    s_novad = settings.model_copy(update={"whisper": settings.whisper.model_copy(update={"vad_filter": False})})
    assert DialoguePipeline(s_novad, downloader=FakeDownloader(sample_video), verifiers=[]).retry_transcriber is None


def test_default_verify_transcriber_runs_without_vad(settings):
    from dialogue_locator.pipeline.pipeline import build_default_verifiers

    verifiers = build_default_verifiers(settings)
    assert verifiers and verifiers[0]._transcriber._config.vad_filter is False


# --------------------------------------------------------------------------- #
# two-phase download
# --------------------------------------------------------------------------- #
def test_local_audio_only_source_found_without_frame(settings, sample_wav):
    # A local audio file can be searched; there is just no frame to extract.
    dl = FakeDownloader(sample_wav, has_video=False)
    p, _ = make_pipeline(settings, sample_wav, TRANSCRIPT, downloader=dl)
    result = p.run(PipelineRequest(source=str(sample_wav), dialogue=DIALOGUE))
    assert result.found
    assert result.frame is None and result.video is None
    assert dl.video_calls == 0
    assert any("no video stream" in w for w in result.warnings)
    assert result.timestamp == result.match.timestamp


def test_url_with_audio_only_search_media_still_fetches_video(settings, sample_video):
    # An audio-only *download* does not mean the *source* has no video: URLs
    # always get the full-quality fetch once a match is found.
    dl = FakeDownloader(sample_video, has_video=False)
    p, _ = make_pipeline(settings, sample_video, TRANSCRIPT, downloader=dl)
    result = p.run(PipelineRequest(source="https://example.com/v", dialogue=DIALOGUE))
    assert result.found
    assert dl.calls == 1 and dl.video_calls == 1
    assert result.frame is not None and result.video is not None


# --------------------------------------------------------------------------- #
# face detection (V2)
# --------------------------------------------------------------------------- #
def test_face_detected_on_extracted_frame(settings, sample_video):
    fd = FakeFaceDetector()
    p, _ = make_pipeline(settings, sample_video, TRANSCRIPT, face_detector=fd)
    events: list[ProgressEvent] = []
    req = PipelineRequest(source=str(sample_video), dialogue=DIALOGUE)
    result = p.run(req, progress=events.append)
    assert result.face_present is True
    assert result.face_detection.faces == ONE_FACE
    assert fd.paths == [result.frame.image_path]  # ran on the saved output frame
    face_events = [e for e in events if e.stage is PipelineStage.FACE_DETECTION]
    assert any("face(s) visible" in e.message for e in face_events)
    d = result.to_dict()
    assert d["face_present"] is True and d["face_detection"]["face_count"] == 1


def test_no_face_makes_result_not_onscreen(settings, sample_video):
    no_faces = FakeFaceDetector(faces=())
    p, _ = make_pipeline(settings, sample_video, TRANSCRIPT, face_detector=no_faces)
    result = p.run(PipelineRequest(source=str(sample_video), dialogue=DIALOGUE))
    assert result.status is ResultStatus.NOT_ONSCREEN and not result.found
    assert result.face_present is False and result.face_detection.face_present is False
    # the localisation details are kept for transparency
    assert result.match is not None and result.frame is not None and result.timestamp is not None
    assert result.warnings == []  # "no face" is a verdict, not a failure
    d = result.to_dict()
    assert d["status"] == "not_onscreen" and d["face_present"] is False


def test_face_detection_failure_warns_and_keeps_result(settings, sample_video):
    fd = FakeFaceDetector(error=FaceDetectionError("model download failed"))
    p, _ = make_pipeline(settings, sample_video, TRANSCRIPT, face_detector=fd)
    logger.info(">>> expecting a WARNING: face detector failed, localisation kept")
    result = p.run(PipelineRequest(source=str(sample_video), dialogue=DIALOGUE))
    # fails open: a face-check outage never hides a real localisation
    assert result.status is ResultStatus.FOUND and result.frame is not None
    assert result.face_detection is None and result.face_present is None
    assert any("Face check failed" in w for w in result.warnings)
    assert result.to_dict()["face_present"] is None
    logger.info("<<< warning recorded as expected")


def test_no_face_detection_without_frame(settings, sample_wav):
    fd = FakeFaceDetector()
    dl = FakeDownloader(sample_wav, has_video=False)
    p, _ = make_pipeline(settings, sample_wav, TRANSCRIPT, downloader=dl, face_detector=fd)
    result = p.run(PipelineRequest(source=str(sample_wav), dialogue=DIALOGUE))
    assert result.found and result.frame is None
    assert fd.paths == [] and result.face_detection is None
    assert "face_detection" not in result.stage_timings


# --------------------------------------------------------------------------- #
# mouth movement (V3)
# --------------------------------------------------------------------------- #
def test_mouth_moving_keeps_found(settings, sample_video):
    ma = FakeMouthAnalyzer(mouth_result(moving=True))
    p, _ = make_pipeline(settings, sample_video, TRANSCRIPT, mouth_analyzer=ma)
    result = p.run(PipelineRequest(source=str(sample_video), dialogue=DIALOGUE))
    assert result.status is ResultStatus.FOUND
    assert result.mouth_moving is True and result.warnings == []
    # analysed the clip over the final match window
    video, start, end = ma.calls[0]
    assert video is result.video and (start, end) == (result.match.start, result.match.end)
    d = result.to_dict()
    assert d["mouth_moving"] is True and d["mouth_movement"]["movement_score"] == 0.09


def test_frame_moves_to_where_the_speaker_comes_on_camera(settings, sample_video):
    # The brief asks for the first frame where the character is on camera saying
    # the line. When the line opens on someone else, the frame at its start
    # shows the wrong person; the mouth check knows where the speaker appears.
    ma = FakeMouthAnalyzer(mouth_result(moving=True, movement_start=1.0))
    fd = FakeFaceDetector()
    p, _ = make_pipeline(settings, sample_video, TRANSCRIPT, face_detector=fd, mouth_analyzer=ma)
    result = p.run(PipelineRequest(source=str(sample_video), dialogue=DIALOGUE))

    assert result.status is ResultStatus.FOUND and result.warnings == []
    assert result.frame.frame_number == 25  # 1.0 s at 25 fps, not 0.6 s -> 15
    assert result.timestamp == "00:00:01.000"
    assert result.frame.image_path.is_file()
    # The localisation is untouched: the line still starts where it was heard.
    assert result.match.start == pytest.approx(0.6)
    assert result.to_dict()["mouth_movement"]["movement_start"] == 1.0


def test_moving_the_frame_re_runs_the_face_check_on_it(settings, sample_video):
    # Every visual field must describe the frame being shown; leaving the face
    # box from the old frame would describe the person who was not speaking.
    other_face = (FaceBox(x=200, y=100, width=60, height=60, confidence=0.5),)
    fd = FakeFaceDetector(per_call=[ONE_FACE, other_face])
    ma = FakeMouthAnalyzer(mouth_result(moving=True, movement_start=1.0))
    p, _ = make_pipeline(settings, sample_video, TRANSCRIPT, face_detector=fd, mouth_analyzer=ma)
    result = p.run(PipelineRequest(source=str(sample_video), dialogue=DIALOGUE))
    assert fd.calls == 2  # once on the line's first frame, once on the speaker's
    assert result.face_detection.faces == other_face


def test_moved_frame_never_reports_a_face_and_no_face_at_once(settings, sample_video):
    # The detector is marginal on exactly the faces this feature exists for
    # (one scored 0.30 against a 0.30 threshold), so the window's first frame
    # can come back faceless while the window scan says the mouth is moving.
    # The reported frame must be one the detector confirms, or the result reads
    # as a contradiction.
    fd = FakeFaceDetector(per_call=[ONE_FACE, (), (), ONE_FACE])
    ma = FakeMouthAnalyzer(mouth_result(moving=True, movement_start=1.0))
    p, _ = make_pipeline(settings, sample_video, TRANSCRIPT, face_detector=fd, mouth_analyzer=ma)
    result = p.run(PipelineRequest(source=str(sample_video), dialogue=DIALOGUE))
    assert result.status is ResultStatus.FOUND
    assert result.face_present is True  # agrees with "mouth moving"
    assert result.frame.frame_number == 27  # 1.0 s -> frame 25, skipped 25 and 26


def test_silent_face_frame_moves_off_the_card_the_line_opened_on(settings, sample_video):
    # A line can open over a title card and cut to the speaker immediately.
    # Reporting the card next to "mouth not moving" explains nothing - and made
    # the same line report two different reasons depending on whether its first
    # word was included. The frame moves to the face the verdict is about.
    fd = FakeFaceDetector(per_call=[(), ONE_FACE])
    ma = FakeMouthAnalyzer(mouth_result(moving=False, score=0.003, face_start=1.0))
    p, _ = make_pipeline(settings, sample_video, TRANSCRIPT, face_detector=fd, mouth_analyzer=ma)
    result = p.run(PipelineRequest(source=str(sample_video), dialogue=DIALOGUE))
    assert result.status is ResultStatus.NOT_ONSCREEN
    assert result.frame.frame_number == 25  # 1.0 s, where the face is
    assert result.face_present is True  # and the reason reads consistently


def test_silent_face_already_on_camera_keeps_its_frame(settings, sample_video):
    # The common case: the line opens on the speaker. Nothing to move.
    ma = FakeMouthAnalyzer(mouth_result(moving=False, score=0.003, face_start=1.0))
    fd = FakeFaceDetector()
    p, _ = make_pipeline(settings, sample_video, TRANSCRIPT, face_detector=fd, mouth_analyzer=ma)
    result = p.run(PipelineRequest(source=str(sample_video), dialogue=DIALOGUE))
    assert result.status is ResultStatus.NOT_ONSCREEN
    assert result.frame.frame_number == 15  # floor(0.6 * 25), untouched
    assert fd.calls == 1


def test_frame_stays_when_the_speaker_is_on_camera_from_the_start(settings, sample_video):
    # Movement found at the line's own first frame: nothing to move, and no
    # second extraction or face check to pay for.
    ma = FakeMouthAnalyzer(mouth_result(moving=True, movement_start=0.6))
    fd = FakeFaceDetector()
    p, _ = make_pipeline(settings, sample_video, TRANSCRIPT, face_detector=fd, mouth_analyzer=ma)
    result = p.run(PipelineRequest(source=str(sample_video), dialogue=DIALOGUE))
    assert result.frame.frame_number == 15  # floor(0.6 * 25)
    assert fd.calls == 1  # no second frame to check


def test_failed_frame_move_keeps_the_original_frame(settings, sample_video):
    # Fails open like the checks around it: a re-extraction that cannot run
    # must not lose the answer already in hand.
    ma = FakeMouthAnalyzer(mouth_result(moving=True, movement_start=1.0))
    p, _ = make_pipeline(settings, sample_video, TRANSCRIPT, mouth_analyzer=ma)

    def explode(video, timestamp, dest_path):
        raise FrameExtractionError("disk full")

    original = p.frame_extractor.extract
    p.frame_extractor.extract = lambda *a, **k: (
        explode(*a, **k) if a[1] > 0.9 else original(*a, **k)
    )
    logger.info(">>> expecting a WARNING: the speaker's frame could not be extracted")
    result = p.run(PipelineRequest(source=str(sample_video), dialogue=DIALOGUE))
    assert result.status is ResultStatus.FOUND
    assert result.frame.frame_number == 15  # the line's first frame still stands
    assert any("could not be extracted" in w for w in result.warnings)
    logger.info("<<< warning recorded as expected")


def test_still_mouth_makes_result_not_onscreen(settings, sample_video):
    ma = FakeMouthAnalyzer(mouth_result(moving=False, score=0.003))
    p, _ = make_pipeline(settings, sample_video, TRANSCRIPT, mouth_analyzer=ma)
    result = p.run(PipelineRequest(source=str(sample_video), dialogue=DIALOGUE))
    assert result.status is ResultStatus.NOT_ONSCREEN and not result.found
    assert result.face_present is True and result.mouth_moving is False
    assert result.match is not None and result.frame is not None  # details kept
    assert result.warnings == []


def test_indeterminate_mouth_fails_open_with_warning(settings, sample_video):
    # A face on camera throughout whose frames never line up into a scorable
    # run is genuinely unmeasured, so the localisation stands.
    ma = FakeMouthAnalyzer(mouth_result(moving=None, score=None, frames_with_face=20))
    p, _ = make_pipeline(settings, sample_video, TRANSCRIPT, mouth_analyzer=ma)
    logger.info(">>> expecting a WARNING: mouth movement indeterminate, verdict kept")
    result = p.run(PipelineRequest(source=str(sample_video), dialogue=DIALOGUE))
    assert result.status is ResultStatus.FOUND
    assert result.mouth_moving is None and result.mouth_movement is not None
    assert any("could not be judged" in w for w in result.warnings)
    logger.info("<<< warning recorded as expected")


def test_a_face_nothing_can_landmark_is_not_a_face(settings, sample_video):
    # Regression: BlazeFace fires on blurred rubble at 0.45-0.62, above what it
    # scores some real faces, so the face check alone reported "found" on a
    # frame with no face in it. A box the landmarker cannot land a mesh on for
    # even a handful of frames must not fail open.
    ma = FakeMouthAnalyzer(mouth_result(moving=None, score=None, frames_with_face=4))
    p, _ = make_pipeline(settings, sample_video, TRANSCRIPT, mouth_analyzer=ma)
    result = p.run(PipelineRequest(source=str(sample_video), dialogue=DIALOGUE))
    assert result.status is ResultStatus.NOT_ONSCREEN
    assert result.face_present is True  # V2 was fooled; the window scan was not
    assert result.warnings == []


def test_mouth_analyzer_failure_fails_open_with_warning(settings, sample_video):
    ma = FakeMouthAnalyzer(error=MouthMovementError("landmarker exploded"))
    p, _ = make_pipeline(settings, sample_video, TRANSCRIPT, mouth_analyzer=ma)
    logger.info(">>> expecting a WARNING: mouth analyzer failed, verdict kept")
    result = p.run(PipelineRequest(source=str(sample_video), dialogue=DIALOGUE))
    assert result.status is ResultStatus.FOUND and result.mouth_movement is None
    assert any("Mouth check failed" in w for w in result.warnings)
    logger.info("<<< warning recorded as expected")


def test_line_starting_off_camera_is_onscreen_once_the_speaker_appears(settings, sample_video):
    # A line can open on a title card or a cutaway and cut to the speaker a
    # frame later. Judging only the first frame called that "no face"; the
    # window scan finds the speaker and the verdict follows the whole line.
    ma = FakeMouthAnalyzer(mouth_result(moving=True, movement_start=1.0))
    fd = FakeFaceDetector(per_call=[(), ONE_FACE])  # nothing at the start, a face later
    p, _ = make_pipeline(settings, sample_video, TRANSCRIPT, face_detector=fd, mouth_analyzer=ma)
    result = p.run(PipelineRequest(source=str(sample_video), dialogue=DIALOGUE))

    assert result.status is ResultStatus.FOUND
    assert result.frame.frame_number == 25  # moved to where the speaker appears
    assert result.face_present is True  # re-checked on that frame
    assert ma.calls, "the mouth check must run even with no face in the first frame"


def test_no_face_anywhere_in_the_line_stays_not_onscreen(settings, sample_video):
    # The other side of the same change: with nothing to judge in the window
    # either, V2's verdict stands - and is not downgraded to a warning.
    ma = FakeMouthAnalyzer(mouth_result(moving=None, score=None))
    p, _ = make_pipeline(settings, sample_video, TRANSCRIPT,
                         face_detector=FakeFaceDetector(faces=()), mouth_analyzer=ma)
    result = p.run(PipelineRequest(source=str(sample_video), dialogue=DIALOGUE))
    assert result.status is ResultStatus.NOT_ONSCREEN
    assert result.mouth_movement is not None  # the window was actually scanned
    assert result.warnings == []


def test_unmovable_speaker_frame_does_not_claim_onscreen(settings, sample_video):
    # Mouth moving, first frame faceless, and the speaker's frame cannot be
    # extracted: claiming "onscreen" over a frame that shows no face would be a
    # lie, so the verdict stays not_onscreen.
    ma = FakeMouthAnalyzer(mouth_result(moving=True, movement_start=1.0))
    p, _ = make_pipeline(settings, sample_video, TRANSCRIPT,
                         face_detector=FakeFaceDetector(faces=()), mouth_analyzer=ma)

    original = p.frame_extractor.extract
    p.frame_extractor.extract = lambda *a, **k: (
        (_ for _ in ()).throw(FrameExtractionError("disk full")) if a[1] > 0.9 else original(*a, **k)
    )
    logger.info(">>> expecting a WARNING: the speaker's frame could not be extracted")
    result = p.run(PipelineRequest(source=str(sample_video), dialogue=DIALOGUE))
    assert result.status is ResultStatus.NOT_ONSCREEN
    assert any("could not be extracted" in w for w in result.warnings)
    logger.info("<<< warning recorded as expected")


def test_face_stage_disabled_skips_all_visual_checks(settings, sample_video):
    s = settings.model_copy(
        update={"face_detection": settings.face_detection.model_copy(update={"enabled": False})}
    )
    fd, ma = FakeFaceDetector(), FakeMouthAnalyzer()
    p, _ = make_pipeline(s, sample_video, TRANSCRIPT, face_detector=fd, mouth_analyzer=ma)
    result = p.run(PipelineRequest(source=str(sample_video), dialogue=DIALOGUE))
    assert result.status is ResultStatus.FOUND and result.frame is not None
    assert fd.paths == [] and ma.calls == []
    assert result.face_detection is None and result.mouth_movement is None
    assert "face_detection" not in result.stage_timings
    assert "mouth_movement" not in result.stage_timings


def test_mouth_stage_disabled_skips_only_mouth(settings, sample_video):
    s = settings.model_copy(
        update={"mouth_movement": settings.mouth_movement.model_copy(update={"enabled": False})}
    )
    ma = FakeMouthAnalyzer()
    p, _ = make_pipeline(s, sample_video, TRANSCRIPT, mouth_analyzer=ma)
    result = p.run(PipelineRequest(source=str(sample_video), dialogue=DIALOGUE))
    assert result.status is ResultStatus.FOUND
    assert result.face_present is True  # face check still ran
    assert ma.calls == [] and result.mouth_movement is None
    assert "mouth_movement" not in result.stage_timings


def test_face_check_failure_skips_mouth_check(settings, sample_video):
    # Fail-open face check leaves face_detection None: mouth check has no face
    # confirmation to build on, so it must not run.
    fd = FakeFaceDetector(error=FaceDetectionError("boom"))
    ma = FakeMouthAnalyzer()
    p, _ = make_pipeline(settings, sample_video, TRANSCRIPT, face_detector=fd, mouth_analyzer=ma)
    logger.info(">>> expecting a WARNING: face check failed")
    result = p.run(PipelineRequest(source=str(sample_video), dialogue=DIALOGUE))
    assert result.status is ResultStatus.FOUND
    assert ma.calls == [] and result.mouth_movement is None
    logger.info("<<< warning recorded as expected")


# --------------------------------------------------------------------------- #
# verification outcomes
# --------------------------------------------------------------------------- #
def test_rejected_verifier_keeps_first_pass_and_warns(settings, sample_video):
    v = ScriptedVerifier(VerificationStatus.REJECTED, shift=1.0, score=60)
    p, _ = make_pipeline(settings, sample_video, TRANSCRIPT, verifiers=[v])
    logger.info(">>> expecting a WARNING: verifier rejected, first-pass timestamp kept")
    result = p.run(PipelineRequest(source=str(sample_video), dialogue=DIALOGUE))
    assert result.found
    assert result.match is result.first_pass
    assert len(result.warnings) == 1 and "rejected" in result.warnings[0]
    assert result.frame.frame_number == 15  # floor(0.6 * 25)
    logger.info("<<< warning recorded as expected")


def test_failed_verifier_warns(settings, sample_video):
    p, _ = make_pipeline(settings, sample_video, TRANSCRIPT, verifiers=[ScriptedVerifier(VerificationStatus.FAILED)])
    logger.info(">>> expecting a WARNING: verifier failed")
    result = p.run(PipelineRequest(source=str(sample_video), dialogue=DIALOGUE))
    assert result.found and result.match is result.first_pass
    assert result.warnings and "failed" in result.warnings[0]
    logger.info("<<< warning recorded as expected")


def test_verifier_chain_passes_refined_candidate_along(settings, sample_video):
    v1 = ScriptedVerifier(VerificationStatus.CONFIRMED, shift=-0.1)
    v2 = ScriptedVerifier(VerificationStatus.CONFIRMED, shift=+0.3)  # e.g. V2 visual verifier
    p, _ = make_pipeline(settings, sample_video, TRANSCRIPT, verifiers=[v1, v2])
    result = p.run(PipelineRequest(source=str(sample_video), dialogue=DIALOGUE))
    assert v2.seen[0].start == pytest.approx(0.5)  # saw v1's refinement
    assert result.match.start == pytest.approx(0.8)
    assert len(result.verifications) == 2


def test_skipped_verifier_no_warning(settings, sample_video):
    p, _ = make_pipeline(settings, sample_video, TRANSCRIPT, verifiers=[ScriptedVerifier(VerificationStatus.SKIPPED)])
    result = p.run(PipelineRequest(source=str(sample_video), dialogue=DIALOGUE))
    assert result.warnings == [] and result.match is result.first_pass


# --------------------------------------------------------------------------- #
# errors
# --------------------------------------------------------------------------- #
def test_invalid_dialogue_fails_before_download(settings, sample_video):
    dl = FakeDownloader(sample_video)
    p, _ = make_pipeline(settings, sample_video, TRANSCRIPT, downloader=dl)
    with expect_error(InvalidDialogueError):
        p.run(PipelineRequest(source=str(sample_video), dialogue="   "))
    assert dl.calls == 0


def test_invalid_url_fails_before_download(settings, sample_video):
    dl = FakeDownloader(sample_video)
    p, _ = make_pipeline(settings, sample_video, TRANSCRIPT, downloader=dl)
    with expect_error(InvalidURLError):
        p.run(PipelineRequest(source="http://nohost", dialogue=DIALOGUE))
    assert dl.calls == 0


def test_download_error_propagates_with_stage(settings, sample_video):
    dl = FakeDownloader(sample_video, error=DownloadError("gone"))
    p, _ = make_pipeline(settings, sample_video, TRANSCRIPT, downloader=dl)
    with expect_error(DownloadError) as exc:
        p.run(PipelineRequest(source="https://example.com/v", dialogue=DIALOGUE))
    assert exc.value.stage == "download"


def test_unexpected_exception_is_wrapped_with_stage(settings, sample_video):
    class Exploding(ScriptedTranscriber):
        def transcribe(self, *a, **k):
            raise RuntimeError("kaboom")
            yield  # noqa: RET503

    p = DialoguePipeline(settings, downloader=FakeDownloader(sample_video), fast_transcriber=Exploding(""), verifiers=[])
    with expect_error(DialogueLocatorError, match="kaboom") as exc:
        p.run(PipelineRequest(source=str(sample_video), dialogue=DIALOGUE))
    assert exc.value.stage == "transcription"


def test_cancellation_during_transcription(settings, sample_video):
    p, tr = make_pipeline(settings, sample_video, TRANSCRIPT)
    calls = {"n": 0}

    def should_cancel():
        calls["n"] += 1
        return calls["n"] > 3

    logger.info(">>> expecting a WARNING: job cancelled mid-transcription")
    with pytest.raises(PipelineCancelledError) as exc:
        p.run(PipelineRequest(source=str(sample_video), dialogue=DIALOGUE), should_cancel=should_cancel)
    assert exc.value.stage == "transcription"
    assert tr.pulled < len(TRANSCRIPT.split())
    logger.info("<<< cancelled as expected")


# --------------------------------------------------------------------------- #
# caching / cleanup
# --------------------------------------------------------------------------- #
def test_media_cached_per_source_and_cleanup_when_not_keeping(settings, sample_video):
    p, _ = make_pipeline(settings, sample_video, TRANSCRIPT)
    r1 = PipelineRequest(source=str(sample_video), dialogue=DIALOGUE, job_id="a")
    r2 = PipelineRequest(source=str(sample_video), dialogue="give me problems", job_id="b")
    assert r1.source_key == r2.source_key
    p.run(r1)
    work = settings.storage.work_dir / r1.source_key
    assert (work / "audio.wav").is_file()  # kept (keep_intermediate=True default)

    s2 = settings.model_copy(update={"storage": settings.storage.model_copy(update={"keep_intermediate": False})})
    p2, _ = make_pipeline(s2, sample_video, TRANSCRIPT)
    p2.run(r2)
    assert not work.exists()
    assert (s2.storage.output_dir / "b" / "frame.jpg").is_file()


def test_warm_up_loads_fast_and_verifier_models(settings, sample_video):
    from dialogue_locator.config import MatchingConfig, VerificationConfig
    from dialogue_locator.verification.asr_verifier import AsrVerifier

    fast, big = ScriptedTranscriber(""), ScriptedTranscriber("")
    p = DialoguePipeline(settings, downloader=FakeDownloader(sample_video), fast_transcriber=fast,
                         verifiers=[AsrVerifier(big, MatchingConfig(), VerificationConfig())])
    p.warm_up()
    assert fast.warmed == 1 and big.warmed == 1
