"""Pipeline orchestration tests with fake stages (real ffmpeg/OpenCV for audio + frame)."""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest

from dialogue_locator.config import Settings
from dialogue_locator.exceptions import (
    DialogueLocatorError,
    DownloadError,
    InvalidDialogueError,
    InvalidURLError,
    PipelineCancelledError,
)
from dialogue_locator.models import (
    MatchCandidate,
    MediaInfo,
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


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(storage={"work_dir": tmp_path / "work", "output_dir": tmp_path / "out"})


def make_pipeline(settings: Settings, sample_video: Path, transcript: str, verifiers=None, downloader=None) -> tuple[DialoguePipeline, ScriptedTranscriber]:
    tr = ScriptedTranscriber(transcript)
    p = DialoguePipeline(
        settings,
        downloader=downloader or FakeDownloader(sample_video),
        fast_transcriber=tr,
        verifiers=[] if verifiers is None else verifiers,
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
    for stage in ("input", "download", "download_video", "audio", "transcription", "verification", "frame", "total"):
        assert stage in result.stage_timings
    stages = [e.stage for e in events]
    assert stages[0] is PipelineStage.INPUT and stages[-1] is PipelineStage.DONE
    assert PipelineStage.MATCHING in stages and PipelineStage.FRAME in stages
    assert result.video.source_url == "https://example.com/v"
    # verification is audio-only and runs before any video is downloaded ...
    assert verifier.contexts[0].video is None
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
