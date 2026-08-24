import json
from pathlib import Path

import pytest

from dialogue_locator import cli
from dialogue_locator.config import Settings, reset_settings_cache
from dialogue_locator.exceptions import DownloadError
from dialogue_locator.models import FrameInfo, LocalizationResult, MatchCandidate, ResultStatus


def _found() -> LocalizationResult:
    m = MatchCandidate(score=97.5, start=312.48, end=314.0, matched_text="My mind rebels at stagnation.")
    return LocalizationResult(
        ResultStatus.FOUND, "My mind rebels at stagnation", "https://ok.ru/video/1", match=m, first_pass=m,
        frame=FrameInfo(frame_number=7812, timestamp=312.48, fps=25.0, image_path=Path("out/frame.jpg")),
        transcribed_seconds=320.0,
    )


def _not_found() -> LocalizationResult:
    return LocalizationResult(
        ResultStatus.NOT_FOUND, "nothing here", "https://ok.ru/video/1",
        near_misses=[MatchCandidate(score=44.1, start=8.84, end=10.0, matched_text="that is why")],
        transcribed_seconds=3300.0,
    )


class FakePipeline:
    result = None
    error = None
    last_settings = None

    def __init__(self, settings):
        FakePipeline.last_settings = settings

    def run(self, request, progress=None, should_cancel=None):
        if progress:
            from dialogue_locator.models import PipelineStage, ProgressEvent
            progress(ProgressEvent(PipelineStage.DOWNLOAD, "x", 0.5))
        if self.error:
            raise self.error
        return self.result


@pytest.fixture(autouse=True)
def _isolated_settings(tmp_path, monkeypatch):
    monkeypatch.setenv("DL_STORAGE__OUTPUT_DIR", str(tmp_path / "out"))
    monkeypatch.setenv("DL_STORAGE__WORK_DIR", str(tmp_path / "work"))
    reset_settings_cache()
    yield
    reset_settings_cache()


def test_format_found():
    text = cli.format_result(_found())
    assert "Timestamp : 00:05:12.480" in text
    assert "Frame     : 7812" in text
    assert 'Text      : "My mind rebels at stagnation."' in text
    assert "Scanned   : 320.0s" in text


def test_format_not_found():
    text = cli.format_result(_not_found())
    assert 'Not found : "nothing here"' in text and "44.1" in text and "00:00:08.840" in text


def test_main_found_writes_result_json(capsys, tmp_path):
    FakePipeline.result, FakePipeline.error = _found(), None
    code = cli.main(["https://ok.ru/video/1", "My mind rebels at stagnation"], pipeline_factory=FakePipeline)
    out = capsys.readouterr()
    assert code == cli.EXIT_FOUND
    assert "Frame     : 7812" in out.out
    results = list((tmp_path / "out").glob("*/result.json"))
    assert len(results) == 1 and json.loads(results[0].read_text())["frame_number"] == 7812


def test_main_not_found_exit_code(capsys):
    FakePipeline.result, FakePipeline.error = _not_found(), None
    assert cli.main(["https://ok.ru/video/1", "nothing here", "-q"], pipeline_factory=FakePipeline) == cli.EXIT_NOT_FOUND
    assert "Not found" in capsys.readouterr().out


def test_main_error_exit_code(capsys):
    FakePipeline.result, FakePipeline.error = None, DownloadError("blocked")
    assert cli.main(["https://ok.ru/video/1", "some words", "--json"], pipeline_factory=FakePipeline) == cli.EXIT_ERROR
    out = capsys.readouterr()
    assert "ERROR [download]: blocked" in out.err
    assert json.loads(out.out)["stage"] == "download"


def test_main_json_output(capsys):
    FakePipeline.result, FakePipeline.error = _found(), None
    cli.main(["https://ok.ru/video/1", "some words", "--json", "-q"], pipeline_factory=FakePipeline)
    data = json.loads(capsys.readouterr().out)
    assert data["status"] == "found" and data["timestamp"] == "00:05:12.480"


def test_overrides_reach_settings():
    FakePipeline.result, FakePipeline.error = _found(), None
    cli.main(["u", "d", "-q", "--fast-model", "base", "--verify-model", "small", "--no-verify", "--threshold", "70",
              "--max-height", "480", "-v"], pipeline_factory=FakePipeline)
    s: Settings = FakePipeline.last_settings
    assert s.whisper.fast_model == "base" and s.whisper.verify_model == "small"
    assert s.verification.enabled is False and s.matching.match_threshold == 70
    assert s.download.max_height == 480 and s.logging.level == "DEBUG"
