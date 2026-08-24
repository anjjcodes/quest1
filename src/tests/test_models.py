from pathlib import Path

import pytest

from dialogue_locator.exceptions import DownloadError, NoMatchFoundError
from dialogue_locator.models import (
    FrameInfo,
    LocalizationResult,
    MatchCandidate,
    ResultStatus,
    VerificationOutcome,
    VerificationStatus,
    Word,
    format_timestamp,
)


@pytest.mark.parametrize(
    "seconds, expected",
    [
        (0, "00:00:00.000"),
        (-5, "00:00:00.000"),
        (1.9996, "00:00:02.000"),
        (59.9994, "00:00:59.999"),
        (3725.4567, "01:02:05.457"),
        (3599.9999, "01:00:00.000"),
    ],
)
def test_format_timestamp(seconds, expected):
    assert format_timestamp(seconds) == expected


def _candidate() -> MatchCandidate:
    words = (Word("My", 10.0, 10.2), Word("mind", 10.2, 10.5))
    return MatchCandidate(score=88.5, start=10.0, end=10.5, matched_text="My mind", words=words)


def test_candidate_to_dict_optionally_includes_words():
    c = _candidate()
    assert "words" not in c.to_dict()
    assert len(c.to_dict(include_words=True)["words"]) == 2
    assert c.timestamp == "00:00:10.000"


def test_result_not_found_shape():
    r = LocalizationResult(ResultStatus.NOT_FOUND, "x y", "http://v", near_misses=[_candidate()])
    d = r.to_dict()
    assert not r.found
    assert d["timestamp"] is None and d["frame_number"] is None
    assert len(d["near_misses"]) == 1


def test_result_timestamp_prefers_frame_over_match():
    c = _candidate()
    frame = FrameInfo(frame_number=250, timestamp=10.04, fps=25.0, image_path=Path("f.jpg"))
    r = LocalizationResult(ResultStatus.FOUND, "x y", "http://v", match=c)
    assert r.timestamp == "00:00:10.000"
    r.frame = frame
    assert r.timestamp == "00:00:10.040"
    assert r.frame_number == 250
    assert r.matched_text == "My mind"


def test_verification_outcome_serialises():
    v = VerificationOutcome("asr_large_model", VerificationStatus.REJECTED, score=70.123, message="drop")
    d = v.to_dict()
    assert d["status"] == "rejected" and d["score"] == 70.12 and d["refined"] is None


def test_exceptions_to_dict():
    e = DownloadError("boom", details={"url": "u"})
    assert e.to_dict() == {"error": "DownloadError", "stage": "download", "message": "boom", "details": {"url": "u"}}
    assert str(e) == "[download] boom"
    n = NoMatchFoundError(near_misses=[{"score": 1}])
    assert n.to_dict()["near_misses"] == [{"score": 1}]
