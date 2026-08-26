import pytest
from pydantic import ValidationError

from dialogue_locator.config import (
    UNLIMITED_OCCURRENCES,
    MatchingConfig,
    Settings,
    get_settings,
    reset_settings_cache,
)


def test_defaults_load():
    s = Settings()
    assert s.whisper.fast_model == "base"
    assert s.whisper.verify_model == "small"
    assert s.matching.match_threshold == 80.0
    assert s.verification.search_window_seconds == 12.0
    assert s.verification.skip_above_score == 90.0
    assert s.whisper.fast_beam_size == 1 and s.whisper.beam_size == 5
    assert s.audio.sample_rate == 16_000


def test_env_override_nested(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("DL_MATCHING__MATCH_THRESHOLD", "85")
    monkeypatch.setenv("DL_WHISPER__FAST_MODEL", "small")
    reset_settings_cache()
    try:
        s = get_settings()
        assert s.matching.match_threshold == 85.0
        assert s.whisper.fast_model == "small"
    finally:
        reset_settings_cache()


def test_validation_rejects_out_of_range():
    with pytest.raises(ValidationError):
        MatchingConfig(match_threshold=150)
    with pytest.raises(ValidationError):
        Settings(server={"port": 0})


def test_max_occurrences_accepts_a_count_or_unlimited():
    assert MatchingConfig(max_occurrences=1).max_occurrences == 1
    assert MatchingConfig(max_occurrences=5).max_occurrences == 5
    # -1 means "evaluate every occurrence in the video"
    assert MatchingConfig(max_occurrences=UNLIMITED_OCCURRENCES).max_occurrences == -1


@pytest.mark.parametrize("value", [0, -2, -10])
def test_max_occurrences_rejects_zero_and_other_negatives(value):
    with pytest.raises(ValidationError, match="max_occurrences"):
        MatchingConfig(max_occurrences=value)


def test_get_settings_is_cached():
    reset_settings_cache()
    assert get_settings() is get_settings()
    reset_settings_cache()


def test_ensure_directories(tmp_path):
    s = Settings(storage={"work_dir": tmp_path / "w", "output_dir": tmp_path / "o"})
    s.ensure_directories()
    assert (tmp_path / "w").is_dir() and (tmp_path / "o").is_dir()
