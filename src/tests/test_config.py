import pytest
from pydantic import ValidationError

from dialogue_locator.config import MatchingConfig, Settings, get_settings, reset_settings_cache


def test_defaults_load():
    s = Settings()
    assert s.whisper.fast_model == "small"
    assert s.whisper.verify_model == "medium"
    assert s.matching.match_threshold == 80.0
    assert s.verification.search_window_seconds == 20.0
    assert s.audio.sample_rate == 16_000


def test_env_override_nested(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("DL_MATCHING__MATCH_THRESHOLD", "85")
    monkeypatch.setenv("DL_WHISPER__FAST_MODEL", "base")
    reset_settings_cache()
    try:
        s = get_settings()
        assert s.matching.match_threshold == 85.0
        assert s.whisper.fast_model == "base"
    finally:
        reset_settings_cache()


def test_validation_rejects_out_of_range():
    with pytest.raises(ValidationError):
        MatchingConfig(match_threshold=150)
    with pytest.raises(ValidationError):
        Settings(server={"port": 0})


def test_get_settings_is_cached():
    reset_settings_cache()
    assert get_settings() is get_settings()
    reset_settings_cache()


def test_ensure_directories(tmp_path):
    s = Settings(storage={"work_dir": tmp_path / "w", "output_dir": tmp_path / "o"})
    s.ensure_directories()
    assert (tmp_path / "w").is_dir() and (tmp_path / "o").is_dir()
