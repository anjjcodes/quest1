"""Application configuration.

All tunable parameters live here, grouped by concern. Values can be overridden
without code changes via environment variables (prefix ``DL_``, nested keys
joined by ``__``) or a ``.env`` file in the working directory, e.g.::

    DL_WHISPER__FAST_MODEL=base
    DL_MATCHING__MATCH_THRESHOLD=85
    DL_STORAGE__WORK_DIR=/tmp/dialogue-locator

Modules must never hard-code these values; they receive a config object (or a
sub-config) via constructor / function argument so they can be unit-tested with
custom settings.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class DownloadConfig(BaseModel):
    """Settings for fetching the source video (yt-dlp)."""

    max_height: int = Field(
        720,
        ge=144,
        description="Maximum video height to download. Lower = faster download; "
        "the extracted frame will have this resolution.",
    )
    socket_timeout_seconds: int = Field(30, ge=1)
    retries: int = Field(3, ge=0)
    container: str = Field(
        "mp4",
        description="Preferred container. Ensures a format OpenCV/FFmpeg can seek in.",
    )


class AudioConfig(BaseModel):
    """Settings for audio extraction (FFmpeg)."""

    sample_rate: int = Field(16_000, description="Whisper expects 16 kHz mono PCM.")
    channels: int = Field(1, ge=1, le=2)
    ffmpeg_binary: str = "ffmpeg"
    ffprobe_binary: str = "ffprobe"
    timeout_seconds: int = Field(600, ge=1)


class WhisperConfig(BaseModel):
    """Settings for Faster-Whisper transcription.

    Two models are used:
    * ``fast_model``   - streamed over the audio from the start until the first
      match; speed matters more than accuracy here.
    * ``verify_model`` - larger model run only on a short window around a
      candidate match to confirm the text and tighten word timestamps.
    """

    fast_model: str = Field("small", description="Model for the streaming pass.")
    verify_model: str = Field("medium", description="Model for the verification pass.")
    device: Literal["auto", "cpu", "cuda"] = "auto"
    compute_type: str = Field(
        "int8",
        description="ctranslate2 compute type. int8 is fastest on CPU (incl. Apple Silicon); "
        "use float16 or int8_float16 on CUDA GPUs; float32 for maximum accuracy.",
    )
    cpu_threads: int = Field(0, ge=0, description="0 = let ctranslate2 decide.")
    language: str | None = Field(
        "en",
        description="Force a language (ISO 639-1) or None to auto-detect. Forcing "
        "avoids a detection step and mis-detections on music intros.",
    )
    beam_size: int = Field(5, ge=1)
    vad_filter: bool = Field(
        True,
        description="Skip silent regions. Speeds up long videos with quiet stretches.",
    )
    vad_min_silence_ms: int = Field(500, ge=0)
    condition_on_previous_text: bool = Field(
        False,
        description="False reduces hallucination loops in long streams.",
    )
    download_root: Path | None = Field(
        None,
        description="Where to cache model weights. None = Hugging Face default cache.",
    )


class MatchingConfig(BaseModel):
    """Settings for fuzzy matching of the transcript against the dialogue."""

    match_threshold: float = Field(
        80.0,
        ge=0,
        le=100,
        description="Minimum RapidFuzz score (0-100) for a window to count as a match.",
    )
    window_tolerance: int = Field(
        2,
        ge=0,
        description="Window sizes tried = dialogue word count +/- this many words, to "
        "absorb ASR word splits/merges.",
    )
    min_dialogue_words: int = Field(
        2,
        ge=1,
        description="Reject dialogues shorter than this: single words match too easily.",
    )
    top_k_near_misses: int = Field(
        3,
        ge=1,
        description="How many best-scoring windows to report when nothing crosses the "
        "threshold.",
    )


class VerificationConfig(BaseModel):
    """Settings for the second-pass (large model) verification stage."""

    enabled: bool = True
    search_window_seconds: float = Field(
        20.0,
        gt=0,
        description="Re-transcribe +/- this many seconds around the candidate.",
    )
    max_score_drop: float = Field(
        5.0,
        ge=0,
        description="Accept the verifier's timestamp only if its score is not more than "
        "this many points below the first-pass score.",
    )


class FrameConfig(BaseModel):
    """Settings for frame extraction."""

    image_format: Literal["jpg", "png"] = "jpg"
    jpeg_quality: int = Field(92, ge=1, le=100)


class StorageConfig(BaseModel):
    """Where intermediate and output files are written."""

    work_dir: Path = Field(
        Path("data/work"),
        description="Per-job scratch space: downloaded video, extracted audio.",
    )
    output_dir: Path = Field(
        Path("data/output"),
        description="Per-job results: extracted frame image and result JSON.",
    )
    keep_intermediate: bool = Field(
        True,
        description="Keep downloaded video/audio after a job. Repeated runs on the same URL "
        "(e.g. a different dialogue) then skip download and extraction. Set False to save disk.",
    )

    @field_validator("work_dir", "output_dir", mode="before")
    @classmethod
    def _expand(cls, value: str | Path) -> Path:
        return Path(value).expanduser()


class ServerConfig(BaseModel):
    """FastAPI / uvicorn settings."""

    host: str = "127.0.0.1"
    port: int = Field(8000, ge=1, le=65535)
    max_concurrent_jobs: int = Field(
        1,
        ge=1,
        description="Whisper is CPU/GPU heavy; more than 1 rarely helps on one machine.",
    )
    job_retention_seconds: int = Field(
        3600, ge=0, description="How long finished jobs stay queryable via the API."
    )


class LoggingConfig(BaseModel):
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    fmt: str = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"


class Settings(BaseSettings):
    """Root settings object. Use :func:`get_settings` to obtain the shared instance."""

    model_config = SettingsConfigDict(
        env_prefix="DL_",
        env_nested_delimiter="__",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    download: DownloadConfig = Field(default_factory=DownloadConfig)
    audio: AudioConfig = Field(default_factory=AudioConfig)
    whisper: WhisperConfig = Field(default_factory=WhisperConfig)
    matching: MatchingConfig = Field(default_factory=MatchingConfig)
    verification: VerificationConfig = Field(default_factory=VerificationConfig)
    frame: FrameConfig = Field(default_factory=FrameConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    server: ServerConfig = Field(default_factory=ServerConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    def ensure_directories(self) -> None:
        """Create storage directories if they do not exist."""
        self.storage.work_dir.mkdir(parents=True, exist_ok=True)
        self.storage.output_dir.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings instance (loaded once from env/.env)."""
    return Settings()


def reset_settings_cache() -> None:
    """Clear the cached settings. Intended for tests that change env vars."""
    get_settings.cache_clear()
