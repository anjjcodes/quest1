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
    """Settings for fetching the source media (yt-dlp).

    Two fetches happen per job: a cheap *search* fetch (audio-only when the
    host offers it) that feeds transcription/matching/verification, and a
    *video* fetch at full quality that happens only after a match is confirmed
    and exists solely to extract the output frame.
    """

    max_height: int = Field(
        1080,
        ge=144,
        description="Height cap for the full-quality video fetch. This bounds the "
        "resolution of the extracted frame and the size of the one large download; "
        "it does not affect search speed (the search runs on audio).",
    )
    search_max_height: int = Field(
        360,
        ge=144,
        description="Fallback cap for the search fetch when the host has no "
        "audio-only stream (e.g. progressive-only hosts like ok.ru): the lowest "
        "video rendition no taller than this is used instead.",
    )
    clip_padding_seconds: float = Field(
        5.0,
        ge=0,
        description="Seconds of video downloaded either side of the verified match "
        "window by the clip fetch, so the target frame is safely inside the clip.",
    )
    progress_interval_seconds: float = Field(
        2.0,
        gt=0,
        description="Minimum seconds between download progress events. Time-based so a "
        "slow host still shows movement (MB, speed, ETA) instead of minutes-long silent "
        "percent buckets.",
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

    fast_model: str = Field(
        "base",
        description="Model for the streaming pass. 'base' runs ~12x realtime on Apple Silicon CPU "
        "(vs ~3x for 'small') with near-identical hit rate; the verify model fixes wording/timing.",
    )
    verify_model: str = Field(
        "small",
        description="Model for the verification pass. 'small' confirms/refines a "
        "fuzzy match about as reliably as 'medium' at ~3x the speed on CPU; bump "
        "to 'medium' for maximum wording accuracy.",
    )
    device: Literal["auto", "cpu", "cuda"] = "auto"
    compute_type: str = Field(
        "int8",
        description="ctranslate2 compute type. int8 is fastest on CPU (incl. Apple Silicon); "
        "use float16 or int8_float16 on CUDA GPUs; float32 for maximum accuracy.",
    )
    cpu_threads: int = Field(
        0,
        ge=0,
        description="Intra-op threads for CPU inference. 0 = use all CPU cores "
        "(ctranslate2's own default is only 4, leaving most of a modern CPU idle).",
    )
    language: str | None = Field(
        "en",
        description="Force a language (ISO 639-1) or None to auto-detect. Forcing "
        "avoids a detection step and mis-detections on music intros.",
    )
    beam_size: int = Field(
        5,
        ge=1,
        description="Beam width for the verification pass, where accuracy matters.",
    )
    fast_beam_size: int = Field(
        1,
        ge=1,
        description="Beam width for the streaming search pass. Greedy (1) decodes "
        "~1.5-2x faster than beam 5; the fuzzy matcher absorbs the slightly "
        "rougher wording and the verification pass re-checks it anyway.",
    )
    vad_filter: bool = Field(
        True,
        description="Skip silent regions. Speeds up long videos with quiet stretches.",
    )
    vad_min_silence_ms: int = Field(500, ge=0)
    retry_without_vad: bool = Field(
        True,
        description="If the streaming pass ends with no match while vad_filter is on, "
        "re-run the scan once with the VAD disabled before reporting not_found. Loud "
        "music/effects can make the VAD discard real speech (e.g. action scenes); the "
        "retry hears everything, at the cost of one extra pass paid only on a miss.",
    )
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
        12.0,
        gt=0,
        description="Re-transcribe +/- this many seconds around the candidate. "
        "Fast-pass timestamps are rarely off by more than a couple of seconds, "
        "so this bounds the (expensive) large-model transcription generously "
        "while keeping it short.",
    )
    skip_above_score: float | None = Field(
        90.0,
        ge=0,
        le=100,
        description="Skip the large-model re-transcription when the first pass "
        "already scored at least this: verification exists to check uncertain "
        "matches, not to re-prove near-perfect ones. Calibrated for the greedy "
        "fast pass, whose correct matches score ~94+ while the genuinely "
        "uncertain band sits at 80-90. None = always verify.",
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


class FaceDetectionConfig(BaseModel):
    """Settings for the V2 face-presence check (MediaPipe BlazeFace).

    The detector answers one question about the matched frame: is at least one
    human face visible? MediaPipe's Tasks API needs a small ``.tflite`` model
    file; like the Whisper weights it is fetched once on first use and cached
    at ``model_path``.
    """

    enabled: bool = Field(
        True,
        description="Run the face check on the matched frame. Disabling it also "
        "disables the mouth-movement check, which builds on a confirmed face.",
    )
    min_detection_confidence: float = Field(
        0.3,
        ge=0,
        le=1,
        description="Minimum BlazeFace score for a detection to count as a face. "
        "Calibrated on real frames: faces in cinematic mid/wide shots score as low "
        "as ~0.34 (close-ups ~0.97) while non-face junk stays at or below ~0.2, so "
        "0.3 separates them; the model's usual 0.5 default misses distant faces.",
    )
    model_path: Path = Field(
        Path("data/models/blaze_face_short_range.tflite"),
        description="Where the BlazeFace model file is cached. Downloaded from "
        "model_url on first use if missing (~230 KB, one-time).",
    )
    model_url: str = Field(
        "https://storage.googleapis.com/mediapipe-models/face_detector/"
        "blaze_face_short_range/float16/latest/blaze_face_short_range.tflite",
        description="Where to fetch the model file from when model_path is missing.",
    )
    download_timeout_seconds: int = Field(60, ge=1)

    @field_validator("model_path", mode="before")
    @classmethod
    def _expand(cls, value: str | Path) -> Path:
        return Path(value).expanduser()


class MouthMovementConfig(BaseModel):
    """Settings for the V3 mouth-movement check (MediaPipe Face Landmarker).

    Tracks the lip landmarks across the video frames of the matched dialogue
    window and decides whether the mouth moves significantly. The per-frame
    signal is *mouth openness*: the inner-lip gap (landmarks 13-14) divided by
    the mouth width (61-291), which is scale- and distance-invariant. The
    movement score is the standard deviation of that signal over the window;
    measured values: ~0.09 while speaking vs ~0.001 landmark jitter on a
    static face, so the default threshold sits well between the two.
    """

    enabled: bool = Field(
        True,
        description="Run the mouth-movement check. Only runs when the face "
        "check is enabled and confirmed a face.",
    )
    movement_threshold: float = Field(
        0.02,
        gt=0,
        description="Minimum standard deviation of mouth openness to count as "
        "significant movement.",
    )
    min_face_frames: int = Field(
        5,
        ge=2,
        description="Minimum frames with a detected face needed for a verdict; "
        "fewer makes the result indeterminate rather than a false 'not moving'.",
    )
    max_window_seconds: float = Field(
        10.0,
        gt=0,
        description="Cap on how much of the dialogue window is analysed; long "
        "lines are judged on their first seconds.",
    )
    min_face_confidence: float = Field(
        0.3,
        ge=0,
        le=1,
        description="Face detection/presence confidence for the landmarker, "
        "lowered from MediaPipe's 0.5 default for the same reason as "
        "face_detection.min_detection_confidence: faces in mid/wide shots "
        "score lower than close-ups.",
    )
    model_path: Path = Field(
        Path("data/models/face_landmarker.task"),
        description="Where the Face Landmarker model file is cached. Downloaded "
        "from model_url on first use if missing (~3.7 MB, one-time).",
    )
    model_url: str = Field(
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
        "face_landmarker/float16/latest/face_landmarker.task",
        description="Where to fetch the model file from when model_path is missing.",
    )
    download_timeout_seconds: int = Field(120, ge=1)

    @field_validator("model_path", mode="before")
    @classmethod
    def _expand(cls, value: str | Path) -> Path:
        return Path(value).expanduser()


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
    face_detection: FaceDetectionConfig = Field(default_factory=FaceDetectionConfig)
    mouth_movement: MouthMovementConfig = Field(default_factory=MouthMovementConfig)
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
