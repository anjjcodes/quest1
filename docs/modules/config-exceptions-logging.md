# Module: `config.py`, `exceptions.py`, `logging_config.py`

The foundation layer. Everything else imports from here; these three import nothing from the
rest of the package.

---

## `config.py` — settings

### Structure

```python
class Settings(BaseSettings):            # pydantic-settings, env prefix DL_, nested delimiter __
    download: DownloadConfig             # yt-dlp: max_height, search_max_height, clip_padding, retries, timeout
    audio: AudioConfig                   # ffmpeg: sample_rate, channels, binaries, timeout
    whisper: WhisperConfig               # fast_model, verify_model, device, compute_type, language, VAD…
    matching: MatchingConfig             # match_threshold, window_tolerance, min_dialogue_words, top_k
    verification: VerificationConfig     # enabled, search_window_seconds, skip_above_score, max_score_drop
    frame: FrameConfig                   # image_format, jpeg_quality
    face_detection: FaceDetectionConfig  # V2: enabled, min_detection_confidence, model path/url
    mouth_movement: MouthMovementConfig  # V3: enabled, movement_threshold, min_face_frames, model path/url
    storage: StorageConfig               # work_dir, output_dir, keep_intermediate
    server: ServerConfig                 # host, port, max_concurrent_jobs, job_retention_seconds
    logging: LoggingConfig               # level, fmt

    def ensure_directories(self) -> None

def get_settings() -> Settings           # lru_cache(maxsize=1)
def reset_settings_cache() -> None       # for tests that change env vars
```

Full field list with defaults: [Configuration](../04-configuration.md).

### Design

* **One sub-model per stage.** A stage's constructor takes only its own sub-model
  (`AudioExtractor(AudioConfig)`, `StreamingMatcher(dialogue, MatchingConfig)`). Tests build the
  sub-model directly with custom values; no environment juggling.
* **Fields carry a `Field(description=…)`** wherever the meaning is not obvious from the name, so
  documentation can be generated from the model (that is how the reference table in doc 4 is
  produced). Self-explanatory leaves such as `audio.channels` or `server.port` leave it empty.
* **Validation at the edges**: `ge/le` bounds, `Literal` enums, a `field_validator` that expands
  `~` in paths. A typo like `DL_MATCHING__MATCH_THRESHOLD=150` fails at startup.
* `compute_type` defaults to `int8` because ctranslate2 on Apple Silicon/CPU cannot run float16
  and silently converts to float32 (slow); int8 is the fast path on CPU and still valid on CUDA.
* `keep_intermediate=True` by default: repeated runs on one video (different dialogue) reuse the
  download and the audio.

---

## `exceptions.py` — error hierarchy

```
DialogueLocatorError(message, *, details)    .stage  .to_dict()
├── InvalidInputError            stage="input"
│   ├── InvalidURLError
│   └── InvalidDialogueError
├── ConfigurationError           stage="config"      (missing ffmpeg, etc.)
├── DownloadError                stage="download"
│   └── UnsupportedVideoError
├── AudioExtractionError         stage="audio"
├── TranscriptionError           stage="transcription"
├── VerificationError            stage="verification"
├── PipelineCancelledError       stage set to the stage being run when cancelled
├── FrameExtractionError         stage="frame"
├── FaceDetectionError           stage="face_detection"   (V2; fails open in the pipeline)
└── MouthMovementError           stage="mouth_movement"   (V3; fails open in the pipeline)
```

### Contract

* Every stage raises the most specific subclass. Third-party exceptions (yt-dlp, ffmpeg,
  ctranslate2, cv2, `wave`) are caught **at the stage boundary** and re-raised `from` the
  original, so logs keep the real traceback while callers see one stable type.
* `to_dict()` → `{"error": <class>, "stage": …, "message": …, "details": {...}}`. The API returns
  exactly this JSON; the CLI prints `ERROR [stage]: message`; the UI turns `stage` into a hint.
* Before raising, every boundary **logs** — `WARNING` for user/input problems, `ERROR` for
  system/tool failures — so a failure is visible in the server log even when the caller swallows it.
* "Not found" is never an exception: the pipeline returns a
  `LocalizationResult(status=NOT_FOUND)` with near-misses, because a legitimate answer is not a crash.

---

## `logging_config.py`

```python
def configure_logging(config: LoggingConfig | None = None) -> None
```

* Installs one stderr handler on the root logger with `LoggingConfig.fmt`; idempotent (its own
  handlers are tagged and replaced on re-call, so uvicorn reloads don't duplicate lines).
* Pins noisy third-party loggers (`faster_whisper`, `yt_dlp`, `httpx`, `httpcore`, `urllib3`,
  `numba`) to `WARNING` unless the app itself is at `DEBUG`.
* Every module uses `logging.getLogger(__name__)`, so lines are attributable:
  `dialogue_locator.transcription.faster_whisper | [faster_whisper:base] transcribing 1095.4s…`.

Logging conventions used across the codebase:

| Level | Used for |
|---|---|
| `DEBUG` | commands executed (ffmpeg/ffprobe argv), per-segment ASR text, per-window best scores, cache hits |
| `INFO` | stage begin/end with timing, match found, verifier decision, files written, job lifecycle |
| `WARNING` | rejected input, verifier rejected/failed, fallbacks taken (ffmpeg frame path, even-spread word timing), clamping |
| `ERROR` | anything that raises a `DialogueLocatorError` from a tool/system failure |

yt-dlp's own messages are routed through a `_YtDlpLoggerAdapter` into the `yt_dlp` logger so they
obey the same configuration.
