# 4. Configuration

All tunables live in `src/dialogue_locator/config.py` as nested pydantic models under one
`Settings` object. Nothing in the pipeline hard-codes a model name, threshold or path.

## Precedence (highest wins)

1. CLI flags (`--fast-model`, `--threshold`, …) — applied with `Settings.model_copy` in `cli.py`
2. Environment variables `DL_<SECTION>__<FIELD>` (double underscore between levels)
3. A `.env` file in the working directory (same names)
4. Defaults below

Examples:

```bash
DL_WHISPER__FAST_MODEL=small DL_MATCHING__MATCH_THRESHOLD=85 dialogue-locator <url> "<line>"
echo 'DL_DOWNLOAD__MAX_HEIGHT=360' >> .env      # picked up by CLI and server alike
```

`get_settings()` returns a cached instance (`functools.lru_cache`); tests call
`reset_settings_cache()` after changing the environment. Values are validated on load — an
out-of-range threshold or unknown device is a startup error, not a runtime surprise.

## Reference (generated from the models)

| Env variable | Field | Default | Type | Description |
|---|---|---|---|---|
| `DL_DOWNLOAD__MAX_HEIGHT` | `download.max_height` | `720` | `int` | Maximum video height to download. Lower = faster download; the extracted frame will have this resolution. |
| `DL_DOWNLOAD__SOCKET_TIMEOUT_SECONDS` | `download.socket_timeout_seconds` | `30` | `int` |  |
| `DL_DOWNLOAD__RETRIES` | `download.retries` | `3` | `int` |  |
| `DL_DOWNLOAD__CONTAINER` | `download.container` | `mp4` | `str` | Preferred container. Ensures a format OpenCV/FFmpeg can seek in. |
| `DL_AUDIO__SAMPLE_RATE` | `audio.sample_rate` | `16000` | `int` | Whisper expects 16 kHz mono PCM. |
| `DL_AUDIO__CHANNELS` | `audio.channels` | `1` | `int` |  |
| `DL_AUDIO__FFMPEG_BINARY` | `audio.ffmpeg_binary` | `ffmpeg` | `str` |  |
| `DL_AUDIO__FFPROBE_BINARY` | `audio.ffprobe_binary` | `ffprobe` | `str` |  |
| `DL_AUDIO__TIMEOUT_SECONDS` | `audio.timeout_seconds` | `600` | `int` |  |
| `DL_WHISPER__FAST_MODEL` | `whisper.fast_model` | `base` | `str` | Model for the streaming pass. 'base' runs ~12x realtime on Apple Silicon CPU (vs ~3x for 'small') with near-identical hit rate; the verify model fixes wording/timing. |
| `DL_WHISPER__VERIFY_MODEL` | `whisper.verify_model` | `medium` | `str` | Model for the verification pass. |
| `DL_WHISPER__DEVICE` | `whisper.device` | `auto` | `Literal['auto', 'cpu', 'cuda']` |  |
| `DL_WHISPER__COMPUTE_TYPE` | `whisper.compute_type` | `int8` | `str` | ctranslate2 compute type. int8 is fastest on CPU (incl. Apple Silicon); use float16 or int8_float16 on CUDA GPUs; float32 for maximum accuracy. |
| `DL_WHISPER__CPU_THREADS` | `whisper.cpu_threads` | `0` | `int` | 0 = let ctranslate2 decide. |
| `DL_WHISPER__LANGUAGE` | `whisper.language` | `en` | `str | None` | Force a language (ISO 639-1) or None to auto-detect. Forcing avoids a detection step and mis-detections on music intros. |
| `DL_WHISPER__BEAM_SIZE` | `whisper.beam_size` | `5` | `int` |  |
| `DL_WHISPER__VAD_FILTER` | `whisper.vad_filter` | `True` | `bool` | Skip silent regions. Speeds up long videos with quiet stretches. |
| `DL_WHISPER__VAD_MIN_SILENCE_MS` | `whisper.vad_min_silence_ms` | `500` | `int` |  |
| `DL_WHISPER__CONDITION_ON_PREVIOUS_TEXT` | `whisper.condition_on_previous_text` | `False` | `bool` | False reduces hallucination loops in long streams. |
| `DL_WHISPER__DOWNLOAD_ROOT` | `whisper.download_root` | `None` | `Path | None` | Where to cache model weights. None = Hugging Face default cache. |
| `DL_MATCHING__MATCH_THRESHOLD` | `matching.match_threshold` | `80.0` | `float` | Minimum RapidFuzz score (0-100) for a window to count as a match. |
| `DL_MATCHING__WINDOW_TOLERANCE` | `matching.window_tolerance` | `2` | `int` | Window sizes tried = dialogue word count +/- this many words, to absorb ASR word splits/merges. |
| `DL_MATCHING__MIN_DIALOGUE_WORDS` | `matching.min_dialogue_words` | `2` | `int` | Reject dialogues shorter than this: single words match too easily. |
| `DL_MATCHING__TOP_K_NEAR_MISSES` | `matching.top_k_near_misses` | `3` | `int` | How many best-scoring windows to report when nothing crosses the threshold. |
| `DL_VERIFICATION__ENABLED` | `verification.enabled` | `True` | `bool` |  |
| `DL_VERIFICATION__SEARCH_WINDOW_SECONDS` | `verification.search_window_seconds` | `20.0` | `float` | Re-transcribe +/- this many seconds around the candidate. |
| `DL_VERIFICATION__MAX_SCORE_DROP` | `verification.max_score_drop` | `5.0` | `float` | Accept the verifier's timestamp only if its score is not more than this many points below the first-pass score. |
| `DL_FRAME__IMAGE_FORMAT` | `frame.image_format` | `jpg` | `Literal['jpg', 'png']` |  |
| `DL_FRAME__JPEG_QUALITY` | `frame.jpeg_quality` | `92` | `int` |  |
| `DL_STORAGE__WORK_DIR` | `storage.work_dir` | `data/work` | `Path` | Per-job scratch space: downloaded video, extracted audio. |
| `DL_STORAGE__OUTPUT_DIR` | `storage.output_dir` | `data/output` | `Path` | Per-job results: extracted frame image and result JSON. |
| `DL_STORAGE__KEEP_INTERMEDIATE` | `storage.keep_intermediate` | `True` | `bool` | Keep downloaded video/audio after a job. Repeated runs on the same URL (e.g. a different dialogue) then skip download and extraction. Set False to save disk. |
| `DL_SERVER__HOST` | `server.host` | `127.0.0.1` | `str` |  |
| `DL_SERVER__PORT` | `server.port` | `8000` | `int` |  |
| `DL_SERVER__MAX_CONCURRENT_JOBS` | `server.max_concurrent_jobs` | `1` | `int` | Whisper is CPU/GPU heavy; more than 1 rarely helps on one machine. |
| `DL_SERVER__JOB_RETENTION_SECONDS` | `server.job_retention_seconds` | `3600` | `int` | How long finished jobs stay queryable via the API. |
| `DL_LOGGING__LEVEL` | `logging.level` | `INFO` | `Literal['DEBUG', 'INFO', 'WARNING', 'ERROR']` |  |
| `DL_LOGGING__FMT` | `logging.fmt` | `%(asctime)s | %(levelname)-8s | %(name)s | %(message)s` | `str` |  |

## Choosing values

* **`whisper.fast_model`** — trade speed for first-pass accuracy. Measured on Apple M2 CPU/int8:
  `tiny` ≈ 40×, `base` ≈ 12×, `small` ≈ 3× realtime. Because `verify_model` re-checks the hit,
  `base` is the default. Use `small` for noisy audio or heavy accents.
* **`whisper.verify_model`** — `medium` is a good CPU compromise; `large-v3` on a GPU with
  `compute_type=float16` is the accuracy ceiling.
* **`matching.match_threshold`** — 80 lets typical ASR errors through (`reveals a` for `rebels at`
  still scores 93) while partial windows stay below (a 3-of-5-word window scores ~67). Lower it
  for very noisy audio; raise it for short dialogues that could match by chance.
* **`matching.window_tolerance`** — how many extra/fewer words a window may contain. 2 absorbs
  split/merged words; larger values slow matching slightly and admit more spurious edges.
* **`verification.max_score_drop`** — 5 points. If the large model scores more than this below
  the first pass, its result is *rejected* and the first-pass timestamp is kept with a warning.
* **`download.max_height`** — 720 by default; the extracted frame has this resolution. Lower it
  on slow links (a 55-minute 720p file can be ~1.5 GB).
* **`storage.keep_intermediate`** — `True` keeps `data/work/<source>/` so a second dialogue on the
  same video skips download and audio extraction.
