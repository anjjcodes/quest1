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
echo 'DL_VERIFICATION__SKIP_ABOVE_SCORE=100' >> .env   # picked up by CLI and server alike
```

`get_settings()` returns a cached instance (`functools.lru_cache`); tests call
`reset_settings_cache()` after changing the environment. Values are validated on load — an
out-of-range threshold or unknown device is a startup error, not a runtime surprise.

## Reference (generated from the models)

| Env variable | Field | Default | Type | Description |
|---|---|---|---|---|
| `DL_DOWNLOAD__MAX_HEIGHT` | `download.max_height` | `1080` | `int` | Height cap for the full-quality video fetch. This bounds the resolution of the extracted frame and the size of the one large download; it does not affect search speed (the search runs on audio). |
| `DL_DOWNLOAD__SEARCH_MAX_HEIGHT` | `download.search_max_height` | `360` | `int` | Fallback cap for the search fetch when the host has no audio-only stream (e.g. progressive-only hosts like ok.ru): the lowest video rendition no taller than this is used instead. |
| `DL_DOWNLOAD__CLIP_PADDING_SECONDS` | `download.clip_padding_seconds` | `5.0` | `float` | Seconds of video downloaded either side of the verified match window by the clip fetch, so the target frame is safely inside the clip. |
| `DL_DOWNLOAD__PROGRESS_INTERVAL_SECONDS` | `download.progress_interval_seconds` | `2.0` | `float` | Minimum seconds between download progress events. Time-based so a slow host still shows movement (MB, speed, ETA) instead of minutes-long silent percent buckets. |
| `DL_DOWNLOAD__SOCKET_TIMEOUT_SECONDS` | `download.socket_timeout_seconds` | `30` | `int` |  |
| `DL_DOWNLOAD__RETRIES` | `download.retries` | `3` | `int` |  |
| `DL_DOWNLOAD__CONTAINER` | `download.container` | `mp4` | `str` | Preferred container. Ensures a format OpenCV/FFmpeg can seek in. |
| `DL_AUDIO__SAMPLE_RATE` | `audio.sample_rate` | `16000` | `int` | Whisper expects 16 kHz mono PCM. |
| `DL_AUDIO__CHANNELS` | `audio.channels` | `1` | `int` |  |
| `DL_AUDIO__FFMPEG_BINARY` | `audio.ffmpeg_binary` | `ffmpeg` | `str` |  |
| `DL_AUDIO__FFPROBE_BINARY` | `audio.ffprobe_binary` | `ffprobe` | `str` |  |
| `DL_AUDIO__TIMEOUT_SECONDS` | `audio.timeout_seconds` | `600` | `int` |  |
| `DL_WHISPER__FAST_MODEL` | `whisper.fast_model` | `base` | `str` | Model for the streaming pass. 'base' runs ~12x realtime on Apple Silicon CPU (vs ~3x for 'small') with near-identical hit rate; the verify model fixes wording/timing. |
| `DL_WHISPER__VERIFY_MODEL` | `whisper.verify_model` | `small` | `str` | Model for the verification pass. 'small' confirms/refines a fuzzy match about as reliably as 'medium' at ~3x the speed on CPU; bump to 'medium' for maximum wording accuracy. |
| `DL_WHISPER__DEVICE` | `whisper.device` | `auto` | `Literal['auto', 'cpu', 'cuda']` |  |
| `DL_WHISPER__COMPUTE_TYPE` | `whisper.compute_type` | `int8` | `str` | ctranslate2 compute type. int8 is fastest on CPU (incl. Apple Silicon); use float16 or int8_float16 on CUDA GPUs; float32 for maximum accuracy. |
| `DL_WHISPER__CPU_THREADS` | `whisper.cpu_threads` | `0` | `int` | Intra-op threads for CPU inference. 0 = use all CPU cores (ctranslate2's own default is only 4, leaving most of a modern CPU idle). |
| `DL_WHISPER__LANGUAGE` | `whisper.language` | `en` | `str | None` | Force a language (ISO 639-1) or None to auto-detect. Forcing avoids a detection step and mis-detections on music intros. |
| `DL_WHISPER__BEAM_SIZE` | `whisper.beam_size` | `5` | `int` | Beam width for the verification pass, where accuracy matters. |
| `DL_WHISPER__FAST_BEAM_SIZE` | `whisper.fast_beam_size` | `1` | `int` | Beam width for the streaming search pass. Greedy (1) decodes ~1.5-2x faster than beam 5; the fuzzy matcher absorbs the slightly rougher wording and the verification pass re-checks it anyway. |
| `DL_WHISPER__VAD_FILTER` | `whisper.vad_filter` | `True` | `bool` | Skip silent regions. Speeds up long videos with quiet stretches. |
| `DL_WHISPER__VAD_MIN_SILENCE_MS` | `whisper.vad_min_silence_ms` | `500` | `int` |  |
| `DL_WHISPER__RETRY_WITHOUT_VAD` | `whisper.retry_without_vad` | `True` | `bool` | If the streaming pass ends with no match while vad_filter is on, re-run the scan once with the VAD disabled before reporting not_found. Loud music/effects can make the VAD discard real speech (e.g. action scenes); the retry hears everything, at the cost of one extra pass paid only on a miss. |
| `DL_WHISPER__CONDITION_ON_PREVIOUS_TEXT` | `whisper.condition_on_previous_text` | `False` | `bool` | False reduces hallucination loops in long streams. |
| `DL_WHISPER__DOWNLOAD_ROOT` | `whisper.download_root` | `None` | `Path | None` | Where to cache model weights. None = Hugging Face default cache. |
| `DL_MATCHING__MATCH_THRESHOLD` | `matching.match_threshold` | `80.0` | `float` | Minimum RapidFuzz score (0-100) for a window to count as a match. |
| `DL_MATCHING__WINDOW_TOLERANCE` | `matching.window_tolerance` | `2` | `int` | Window sizes tried = dialogue word count +/- this many words, to absorb ASR word splits/merges. |
| `DL_MATCHING__MIN_DIALOGUE_WORDS` | `matching.min_dialogue_words` | `2` | `int` | Reject dialogues shorter than this: single words match too easily. |
| `DL_MATCHING__TOP_K_NEAR_MISSES` | `matching.top_k_near_misses` | `3` | `int` | How many best-scoring windows to report when nothing crosses the threshold. |
| `DL_VERIFICATION__ENABLED` | `verification.enabled` | `True` | `bool` |  |
| `DL_VERIFICATION__SEARCH_WINDOW_SECONDS` | `verification.search_window_seconds` | `12.0` | `float` | Re-transcribe +/- this many seconds around the candidate. Fast-pass timestamps are rarely off by more than a couple of seconds, so this bounds the (expensive) large-model transcription generously while keeping it short. |
| `DL_VERIFICATION__SKIP_ABOVE_SCORE` | `verification.skip_above_score` | `90.0` | `float | None` | Skip the large-model re-transcription when the first pass already scored at least this: verification exists to check uncertain matches, not to re-prove near-perfect ones. Calibrated for the greedy fast pass, whose correct matches score ~94+ while the genuinely uncertain band sits at 80-90. None = always verify. |
| `DL_VERIFICATION__MAX_SCORE_DROP` | `verification.max_score_drop` | `5.0` | `float` | Accept the verifier's timestamp only if its score is not more than this many points below the first-pass score. |
| `DL_FRAME__IMAGE_FORMAT` | `frame.image_format` | `jpg` | `Literal['jpg', 'png']` |  |
| `DL_FRAME__JPEG_QUALITY` | `frame.jpeg_quality` | `92` | `int` |  |
| `DL_FACE_DETECTION__ENABLED` | `face_detection.enabled` | `True` | `bool` | Run the face check on the matched frame. Disabling it also disables the mouth-movement check, which builds on a confirmed face. |
| `DL_FACE_DETECTION__MIN_DETECTION_CONFIDENCE` | `face_detection.min_detection_confidence` | `0.3` | `float` | Minimum BlazeFace score for a detection to count as a face. Calibrated on real frames: faces in cinematic mid/wide shots score as low as ~0.34 (close-ups ~0.97) while non-face junk stays at or below ~0.2, so 0.3 separates them; the model's usual 0.5 default misses distant faces. |
| `DL_FACE_DETECTION__MODEL_PATH` | `face_detection.model_path` | `data/models/blaze_face_short_range.tflite` | `Path` | Where the BlazeFace model file is cached. Downloaded from model_url on first use if missing (~230 KB, one-time). |
| `DL_FACE_DETECTION__MODEL_URL` | `face_detection.model_url` | `https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/latest/blaze_face_short_range.tflite` | `str` | Where to fetch the model file from when model_path is missing. |
| `DL_FACE_DETECTION__DOWNLOAD_TIMEOUT_SECONDS` | `face_detection.download_timeout_seconds` | `60` | `int` |  |
| `DL_MOUTH_MOVEMENT__ENABLED` | `mouth_movement.enabled` | `True` | `bool` | Run the mouth-movement check. Only runs when the face check is enabled and confirmed a face. |
| `DL_MOUTH_MOVEMENT__MOVEMENT_THRESHOLD` | `mouth_movement.movement_threshold` | `0.02` | `float` | Minimum standard deviation of mouth openness to count as significant movement. |
| `DL_MOUTH_MOVEMENT__MIN_FACE_FRAMES` | `mouth_movement.min_face_frames` | `5` | `int` | Minimum frames with a detected face needed for a verdict; fewer makes the result indeterminate rather than a false 'not moving'. |
| `DL_MOUTH_MOVEMENT__MAX_WINDOW_SECONDS` | `mouth_movement.max_window_seconds` | `10.0` | `float` | Cap on how much of the dialogue window is analysed; long lines are judged on their first seconds. |
| `DL_MOUTH_MOVEMENT__MIN_FACE_CONFIDENCE` | `mouth_movement.min_face_confidence` | `0.3` | `float` | Face detection/presence confidence for the landmarker, lowered from MediaPipe's 0.5 default for the same reason as face_detection.min_detection_confidence: faces in mid/wide shots score lower than close-ups. |
| `DL_MOUTH_MOVEMENT__MODEL_PATH` | `mouth_movement.model_path` | `data/models/face_landmarker.task` | `Path` | Where the Face Landmarker model file is cached. Downloaded from model_url on first use if missing (~3.7 MB, one-time). |
| `DL_MOUTH_MOVEMENT__MODEL_URL` | `mouth_movement.model_url` | `https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task` | `str` | Where to fetch the model file from when model_path is missing. |
| `DL_MOUTH_MOVEMENT__DOWNLOAD_TIMEOUT_SECONDS` | `mouth_movement.download_timeout_seconds` | `120` | `int` |  |
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
* **`whisper.verify_model`** — `small` by default; `medium` is a good CPU compromise when
  accuracy matters more than latency, and `large-v3` on a GPU with `compute_type=float16` is the
  accuracy ceiling.
* **`verification.skip_above_score`** — 90 by default, so the large model never runs on a first
  pass that already scored 90 or more. That is a deliberate speed trade: it means verification is
  skipped on most clean hits. Set it to `None` to verify every match (see decision #23), or raise
  it to 100 to verify everything short of a perfect score.
* **`matching.match_threshold`** — 80 lets typical ASR errors through (`reveals a` for `rebels at`
  still scores 93) while partial windows stay below (a 3-of-5-word window scores ~67). Lower it
  for very noisy audio; raise it for short dialogues that could match by chance.
* **`matching.window_tolerance`** — how many extra/fewer words a window may contain. 2 absorbs
  split/merged words; larger values slow matching slightly and admit more spurious edges.
* **`verification.max_score_drop`** — 5 points. If the large model scores more than this below
  the first pass, its result is *rejected* and the first-pass timestamp is kept with a warning.
* **`download.max_height`** — 1080 by default, and it caps only the short clip fetched *after*
  a match is verified, so lowering it does not speed up the search; it just gives you a
  lower-resolution frame. The search itself downloads audio only.
* **`download.search_max_height`** — 360 by default, and it only matters on hosts with no
  audio-only stream, where the search falls back to the smallest video rendition carrying audio.
* **`storage.keep_intermediate`** — `True` keeps `data/work/<source>/` so a second dialogue on the
  same video skips download and audio extraction.
