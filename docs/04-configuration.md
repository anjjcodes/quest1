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
| `DL_WHISPER__FAST_MODEL` | `whisper.fast_model` | `base` | `str` | Model for the streaming pass. On Apple Silicon performance cores 'base' scans at ~11x realtime inside a live server run and ~40-60x isolated on an idle machine (vs ~3x for 'small'), with near-identical hit rate; the verify model fixes wording/timing. |
| `DL_WHISPER__VERIFY_MODEL` | `whisper.verify_model` | `small` | `str` | Model for the verification pass. 'small' confirms/refines a fuzzy match about as reliably as 'medium' at ~3x the speed on CPU; bump to 'medium' for maximum wording accuracy. |
| `DL_WHISPER__DEVICE` | `whisper.device` | `auto` | `Literal['auto', 'cpu', 'cuda']` |  |
| `DL_WHISPER__COMPUTE_TYPE` | `whisper.compute_type` | `int8` | `str` | ctranslate2 compute type. int8 is fastest on CPU (incl. Apple Silicon); use float16 or int8_float16 on CUDA GPUs; float32 for maximum accuracy. |
| `DL_WHISPER__CPU_THREADS` | `whisper.cpu_threads` | `0` | `int` | Intra-op threads for CPU inference. 0 = size the pool to the machine's performance cores (see transcription.faster_whisper.performance_cores). Deliberately not every core: on Apple Silicon the efficiency cores drag every parallel region down to their own speed, and an M2 measured 5.4 s at 4 threads against 90.9 s at 8 for the same scan. |
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
| `DL_MATCHING__MAX_OCCURRENCES` | `matching.max_occurrences` | `1` | `int` | How many occurrences of the dialogue to evaluate before settling. 1 keeps the V1 behaviour: the first audible occurrence is judged and reported whatever the visual verdict. Higher values keep scanning past an occurrence that came back not_onscreen and report the first one actually delivered on camera, falling back to the first occurrence when none are. -1 means no limit: keep going until the audio runs out, for a video whose repeat count is not known up front. Costs one clip download and one visual pass per rejected occurrence, and forfeits the early stop - a line that is never onscreen transcribes the whole track. |
| `DL_MATCHING__MIN_TAIL_SECONDS` | `matching.min_tail_seconds` | `1.0` | `float` | Stop looking for further occurrences when less than this much audio remains after the previous one: a fragment that short cannot hold a dialogue, and an empty slice makes the decoder raise rather than return nothing. |
| `DL_VERIFICATION__ENABLED` | `verification.enabled` | `True` | `bool` |  |
| `DL_VERIFICATION__SEARCH_WINDOW_SECONDS` | `verification.search_window_seconds` | `12.0` | `float` | Re-transcribe +/- this many seconds around the candidate. Fast-pass timestamps are rarely off by more than a couple of seconds, so this bounds the (expensive) large-model transcription generously while keeping it short. |
| `DL_VERIFICATION__SKIP_ABOVE_SCORE` | `verification.skip_above_score` | `90.0` | `float | None` | Skip the large-model re-transcription when the first pass already scored at least this: verification exists to check uncertain matches, not to re-prove near-perfect ones. Calibrated for the greedy fast pass, whose correct matches score ~94+ while the genuinely uncertain band sits at 80-90. None = always verify. |
| `DL_VERIFICATION__MAX_SCORE_DROP` | `verification.max_score_drop` | `5.0` | `float` | Accept the verifier's timestamp only if its score is not more than this many points below the first-pass score. |
| `DL_FRAME__IMAGE_FORMAT` | `frame.image_format` | `jpg` | `Literal['jpg', 'png']` |  |
| `DL_FRAME__JPEG_QUALITY` | `frame.jpeg_quality` | `92` | `int` |  |
| `DL_FACE_DETECTION__ENABLED` | `face_detection.enabled` | `True` | `bool` | Run the face check on the matched frame. Disabling it also disables the mouth-movement check, which needs this detector's boxes to crop to. Note the face check no longer decides the verdict on its own: a line can open on a title card and cut to the speaker, so the mouth-movement stage scans the whole window and settles it. |
| `DL_FACE_DETECTION__MIN_DETECTION_CONFIDENCE` | `face_detection.min_detection_confidence` | `0.3` | `float` | Minimum BlazeFace score for a detection to count as a face. Faces in cinematic mid/wide shots score as low as ~0.30 (close-ups ~0.97), so the model's usual 0.5 default misses distant faces. This threshold cannot separate faces from non-faces on its own: blurred rubble in a low-quality source measured 0.45-0.62, above what real faces score here. What rejects those is the landmarker's second opinion on the crop (mouth_movement.min_face_confidence), so keep this loose and let that stage decide. |
| `DL_FACE_DETECTION__MODEL_PATH` | `face_detection.model_path` | `data/models/blaze_face_short_range.tflite` | `Path` | Where the BlazeFace model file is cached. Downloaded from model_url on first use if missing (~230 KB, one-time). |
| `DL_FACE_DETECTION__MODEL_URL` | `face_detection.model_url` | `https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/latest/blaze_face_short_range.tflite` | `str` | Where to fetch the model file from when model_path is missing. |
| `DL_FACE_DETECTION__DOWNLOAD_TIMEOUT_SECONDS` | `face_detection.download_timeout_seconds` | `60` | `int` |  |
| `DL_MOUTH_MOVEMENT__ENABLED` | `mouth_movement.enabled` | `True` | `bool` | Run the mouth-movement check. Only runs when the face check is enabled and confirmed a face. |
| `DL_MOUTH_MOVEMENT__MOVEMENT_THRESHOLD` | `mouth_movement.movement_threshold` | `0.03` | `float` | Minimum movement score to count as speaking. Calibrated on the cached corpus of real matches: faces that are present but silent - still faces, a voice-over read with the mouth shut, a listener whose head turns through a reaction shot - score 0.000-0.021, and faces speaking on camera 0.046-0.117. 0.03 sits near the middle of that gap. It moved up with min_face_confidence: a stricter landmarker re-detects more often and smooths less, so a still face carries more jitter. |
| `DL_MOUTH_MOVEMENT__MIN_FACE_FRAMES` | `mouth_movement.min_face_frames` | `5` | `int` | Minimum frames with a detected face needed for a verdict; fewer makes the result indeterminate rather than a false 'not moving'. |
| `DL_MOUTH_MOVEMENT__MAX_WINDOW_SECONDS` | `mouth_movement.max_window_seconds` | `10.0` | `float` | Cap on how much of the dialogue window is analysed; long lines are judged on their first seconds. |
| `DL_MOUTH_MOVEMENT__SCORE_WINDOW_SECONDS` | `mouth_movement.score_window_seconds` | `0.5` | `float` | Length of the sliding window the movement score is measured over. The score is the largest standard deviation of any such window, so a short burst of speech is not diluted by the rest of the line (a reaction shot before the camera cuts to the speaker) and a hard cut between two still faces cannot inflate one long window into a verdict. |
| `DL_MOUTH_MOVEMENT__CROP_PADDING` | `mouth_movement.crop_padding` | `0.6` | `float` | How much of the face box size to add around it before landmarking, as a fraction. The landmarker needs the chin and forehead, not just the detector's tight box. |
| `DL_MOUTH_MOVEMENT__MIN_CROP_SIZE` | `mouth_movement.min_crop_size` | `256` | `int` | Face crops shorter/narrower than this are upscaled to it before landmarking. The landmarker downsamples its input, so a face that is small in a 1920x1080 frame is lost unless it is cropped out and enlarged - the failure that made cinematic wide shots read as 'no face'. |
| `DL_MOUTH_MOVEMENT__BOX_CARRY_SECONDS` | `mouth_movement.box_carry_seconds` | `0.4` | `float` | How long a face box is reused on frames where detection finds nothing. BlazeFace flickers on faces that are small in the frame (a 290px face in 1920x1080 scores ~0.35, on and off frame to frame), while the landmarker is reliable on a crop, so the last known box is carried forward and the landmarker confirms or rejects it. Bounded so a cut cannot attribute a new shot's face to the previous run; 0 disables. |
| `DL_MOUTH_MOVEMENT__SHOT_CHANGE_SHIFT` | `mouth_movement.shot_change_shift` | `0.5` | `float` | A face box that jumps by more than this fraction of its own width between adjacent frames is a different shot, not motion. Scoring never spans a shot change: the step between two faces' resting mouths would otherwise read as movement. |
| `DL_MOUTH_MOVEMENT__SHOT_CHANGE_SCALE` | `mouth_movement.shot_change_scale` | `2.0` | `float` | A face box that grows or shrinks by more than this factor between adjacent frames is a different shot, like shot_change_shift. Loose because the box itself is noisy on a marginal face: 267px and 416px boxes were measured on one unmoving face six frames apart, and reading that as a cut chopped a single take into unscorable pieces. |
| `DL_MOUTH_MOVEMENT__MAX_GAP_SECONDS` | `mouth_movement.max_gap_seconds` | `0.12` | `float` | A run of frames tolerates gaps up to this long where the landmarker found nothing. A one- or two-frame dropout is a miss, not a cut, and splitting on it leaves half-syllables that the trend fit then flattens to nothing. Longer gaps still split the run. |
| `DL_MOUTH_MOVEMENT__MIN_FACE_CONFIDENCE` | `mouth_movement.min_face_confidence` | `0.4` | `float` | Face detection/presence confidence for the landmarker. This is the second opinion on whether a crop really holds a face, and it must be strict: BlazeFace fires on blurred rubble at 0.45-0.62, higher than it scores some real faces, so it cannot police itself. Raising this rejects those crops (18 landmarked frames drop to 4) - the crop already fixed what lowering it used to be for. 0.4 is the measured middle: 0.3 lets the rubble through at 0.101, while MediaPipe's 0.5 default loses a real speaker in a soft, upscaled TV master (0.053 -> 0.022, below the movement threshold), and 0.7 reads a listener's landmark noise as speech. |
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
| `DL_LOGGING__FMT` | `logging.fmt` | `%(asctime)s \| %(levelname)-8s \| %(name)s \| %(message)s` | `str` |  |

## Choosing values

* **`whisper.fast_model`** — trade speed for first-pass accuracy. Measured on Apple M2 CPU/int8:
  `tiny` ≈ 40×, `base` ≈ 11× in a live run (≈ 40-60× isolated), `small` ≈ 3× realtime. Because `verify_model` re-checks the hit,
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
