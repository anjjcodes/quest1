# 6. Testing

309 tests under `src/tests/`, all offline by default (302 pass, 7 skip without opt-in env
vars: 5 need `DL_RUN_NETWORK_TESTS=1`, 2 need `DL_RUN_MODEL_TESTS=1`), ~17 s. Configured in `pyproject.toml`
(`[tool.pytest.ini_options]`: `testpaths=src/tests`, `pythonpath=src`, live logging on).

## Run

```bash
python -m pytest                                   # live INFO logs per test
python -m pytest -q --log-cli-level=WARNING        # quiet
python -m pytest --log-cli-level=DEBUG             # ffmpeg/ffprobe argv, per-segment ASR, per-window scores
python -m pytest src/tests/test_matching.py -k settling --log-cli-level=INFO   # one behaviour, watch it happen
DL_RUN_NETWORK_TESTS=1 python -m pytest -m network    # 5 tests: 3 real yt-dlp downloads + 2 MediaPipe model fetches
DL_RUN_MODEL_TESTS=1  python -m pytest -k real        # real tiny/base Whisper models; macOS `say` synthesises speech
```

## Files

| File | Tests | Covers |
|---|---|---|
| `test_config.py` | 5 | defaults, nested env override, validation, cache, `ensure_directories` |
| `test_models.py` | 11 | `format_timestamp` edge cases, `to_dict` shapes, `timestamp` property precedence, exception `to_dict` |
| `test_probe.py` | 11 | fps fraction parsing, video/silent/audio-only probes, missing/non-media files, missing binary |
| `test_downloader.py` | 34 | URL accept/reject matrix, local path, audio-only file, **fake `YoutubeDL`** (success + progress + format string; four error mappings), cache reuse ignoring `.part`, 3 network tests |
| `test_audio.py` | 11 | WAV params via `wave`, config respected, reuse vs force, no audio stream, missing ffmpeg, clips (from WAV/video, clamped start, bad range, past end), **ffmpeg timeout → kill** |
| `test_transcription.py` | 23 | `load_pcm` (mono/stereo/wrong rate/garbage), **fake Whisper model**: word stripping/offset/indices, decode options follow config, laziness + early stop, progress monotonic, decode crash & start failure wrapped, path input, model cache keys, load failure wrapped, ABC, no-alignment fallback (with `caplog`), 1 real-model test |
| `test_matching.py` | 43 | normalisation matrix, window sizes, exact match indices/timestamps, early-stop count, five ASR-error patterns, dialogue at start/end of stream, first occurrence, best-size selection, settling on/off, threshold, custom scorer, near-miss ordering/non-overlap/top-k, tracker semantics, `best_match` |
| `test_verification.py` | 16 | `slice_audio` clamping, confirmed with refined absolute times and correct clip/offset, rejected on >5 drop, small drop confirmed, higher score confirmed, `TranscriptionError` → FAILED, no words → REJECTED, disabled → SKIPPED (model untouched), edge clamping, beyond audio → FAILED, `to_dict`, 1 real tiny→base test |
| `test_frames.py` | 22 | frame arithmetic (incl. 29.97 fps), roundtrip, JPG/PNG, different frames differ, **OpenCV vs ffmpeg pixel agreement**, forced fallback (`caplog`), fps from file, clamp past end / negative, unreadable/missing video, unwritable output, `read_frame` |
| `test_pipeline.py` | 43 | found end-to-end with fakes (refined timestamp used, frame at refined time, early stop, timings, progress order), `save_result`, not found + near-misses, rejected/failed/skipped verifier folding, verifier chain passes refinement along, invalid dialogue/URL fail **before** download, download error stage, unexpected exception wrapped with stage, cancellation, per-source cache + cleanup, `warm_up` loads both models, and every onscreen verdict path: still mouth -> `not_onscreen`, a face nothing can landmark -> `not_onscreen`, a line that opens off camera -> `found` once the speaker appears, the answer frame moving to the speaker (and the face check re-running there), a moved frame never reporting a face and no face at once, both checks failing open on error |
| `test_face_detector.py` | 19 | box clamping and best-first sort, bad input rejected, **fake MediaPipe detector**, threshold respected, model file download (atomic `.part` write, too-small response rejected), 1 network test |
| `test_mouth_movement.py` | 35 | openness maths, moving vs still verdicts, indeterminate under `min_face_frames`, window clamping, invalid window rejected, **fake landmarker + fake face detector**; and the scoring rules each real failure produced: a drifting mouth is not a speaking mouth, oscillation survives detrending, best window beats whole window, speech at the end of a line still counts, cuts between two still faces, gaps in the face, box carry and its expiry, crop upscaling and the whole-frame fallback, calibration pinned between the measured clusters, 1 network test |
| `test_cli.py` | 12 | output format found/not found, `result.json` written, exit codes, `--json`, flag overrides reach `Settings` |
| `test_api.py` | 24 | see [api](modules/api.md#testing-the-api) |

## Fixtures and helpers (`conftest.py`)

* `sample_video` (3 s, 320×240 @ 25 fps, sine audio), `silent_video` (no audio stream),
  `sample_wav` (4 s, 16 kHz mono), `not_media` — **generated with ffmpeg at session start**, so no
  binary fixtures live in git and the tests exercise the real tools.
* `requires_ffmpeg` skip marker; `network` marker auto-skipped unless `DL_RUN_NETWORK_TESTS=1`.
* `expect_error(ExcType, match=…)` — `pytest.raises` that logs `>>> expecting X` before and
  `<<< got X as expected` after. Convention: **an `ERROR` line in a passing test must sit between
  those two markers**; anything else is a real problem. An autouse fixture logs
  `===== START/END <test id> =====` so module logs are attributable.

## Fakes used instead of mocks

Small hand-written fakes keep tests readable and pin the *contracts*:

* `FakeWhisperModel` — lazy segment generator; counts segments pulled (proves early stop).
* `FakeTranscriber` / `ScriptedTranscriber` — preset words, records clip length and offset.
* `ScriptedVerifier` — returns a chosen status with an optional time shift.
* `FakeDownloader`, `_FakeYDL` (patched in place of `yt_dlp.YoutubeDL`).
* `FakePipeline` (CLI and API tests) — modes found/not_found/error/slow/crash; `slow` blocks on an
  `Event` so cancellation and queueing can be tested deterministically.
* `FakeFaceDetector` / `FakeMouthAnalyzer` (`test_pipeline.py`) — return a chosen V2/V3 verdict, or
  raise, so the gating and the fail-open paths are both testable without MediaPipe.
* `FakeMpDetector` (`test_face_detector.py`) / `FakeLandmarker` (`test_mouth_movement.py`) — stand in
  for the MediaPipe objects so the vision maths is tested without model weights.

Monkeypatching is done at the narrowest seam (e.g. `AudioExtractor._run_ffmpeg`, not
`subprocess.Popen`, which would also break ffprobe — that mistake was made once and is why).

## Adding a test

1. Put pure-logic tests next to the module's file; use the ffmpeg fixtures for anything touching
   media; use a fake for anything touching a model or the network.
2. If the test's purpose is a failure path, wrap it in `expect_error(...)` (or log
   `>>> expecting a WARNING …` and assert with `caplog`).
3. Mark real-network/real-model tests with `@pytest.mark.network` / the `DL_RUN_MODEL_TESTS` skip.
