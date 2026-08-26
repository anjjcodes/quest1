# Module: `pipeline/` and `cli.py`

## `pipeline/pipeline.py`

```python
@dataclass
class PipelineRequest:
    source: str; dialogue: str
    job_id: str = <uuid12>                 # output directory name
    reuse_cached_media: bool = True
    source_key -> str                      # sha1(source)[:16] → data/work/<key>/

def build_default_verifiers(settings, verify_transcriber=None) -> list[Verifier]   # [] if verification disabled

class DialoguePipeline:
    def __init__(self, settings, *, downloader=None, audio_extractor=None, fast_transcriber=None,
                 verifiers=None, retry_transcriber=None, frame_extractor=None,
                 face_detector=None, mouth_analyzer=None)  # every stage injectable (tests use fakes)
    def warm_up(self)                                     # load fast + verify models now
    def run(self, request, progress=None, should_cancel=None) -> LocalizationResult

def save_result(result, path) -> Path                    # result.to_dict() as JSON
```

`DialoguePipeline` is stateless; each `run()` creates a private `_Run` holding timings, warnings,
the current stage and the directories. `_Run._execute` reads as the stage list; each stage is a
`_stage_*` method that owns its timing and progress events.

### Stage sequence in `_Run._execute`

| # | Stage | What happens | Fails with |
|---|---|---|---|
| 1 | `input` | `StreamingMatcher(dialogue)` (validates dialogue) and `validate_url` if the source looks like a URL. **Before any I/O.** | `InvalidDialogueError`, `InvalidURLError` |
| 2 | `download` | `downloader.fetch_search_media(source, work_dir/<source_key>, reuse)` — audio-only when the host offers it | `DownloadError`, `UnsupportedVideoError` |
| 3 | `audio` | `audio_extractor.extract(...)`, then `load_pcm` into memory | `AudioExtractionError`, `TranscriptionError` (bad WAV) |
| 4 | `transcription` (+`matching`) | `warm_up()`, then `for word in transcriber.transcribe(samples): matcher.feed(word)`; break on match; `stream.close()` in a `finally` so the decoder stops even on error/cancel; `matcher.finish()` if the stream ended. On a miss with the VAD on: one retry scan with the VAD off (decision #24) | `TranscriptionError` |
|   | not found → | return `LocalizationResult(NOT_FOUND, near_misses, transcribed_seconds)` | — |
| 5 | `verification` | build `VerificationContext`; for each verifier: fold the outcome (see below) | never raises (verifiers return FAILED) |
|   | *guard* | stages 6–9 run only when the source is a URL or the probed media has a video stream; an audio-only local file instead adds the warning "Source has no video stream; returning the timestamp without a frame." Setting `face_detection.enabled=False` skips **both** visual stages | — |
| 6 | `download_video` | `downloader.fetch_video_clip(source, match.start, match.end, ...)` — a few seconds of full quality around the *verified* match | `DownloadError` |
| 7 | `frame` | `frame_extractor.extract(clip, match.start, output_dir/<job>/frame)` | `FrameExtractionError` |
| 8 | `face_detection` (V2) | `face_detector.detect_file(frame)`; no face → status `NOT_ONSCREEN`; a crashed check fails open with a warning | never fails the job |
| 9 | `mouth_movement` (V3) | only when V2 confirmed a face: `mouth_analyzer.analyze(clip, match.start, match.end)`; not moving → `NOT_ONSCREEN`; indeterminate/crash fails open with a warning | never fails the job |
| 10 | `done` | timings (`total`), final progress event | — |

Folding rule: `CONFIRMED` + `refined` → candidate := refined (logged when the timestamp moves);
`REJECTED`/`FAILED` → `warnings.append(f"{verifier}: {status} - {message}")`.

### Cross-cutting

* **Timings**: `stage_timings[stage]` per stage and `total` — shown in the UI's timing bar.
* **Progress**: a `ProgressEvent` at each stage start, plus pass-through of the sub-stage events
  (download %, audio %, transcription position, match found / not found).
* **Cancellation**: `should_cancel()` is polled after each stage and after every word; raises
  `PipelineCancelledError` with `stage` set to the running stage (logged at WARNING, not ERROR).
* **Last-resort boundary**: any non-`DialogueLocatorError` is logged with traceback and re-raised
  as `DialogueLocatorError` with the current stage, so interfaces never leak raw stack traces.
* **Cleanup**: `work_dir` is removed after the run only when `storage.keep_intermediate=False`.

### Directory layout it produces

```
data/work/<sha1(source)[:16]>/media.* + audio.wav + clip_<ms>_<ms>.mp4   # per source, shared across dialogues
data/output/<job_id>/frame.jpg + result.json                             # per job
```

## `cli.py`

```
dialogue-locator <source> "<dialogue>" [--fast-model M] [--verify-model M] [--no-verify]
                 [--threshold N] [--max-height H] [--output-dir D] [--work-dir D] [--no-cache]
                 [--json] [-q] [-v]
```

* `apply_overrides()` copies `Settings` with `model_copy(update=…)` — flags are just another
  override layer above env/.env; config stays the single source of truth.
* Prints progress events to stderr (`[  transcription]  22%  Transcribed 00:04:03 / 00:18:15`),
  the result in the PS format to stdout, and writes `result.json` next to the frame.
* Exit codes: `0` found, `2` not found (near-misses printed), `3` found in audio but not onscreen
  (no face, or mouth not moving; details still printed), `1` error (`ERROR [stage]: message`).
* `main(argv, pipeline_factory=DialoguePipeline)` — the factory parameter lets tests inject a fake
  pipeline without monkeypatching.
