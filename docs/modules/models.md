# Module: `models.py`

Plain `dataclasses` (no pydantic, no FastAPI) that every stage exchanges. Keeping the core
types framework-free is what lets the pipeline run from the CLI, the API, or a test with equal
ease. Each type has a `to_dict()` that produces the JSON written to `result.json` and returned by
the API.

## Helper

```python
def format_timestamp(seconds: float) -> str   # "HH:MM:SS.mmm"; clamps negatives to 0; rounds .9996 → next second
```

## Enums

* `PipelineStage`: `input, download, audio, transcription, matching, verification, frame, done`
  — the vocabulary used by `ProgressEvent`, exceptions and the UI stepper.
* `VerificationStatus`: `confirmed | rejected | skipped | failed`.
* `ResultStatus`: `found | not_found`.

## Media

| Type | Fields | Produced by |
|---|---|---|
| `VideoInfo` | `path, source_url, title, duration, fps, width, height, frame_count` | `VideoDownloader.fetch` (facts from ffprobe, not the website) |
| `AudioInfo` | `path, sample_rate, channels, duration` | `AudioExtractor.extract` / `extract_clip` |

## Transcription

```python
@dataclass(frozen=True)
class Word:
    text: str                     # raw ASR token, punctuation kept ("there,") — normalisation is the matcher's job
    start: float; end: float      # ABSOLUTE seconds into the video (offset already applied)
    probability: float | None
    segment_index: int | None     # which ASR segment produced it (debugging only)
```

## Matching

```python
@dataclass(frozen=True)
class MatchCandidate:
    score: float                  # RapidFuzz 0–100
    start: float; end: float      # first word start, last word end
    matched_text: str             # raw words joined — what was heard
    words: tuple[Word, ...]
    word_index_start/end: int | None   # indices into the word stream (end exclusive)
    timestamp -> str              # property, formatted start
```

The same type serves accepted matches *and* near-misses (a near-miss is just a candidate below
threshold), so the "top-3 closest windows" report needs no extra type.

## Verification

```python
@dataclass(frozen=True)
class VerificationOutcome:
    verifier: str                 # "asr_large_model", later "visual_speaker" …
    status: VerificationStatus
    score: float | None
    refined: MatchCandidate | None   # the verifier's own localisation, if it produced one
    message: str | None
    details: dict                 # model, clip bounds, timing, shift_seconds …
```

Deliberately generic: a V2 visual verifier returns the same shape with `refined.start` = first
frame where the speaker is visible.

## Frame

```python
@dataclass(frozen=True)
class FrameInfo:
    frame_number: int
    timestamp: float              # the frame's OWN presentation time (frame_number / fps)
    fps: float
    image_path: Path
    width, height: int | None
```

## Result

```python
@dataclass
class LocalizationResult:
    status: ResultStatus; dialogue: str; source_url: str; video: VideoInfo | None
    match: MatchCandidate | None          # final (possibly refined) localisation
    first_pass: MatchCandidate | None     # what the fast model found, before verification
    verifications: list[VerificationOutcome]
    frame: FrameInfo | None
    near_misses: list[MatchCandidate]     # populated when NOT_FOUND
    warnings: list[str]                   # e.g. "asr_large_model: rejected - …"
    stage_timings: dict[str, float]       # seconds per stage + "total"
    transcribed_seconds: float | None     # how far the fast pass got before stopping
    # properties: found, timestamp (prefers frame time), frame_number, matched_text
```

Keeping `first_pass` *and* `match` lets the UI show "100.0 at 00:09:06.800 → refined by verifier"
and lets you audit a rejected verification.

## Progress

```python
@dataclass(frozen=True)
class ProgressEvent:
    stage: PipelineStage; message: str; fraction: float | None; details: dict
ProgressCallback = Callable[[ProgressEvent], None]
```

The single seam through which the pipeline talks to the outside world. The CLI prints events;
the `JobManager` stores them (latest + a 50-event ring buffer) for the UI to poll.
