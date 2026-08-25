# 8. Extending

## A. V2 — visual "speaker on camera" verification

Goal: after the ASR verifier has pinned the spoken line, confirm the character is visible and,
if needed, move the frame to the first moment they are on screen. Everything needed already
exists as a seam.

```python
# src/dialogue_locator/verification/visual_verifier.py
from dialogue_locator.media.frames import FrameExtractor, timestamp_to_frame
from dialogue_locator.models import MatchCandidate, VerificationOutcome, VerificationStatus
from dialogue_locator.verification.base import VerificationContext, Verifier

class VisualSpeakerVerifier(Verifier):
    name = "visual_speaker"

    def __init__(self, frame_extractor: FrameExtractor, config: VisualConfig):
        self._frames, self._config = frame_extractor, config

    def verify(self, candidate: MatchCandidate, context: VerificationContext) -> VerificationOutcome:
        video = context.video                                  # VideoInfo: path, fps, …
        if video is None:
            return VerificationOutcome(self.name, VerificationStatus.SKIPPED, message="no video")
        start = timestamp_to_frame(candidate.start, video.fps)
        end = timestamp_to_frame(candidate.end, video.fps)
        for n in range(start, end + 1, self._config.frame_step):
            image = self._frames.read_frame(video, n, video.fps)   # BGR ndarray, same seek logic as extract()
            if self._speaker_visible(image):                       # face / lip-activity detector of your choice
                refined = MatchCandidate(score=candidate.score, start=n / video.fps, end=candidate.end,
                                         matched_text=candidate.matched_text, words=candidate.words)
                return VerificationOutcome(self.name, VerificationStatus.CONFIRMED, score=…, refined=refined,
                                           message=f"speaker visible from frame {n}")
        return VerificationOutcome(self.name, VerificationStatus.REJECTED, score=0.0,
                                   message="speaker not visible while the line is spoken")
```

Wire it in (`pipeline/pipeline.py`):

```python
def build_default_verifiers(settings, verify_transcriber=None):
    verifiers = []
    if settings.verification.enabled:
        verifiers.append(AsrVerifier(...))
    if settings.visual.enabled:                       # new VisualConfig block in config.py
        verifiers.append(VisualSpeakerVerifier(FrameExtractor(settings.frame), settings.visual))
    return verifiers
```

That is the whole change. The pipeline already folds a `CONFIRMED`+`refined` outcome into the
candidate (so the frame is extracted at the refined time), records `REJECTED` as a warning, shows
each verifier as a badge in the UI, and `test_verifier_chain_passes_refined_candidate_along`
already exercises a two-verifier chain. Put a `VisualConfig` next to the other config blocks and
the env override `DL_VISUAL__…` comes for free.

Design notes for the detector: `read_frame` re-opens the file per call — for dense scanning,
open one `cv2.VideoCapture` and step frames sequentially (add a `iter_frames(video, start, end)`
helper to `frames.py`). Keep the detector behind a small interface so the verifier can be tested
with a fake, exactly like `AsrVerifier` is tested with `FakeTranscriber`.

## B. Swap the ASR engine

Implement `transcription.base.Transcriber`:

```python
class WhisperCppTranscriber(Transcriber):
    name = "whisper.cpp:base"
    def transcribe(self, audio, *, offset=0.0, progress=None) -> Iterator[Word]:
        for seg in self._engine.stream(self.as_samples(audio)):          # must be lazy
            for w in seg.words:
                yield Word(w.text.strip(), w.start + offset, w.end + offset, w.p)
```

and pass it: `DialoguePipeline(settings, fast_transcriber=WhisperCppTranscriber(...))`. Honour the
contract: chronological words, absolute times via `offset`, stop when the consumer stops.

## C. Try another matching scorer

`StreamingMatcher(dialogue, config, scorer=fuzz.token_sort_ratio)` or any
`Callable[[str, str], float]` returning 0–100. `best_match` takes the same argument. Re-run
`test_matching.py` — the ASR-error and partial-window tests tell you immediately whether the new
scorer keeps the 80 threshold meaningful.

## D. Audio-first download (planned optimisation)

Today the whole video is fetched before anything else. For a 55-minute 720p file that is ~1.5 GB
for one frame. The plan:

1. `VideoDownloader.fetch_audio(source, dest)` — yt-dlp `bestaudio` (~50 MB/hour) → `AudioInfo`.
2. Pipeline: `input → audio-download → transcription/matching → verification →
   video-section-download → frame`.
3. `VideoDownloader.fetch_section(source, start, end, dest)` — yt-dlp `--download-sections
   "*start-end" --force-keyframes-at-cuts` so section time 0 == `start` exactly (re-encoded cut);
   `FrameExtractor.extract(section_video, t - start, …)` with the frame number computed from the
   absolute `t × fps`.

Nothing in transcription/matching/verification changes; only `media/downloader.py` and the
stage order in `pipeline.py`.

## E. Persistent job store

Replace `api/jobs.py`'s dict with Redis/SQLite keyed by `job_id`, keeping the same `JobManager`
methods. `JobResponse` already carries everything the UI needs; `result.json` on disk is the
canonical result document.

## F. Add a pipeline stage

1. Add a `PipelineStage` member (`models.py`) — it appears in logs, events, errors and the UI stepper
   (add an `<li data-stage>` in `index.html`).
2. Implement the stage as a class with an injected config sub-model; raise a
   `DialogueLocatorError` subclass with the new `stage` name.
3. Call it from `_Run._execute` between `_begin(...)` / `_end(...)`; add it to `DialoguePipeline.__init__`
   as an injectable dependency; add a fake in `test_pipeline.py`.
