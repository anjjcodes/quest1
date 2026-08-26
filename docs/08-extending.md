# 8. Extending

## A. Add another visual validation stage (V4+)

V2 (face presence, `vision/face_detector.py`) and V3 (mouth movement,
`vision/mouth_movement.py`) are the worked examples: each is a small analyser class with its own
config block, called from a `_stage_*` method in `pipeline/pipeline.py` that sets the verdict
and **fails open** (a crashed check adds a warning, it never loses the localisation). To add,
say, a "subtitle text on screen" check (V4):

1. Write the analyser in `vision/` — a class that takes its config in the constructor, works on
   the clip `VideoInfo` and/or the extracted frame, and raises only its own
   `DialogueLocatorError` subclass. Follow `MouthMovementAnalyzer`: lazy `mediapipe`-style
   imports, model file via `vision/model_files.ensure_model_file` if one is needed.
2. Add a config block in `config.py` (`enabled` flag + thresholds) — the `DL_…` env override
   comes for free — and a result dataclass in `models.py` plus a field on `LocalizationResult`.
3. Add a `PipelineStage` member and a `_stage_*` method in `_Run`, called after the stages it
   builds on (see the mouth check: it only runs when the face check confirmed a face). Decide
   what a negative verdict means for `ResultStatus` and keep the fail-open pattern.
4. Inject the analyser through `DialoguePipeline.__init__` like `face_detector` /
   `mouth_analyzer`, add a fake in `test_pipeline.py`, and (optionally) a toggle in
   `StageToggles` (`api/schemas.py`) so the UI can switch it per job.

The `Verifier` chain (`verification/base.py`) is the seam for a different kind of check: another
*audio* judge of "is this really the line, and when does it start" (e.g. a second ASR engine).
The pipeline already folds a `CONFIRMED`+`refined` outcome into the candidate, records
`REJECTED`/`FAILED` as warnings, and `test_verifier_chain_passes_refined_candidate_along`
exercises a two-verifier chain.

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

## D. Persistent job store

Replace `api/jobs.py`'s dict with Redis/SQLite keyed by `job_id`, keeping the same `JobManager`
methods. `JobResponse` already carries everything the UI needs; `result.json` on disk is the
canonical result document.

## E. Add a pipeline stage (general recipe)

1. Add a `PipelineStage` member (`models.py`) — it appears in logs, events, errors and the UI stepper
   (add an `<li data-stage>` in `index.html`).
2. Implement the stage as a class with an injected config sub-model; raise a
   `DialogueLocatorError` subclass with the new `stage` name.
3. Call it from `_Run._execute` between `_begin(...)` / `_end(...)`; add it to `DialoguePipeline.__init__`
   as an injectable dependency; add a fake in `test_pipeline.py`.
