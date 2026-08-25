# Module: `verification/`

Second-pass checks that **confirm, refine or reject** the first-pass candidate. This package is
also the designed extension point for V2 ("is the speaker on camera").

## `base.py`

```python
@dataclass
class VerificationContext:
    dialogue: str
    audio_samples: np.ndarray          # full track, float32 16 kHz — already in memory
    audio_path: Path
    sample_rate: int = 16_000
    video: VideoInfo | None = None     # lets visual verifiers open context.video.path
    extra: dict = {}                   # free-form for future verifiers
    audio_duration -> float
    def slice_audio(start, end) -> (samples, actual_start, actual_end)   # clamped to the track

class Verifier(ABC):
    name: str
    @abstractmethod
    def verify(self, candidate: MatchCandidate, context) -> VerificationOutcome
```

Contract: `verify()` **must not raise** for expected failures — return
`VerificationStatus.FAILED` with a message. A broken verifier must never lose a result the fast
pass already found. The pipeline folds outcomes in order (see pipeline doc): `CONFIRMED` with a
`refined` candidate replaces the current candidate; `REJECTED`/`FAILED` append a warning and keep
it; `SKIPPED` is silent.

## `asr_verifier.py`

```python
class AsrVerifier(Verifier):            # name = "asr_large_model"
    def __init__(self, transcriber: Transcriber, matching_config, config: VerificationConfig)
    def verify(candidate, context) -> VerificationOutcome
```

Steps:

1. `context.slice_audio(candidate.start − W, candidate.end + W)` with `W = search_window_seconds`
   (20 s) — a numpy slice, no ffmpeg, no disk.
2. `transcriber.transcribe_all(clip, offset=clip_start)` with the `verify_model` (default `medium`),
   so words carry absolute times.
3. `best_match(words, dialogue, matching_config)` — the strongest window in the clip.
4. Decision:
   * `best.score ≥ candidate.score − max_score_drop` (5) → **CONFIRMED**, `refined=best`
     (the pipeline uses the large model's timestamps and text).
   * otherwise → **REJECTED**, `refined=best` kept for diagnostics, message
     *"keeping first-pass timestamp"* → pipeline warning.
   * no words in the clip → REJECTED with score 0; `TranscriptionError` → **FAILED**;
     `enabled=False` → **SKIPPED** without loading the model; empty window (candidate beyond the
     audio) → FAILED.
5. `details` records `model, clip_start, clip_end, first_pass_score, seconds, clip_words,
   shift_seconds` — enough for the UI to explain why a timestamp moved.

### What the verifier catches (real runs)

* **Repetition ambiguity** (JFK, "We choose to go to the moon…" said three times; only the third
  continues with "in this decade…"): fast pass pinned the start to an earlier repeat; `medium`
  re-transcribed cleanly and moved it **+6.4 s** to the true first occurrence of the full line,
  score 100 → confirmed.
* **Large model being *wrong***: zoo clip, `base/small` heard "long trunks" (100), `medium` heard
  "long fronts" (92.9, a 7.1-point drop) → rejected, first-pass timestamp kept, warning shown.
  The 5-point rule works in both directions: it protects against the fast model *and* against the
  large model.
