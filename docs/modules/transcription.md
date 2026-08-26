# Module: `transcription/`

Speech-to-text as a **lazy stream of timestamped words**. Two files: the contract
(`base.py`) and the Faster-Whisper implementation (`faster_whisper.py`).

## `base.py`

```python
TARGET_SAMPLE_RATE = 16_000

def load_pcm(path, expected_rate=16_000) -> np.ndarray     # 16-bit WAV → float32 in [-1, 1]; downmixes stereo
def pcm_duration(samples, rate=16_000) -> float

class Transcriber(ABC):
    name: str                                              # e.g. "faster_whisper:base"
    def warm_up(self) -> None                              # optional: load models now
    @abstractmethod
    def transcribe(self, audio: np.ndarray | Path, *, offset=0.0, progress=None) -> Iterator[Word]
    def transcribe_all(self, audio, *, offset=0.0) -> list[Word]
    @staticmethod
    def as_samples(audio) -> np.ndarray
```

### The streaming contract (the important part)

* `transcribe()` is a **generator**. Words are yielded in chronological order as soon as the
  model finishes each segment, with `offset` added to every timestamp.
* The consumer may `break`/`close()` at any time, and the implementation **must not do work beyond
  what was consumed**. This is what makes "stop at the first match" cost only the audio up to
  the match. Test: `test_streaming_is_lazy_and_stops_early` proves only one segment is decoded when
  one word is pulled.
* `offset` lets the verifier transcribe a clip cut from the middle of the track and still get
  absolute video times — no timeline arithmetic leaks into the matcher.

### Why we decode the WAV ourselves

`load_pcm` uses the stdlib `wave` + numpy instead of letting faster-whisper decode via PyAV:
PCM decoding is trivial, it validates the format up front (wrong rate → `TranscriptionError`
telling you to run `AudioExtractor`), avoids a second FFmpeg binding at runtime, and yields an
in-memory array the verifier can slice without touching disk.

## `faster_whisper.py`

```python
class WhisperModelCache:                       # process-wide, thread-safe
    def get(self, model_name, config: WhisperConfig) -> WhisperModel   # keyed by (name, device, compute_type, cpu_threads, download_root)
    def clear(self)
default_model_cache = WhisperModelCache()

class FasterWhisperTranscriber(Transcriber):
    def __init__(self, model_name, config: WhisperConfig, model_cache=None)
    def warm_up(self)
    def transcribe(self, audio, *, offset=0.0, progress=None) -> Iterator[Word]
```

### Decode options (from `WhisperConfig`)

`word_timestamps=True`, `beam_size`, `vad_filter` (+ `min_silence_duration_ms`),
`condition_on_previous_text` (default **False** — prevents the repetition loops Whisper falls into
on long streams), `language` (default `en`; `None` → auto-detect).

### Word handling

* Faster-Whisper returns words like `" there,"` — leading space and punctuation. We strip
  whitespace only; punctuation is left for the matcher's normaliser. Empty tokens are dropped.
* If a segment has **no word alignment** (rare Whisper failure), its text is split and the words
  are spread evenly across the segment's time span, with a WARNING. Matching still works;
  timestamps degrade gracefully instead of the segment vanishing. Tested with `caplog`.
* VAD-filtered timestamps are already mapped back to the original timeline by faster-whisper.

### Progress

Every 2 % of the audio, `ProgressEvent(TRANSCRIPTION, "Transcribed HH:MM:SS / HH:MM:SS", fraction,
{words, position})`. The UI shows this as the progress bar during the streaming pass.

### Errors

Model load failure (no network, bad name) → `TranscriptionError("Failed to load …")`. A crash while
decoding → `TranscriptionError("… while decoding")`, after logging how many words were produced.
`GeneratorExit` (consumer stopped) is logged at INFO and re-raised — it is not an error.

### Performance (Apple M2, CPU, int8, steady state)

| model | realtime factor | notes |
|---|---|---|
| tiny | ~40× | rough text |
| base | ~11× live, ~40-60× isolated | **streaming default** (greedy; ~1.4× slower at beam 5) |
| small | ~3× | **verification default** (`verify_model`); the ±12 s window (≈ 28 s of audio) took 21.4 s in the reference run |
| medium | ~0.6× | optional, slower verification |

"Live" is inside a running server (API process, warm-up thread, both models resident);
"isolated" is a single process on an idle machine. The reference run `df328328a2e7` scanned
330.3 s of audio in 29.5 s.

These assume `cpu_threads` is sized to the machine's **performance** cores, which is what the
default of `0` now does. Handing ctranslate2 every core instead costs ~17× on a big.LITTLE CPU
(M2: 5.4 s vs 90.9 s for the same 331 s scan) — see `performance_cores()` in
`transcription/faster_whisper.py`.

Beam size 1 vs 5 is worth ~1.4× on the streaming pass (8.1 s vs 11.3 s over 331 s of audio,
identical output), which is why `fast_beam_size` defaults to 1 while the verify pass keeps
the full beam.
