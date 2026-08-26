# Module: `matching/`

Pure Python, no models, fully unit-tested (43 tests, ~0.2 s). Two files.

## `normalize.py`

```python
def normalize_text(text) -> str      # canonical comparable form
def tokenize(text) -> list[str]      # normalise + split
def normalize_word(word) -> str      # for a single ASR token (may return "")
```

Rules, applied identically to the dialogue and to every ASR word:

1. Unicode NFKC (full-width → ASCII, ligatures)
2. curly quotes/apostrophes → straight `'`
3. lower-case
4. hyphens/dashes → space (`well-known` → `well known`)
5. drop everything except `[a-z0-9' ]`
6. drop apostrophes not inside a word (`'hello'` → `hello`, `don't` stays)
7. collapse whitespace

V1 limitation: non-ASCII letters are removed (English only). Widening rule 5 is a one-regex change.

## `matcher.py`

### Types

```python
@dataclass(frozen=True)
class TargetDialogue: raw, normalized, tokens; word_count
    @classmethod parse(dialogue, config) -> TargetDialogue    # InvalidDialogueError if empty or < min_dialogue_words

class WindowScorer:                       # shared by streaming and batch modes
    window_sizes: tuple[int, ...]         # n-t … n+t, never < 1
    def best_window_ending_at(words, tokens, end_index) -> MatchCandidate | None

class NearMissTracker:                    # keeps the k best MUTUALLY NON-OVERLAPPING windows
    def offer(cand); items

class StreamingMatcher:
    def __init__(self, dialogue, config: MatchingConfig, scorer=fuzz.ratio)
    def feed(self, word: Word) -> MatchCandidate | None     # returns the FINAL match once confirmed
    def feed_many(self, words) -> MatchCandidate | None
    def finish(self) -> MatchCandidate | None               # end of stream: confirm a settling candidate
    match, near_misses, best_score, words_seen

def best_match(words, dialogue, config, scorer=fuzz.ratio) -> MatchCandidate | None   # batch; best window of any score
```

### Algorithm

Let `n` = dialogue word count, `t` = `window_tolerance` (default 2).

1. For each new word at stream position `i`, score every window **ending at `i`** with sizes
   `n−t … n+t`: `fuzz.ratio(normalised window text, normalised dialogue)`. Each window is thus
   scored exactly once, when its last word arrives.
2. Keep the **best-scoring size** at that position. This matters: with the target
   `my mind rebels at stagnation`, the 6-word window `the my mind rebels at stagnation` scores 93.3
   and the 4-word `mind rebels at stagnation` 94.3 — both clear 80 — but the 5-word window scores
   100 and pins the start timestamp to the right word.
3. If the best score ≥ `match_threshold` → it becomes a **pending** candidate and the matcher
   **settles**: it consumes up to `t` more words and replaces the candidate with any *overlapping*
   window that scores strictly higher. Then the match is final and `feed()` returns it.
   Motivation: for long dialogues a window missing its last word already clears 80; without
   settling `matched_text` would be truncated. Log from the test: `81.7 → 89.1 → 94.4 → 100.0`.
   `window_tolerance=0` disables settling; `finish()` flushes a candidate still settling at the
   end of the stream.
4. Below threshold → offered to the `NearMissTracker`. Overlapping windows describe the same
   region, so only the best per region is kept — otherwise the "top 3" would be three
   near-identical windows from one spot.
5. **First occurrence wins** by construction: the stream is chronological and we stop at the first
   confirmed match.

### Scoring facts (RapidFuzz `ratio`, target `my mind rebels at stagnation`)

| candidate text | score | meaning |
|---|---|---|
| `my mind rebels at stagnation` | 100 | exact |
| `my mind reveals a stagnation` | 92.9 | ASR substitution |
| `my mind rebels at stag nation` | 98.2 | split word |
| `my mine rebels at stagnation` | 96.4 | small error |
| `your mind rebels at stagnation` | 93.1 | wrong first word |
| `my mind rebels` | 66.7 | partial — below threshold, good |
| `i need work my mind rebels` | 51.9 | wrong window |

80 therefore separates "real match with ASR noise" from "partial/wrong window" comfortably.

### `best_match` (batch)

Used by the verifier on the re-transcribed clip: scores every window and returns the single best
regardless of threshold (the verifier compares it to the first-pass score). Ties → earlier window.

### Why the matcher stays permissive

A "maximum time gap inside a window" rule was considered and rejected: in the JFK case the fast
model dropped a repeated phrase, and a gap rule would have turned a verifiable candidate into
"not found" — the verifier would never have run. The first pass generates candidates; the
verifier provides precision. See the decision log.
