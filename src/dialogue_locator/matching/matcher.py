"""Sliding-window fuzzy matching of a word stream against a target dialogue.

Two entry points:

* :class:`StreamingMatcher` - incremental. ``feed(word)`` is called for every
  transcribed word as it arrives and returns the *first* confirmed match, so
  the caller can stop transcribing immediately. Also tracks the best
  near-misses for the "nothing found" report.
* :func:`best_match` - batch. Scores every window in a finished word list and
  returns the best one. Used by the verifier on the re-transcribed clip.

Algorithm
---------
Let ``n`` = number of words in the (normalised) dialogue and ``t`` =
``window_tolerance``. For every stream position ``i`` we look at the windows
*ending* at ``i`` with sizes ``n-t .. n+t`` (ASR may split or merge words) and
score each as ``fuzz.ratio(normalised window text, normalised dialogue)``.
Each window is therefore scored exactly once, when its last word arrives.

At a given end position several sizes can clear the threshold (a window with
one spurious leading word still scores ~93), so we take the *best-scoring*
size at that position - that is what pins the start timestamp to the right
word.

Settling: once a position produces a score >= threshold we do not return
immediately. We consume up to ``t`` further words and, if a window overlapping
the candidate scores strictly higher (e.g. the ASR emitted the final word one
position later), we prefer it. Then the match is final. This costs at most
``t`` extra words of transcription and prevents truncated matches on long
dialogues.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable
from dataclasses import dataclass

from rapidfuzz import fuzz

from dialogue_locator.config import MatchingConfig
from dialogue_locator.exceptions import InvalidDialogueError
from dialogue_locator.matching.normalize import normalize_word, tokenize
from dialogue_locator.models import MatchCandidate, Word, format_timestamp

logger = logging.getLogger(__name__)

#: Scorer signature: (candidate_text, target_text) -> 0..100
Scorer = Callable[[str, str], float]


# --------------------------------------------------------------------------- #
# Target dialogue
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class TargetDialogue:
    """The dialogue to look for, in raw and normalised form."""

    raw: str
    normalized: str
    tokens: tuple[str, ...]

    @property
    def word_count(self) -> int:
        return len(self.tokens)

    @classmethod
    def parse(cls, dialogue: str, config: MatchingConfig) -> TargetDialogue:
        raw = (dialogue or "").strip()
        tokens = tokenize(raw)
        if not raw or not tokens:
            logger.warning("Rejected dialogue: empty after normalisation (%r)", dialogue)
            raise InvalidDialogueError("Dialogue text is empty.", details={"dialogue": dialogue})
        if len(tokens) < config.min_dialogue_words:
            logger.warning(
                "Rejected dialogue %r: %d word(s), minimum is %d", raw, len(tokens), config.min_dialogue_words
            )
            raise InvalidDialogueError(
                f"Dialogue must contain at least {config.min_dialogue_words} words "
                f"(got {len(tokens)}): single words match far too easily.",
                details={"dialogue": raw, "word_count": len(tokens)},
            )
        return cls(raw=raw, normalized=" ".join(tokens), tokens=tuple(tokens))


# --------------------------------------------------------------------------- #
# Window scoring (shared by streaming and batch modes)
# --------------------------------------------------------------------------- #
class WindowScorer:
    """Scores word windows against a :class:`TargetDialogue`."""

    def __init__(self, target: TargetDialogue, config: MatchingConfig, scorer: Scorer = fuzz.ratio) -> None:
        self.target = target
        self.config = config
        self._scorer = scorer
        n, t = target.word_count, config.window_tolerance
        self.window_sizes: tuple[int, ...] = tuple(k for k in range(n - t, n + t + 1) if k >= 1)

    def score_text(self, text: str) -> float:
        return float(self._scorer(text, self.target.normalized))

    def best_window_ending_at(
        self, words: list[Word], tokens: list[str], end_index: int
    ) -> MatchCandidate | None:
        """Best-scoring window (over all sizes) whose last word is ``words[end_index]``."""
        best: MatchCandidate | None = None
        for size in self.window_sizes:
            start = end_index - size + 1
            if start < 0:
                continue
            text = " ".join(tok for tok in tokens[start : end_index + 1] if tok)
            if not text:
                continue
            score = self.score_text(text)
            if best is None or score > best.score:
                best = self.make_candidate(words, score, start, end_index + 1)
        return best

    @staticmethod
    def make_candidate(words: list[Word], score: float, start: int, end: int) -> MatchCandidate:
        window = words[start:end]
        return MatchCandidate(
            score=score,
            start=window[0].start,
            end=window[-1].end,
            matched_text=" ".join(w.text for w in window),
            words=tuple(window),
            word_index_start=start,
            word_index_end=end,
        )


# --------------------------------------------------------------------------- #
# Near-miss tracking
# --------------------------------------------------------------------------- #
class NearMissTracker:
    """Keeps the ``k`` best-scoring, mutually non-overlapping windows."""

    def __init__(self, k: int) -> None:
        self._k = k
        self._items: list[MatchCandidate] = []

    def offer(self, cand: MatchCandidate) -> None:
        # Overlapping windows describe the same region: keep only the best of them.
        for i, existing in enumerate(self._items):
            if _overlaps(existing, cand):
                if cand.score > existing.score:
                    self._items[i] = cand
                    self._items.sort(key=lambda c: -c.score)
                return
        self._items.append(cand)
        self._items.sort(key=lambda c: -c.score)
        del self._items[self._k :]

    @property
    def items(self) -> list[MatchCandidate]:
        return list(self._items)


def _overlaps(a: MatchCandidate, b: MatchCandidate) -> bool:
    if a.word_index_start is None or b.word_index_start is None:
        return not (a.end <= b.start or b.end <= a.start)
    return a.word_index_start < b.word_index_end and b.word_index_start < a.word_index_end  # type: ignore[operator]


# --------------------------------------------------------------------------- #
# Streaming matcher
# --------------------------------------------------------------------------- #
class StreamingMatcher:
    """Find the first occurrence of a dialogue in a stream of words.

    Usage::

        matcher = StreamingMatcher("My mind rebels at stagnation", config)
        for word in transcriber.transcribe(audio):
            if (match := matcher.feed(word)) is not None:
                break
        match = match or matcher.finish()      # flush a pending candidate at end of stream
        if match is None:
            report(matcher.near_misses)
    """

    def __init__(self, dialogue: str, config: MatchingConfig, scorer: Scorer = fuzz.ratio) -> None:
        self.config = config
        self.target = TargetDialogue.parse(dialogue, config)
        self._scorer = WindowScorer(self.target, config, scorer)
        self._words: list[Word] = []
        self._tokens: list[str] = []
        self._near = NearMissTracker(config.top_k_near_misses)
        self._pending: MatchCandidate | None = None
        self._settle_left = 0
        self._final: MatchCandidate | None = None
        self.best_score: float = 0.0
        logger.info(
            "Matcher ready: target=%r (%d words), window sizes %s, threshold %.1f",
            self.target.normalized,
            self.target.word_count,
            list(self._scorer.window_sizes),
            config.match_threshold,
        )

    # ------------------------------------------------------------------ #
    @property
    def words_seen(self) -> int:
        return len(self._words)

    @property
    def match(self) -> MatchCandidate | None:
        """The confirmed match, if any (set by ``feed``/``finish``)."""
        return self._final

    @property
    def near_misses(self) -> list[MatchCandidate]:
        """Best windows below the threshold, highest score first."""
        return self._near.items

    # ------------------------------------------------------------------ #
    def feed(self, word: Word) -> MatchCandidate | None:
        """Consume one word. Returns the final match once confirmed, else ``None``."""
        if self._final is not None:
            return self._final

        self._words.append(word)
        self._tokens.append(normalize_word(word.text))
        end_index = len(self._words) - 1

        cand = self._scorer.best_window_ending_at(self._words, self._tokens, end_index)
        if cand is None:
            return None

        if cand.score > self.best_score:
            self.best_score = cand.score
            logger.debug(
                "New best %.1f at %s: %r", cand.score, format_timestamp(cand.start), cand.matched_text
            )

        if self._pending is None:
            if cand.score >= self.config.match_threshold:
                logger.info(
                    "Candidate match %.1f at %s (words %d-%d): %r - settling over next %d word(s)",
                    cand.score,
                    format_timestamp(cand.start),
                    cand.word_index_start,
                    cand.word_index_end,
                    cand.matched_text,
                    self.config.window_tolerance,
                )
                self._pending = cand
                self._settle_left = self.config.window_tolerance
                if self._settle_left == 0:
                    return self._finalize()
            else:
                self._near.offer(cand)
            return None

        # Settling: prefer a strictly better overlapping window.
        if cand.score > self._pending.score and _overlaps(cand, self._pending):
            logger.info(
                "Settling improved candidate %.1f -> %.1f: %r",
                self._pending.score,
                cand.score,
                cand.matched_text,
            )
            self._pending = cand
        self._settle_left -= 1
        if self._settle_left <= 0:
            return self._finalize()
        return None

    def feed_many(self, words: Iterable[Word]) -> MatchCandidate | None:
        """Feed words until a match is confirmed or the iterable is exhausted."""
        for word in words:
            if (match := self.feed(word)) is not None:
                return match
        return self.finish()

    def finish(self) -> MatchCandidate | None:
        """Signal end of stream: confirm a still-settling candidate, if any."""
        if self._final is None and self._pending is not None:
            return self._finalize()
        if self._final is None:
            logger.info(
                "No match: %d words scanned, best score %.1f (threshold %.1f), %d near-miss(es) kept",
                len(self._words),
                self.best_score,
                self.config.match_threshold,
                len(self._near.items),
            )
        return self._final

    def _finalize(self) -> MatchCandidate:
        assert self._pending is not None
        self._final = self._pending
        self._pending = None
        logger.info(
            "MATCH %.1f at %s - %s (words %d-%d): %r",
            self._final.score,
            format_timestamp(self._final.start),
            format_timestamp(self._final.end),
            self._final.word_index_start,
            self._final.word_index_end,
            self._final.matched_text,
        )
        return self._final


# --------------------------------------------------------------------------- #
# Batch matcher
# --------------------------------------------------------------------------- #
def best_match(
    words: list[Word], dialogue: str, config: MatchingConfig, scorer: Scorer = fuzz.ratio
) -> MatchCandidate | None:
    """Return the single best-scoring window over ``words`` (any score), or ``None``.

    Used by the verification stage, which wants the best window inside a short
    clip rather than the first one above a threshold. Ties are resolved in
    favour of the earlier window.
    """
    target = TargetDialogue.parse(dialogue, config)
    window_scorer = WindowScorer(target, config, scorer)
    tokens = [normalize_word(w.text) for w in words]
    best: MatchCandidate | None = None
    for end_index in range(len(words)):
        cand = window_scorer.best_window_ending_at(words, tokens, end_index)
        if cand is not None and (best is None or cand.score > best.score):
            best = cand
    if best is not None:
        logger.debug(
            "best_match over %d words: %.1f at %s: %r",
            len(words),
            best.score,
            format_timestamp(best.start),
            best.matched_text,
        )
    return best
