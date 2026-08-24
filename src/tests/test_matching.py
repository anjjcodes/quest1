"""Tests for text normalisation and the sliding-window matcher (pure Python, no models)."""

from __future__ import annotations

import logging

import pytest

from dialogue_locator.config import MatchingConfig
from dialogue_locator.exceptions import InvalidDialogueError
from dialogue_locator.matching.matcher import (
    NearMissTracker,
    StreamingMatcher,
    TargetDialogue,
    WindowScorer,
    best_match,
)
from dialogue_locator.matching.normalize import normalize_text, normalize_word, tokenize
from dialogue_locator.models import MatchCandidate, Word
from tests.conftest import expect_error

logger = logging.getLogger("tests")

DIALOGUE = "My mind rebels at stagnation"


# --------------------------------------------------------------------------- #
# normalize
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "raw, expected",
    [
        ("My mind rebels at stagnation.", "my mind rebels at stagnation"),
        ("  Hello,   WORLD!!  ", "hello world"),
        ("don't stop", "don't stop"),
        ("don’t stop", "don't stop"),  # curly apostrophe
        ("'quoted' word", "quoted word"),  # lone apostrophes dropped
        ("“Smart quotes”", "smart quotes"),
        ("well-known re—sult", "well known re sult"),
        ("café naïve", "caf na ve"),  # non-ASCII letters dropped (English-only V1)
        ("Ｈｅｌｌｏ", "hello"),  # full-width -> ASCII via NFKC
        ("...", ""),
        ("", ""),
        ("Room 101, now!", "room 101 now"),
    ],
)
def test_normalize_text(raw, expected):
    assert normalize_text(raw) == expected


def test_tokenize_and_normalize_word():
    assert tokenize("  My mind, rebels...  ") == ["my", "mind", "rebels"]
    assert tokenize("") == []
    assert normalize_word(" stagnation.") == "stagnation"
    assert normalize_word("...") == ""


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def make_words(text: str, start: float = 0.0, step: float = 0.5) -> list[Word]:
    """Turn a sentence into Words with evenly spaced timestamps."""
    out = []
    for i, tok in enumerate(text.split()):
        t = start + i * step
        out.append(Word(text=tok, start=t, end=t + step * 0.8))
    return out


def run(text: str, dialogue: str = DIALOGUE, **cfg) -> tuple[StreamingMatcher, MatchCandidate | None]:
    matcher = StreamingMatcher(dialogue, MatchingConfig(**cfg))
    return matcher, matcher.feed_many(make_words(text))


# --------------------------------------------------------------------------- #
# TargetDialogue / config
# --------------------------------------------------------------------------- #
def test_target_dialogue_parse():
    t = TargetDialogue.parse("  My mind, rebels at STAGNATION!  ", MatchingConfig())
    assert t.normalized == "my mind rebels at stagnation"
    assert t.word_count == 5
    assert t.raw == "My mind, rebels at STAGNATION!"


@pytest.mark.parametrize("bad", ["", "   ", "...", "!!!"])
def test_empty_dialogue_rejected(bad):
    with expect_error(InvalidDialogueError, match="empty"):
        StreamingMatcher(bad, MatchingConfig())


def test_too_short_dialogue_rejected():
    with expect_error(InvalidDialogueError, match="at least 2 words") as exc:
        StreamingMatcher("stagnation", MatchingConfig())
    assert exc.value.stage == "input"
    # but allowed when configured
    StreamingMatcher("stagnation", MatchingConfig(min_dialogue_words=1))


def test_window_sizes_follow_tolerance():
    cfg = MatchingConfig(window_tolerance=2)
    assert WindowScorer(TargetDialogue.parse(DIALOGUE, cfg), cfg).window_sizes == (3, 4, 5, 6, 7)
    cfg = MatchingConfig(window_tolerance=0)
    assert WindowScorer(TargetDialogue.parse(DIALOGUE, cfg), cfg).window_sizes == (5,)
    cfg = MatchingConfig(window_tolerance=3)  # never below 1 word
    assert WindowScorer(TargetDialogue.parse("two words", cfg), cfg).window_sizes == (1, 2, 3, 4, 5)


# --------------------------------------------------------------------------- #
# StreamingMatcher: matches
# --------------------------------------------------------------------------- #
def test_exact_match_pins_first_word_timestamp():
    m, match = run("I need work. My mind rebels at stagnation. Give me problems!")
    assert match is not None
    assert match.score == 100.0
    assert match.matched_text == "My mind rebels at stagnation."
    assert match.word_index_start == 3 and match.word_index_end == 8
    assert match.start == pytest.approx(1.5)  # word 3 at 0.5 s steps
    assert m.match is match


def test_stops_consuming_after_match():
    """After the match is final, further words are ignored (stream can be closed)."""
    words = make_words("x y my mind rebels at stagnation a b c d e f")
    m = StreamingMatcher(DIALOGUE, MatchingConfig(window_tolerance=2))
    consumed = 0
    for w in words:
        consumed += 1
        if m.feed(w) is not None:
            break
    # match confirmed at word index 6, then settled over 2 more words -> 9 consumed
    assert consumed == 9
    assert m.words_seen == 9


@pytest.mark.parametrize(
    "asr_text",
    [
        "my mind reveals a stagnation",  # substitutions
        "my mind rebels at stag nation",  # split word
        "my mine rebels at stagnation",  # small typo
        "my mindrebels at stagnation",  # merged words
        "your mind rebels at stagnation",  # first word wrong
    ],
)
def test_asr_errors_still_match(asr_text):
    _, match = run(f"blah blah {asr_text} blah")
    assert match is not None, asr_text
    assert match.score >= 80


def test_dialogue_at_very_start_of_stream():
    _, match = run("my mind rebels at stagnation and then more")
    assert match is not None and match.word_index_start == 0


def test_dialogue_at_very_end_of_stream_is_flushed_by_finish():
    """Match found on the last word: settle window never fills, finish() must confirm it."""
    m = StreamingMatcher(DIALOGUE, MatchingConfig(window_tolerance=2))
    result = None
    for w in make_words("some words my mind rebels at stagnation"):
        result = m.feed(w)
    assert result is None  # still settling
    assert m.finish() is not None
    assert m.match.matched_text == "my mind rebels at stagnation"


def test_first_occurrence_wins():
    _, match = run("my mind rebels at stagnation ... later ... my mind rebels at stagnation")
    assert match is not None and match.word_index_start == 0


def test_best_size_is_chosen_not_first_size_over_threshold():
    """'the my mind rebels at stagnation' (size 6) scores ~93; size 5 scores 100 -> start at 'my'."""
    _, match = run("the my mind rebels at stagnation")
    assert match is not None
    assert match.score == 100.0
    assert match.matched_text == "my mind rebels at stagnation"
    assert match.word_index_start == 1


def test_settling_prefers_later_complete_window():
    """Long dialogue: a window missing the last word clears the threshold one position early."""
    dialogue = "we choose to go to the moon in this decade and do the other things"
    words = make_words("ladies and gentlemen " + dialogue + " not because they are easy")
    m = StreamingMatcher(dialogue, MatchingConfig(window_tolerance=2))
    match = m.feed_many(words)
    assert match is not None
    assert match.score == 100.0
    assert match.matched_text == dialogue


def test_settling_disabled_with_zero_tolerance():
    m = StreamingMatcher(DIALOGUE, MatchingConfig(window_tolerance=0))
    words = make_words("my mind rebels at stagnation")
    assert m.feed_many(words[:4]) is None
    # exact-size window completes on the 5th word and is returned immediately
    assert m.feed(words[4]) is not None


def test_case_and_punctuation_insensitive():
    _, match = run("MY MIND, REBELS... AT STAGNATION!", dialogue="my mind rebels at stagnation")
    assert match is not None and match.score == 100.0
    assert match.matched_text == "MY MIND, REBELS... AT STAGNATION!"  # raw ASR text preserved


def test_threshold_is_configurable():
    text = "my mind reveals a station"  # ~85 vs target
    _, match = run(text, match_threshold=80)
    assert match is not None
    _, match = run(text, match_threshold=95)
    assert match is None


def test_custom_scorer_is_used():
    m = StreamingMatcher(DIALOGUE, MatchingConfig(), scorer=lambda a, b: 100.0)
    assert m.feed_many(make_words("anything at all works")) is not None


# --------------------------------------------------------------------------- #
# StreamingMatcher: no match + near misses
# --------------------------------------------------------------------------- #
def test_no_match_reports_top_k_non_overlapping_near_misses():
    text = (
        "the weather is nice today "  # unrelated
        "my mind rebels quite often "  # partial (~75)
        "some more filler words here "
        "his mind revolts at stagnant water "  # different partial
        "and finally nothing relevant"
    )
    m, match = run(text, top_k_near_misses=3)
    assert match is None
    near = m.near_misses
    assert 1 <= len(near) <= 3
    scores = [c.score for c in near]
    assert scores == sorted(scores, reverse=True)
    assert all(c.score < 80 for c in near)
    # mutually non-overlapping regions
    for i, a in enumerate(near):
        for b in near[i + 1 :]:
            assert a.word_index_end <= b.word_index_start or b.word_index_end <= a.word_index_start
    assert m.best_score == near[0].score
    logger.info("near misses: %s", [(round(c.score, 1), c.matched_text) for c in near])


def test_near_misses_empty_for_empty_stream():
    m = StreamingMatcher(DIALOGUE, MatchingConfig())
    assert m.finish() is None
    assert m.near_misses == [] and m.best_score == 0.0 and m.words_seen == 0


def test_near_miss_tracker_keeps_best_per_region():
    tr = NearMissTracker(k=2)

    def cand(score, s, e):
        return MatchCandidate(score=score, start=s, end=e, matched_text="", word_index_start=s, word_index_end=e)

    tr.offer(cand(50, 0, 5))
    tr.offer(cand(60, 2, 7))  # overlaps -> replaces
    tr.offer(cand(55, 10, 15))
    tr.offer(cand(40, 20, 25))  # 3rd region, k=2 -> dropped
    assert [(c.score, c.word_index_start) for c in tr.items] == [(60, 2), (55, 10)]
    tr.offer(cand(58, 12, 16))  # overlaps region 2 with higher score -> replaces
    assert [(c.score, c.word_index_start) for c in tr.items] == [(60, 2), (58, 12)]
    tr.offer(cand(30, 12, 16))  # lower than existing overlapping -> ignored
    assert len(tr.items) == 2 and tr.items[1].score == 58


def test_empty_normalized_words_are_skipped_in_window_text():
    """ASR tokens like '...' normalise to '' and must not pad the window text."""
    _, match = run("my ... mind rebels at stagnation")
    assert match is not None
    assert match.score >= 95


# --------------------------------------------------------------------------- #
# best_match (batch, for the verifier)
# --------------------------------------------------------------------------- #
def test_best_match_returns_highest_score_not_first():
    words = make_words("my mind rebels quite a lot and my mind rebels at stagnation")
    best = best_match(words, DIALOGUE, MatchingConfig())
    assert best is not None and best.score == 100.0
    assert best.matched_text == "my mind rebels at stagnation"


def test_best_match_below_threshold_still_returned():
    words = make_words("something completely unrelated here")
    best = best_match(words, DIALOGUE, MatchingConfig())
    assert best is not None and best.score < 80


def test_best_match_empty():
    assert best_match([], DIALOGUE, MatchingConfig()) is None
