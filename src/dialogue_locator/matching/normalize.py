"""Text normalisation shared by the matcher and the verifier.

Both the target dialogue and the ASR output are normalised the same way
before scoring, so differences in case, punctuation, quotes or Unicode form
never count against a match. The rules are intentionally conservative: we
only remove what carries no spoken information.

Rules
-----
1. Unicode NFKC (full-width chars, ligatures -> ASCII equivalents)
2. curly quotes / apostrophes -> straight ``'``
3. lower-case
4. hyphens and en/em dashes -> space (``well-known`` -> ``well known``)
5. drop every character that is not a letter, digit, apostrophe or space
6. drop apostrophes that are not inside a word (``'hello'`` -> ``hello``,
   ``don't`` stays ``don't``)
7. collapse whitespace
"""

from __future__ import annotations

import re
import unicodedata

_QUOTES = {
    "‘": "'",  # left single
    "’": "'",  # right single (most common apostrophe in subtitles)
    "‚": "'",
    "‛": "'",
    "′": "'",  # prime
    "“": '"',
    "”": '"',
    "„": '"',
}
_DASHES = re.compile(r"[-‐-―−]+")
_NOT_ALLOWED = re.compile(r"[^a-z0-9' ]+")
_LONE_APOSTROPHE = re.compile(r"(?<![a-z0-9])'|'(?![a-z0-9])")
_WHITESPACE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    """Return the canonical, comparable form of ``text`` (may be empty)."""
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = "".join(_QUOTES.get(ch, ch) for ch in text)
    text = text.lower()
    text = _DASHES.sub(" ", text)
    text = _NOT_ALLOWED.sub(" ", text)
    text = _LONE_APOSTROPHE.sub("", text)
    return _WHITESPACE.sub(" ", text).strip()


def tokenize(text: str) -> list[str]:
    """Normalise and split into words. Empty tokens are dropped."""
    normalized = normalize_text(text)
    return normalized.split(" ") if normalized else []


def normalize_word(word: str) -> str:
    """Normalise a single ASR word. May return ``""`` (e.g. a lone ``"..."``).

    Note: a single ASR "word" can occasionally contain a space (``"New York"``);
    we keep it as one token so indices stay aligned with the ``Word`` stream.
    """
    return normalize_text(word)
