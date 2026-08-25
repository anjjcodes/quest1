"""Second-pass ASR verification with a larger Whisper model.

The streaming pass uses a small, fast model, so individual words may be wrong
and word timestamps a little loose. Once it finds a candidate, this verifier:

1. cuts ``+/- search_window_seconds`` of audio around the candidate from the
   in-memory PCM (no disk I/O, no ffmpeg),
2. transcribes only that clip with the larger ``verify_model`` (absolute
   timestamps via ``offset=clip_start``),
3. picks the best-scoring window in the clip (:func:`best_match`), and
4. CONFIRMS (using the large model's timestamps) if its score is not more than
   ``max_score_drop`` points below the first-pass score; otherwise REJECTS and
   the pipeline keeps the first-pass timestamp with a warning.

A transcription failure yields ``FAILED`` rather than an exception: a broken
verifier must never lose a result the fast pass already found.
"""

from __future__ import annotations

import logging
import time

from dialogue_locator.config import MatchingConfig, VerificationConfig
from dialogue_locator.exceptions import DialogueLocatorError
from dialogue_locator.matching.matcher import best_match
from dialogue_locator.models import (
    MatchCandidate,
    VerificationOutcome,
    VerificationStatus,
    format_timestamp,
)
from dialogue_locator.transcription.base import Transcriber
from dialogue_locator.verification.base import VerificationContext, Verifier

logger = logging.getLogger(__name__)


class AsrVerifier(Verifier):
    name = "asr_large_model"

    def __init__(
        self,
        transcriber: Transcriber,
        matching_config: MatchingConfig,
        config: VerificationConfig,
    ) -> None:
        self._transcriber = transcriber
        self._matching = matching_config
        self._config = config

    def verify(self, candidate: MatchCandidate, context: VerificationContext) -> VerificationOutcome:
        if not self._config.enabled:
            logger.info("[%s] disabled by configuration; skipping", self.name)
            return VerificationOutcome(self.name, VerificationStatus.SKIPPED, message="Verification disabled")

        skip = self._config.skip_above_score
        if skip is not None and candidate.score >= skip:
            logger.info(
                "[%s] first-pass score %.1f >= %.1f; skipping re-transcription",
                self.name,
                candidate.score,
                skip,
            )
            return VerificationOutcome(
                self.name,
                VerificationStatus.SKIPPED,
                score=candidate.score,
                message=f"First-pass score {candidate.score:.1f} >= {skip:.1f}; no re-check",
                details={"first_pass_score": round(candidate.score, 2), "skip_above_score": skip},
            )

        window = self._config.search_window_seconds
        clip, clip_start, clip_end = context.slice_audio(candidate.start - window, candidate.end + window)
        details = {
            "model": self._transcriber.name,
            "clip_start": round(clip_start, 3),
            "clip_end": round(clip_end, 3),
            "first_pass_score": round(candidate.score, 2),
        }
        if clip.shape[0] == 0:
            logger.error("[%s] empty audio window %.2f-%.2f for candidate at %s", self.name, clip_start, clip_end, candidate.timestamp)
            return VerificationOutcome(
                self.name, VerificationStatus.FAILED, message="Empty audio window", details=details
            )

        logger.info(
            "[%s] re-transcribing %s - %s (%.1fs) with %s around candidate at %s",
            self.name,
            format_timestamp(clip_start),
            format_timestamp(clip_end),
            clip_end - clip_start,
            self._transcriber.name,
            candidate.timestamp,
        )
        t0 = time.perf_counter()
        try:
            words = self._transcriber.transcribe_all(clip, offset=clip_start)
        except DialogueLocatorError as exc:
            logger.error("[%s] transcription failed: %s", self.name, exc.message)
            details["error"] = exc.to_dict()
            return VerificationOutcome(
                self.name,
                VerificationStatus.FAILED,
                message=f"Verifier transcription failed: {exc.message}",
                details=details,
            )
        details["seconds"] = round(time.perf_counter() - t0, 2)
        details["clip_words"] = len(words)

        best = best_match(words, context.dialogue, self._matching) if words else None
        if best is None:
            logger.warning("[%s] large model produced no words in the window; rejecting", self.name)
            return VerificationOutcome(
                self.name,
                VerificationStatus.REJECTED,
                score=0.0,
                message="Large model produced no words in the search window",
                details=details,
            )

        details["shift_seconds"] = round(best.start - candidate.start, 3)
        drop = candidate.score - best.score
        floor = candidate.score - self._config.max_score_drop

        if best.score >= floor:
            logger.info(
                "[%s] CONFIRMED %.1f (first pass %.1f) at %s (shift %+.3fs): %r",
                self.name,
                best.score,
                candidate.score,
                best.timestamp,
                best.start - candidate.start,
                best.matched_text,
            )
            return VerificationOutcome(
                self.name,
                VerificationStatus.CONFIRMED,
                score=best.score,
                refined=best,
                message=f"Large model agrees (score {best.score:.1f} vs {candidate.score:.1f})",
                details=details,
            )

        logger.warning(
            "[%s] REJECTED: large model best %.1f is %.1f below first pass %.1f (allowed %.1f): %r",
            self.name,
            best.score,
            drop,
            candidate.score,
            self._config.max_score_drop,
            best.matched_text,
        )
        return VerificationOutcome(
            self.name,
            VerificationStatus.REJECTED,
            score=best.score,
            refined=best,
            message=(
                f"Large model scored {best.score:.1f}, more than {self._config.max_score_drop:.0f} "
                f"points below the first pass ({candidate.score:.1f}); keeping first-pass timestamp"
            ),
            details=details,
        )
