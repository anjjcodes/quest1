"""Dialogue Locator.

Finds the first video frame at which a given spoken dialogue occurs, using
streaming ASR (Faster-Whisper), fuzzy matching (RapidFuzz) and frame
extraction (FFmpeg / OpenCV).

The package is organised as a pipeline of independent stages (download ->
audio extraction -> streaming transcription + matching -> verification ->
frame extraction). Each stage lives in its own sub-package so it can be
tested, replaced or extended in isolation.
"""

__version__ = "1.0.0"
