# 1. Overview

## The problem

Given a video URL (or file) and a line of dialogue, find the **first video frame** at which the
line is spoken and return:

* the timestamp (`HH:MM:SS.mmm`),
* the frame number,
* the matched text as heard in the video,
* the frame as an image.

Reference input from the problem statement: `https://ok.ru/video/248244667877`,
dialogue *"My mind rebels at stagnation"*. The evaluators may use a different video/dialogue, so
nothing is hard-coded for that case.

## The approach in one paragraph

Download the video (yt-dlp), extract 16 kHz mono audio (FFmpeg), then **stream** speech-to-text
with a small Faster-Whisper model. Every transcribed word is fed to a sliding-window fuzzy
matcher (RapidFuzz) as soon as it is decoded; the first window whose score reaches the threshold
(80/100) is the candidate, and the decoder is stopped there — so a line five minutes into a
55-minute video costs five minutes of transcription, not fifty-five. A **larger** Whisper model
then re-transcribes only ±20 s around the candidate to confirm the words and tighten the word
timestamps. The confirmed start time is converted to a frame number (`floor(t × fps)`) and that
frame is read with OpenCV and saved as an image. If nothing reaches the threshold, the three best
non-overlapping windows are reported so the user can tell whether the dialogue text or the video
is wrong.

## How the PS evaluation questions are answered

| Question | Answer | Where |
|---|---|---|
| How does the solution determine **where to look** in the video? | It doesn't guess — it listens from the start and stops at the first fuzzy match, using VAD to skip silence. | [matching](modules/matching.md), [transcription](modules/transcription.md) |
| How does it determine the **relevant frame**? | Start time of the first matched word (refined by the large model) → `floor(t × fps)`; frame read by index with OpenCV, ffmpeg fallback verified pixel-for-pixel. | [media § frames](modules/media.md#frames) |
| How does it **extract the text**? | Word-level ASR (Faster-Whisper, word timestamps) — the returned text is what the *large* model heard in the confirmed window. | [transcription](modules/transcription.md), [verification](modules/verification.md) |
| How are **ambiguous / uncertain** results handled? | Two-pass design (fast candidate → large-model confirmation with a ±5-point rule), warnings when the verifier disagrees, near-miss report when nothing matches, first-occurrence semantics. See the JFK case study. | [decision log](09-decision-log.md), [verification](modules/verification.md) |
| Prompts used with an LLM | `prompts.txt` in the repository root. | — |

## Versions

* **V1 (this code)** — spoken-dialogue localisation and frame extraction.
* **V2 (planned)** — "is the speaker on camera" visual verification, designed as an additional
  `Verifier` so nothing upstream changes. See [Extending](08-extending.md).

## Vital statistics

* ~6.5k lines under `src/` including tests; 148 tests (`pytest`), all offline by default.
* Interfaces: CLI (`dialogue-locator`), HTTP API (`dialogue-locator-server`, FastAPI, OpenAPI at
  `/docs`), single-page web UI at `/`.
* Measured on an Apple M2 (CPU, int8): streaming pass `base` ≈ 12× realtime, `small` ≈ 3×;
  verification with `medium` ≈ 0.6× realtime on a ~40 s window (≈ 30–60 s).
