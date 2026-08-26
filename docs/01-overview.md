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

Download the **audio only** (yt-dlp; a low video rendition on hosts without separate audio
streams), extract 16 kHz mono PCM (FFmpeg), then **stream** speech-to-text with a small
Faster-Whisper model. Every transcribed word is fed to a sliding-window fuzzy matcher
(RapidFuzz) as soon as it is decoded; the first window whose score reaches the threshold
(80/100) is the candidate, and the decoder is stopped there — so a line five minutes into a
55-minute video costs five minutes of transcription, not fifty-five. A **larger** Whisper model
then re-transcribes only ±12 s around the candidate to confirm the words and tighten the word
timestamps. Only now is full-quality video fetched — a few seconds of **clip** around the
verified match. The confirmed start time is converted to a frame number (`floor(t × fps)`),
that frame is read with OpenCV and saved as an image, and two visual checks gate the verdict:
is a face visible in the frame (V2), and do its lips actually move during the line (V3)? A
match that fails them is reported as `not_onscreen` — heard, but not an onscreen dialogue. If
nothing reaches the match threshold, the three best non-overlapping windows are reported so the
user can tell whether the dialogue text or the video is wrong.

## How the PS evaluation questions are answered

| Question | Answer | Where |
|---|---|---|
| How does the solution determine **where to look** in the video? | It doesn't guess — it listens from the start and stops at the first fuzzy match, using VAD to skip silence. | [matching](modules/matching.md), [transcription](modules/transcription.md) |
| How does it determine the **relevant frame**? | Start time of the first matched word (refined by the large model) → `floor(t × fps)`; frame read by index with OpenCV, ffmpeg fallback verified to agree within codec noise. | [media § frames](modules/media.md#frames) |
| How does it **extract the text**? | Word-level ASR (Faster-Whisper, word timestamps) — the returned text is what the *large* model heard in the confirmed window, unless the first pass already scored ≥ 90 (`skip_above_score`), in which case the re-check is skipped and the fast model's text stands. | [transcription](modules/transcription.md), [verification](modules/verification.md) |
| How are **ambiguous / uncertain** results handled? | Two-pass design (fast candidate → large-model confirmation, accepted unless it scores more than 5 points *below* the first pass), warnings when the verifier disagrees, near-miss report when nothing matches, first-occurrence semantics. See the JFK case study. | [decision log](09-decision-log.md), [verification](modules/verification.md) |
| Prompts used with an LLM | `prompts.txt` in the repository root. | — |

## Versions

* **V1** — spoken-dialogue localisation and frame extraction (streaming ASR + fuzzy match +
  large-model verification).
* **V2** — face-presence check on the extracted frame (MediaPipe BlazeFace); no face →
  `not_onscreen`.
* **V3** — mouth-movement check over the matched window (MediaPipe Face Landmarker); face
  present but lips still → `not_onscreen`.

All three are implemented; further visual stages plug in the same way (see
[Extending](08-extending.md)).

## Vital statistics

* ~8.5k lines under `src/` including tests; 270+ tests (`pytest`), all offline by default.
* Interfaces: CLI (`dialogue-locator`), HTTP API (`dialogue-locator-server`, FastAPI, OpenAPI at
  `/docs`), single-page web UI at `/`.
* Measured on an Apple M2 (CPU, int8): streaming pass `base` ≈ 12× realtime, `small` ≈ 3×;
  verification with the default `small` over the ±12 s window (≈ 26 s of audio) took 15.6 s in
  the reference run.
