# 7. Operations & troubleshooting

## Disk layout

| Path | Contents | Lifetime |
|---|---|---|
| `data/work/<sha1(source)[:16]>/` | `media.<ext>` (the audio-first search fetch), `audio.wav`, `clip_<startms>_<endms>.mp4` (+ `.part` while downloading) | kept while `storage.keep_intermediate=True` (default); reused across dialogues for the same source |
| `data/output/<job_id>/` | `frame.jpg`, `result.json` | permanent (delete manually) |
| `data/models/` | MediaPipe weights: `blaze_face_short_range.tflite` (~230 KB), `face_landmarker.task` (~3.7 MB) | downloaded on first V2/V3 run; permanent |
| `~/.cache/huggingface/hub/` | Whisper weights (`models--Systran--faster-whisper-*`) | permanent; `whisper.download_root` relocates |
| `.venv/` | environment | — |

`data/` and `.venv/` are git-ignored. To reclaim space: delete `data/work/*` (re-download on next
run) or set `DL_STORAGE__KEEP_INTERMEDIATE=false`.

## Performance reference (Apple M2, CPU, int8)

| Item | Measured |
|---|---|
| streaming pass, `base` | ≈ 11× realtime in a live server run; ≈ 40-60× isolated on an idle machine |
| streaming pass, `small` | ≈ 3× realtime → worst case ≈ 18 min |
| verification, `small`, ±12 s window | 26.5 s of audio: 21.4 s in the live reference run, 9.7 s isolated (skipped entirely when the first pass scores ≥ 90) |
| thread count (`whisper.cpu_threads`) | 4 P-cores 5.4 s vs 8 cores 90.9 s for the same 331 s scan — see below |
| model load (cached weights) | `base` 1 s, `medium` 3–25 s (first load converts to int8) |
| first-time weight download | `base` 140 MB, `small` 460 MB, `medium` 1.5 GB |
| 55-min ok.ru reference video, line at 5:25, audio cached | **56.7 s** end to end (run `df328328a2e7`) |
| 18-min JFK speech, line at 9:13 | scan stops at 51 % of the audio; same per-stage rates as above |

Reference run `df328328a2e7`, the fastest recorded, with the audio already cached:

| stage | before the `cpu_threads` fix | after |
|---|---|---|
| transcription (330.3 s of audio scanned) | 79.9 s | **29.5 s** |
| verification (±12 s window) | 27.7 s | **21.4 s** |
| face + mouth checks | 2.4 s | 4.2 s |
| download, audio, clip, frame | 3.7 s | 1.5 s |
| **total** | **113.7 s** | **56.7 s** |

So the fix is worth ~2× end to end and ~2.7× on the stage it targets. The remaining gap
between 29.5 s here and the ~8 s the same scan takes isolated is machine load, not the
pipeline: a live run shares the CPU with the API server, the background model warm-up and
both resident models.

> **If transcription looks 10-20x slower than the table above**, check `whisper.cpu_threads`.
> The default of `0` sizes the pool to the machine's *performance* cores. Forcing it to the
> total core count on a big.LITTLE CPU (every Apple Silicon Mac) makes every parallel region
> run at efficiency-core speed: an M2 measured 5.4 s at 4 threads against 90.9 s at 8, with
> byte-identical output. Symptom: `base` running at ~3.6x realtime instead of ~60x.

**Transcription** is now the dominant cost. The audio-first fetch shipped, so the search never
downloads the full video: it pulls the audio stream (tens of MB), and only a verified match
triggers a full-quality clip of a few seconds. `DL_DOWNLOAD__MAX_HEIGHT` no longer affects the
search at all — it caps only that final clip. On hosts that throttle a single connection (ok.ru
serves ~200 KB/s) the audio fetch is still the slow part; download once elsewhere and run against
the local file if that bites.

## Errors you will actually see

| Message | Meaning | What to do |
|---|---|---|
| `[input] Dialogue must contain at least 2 words` | one-word dialogues match too easily | give a longer line, or `DL_MATCHING__MIN_DIALOGUE_WORDS=1` |
| `[input] 'x' is neither an http(s) URL nor an existing file` | typo, or file path not visible to the server | check the path from the server's working directory |
| `[download] … Connection reset by peer` (ok.ru) | the **network** resets TLS to that host — an in-path block (ISP/DNS filter), confirmed with `curl` failing in 0.2 s while other sites work | switch network / VPN, or download the file on a reachable network and use the local path |
| `[download] … Sign in to confirm you're not a bot` / `HTTP 429` (YouTube) | YouTube rate-limiting this IP; more likely without a JS runtime | `brew install deno` (yt-dlp uses it automatically); wait or change IP; last resort `yt-dlp --cookies-from-browser chrome <url>` and use the file |
| `[config] Required binary 'ffmpeg' not found` | FFmpeg missing | `brew install ffmpeg` |
| `[audio] 'x' has no audio stream` | nothing to transcribe | pick a video with speech |
| `not_found` but you know the line is there | first check the near-misses: a ~60 score at the right time means the ASR heard different words; no near-miss anywhere near the timestamp used to mean the VAD dropped the speech under loud music — since v1 the pipeline retries once with the VAD off automatically (`No match with VAD on … retrying without voice-activity filter`) | if it still misses: lower `DL_MATCHING__MATCH_THRESHOLD`, or try `DL_WHISPER__FAST_MODEL=small` |
| `[transcription] Failed to load Whisper model` | first-time weight download failed (offline) or bad model name | check network / `whisper.fast_model` value |
| `Warning: asr_large_model: rejected - Large model scored … keeping first-pass timestamp` | the two models disagree by > 5 points; the fast pass timestamp is used | inspect `first_pass` vs `verifications[0].refined` in the result; usually the fast pass is right, but check the frame |
| UI: `Job … no longer exists on the server` | server restarted; jobs are in-memory | start a new job; results are still in `data/output/` |
| `objc[…]: Class AVFFrameReceiver is implemented in both …` at import | PyAV and OpenCV both bundle libavdevice | harmless; ignore |

## Reading the logs

Every stage logs `---- [stage] …` on entry and `---- [stage] done in Ns` on exit; a job ends with
`Job <id> finished: found in 45.1s (download=1.9s, audio=0.2s, transcription=12.0s, …)`. The
decisive lines:

```
matcher      | Candidate match 83.3 at 00:00:06.840 … - settling over next 2 word(s)
matcher      | MATCH 100.0 at 00:00:06.840 - 00:00:12.360 (words 17-24): '…'
faster_whisper | [faster_whisper:base] transcription stopped early by consumer after 27 words
asr_verifier | [asr_large_model] CONFIRMED 100.0 (first pass 94.4) at 00:09:13.200 (shift +6.400s)
frames       | Frame saved: data/output/<job>/frame.jpg (640x360)
```

`DL_LOGGING__LEVEL=DEBUG` (or `-v` on the CLI) adds ffmpeg/ffprobe command lines, per-segment ASR
text with timestamps, and every improvement of the best score.

## Server operations

* `./run_server.sh [--host H] [--port P] [--reload]` — creates `.venv` if missing, checks FFmpeg.
* Concurrency: `DL_SERVER__MAX_CONCURRENT_JOBS` (default 1). Extra jobs queue; cancel via `DELETE`.
* Finished jobs expire from memory after `DL_SERVER__JOB_RETENTION_SECONDS` (3600).
* Health: `GET /api/health` reports the models in use and the active job count.
* The server is single-process; scale horizontally only with a shared job store (see Extending).

## Reproducing the reference runs

```bash
# 19 s YouTube clip: found at 00:00:07.260, frame 108 (15 fps), face + mouth movement -> onscreen
dialogue-locator "https://www.youtube.com/watch?v=jNQXAC9IVRw" "they have really really really long trunks"

# 18 min JFK Rice speech: found at 00:09:13.203, frame 33159 (59.94 fps), verifier confirmed (+0.29 s)
dialogue-locator "https://www.youtube.com/watch?v=WZyRbnpGyzQ" "We choose to go to the moon in this decade and do the other things"

# problem statement: found at 00:05:25.312, frame 7799 (23.976 fps), face 0.95, mouth 0.074 -> onscreen
dialogue-locator "https://ok.ru/video/248244667877" "My mind rebels at stagnation"
```
