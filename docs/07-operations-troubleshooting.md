# 7. Operations & troubleshooting

## Disk layout

| Path | Contents | Lifetime |
|---|---|---|
| `data/work/<sha1(source)[:16]>/` | `video.mp4`, `audio.wav` (+ `.part` while downloading) | kept while `storage.keep_intermediate=True` (default); reused across dialogues for the same source |
| `data/output/<job_id>/` | `frame.jpg`, `result.json` | permanent (delete manually) |
| `~/.cache/huggingface/hub/` | Whisper weights (`models--Systran--faster-whisper-*`) | permanent; `whisper.download_root` relocates |
| `.venv/` | environment | — |

`data/` and `.venv/` are git-ignored. To reclaim space: delete `data/work/*` (re-download on next
run) or set `DL_STORAGE__KEEP_INTERMEDIATE=false`.

## Performance reference (Apple M2, CPU, int8)

| Item | Measured |
|---|---|
| streaming pass, `base` | ≈ 12× realtime → 55-min video worst case ≈ 5 min |
| streaming pass, `small` | ≈ 3× realtime → worst case ≈ 18 min |
| verification, `medium`, ±20 s window | 30–60 s |
| model load (cached weights) | `base` 1 s, `medium` 3–25 s (first load converts to int8) |
| first-time weight download | `base` 140 MB, `small` 460 MB, `medium` 1.5 GB |
| 18-min JFK speech end-to-end, line at 9:13 | 193 s (download 18 + audio 2 + transcription 116 with `small` + verification 57) |

The single biggest cost on the PS video is the **download**: the 55-minute ok.ru rendition at
720p is ~1.5 GB (≈ 2 h at 200 KB/s). Mitigations today: `DL_DOWNLOAD__MAX_HEIGHT=360`, or download
once elsewhere and run on the local file. Planned: audio-first fetch (see decision log).

## Errors you will actually see

| Message | Meaning | What to do |
|---|---|---|
| `[input] Dialogue must contain at least 2 words` | one-word dialogues match too easily | give a longer line, or `DL_MATCHING__MIN_DIALOGUE_WORDS=1` |
| `[input] 'x' is neither an http(s) URL nor an existing file` | typo, or file path not visible to the server | check the path from the server's working directory |
| `[download] … Connection reset by peer` (ok.ru) | the **network** resets TLS to that host — an in-path block (ISP/DNS filter), confirmed with `curl` failing in 0.2 s while other sites work | switch network / VPN, or download the file on a reachable network and use the local path |
| `[download] … Sign in to confirm you're not a bot` / `HTTP 429` (YouTube) | YouTube rate-limiting this IP; more likely without a JS runtime | `brew install deno` (yt-dlp uses it automatically); wait or change IP; last resort `yt-dlp --cookies-from-browser chrome <url>` and use the file |
| `[config] Required binary 'ffmpeg' not found` | FFmpeg missing | `brew install ffmpeg` |
| `[audio] 'x' has no audio stream` | nothing to transcribe | pick a video with speech |
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
# 19 s YouTube clip: found at 00:00:06.800, frame 102 (15 fps), verifier rejected (medium mis-hears "trunks")
dialogue-locator "https://www.youtube.com/watch?v=jNQXAC9IVRw" "they have really really really long trunks"

# 18 min JFK Rice speech: found at 00:09:13.187, frame 16579 (29.97 fps), verifier shifted +6.4 s and confirmed
dialogue-locator "https://www.youtube.com/watch?v=WZyRbnpGyzQ" "We choose to go to the moon in this decade and do the other things"

# problem statement
dialogue-locator "https://ok.ru/video/248244667877" "My mind rebels at stagnation"
```
