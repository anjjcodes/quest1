# 9. Decision log

Non-obvious choices, in the order they were made, with the evidence. Useful for "why did you…"
questions.

| # | Decision | Alternatives | Why |
|---|---|---|---|
| 1 | **Streaming ASR with early stop** instead of transcribe-then-search | full transcription; fixed chunks | A 55-min video costs only the audio up to the line. No application chunking → a dialogue can span Whisper segments without artificial boundaries. Verified: JFK stopped at 51 % of the video. |
| 2 | **Words, not segments**, as the transcriber's output | segment text | The matcher slides over a flat word stream; segment boundaries become irrelevant; timestamps are per word. |
| 3 | **Decode WAV ourselves** (`load_pcm`) and pass numpy to Whisper | let faster-whisper/PyAV decode | Validates format early, avoids a second FFmpeg binding at runtime, gives the verifier an in-memory array to slice. |
| 4 | `compute_type=int8` default | `default` (float16) | ctranslate2 on Apple Silicon/CPU refuses float16 and silently converts to float32; int8 is the CPU fast path and valid on CUDA. |
| 5 | `condition_on_previous_text=False` | True (Whisper default) | Prevents repetition/hallucination loops on long streams; we match on words, not on paragraph coherence. |
| 6 | **Best-scoring window size per end position** (not first size over threshold) | first size ≥ threshold | Windows with one spurious/missing edge word still score 93–94; picking the best size (100) pins the start to the right word. Test: `the my mind rebels at stagnation` → start at `my`. |
| 7 | **Settling** over `window_tolerance` more words after a candidate | return immediately | Long dialogues clear 80 one word early; without settling the match text is truncated. Cost: ≤ 2 words of audio. Log: 81.7 → 89.1 → 94.4 → 100. |
| 8 | **Non-overlapping near-misses** | top-3 raw windows | Raw top-3 are three near-identical windows from one spot; per-region best gives the user three *different* places to inspect. |
| 9 | Reject a **"max time gap inside a window"** rule | add it | On JFK the fast model dropped a repeated phrase; a gap rule would have produced *not found* and the verifier would never run. The first pass is a candidate generator; precision belongs to the verifier. |
| 10 | **Verifier compares by score with a 5-point tolerance**, and keeps `refined` even when rejecting | always trust the big model | Real data: the big model is sometimes wrong (`medium` heard "fronts" for "trunks", −7.1 → rejected, first pass kept). And sometimes decisively right (JFK +6.4 s, 100 → confirmed). The rule protects in both directions and everything is recorded for audit. |
| 11 | **`floor(t × fps)`** for frame number; report the frame's own time | `round` | Frame k is on screen during `[k/fps, (k+1)/fps)`; `round` would pick the next frame half the time. Timestamp and frame number are then mutually consistent. |
| 12 | **OpenCV seek with landing check + ffmpeg fallback**, proven equal pixel-wise | OpenCV only / ffmpeg only | OpenCV is fast but backend-dependent; ffmpeg is frame-accurate but slower. Test shows mean-abs diff 0.5 for the same frame vs ~5 for a neighbour. |
| 13 | Media facts from **ffprobe**, not the site | yt-dlp metadata | Site fps/duration are often rounded or absent (ok.ru). Frame number = t × fps must use the real file's fps. |
| 14 | **Cache media per source URL** (`sha1`), `keep_intermediate=True` | delete after each job | Re-running a second dialogue on the same 55-min video must not re-download 1.5 GB. |
| 15 | `fast_model` default **`base`** (was `small`) | `small` | Measured 12× vs 3× realtime with near-identical hit rate; the verifier fixes wording/timing. Worst case on the PS video 5 min vs 18 min. |
| 16 | Input validated **before any I/O**, and again synchronously in the API (422) | let the job fail | A one-word dialogue or bad URL fails in milliseconds, not after a download. The API reuses the same validators the pipeline uses. |
| 17 | **Errors carry a `stage`**; boundaries log before raising (WARNING for input, ERROR for tools) | plain exceptions | One `except DialogueLocatorError` in each interface; UI shows "Failed during download"; logs always show the cause even when the caller swallows it. |
| 18 | **Pipeline is framework-free**; progress via callback; every stage injectable | pipeline inside FastAPI handlers | Same code runs from CLI, API and tests with fakes; V2 stages plug in without touching HTTP code. |
| 19 | **In-memory job store**, single worker by default | DB/Redis, multiple workers | V1 scope; Whisper saturates one machine's CPU; the module boundary makes replacement local. |
| 20 | **Native arm64 Python 3.12 venv** | Homebrew (x86_64/Rosetta) Python 3.14 | `onnxruntime` has no macOS x86_64 wheels → faster-whisper won't install; native also runs Whisper faster. Encoded in `run_server.sh`. |
| 21 | Model warm-up in a **background thread** at server start | block startup | UI reachable immediately; the first job waits on the model-cache lock instead. |
| 22 | Web UI is **vanilla HTML/CSS/JS**, light theme, large type | a framework | No build step, served by the same process, one file each; easy to hand-edit. |
| 23 | **Verification stays unconditional** — no confidence ladder that skips or downgrades the large model on a high first-pass score | base→small→medium cascade gated on fuzzy score | The fuzzy score is *text* confidence, not *timing* confidence: JFK scored 94.4 with a 6.4 s-wrong timestamp (repetition collapse) that only a clean re-transcription catches. Small models agreeing are correlated judges, not independent evidence. Verification is a bounded 30–60 s that is <5 % of a long job (Amdahl). If escalation is ever added, invert it: re-run the streaming pass with a bigger model only when nothing reaches the threshold. |
| 24 | **VAD-off retry on not-found** — when the streaming pass ends below the threshold, one extra pass with the voice-activity filter disabled runs before reporting `not_found` (`whisper.retry_without_vad`) | trust a single pass; or ship with VAD off by default | The Silero VAD drops real speech buried under loud score/effects: an Infinity War line at 02:00:20 vanished with VAD on (base, small **and** medium all produced the same 18 words in that 5-minute window — the VAD trims the audio before any model hears it) yet scored 100.0 with VAD off. VAD stays the default because it skips silence (speed) and starves hallucinations on music (match safety); the retry costs one extra scan only when the answer would otherwise be a miss. This is the inversion decision #23 anticipated — escalate the *preprocessing*, not the model. For the same reason the **verify model always runs its ±20 s window with the VAD off**: the window is known to contain speech, and with VAD on `medium` deleted the very line under test and rejected a perfect match (59.7 vs 100.0); with VAD off it confirms at 100.0. |

## Known limitations (V1)

* Downloads use a single connection; hosts that throttle per connection (e.g. ok.ru at
  ~200 KB/s) are slow. Parallel fragments / an external downloader (aria2c) are the planned fix.
* English-only normalisation (non-ASCII letters dropped) — one regex to widen.
* The visual checks judge one face per frame (the most prominent); a crowd scene where a
  background speaker mouths the line could still be judged on the foreground face.
* Jobs are forgotten on server restart; result files persist.
* YouTube access depends on yt-dlp keeping up with YouTube; ok.ru reachability depends on the
  network (ISP-level blocks were observed).
