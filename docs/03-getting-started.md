# 3. Getting started

## Prerequisites

| Requirement | Why | Install (macOS) |
|---|---|---|
| Python 3.11+ (3.12 recommended) | runtime | `uv python install 3.12` or python.org |
| **FFmpeg + ffprobe** on `PATH` | audio extraction, probing, frame fallback | `brew install ffmpeg` |
| `deno` (recommended) | yt-dlp's JavaScript runtime for YouTube challenges | `brew install deno` |
| ~2 GB disk for model weights | `base` ≈ 140 MB, `small` ≈ 460 MB, `medium` ≈ 1.5 GB, cached in `~/.cache/huggingface` | automatic on first use |

### Apple Silicon note (important)

If Homebrew lives in `/usr/local` your Python and `uv` are **x86_64 builds running under Rosetta**.
`onnxruntime` (Faster-Whisper's VAD dependency) ships no macOS x86_64 wheels, so `pip install
faster-whisper` fails. Use a native arm64 interpreter:

```bash
uv python install cpython-3.12-macos-aarch64-none
uv venv .venv -p ~/.local/share/uv/python/cpython-3.12*-macos-aarch64-none/bin/python3.12
```

`run_server.sh` does this automatically when it creates `.venv`.

## Install

```bash
git clone <repo> && cd quest1
uv venv .venv -p 3.12            # see the Apple Silicon note above
source .venv/bin/activate
uv pip install -e ".[dev]"       # or: pip install -e ".[dev]"
```

`-e` installs the package from `src/` in editable mode and registers two console scripts:
`dialogue-locator` (CLI) and `dialogue-locator-server` (API + UI).

## Run — CLI

```bash
dialogue-locator "https://ok.ru/video/248244667877" "My mind rebels at stagnation"
dialogue-locator path/to/video.mp4 "some spoken line" --json      # local file, JSON output
dialogue-locator <url> "<line>" --fast-model small --no-verify -v # overrides, debug logs
```

Output (exit code 0 found / 2 not found / 1 error):

```
Timestamp : 00:09:13.187
Frame     : 16579
Text      : "We choose to go to the moon in this decade and do the other things."
Image     : data/output/7eddba4ca5ff/frame.jpg
Score     : 100.0
Verify    : asr_large_model -> confirmed (100.0)
Scanned   : 556.9s of audio
Result    : data/output/7eddba4ca5ff/result.json
```

## Run — server + web UI

```bash
./run_server.sh                 # http://127.0.0.1:8000  (UI at /, OpenAPI at /docs)
./run_server.sh --port 9000 --reload
```

or without the script: `dialogue-locator-server`.

Submit via HTTP:

```bash
curl -X POST localhost:8000/api/jobs -H 'content-type: application/json' \
     -d '{"source":"https://ok.ru/video/248244667877","dialogue":"My mind rebels at stagnation"}'
curl localhost:8000/api/jobs/<job_id>            # poll
curl -o frame.jpg localhost:8000/api/jobs/<job_id>/frame
```

## Run — tests

```bash
python -m pytest                                  # offline suite with live logs (~10 s)
python -m pytest --log-cli-level=WARNING -q       # quiet
DL_RUN_NETWORK_TESTS=1 python -m pytest -m network    # real yt-dlp downloads
DL_RUN_MODEL_TESTS=1  python -m pytest -k real        # real Whisper models (downloads weights)
```

See [Testing](06-testing.md).

## Where things end up

```
data/work/<sha1(source)[:16]>/video.mp4, audio.wav   # cached media, reused across dialogues
data/output/<job_id>/frame.jpg, result.json          # results
~/.cache/huggingface/hub/models--Systran--faster-whisper-*   # model weights
```

Both `data/` and `.venv/` are git-ignored.
