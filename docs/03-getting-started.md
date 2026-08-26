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

Output (exit code 0 found / 2 not found / 3 heard but not onscreen / 1 error):

```
Timestamp : 00:05:24.603
Frame     : 7782
Text      : "My mind rebels at stagnation."
Image     : data/output/0d7f1c22761a/frame.jpg
Score     : 100.0
Face      : 1 detected (best 0.95)
Mouth     : moving (score 0.087)
Verify    : asr_large_model -> confirmed (100.0)
Scanned   : 330.3s of audio
Result    : data/output/0d7f1c22761a/result.json
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

## Run — from Python

The pipeline is a plain class, so it can be driven from any script or notebook:

```python
from dialogue_locator.config import get_settings
from dialogue_locator.pipeline import DialoguePipeline, PipelineRequest

pipeline = DialoguePipeline(get_settings())
result = pipeline.run(PipelineRequest(source="video.mp4", dialogue="my mind rebels at stagnation"))
print(result.timestamp, result.frame_number, result.matched_text)
```

`result` is a `LocalizationResult` (see [models](modules/models.md)); `result.to_dict()` is the
same record the CLI writes to `result.json` and the API returns.

## Using the modules on their own

Every stage is a plain class that takes its config in the constructor, so each one is usable
by itself, without the pipeline. A few examples:

Download just the audio of any video:

```python
from pathlib import Path

from dialogue_locator.config import DownloadConfig
from dialogue_locator.media.downloader import VideoDownloader

media = VideoDownloader(DownloadConfig()).fetch_search_media("https://...", Path("tmp"))
```

Stream a transcription with word timestamps from any 16 kHz WAV:

```python
from dialogue_locator.config import WhisperConfig
from dialogue_locator.transcription.faster_whisper import FasterWhisperTranscriber

for word in FasterWhisperTranscriber("base", WhisperConfig()).transcribe(Path("audio.wav")):
    print(word.text, word.start)
```

Search any word stream for a phrase:

```python
from dialogue_locator.config import MatchingConfig
from dialogue_locator.matching.matcher import StreamingMatcher

matcher = StreamingMatcher("the phrase to find", MatchingConfig())
for word in words:
    if (match := matcher.feed(word)) is not None:
        print(match.timestamp, match.score)
        break
```

Grab the frame at any timestamp of any video file:

```python
from dialogue_locator.config import FrameConfig
from dialogue_locator.media.frames import FrameExtractor
from dialogue_locator.models import VideoInfo

video = VideoInfo(path=Path("video.mp4"), source_url="video.mp4")
FrameExtractor(FrameConfig()).extract(video, 12.5, Path("frame.jpg"))
```

Check any image for faces, or any clip window for mouth movement:

```python
from dialogue_locator.config import FaceDetectionConfig, MouthMovementConfig
from dialogue_locator.vision import FaceDetector, MouthMovementAnalyzer

print(FaceDetector(FaceDetectionConfig()).detect_file(Path("frame.jpg")).face_present)
print(MouthMovementAnalyzer(MouthMovementConfig()).analyze(video, 12.0, 15.0).moving)
```

The remaining pieces follow the same pattern: `AudioExtractor` turns any video into a clean
16 kHz WAV, `probe_media` reads duration, fps and stream info from any file, and `AsrVerifier`
checks any candidate window against the audio with a bigger model. This independence is also
what makes the test suite work: every module is tested alone, and the pipeline is tested with
fakes standing in for all of them.

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
data/work/<sha1(source)[:16]>/media.*, audio.wav, clip_<ms>_<ms>.mp4   # cached media, reused across dialogues
data/output/<job_id>/frame.jpg, result.json                            # results
~/.cache/huggingface/hub/models--Systran--faster-whisper-*             # Whisper weights
data/models/                                                           # MediaPipe face/mouth models
```

Both `data/` and `.venv/` are git-ignored.
