# Dialogue Locator

Give it a video URL and a line of dialogue, and it finds the exact frame where that **on
screen dialogue** first appears: the timestamp, the frame number, the text as it was heard,
and the frame saved as an image. It even confirms the line really is an on screen dialogue by
checking that a face is visible and its mouth is moving while the words are spoken. Nobody
has to scrub through the video by hand.

I built this for the problem statement *"My mind rebels at stagnation"* (reference video:
`https://ok.ru/video/248244667877`). Nothing in the code is specific to that video though, any
video and dialogue pair works.

```
$ dialogue-locator "https://ok.ru/video/248244667877" "My mind rebels at stagnation"

Timestamp : 00:05:25.312
Frame     : 7799
Text      : "My mind rebels at stagnation."
Image     : data/output/df328328a2e7/frame.jpg
Score     : 100.0
Face      : detected (confidence 0.95)
Mouth     : moving (score 0.074)
Verify    : asr_large_model -> confirmed (100.0)
Scanned   : 330.3s of audio
Result    : data/output/df328328a2e7/result.json
```

That is real output from a run against the reference video.

## Demo

A full run against the reference video, end to end:

https://github.com/user-attachments/assets/ff807958-c069-4899-abad-52d7ea160c01

## What it does

I transcribe the audio with a small Whisper model and compare every word against the target
dialogue while the transcription is still running, and the moment a window of words matches I
stop transcribing. Then I confirm that spot with a bigger model, download a few seconds of
video around the confirmed time to extract the frame, and check the frame for a face and the
clip for mouth movement to make sure the line is actually spoken on screen. If the line is
heard but nobody is visibly saying it, the result says so (`not_onscreen`), and if nothing
matches at all, the three closest transcript windows are reported so you can see what the
video actually says.

If the line is said more than once, `matching.max_occurrences` (`--max-occurrences` on the
command line, or the settings panel in the UI) decides how far to look. It defaults to 1, which
judges the first occurrence and reports it whatever the verdict. Raise it (or set `-1` for all
of them) and an occurrence that turns out not to be onscreen no longer ends the job: the search
picks up where that one finished and the first occurrence actually delivered on camera wins,
falling back to the first when none are.

Where to read more:

- [APPROACH.MD](APPROACH.MD): the full approach step by step, the key decisions and their
  trade offs, assumptions, and what I would improve.
- [docs/02-architecture.md](docs/02-architecture.md): the flow chart, the architecture
  diagram, the class diagram, and the data types between stages.
- [docs/09-decision-log.md](docs/09-decision-log.md): 33 decisions with the evidence behind
  each. The rest of [docs/](docs/README.md) covers every module, configuration, testing and
  operations.
- [prompts.txt](prompts.txt): the LLM prompts I used while building this.
- [test_cases.json](test_cases.json): the videos and dialogues I validated against.

## Setup and run

You need Python 3.11 or newer and FFmpeg. Model weights (about 2 GB for Whisper, a few MB for
MediaPipe) download themselves on first use and are cached.

Installing the package (both options below do it) also registers the two commands used on
this page, `dialogue-locator` and `dialogue-locator-server`. They live inside `.venv`, so they
are available whenever the venv is active, or directly as `.venv/bin/dialogue-locator`.

### macOS and Linux

```bash
brew install ffmpeg          # macOS (Linux: sudo apt install ffmpeg)
./run_server.sh
```

The script does the rest: it checks ffmpeg and ffprobe, creates `.venv` (on Apple Silicon it
uses a native arm64 Python, because some wheels do not exist for Rosetta Pythons), installs
the package, and starts the server at `http://127.0.0.1:8000`.

### Windows

```powershell
winget install ffmpeg
py -3.11 -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev]"
dialogue-locator-server
```

The server comes up at `http://127.0.0.1:8000`.

### Use the UI

Open `http://127.0.0.1:8000` in the browser. Paste the video URL (or a local file path) and
the dialogue, and start the job. The page shows every stage live as it runs (download, audio,
transcription, verification, frame, face check, mouth check), and when it finishes you get the
timestamp, the frame number, the matched text, and the frame image right there. Earlier jobs
stay listed at the bottom so you can reopen them, and the settings button lets you toggle
stages or tweak thresholds for a single job.

The web UI is just one front end. The same pipeline also runs from the terminal
(`dialogue-locator`), through the JSON API (OpenAPI docs at `/docs`), and straight from
Python, where every module is also usable on its own; all of it is shown in
[docs/03-getting-started.md](docs/03-getting-started.md).
