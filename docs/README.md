# Dialogue Locator — Documentation

Reference documentation for the whole codebase under `src/`. Written so that a future
reader (including us in six months) can understand *what* each piece does, *why* it is
built that way, and *how* to change it safely.

| # | Document | Read this when you want to… |
|---|---|---|
| 1 | [Overview](01-overview.md) | understand the problem, the approach, and how the PS evaluation questions are answered |
| 2 | [Architecture](02-architecture.md) | see how the stages fit together, the data flow, and the extension points |
| 3 | [Getting started](03-getting-started.md) | install, run the CLI / server / tests on a fresh machine |
| 4 | [Configuration](04-configuration.md) | change models, thresholds, paths, ports — every knob with its default |
| 5 | Modules | understand one package in depth |
|   | [config, exceptions, logging](modules/config-exceptions-logging.md) | settings object, error hierarchy, log setup |
|   | [models](modules/models.md) | the dataclasses every stage exchanges |
|   | [media](modules/media.md) | probe, download, audio extraction, frame extraction |
|   | [transcription](modules/transcription.md) | the `Transcriber` contract and the Faster-Whisper implementation |
|   | [matching](modules/matching.md) | normalisation and the streaming sliding-window matcher |
|   | [verification](modules/verification.md) | the `Verifier` contract and the large-model ASR verifier |
|   | [pipeline & CLI](modules/pipeline-and-cli.md) | orchestration, timings, cancellation, the command line |
|   | [api](modules/api.md) | FastAPI app, job manager, routes, schemas |
|   | [web UI](modules/web-ui.md) | the HTML/CSS/JS front end |
| 6 | [Testing](06-testing.md) | run, read and extend the test suite |
| 7 | [Operations & troubleshooting](07-operations-troubleshooting.md) | disk layout, caching, performance, network / site errors |
| 8 | [Extending](08-extending.md) | add the V2 visual verifier, swap the ASR engine, add a stage |
| 9 | [Decision log](09-decision-log.md) | the non-obvious choices and the evidence behind them |

Conventions used in these docs:

* Paths are relative to the repository root. Source lives in `src/dialogue_locator/`, tests in `src/tests/`.
* `Settings.<section>.<field>` refers to the configuration object; `DL_<SECTION>__<FIELD>` is the
  matching environment variable.
* "Stage" means one step of the pipeline (`input, download, audio, transcription, matching,
  verification, frame, done`) — the same names appear in logs, progress events, errors and the UI.
