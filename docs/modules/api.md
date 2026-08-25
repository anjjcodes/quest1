# Module: `api/`

A thin FastAPI adapter over the pipeline. **No processing logic lives here**: `routes.py`
converts HTTP ↔ `PipelineRequest` / `LocalizationResult`; `jobs.py` only schedules and records.

## Endpoints (`routes.py`)

| Method | Path | Returns | Notes |
|---|---|---|---|
| `GET` | `/api/health` | `HealthResponse` | version, configured models, threshold, `active_jobs` |
| `POST` | `/api/jobs` | `202 JobResponse` | body `JobCreate{source, dialogue, reuse_cached_media}`; validates **synchronously** (dialogue via `TargetDialogue.parse`; URL-looking sources via `validate_url`; other sources must be existing files) → `422 {error, stage:"input", message}` |
| `GET` | `/api/jobs` | `JobListResponse` | newest first |
| `GET` | `/api/jobs/{id}` | `JobResponse` | status, latest `progress`, `progress_log` (≤ 50), `result` or `error`, `frame_url` |
| `GET` | `/api/jobs/{id}/frame` | image | `image/jpeg` or `image/png`; 404 if no frame |
| `DELETE` | `/api/jobs/{id}` | `JobResponse` | cancels a queued/running job; removes a finished one from memory |
| `GET` | `/`, `/static/*` | web UI | falls back to `{"docs": "/docs"}` if the web dir is missing |

Interactive docs: `/docs` (Swagger) and `/openapi.json`.

## Job lifecycle (`jobs.py`)

```
queued ──(worker free)──▶ running ──▶ done | failed | cancelled
   └──────────(DELETE)──────────────────────▶ cancelled
```

```python
class JobManager:
    def __init__(self, pipeline, max_concurrent=1, retention_seconds=3600)
    def submit(request) -> Job          # ThreadPoolExecutor(max_concurrent)
    def get(job_id) / list() / cancel(job_id) / remove(job_id) / active_count / shutdown()
```

* Worker thread runs `pipeline.run(request, progress=record, should_cancel=job.cancel_event.is_set)`.
* Progress is stored as `(event, server timestamp)`: latest + a 50-event ring buffer.
* `PipelineCancelledError` → `cancelled`; `DialogueLocatorError` → `failed` with `error=exc.to_dict()`;
  any other exception → logged with traceback and reported as `failed` (a worker never dies silently).
* On success `result.json` is written to `output_dir/<job>/` (same file the CLI writes).
* Finished jobs expire from memory after `retention_seconds` (checked lazily on access); the
  files on disk stay. **State is in-memory by design for V1** — a server restart forgets jobs
  (the UI handles the resulting 404 by dropping the stale job). Swapping in Redis/DB touches only
  this module.

## Schemas (`schemas.py`)

`JobCreate`, `JobStatus`, `ProgressSchema`, `CandidateSchema`, `VerificationSchema`, `FrameSchema`,
`ResultSchema` (mirrors `LocalizationResult.to_dict()`, `extra="allow"` so the full record passes
through while the important fields are typed for OpenAPI), `ErrorSchema`, `JobResponse`,
`JobListResponse`, `HealthResponse`.

## App factory (`app.py`)

```python
def create_app(settings=None, pipeline=None, warm_up=True) -> FastAPI
def main() -> None            # `dialogue-locator-server` entry point → uvicorn on settings.server.host/port
```

* `lifespan`: builds the `JobManager`, starts model warm-up in a **daemon thread** (UI reachable
  immediately; the first job waits on the model-cache lock), shuts the pool down on exit.
* Exception handler: `InvalidInputError` → 422, other `DialogueLocatorError` → 500, both as
  `exc.to_dict()`.
* Mounts `web/static` at `/static` and serves `web/templates/index.html` at `/`.

## Testing the API

`src/tests/test_api.py` uses `fastapi.testclient.TestClient` with a `FakePipeline` (modes:
found / not_found / error / slow / crash) — no models, no network. Covered: full found flow incl.
frame download and on-disk result, not-found, failed, crash, five 422 cases, 404s, cancel of a
running job, queued job waiting for a slot and cancelling instantly, delete finished, retention
expiry, OpenAPI route list, UI served.
