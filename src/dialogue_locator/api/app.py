"""FastAPI application factory and server entry point."""

from __future__ import annotations

import logging
import threading
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from dialogue_locator import __version__
from dialogue_locator.api.jobs import JobManager
from dialogue_locator.api.routes import router
from dialogue_locator.config import Settings, get_settings
from dialogue_locator.exceptions import DialogueLocatorError, InvalidInputError
from dialogue_locator.logging_config import configure_logging
from dialogue_locator.pipeline import DialoguePipeline

logger = logging.getLogger(__name__)

WEB_DIR = Path(__file__).resolve().parent.parent / "web"


def create_app(settings: Settings | None = None, pipeline: DialoguePipeline | None = None, warm_up: bool = True) -> FastAPI:
    settings = settings or get_settings()
    configure_logging(settings.logging)
    settings.ensure_directories()
    pipeline = pipeline or DialoguePipeline(settings)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.settings = settings
        app.state.pipeline = pipeline
        app.state.jobs = JobManager(
            pipeline, max_concurrent=settings.server.max_concurrent_jobs, retention_seconds=settings.server.job_retention_seconds
        )
        if warm_up:
            # Load models in the background so the UI is reachable immediately;
            # the first job simply waits on the model cache lock until loading completes.
            threading.Thread(target=_safe_warm_up, args=(pipeline,), name="warm-up", daemon=True).start()
        logger.info("Dialogue Locator API v%s ready on http://%s:%d", __version__, settings.server.host, settings.server.port)
        yield
        app.state.jobs.shutdown()

    app = FastAPI(title="Dialogue Locator", version=__version__, lifespan=lifespan)
    app.include_router(router)

    @app.exception_handler(DialogueLocatorError)
    async def _domain_error(_: Request, exc: DialogueLocatorError) -> JSONResponse:
        code = 422 if isinstance(exc, InvalidInputError) else 500
        return JSONResponse(status_code=code, content=exc.to_dict())

    static_dir = WEB_DIR / "static"
    if static_dir.is_dir():
        app.mount("/static", StaticFiles(directory=static_dir), name="static")

    @app.get("/", include_in_schema=False)
    async def index():
        index_html = WEB_DIR / "templates" / "index.html"
        if index_html.is_file():
            return FileResponse(index_html, media_type="text/html")
        return JSONResponse({"message": "Dialogue Locator API", "docs": "/docs"})

    return app


def _safe_warm_up(pipeline: DialoguePipeline) -> None:
    try:
        pipeline.warm_up()
    except DialogueLocatorError as exc:
        logger.error("Model warm-up failed: %s (jobs will retry on demand)", exc.message)


def main() -> None:
    """``dialogue-locator-server`` console entry point."""
    import uvicorn

    settings = get_settings()
    uvicorn.run(create_app(settings), host=settings.server.host, port=settings.server.port, log_config=None)


if __name__ == "__main__":  # pragma: no cover
    main()
