"""Central logging setup.

Call :func:`configure_logging` once at process start (CLI entry point or
FastAPI startup). Every module then obtains its logger via
``logging.getLogger(__name__)`` so log lines are attributable to a stage,
e.g. ``dialogue_locator.transcription.faster_whisper``.

Third-party loggers that are very chatty (faster_whisper, yt_dlp, httpx)
are pinned to WARNING unless the app itself is in DEBUG.
"""

from __future__ import annotations

import logging
import sys

from dialogue_locator.config import LoggingConfig

_NOISY_LOGGERS = ("faster_whisper", "yt_dlp", "httpx", "httpcore", "urllib3", "numba")


def configure_logging(config: LoggingConfig | None = None) -> None:
    """Configure the root logger. Safe to call multiple times (idempotent)."""
    config = config or LoggingConfig()
    root = logging.getLogger()
    root.setLevel(config.level)

    # Replace existing stream handlers we installed earlier (reload/tests).
    for handler in list(root.handlers):
        if getattr(handler, "_dialogue_locator", False):
            root.removeHandler(handler)

    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter(config.fmt))
    handler._dialogue_locator = True  # type: ignore[attr-defined]
    root.addHandler(handler)

    noisy_level = logging.DEBUG if config.level == "DEBUG" else logging.WARNING
    for name in _NOISY_LOGGERS:
        logging.getLogger(name).setLevel(noisy_level)

    logging.getLogger(__name__).debug("Logging configured at %s", config.level)
