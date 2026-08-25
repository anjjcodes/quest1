"""Shared download-and-cache logic for MediaPipe model files.

MediaPipe's pip package does not bundle its task models; like the Whisper
weights they are fetched once on first use and cached at a configured path.
Used by :mod:`face_detector` (BlazeFace, ~230 KB) and
:mod:`mouth_movement` (Face Landmarker, ~3.7 MB).
"""

from __future__ import annotations

import logging
import urllib.error
import urllib.request
from pathlib import Path

from dialogue_locator.exceptions import DialogueLocatorError

logger = logging.getLogger(__name__)


def ensure_model_file(
    model_path: Path,
    url: str,
    timeout_seconds: float,
    error_cls: type[DialogueLocatorError],
) -> Path:
    """Return ``model_path``, downloading it from ``url`` first if missing.

    Failures raise ``error_cls`` so each caller reports its own pipeline stage.
    """
    if model_path.is_file():
        return model_path

    logger.info("Model missing; downloading %s -> %s", url, model_path)
    try:
        model_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = model_path.with_suffix(model_path.suffix + ".part")
        with urllib.request.urlopen(url, timeout=timeout_seconds) as resp:
            data = resp.read()
        if len(data) < 1024:  # a storage error page, not a model
            raise error_cls(f"Model download returned only {len(data)} bytes", details={"url": url})
        tmp_path.write_bytes(data)
        tmp_path.replace(model_path)  # atomic: never leave a half-written model
    except error_cls:
        raise
    except (urllib.error.URLError, OSError, ValueError) as exc:
        logger.error("Model download failed: %s", exc)
        raise error_cls(
            f"Cannot download model: {exc}", details={"url": url, "model_path": str(model_path)}
        ) from exc
    logger.info("Model saved: %s (%d bytes)", model_path, len(data))
    return model_path
