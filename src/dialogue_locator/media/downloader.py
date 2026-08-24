"""Fetch the source video.

Two inputs are accepted:

* an ``http(s)`` URL supported by yt-dlp (YouTube, ok.ru, Vimeo, direct .mp4 ...)
* a path to a local video file (handy for tests and offline runs)

Either way the result is a :class:`~dialogue_locator.models.VideoInfo` whose
``path`` points at a local, seekable file and whose fps / duration / frame
count come from ffprobe rather than from the site's metadata (which is often
missing or rounded).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import yt_dlp
from yt_dlp.utils import DownloadError as YtDlpDownloadError
from yt_dlp.utils import ExtractorError, UnsupportedError

from dialogue_locator.config import DownloadConfig
from dialogue_locator.exceptions import DownloadError, InvalidURLError, UnsupportedVideoError
from dialogue_locator.media.probe import probe_media
from dialogue_locator.models import PipelineStage, ProgressCallback, ProgressEvent, VideoInfo

logger = logging.getLogger(__name__)

_ALLOWED_SCHEMES = ("http", "https")
_OUTPUT_STEM = "video"


def is_url(source: str) -> bool:
    """True if ``source`` looks like an http(s) URL (as opposed to a local path)."""
    parsed = urlparse(source.strip())
    return parsed.scheme in _ALLOWED_SCHEMES and bool(parsed.netloc)


def validate_url(url: str) -> str:
    """Return the cleaned URL or raise ``InvalidURLError``."""
    cleaned = (url or "").strip()
    if not cleaned:
        logger.warning("Rejected input: empty video URL")
        raise InvalidURLError("Video URL is empty.")
    parsed = urlparse(cleaned)
    if parsed.scheme not in _ALLOWED_SCHEMES:
        logger.warning("Rejected URL %r: unsupported scheme %r", cleaned, parsed.scheme)
        raise InvalidURLError(
            f"URL must start with http:// or https:// (got scheme '{parsed.scheme or 'none'}').",
            details={"url": cleaned},
        )
    if not parsed.netloc or "." not in parsed.netloc:
        logger.warning("Rejected URL %r: no valid host", cleaned)
        raise InvalidURLError("URL has no valid host.", details={"url": cleaned})
    return cleaned


class VideoDownloader:
    """Download (or locate) a video and describe it.

    Args:
        config: download tunables (max height, retries, timeouts).
        ffprobe_binary: name/path of ffprobe used to fill in video properties.
    """

    def __init__(self, config: DownloadConfig, ffprobe_binary: str = "ffprobe") -> None:
        self._config = config
        self._ffprobe = ffprobe_binary

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def fetch(
        self,
        source: str,
        dest_dir: Path,
        progress: ProgressCallback | None = None,
        reuse_existing: bool = True,
    ) -> VideoInfo:
        """Obtain the video for ``source`` and return its :class:`VideoInfo`.

        Args:
            source: http(s) URL or local file path.
            dest_dir: directory to download into (created if missing).
            progress: optional callback for download progress events.
            reuse_existing: if a previous download exists in ``dest_dir``, use it
                instead of re-downloading (speeds up repeated runs on one video).
        """
        source = (source or "").strip()
        if not source:
            logger.warning("Rejected input: empty video source")
            raise InvalidURLError("Video source is empty.")

        if is_url(source):
            path = self._download(validate_url(source), dest_dir, progress, reuse_existing)
            title = self._last_title
        else:
            path = self._resolve_local(source)
            title = path.stem

        return self._describe(path, source, title)

    # ------------------------------------------------------------------ #
    # Local file
    # ------------------------------------------------------------------ #
    @staticmethod
    def _resolve_local(source: str) -> Path:
        path = Path(source).expanduser()
        if not path.is_file():
            logger.warning("Rejected input: %r is neither an http(s) URL nor an existing file", source)
            raise InvalidURLError(
                f"'{source}' is neither an http(s) URL nor an existing file.",
                details={"source": source},
            )
        logger.info("Using local media file: %s", path)
        return path.resolve()

    # ------------------------------------------------------------------ #
    # yt-dlp
    # ------------------------------------------------------------------ #
    def _existing_download(self, dest_dir: Path) -> Path | None:
        candidates = sorted(dest_dir.glob(f"{_OUTPUT_STEM}.*"))
        # yt-dlp leaves .part / .ytdl files behind on interrupted downloads.
        candidates = [
            p for p in candidates if p.suffix not in (".part", ".ytdl") and p.stat().st_size > 0
        ]
        return candidates[0] if candidates else None

    def _ydl_options(self, dest_dir: Path, progress: ProgressCallback | None) -> dict[str, Any]:
        h = self._config.max_height
        ext = self._config.container
        # Prefer a single progressive stream (no merge needed), else merge best
        # video+audio under the height cap, else anything under the cap, else best.
        fmt = (
            f"best[height<={h}][ext={ext}]/"
            f"bestvideo[height<={h}][ext={ext}]+bestaudio[ext=m4a]/"
            f"bestvideo[height<={h}]+bestaudio/"
            f"best[height<={h}]/best"
        )
        opts: dict[str, Any] = {
            "format": fmt,
            "merge_output_format": ext,
            "outtmpl": str(dest_dir / f"{_OUTPUT_STEM}.%(ext)s"),
            "noplaylist": True,
            "quiet": True,
            "no_warnings": True,
            "noprogress": True,
            "socket_timeout": self._config.socket_timeout_seconds,
            "retries": self._config.retries,
            "fragment_retries": self._config.retries,
            "logger": _YtDlpLoggerAdapter(),
            "progress_hooks": [self._make_progress_hook(progress)],
        }
        return opts

    def _download(
        self,
        url: str,
        dest_dir: Path,
        progress: ProgressCallback | None,
        reuse_existing: bool,
    ) -> Path:
        dest_dir.mkdir(parents=True, exist_ok=True)
        self._last_title = None

        if reuse_existing and (existing := self._existing_download(dest_dir)):
            logger.info("Reusing previously downloaded video: %s", existing)
            return existing

        logger.info("Downloading %s (max height %dp) -> %s", url, self._config.max_height, dest_dir)
        self._emit(progress, "Starting download", 0.0, {"url": url})

        try:
            with yt_dlp.YoutubeDL(self._ydl_options(dest_dir, progress)) as ydl:
                info = ydl.extract_info(url, download=True)
                if info is None:
                    raise DownloadError("yt-dlp returned no information for the URL.", details={"url": url})
                # Playlists are disabled, but some extractors still wrap a single entry.
                if "entries" in info:
                    entries = [e for e in info["entries"] if e]
                    if not entries:
                        raise DownloadError("URL resolved to an empty playlist.", details={"url": url})
                    info = entries[0]
                self._last_title = info.get("title")
                path = self._resolve_downloaded_path(ydl, info)
        except UnsupportedError as exc:
            logger.error("Unsupported URL %s: %s", url, exc)
            raise UnsupportedVideoError(
                f"This URL is not supported by the downloader: {url}", details={"url": url}
            ) from exc
        except (YtDlpDownloadError, ExtractorError) as exc:
            logger.error("yt-dlp failed for %s: %s", url, _clean_ytdlp_message(exc))
            raise DownloadError(
                f"Download failed: {_clean_ytdlp_message(exc)}", details={"url": url}
            ) from exc
        except DownloadError:
            raise
        except Exception as exc:  # noqa: BLE001 - boundary: wrap anything from yt-dlp
            logger.exception("Unexpected downloader error for %s", url)
            raise DownloadError(
                f"Unexpected error while downloading: {exc}", details={"url": url}
            ) from exc

        if not path.is_file() or path.stat().st_size == 0:
            logger.error("yt-dlp reported success but %s is missing/empty", path)
            raise DownloadError(
                "Download reported success but no file was produced.",
                details={"url": url, "expected_path": str(path)},
            )

        size_mb = path.stat().st_size / 1_000_000
        logger.info("Download complete: %s (%.1f MB)", path.name, size_mb)
        self._emit(progress, "Download complete", 1.0, {"path": str(path), "size_mb": round(size_mb, 1)})
        return path

    @staticmethod
    def _resolve_downloaded_path(ydl: yt_dlp.YoutubeDL, info: dict[str, Any]) -> Path:
        # After merging, yt-dlp records the final filename here.
        downloads = info.get("requested_downloads") or []
        if downloads and downloads[0].get("filepath"):
            return Path(downloads[0]["filepath"])
        return Path(ydl.prepare_filename(info))

    # ------------------------------------------------------------------ #
    # Progress
    # ------------------------------------------------------------------ #
    def _make_progress_hook(self, progress: ProgressCallback | None):
        last_bucket = {"value": -1}

        def hook(status: dict[str, Any]) -> None:
            if progress is None:
                return
            if status.get("status") == "downloading":
                total = status.get("total_bytes") or status.get("total_bytes_estimate")
                done = status.get("downloaded_bytes")
                if total and done is not None:
                    fraction = min(done / total, 1.0)
                    bucket = int(fraction * 20)  # report every 5 %
                    if bucket != last_bucket["value"]:
                        last_bucket["value"] = bucket
                        self._emit(
                            progress,
                            f"Downloading {fraction:.0%}",
                            fraction,
                            {"downloaded_mb": round(done / 1e6, 1), "total_mb": round(total / 1e6, 1)},
                        )
            elif status.get("status") == "finished":
                self._emit(progress, "Download finished, post-processing", 1.0, {})

        return hook

    @staticmethod
    def _emit(
        progress: ProgressCallback | None, message: str, fraction: float, details: dict[str, Any]
    ) -> None:
        if progress is not None:
            progress(ProgressEvent(PipelineStage.DOWNLOAD, message, fraction, details))

    # ------------------------------------------------------------------ #
    # Describe
    # ------------------------------------------------------------------ #
    def _describe(self, path: Path, source: str, title: str | None) -> VideoInfo:
        probe = probe_media(path, self._ffprobe)
        if not probe.has_video:
            logger.error("No video stream in %s", path)
            raise UnsupportedVideoError(
                f"File has no video stream: {path.name}", details={"path": str(path)}
            )
        info = VideoInfo(
            path=path,
            source_url=source,
            title=title,
            duration=probe.duration,
            fps=probe.fps,
            width=probe.width,
            height=probe.height,
            frame_count=probe.frame_count,
        )
        logger.info(
            "Video ready: %s | %.1fs | %sx%s @ %.3f fps | ~%s frames | audio=%s",
            path.name,
            info.duration or 0.0,
            info.width,
            info.height,
            info.fps or 0.0,
            info.frame_count,
            probe.has_audio,
        )
        return info


class _YtDlpLoggerAdapter:
    """Route yt-dlp's internal messages into our logging hierarchy."""

    _log = logging.getLogger("yt_dlp")

    def debug(self, msg: str) -> None:
        # yt-dlp sends info-level lines prefixed with "[debug]" or plain text.
        if not msg.startswith("[debug]"):
            self._log.debug(msg)

    def info(self, msg: str) -> None:
        self._log.info(msg)

    def warning(self, msg: str) -> None:
        self._log.warning(msg)

    def error(self, msg: str) -> None:
        self._log.error(msg)


def _clean_ytdlp_message(exc: Exception) -> str:
    """yt-dlp prefixes messages with 'ERROR: ' and ANSI codes; strip for users."""
    msg = str(exc)
    for prefix in ("ERROR: ", "\x1b[0;31mERROR:\x1b[0m "):
        if msg.startswith(prefix):
            msg = msg[len(prefix) :]
    return msg.strip()
