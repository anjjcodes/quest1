"""Fetch the source media.

Two inputs are accepted:

* an ``http(s)`` URL supported by yt-dlp (YouTube, ok.ru, Vimeo, direct .mp4 ...)
* a path to a local video file (handy for tests and offline runs)

Two fetches are offered:

* :meth:`VideoDownloader.fetch_search_media` - the cheap pass that feeds
  transcription: audio-only when the host offers it, else the lowest video
  rendition under ``search_max_height``. Returns a
  :class:`~dialogue_locator.models.MediaInfo` (which may have no video stream).
* :meth:`VideoDownloader.fetch` - the full-quality video (capped at
  ``max_height``), needed only for frame extraction once a match is confirmed.
  Returns a :class:`~dialogue_locator.models.VideoInfo`.

For a local file both methods resolve to the same file. Media properties
(fps / duration / frame count) come from ffprobe rather than from the site's
metadata (which is often missing or rounded).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import yt_dlp
from yt_dlp.utils import DownloadError as YtDlpDownloadError
from yt_dlp.utils import ExtractorError, UnsupportedError, download_range_func

from dialogue_locator.config import DownloadConfig
from dialogue_locator.exceptions import DownloadError, InvalidURLError, UnsupportedVideoError
from dialogue_locator.media.probe import probe_media
from dialogue_locator.models import MediaInfo, PipelineStage, ProgressCallback, ProgressEvent, VideoInfo

logger = logging.getLogger(__name__)

_ALLOWED_SCHEMES = ("http", "https")
_VIDEO_STEM = "video"  # full-quality fetch, cached per source
_MEDIA_STEM = "media"  # cheap search fetch, cached separately (different format)


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
        """Obtain the full-quality video for ``source`` and return its :class:`VideoInfo`.

        Args:
            source: http(s) URL or local file path.
            dest_dir: directory to download into (created if missing).
            progress: optional callback for download progress events.
            reuse_existing: if a previous download exists in ``dest_dir``, use it
                instead of re-downloading (speeds up repeated runs on one video).
        """
        path, title = self._obtain(source, dest_dir, progress, reuse_existing,
                                   fmt=self._video_format(), stem=_VIDEO_STEM, label="video")
        return self._describe(path, source, title)

    def fetch_search_media(
        self,
        source: str,
        dest_dir: Path,
        progress: ProgressCallback | None = None,
        reuse_existing: bool = True,
    ) -> MediaInfo:
        """Obtain the cheap search-pass media for ``source`` (audio-only if possible).

        Same arguments as :meth:`fetch`. For URLs this downloads the best
        audio-only stream, falling back to the lowest video rendition under
        ``search_max_height`` on hosts without separate audio streams. For a
        local file it resolves the file itself. Raises
        ``UnsupportedVideoError`` if the result has no audio stream, since
        there is then no dialogue to search.
        """
        path, title = self._obtain(source, dest_dir, progress, reuse_existing,
                                   fmt=self._search_format(), stem=_MEDIA_STEM, label="audio")
        return self._describe_media(path, source, title)

    def fetch_video_clip(
        self,
        source: str,
        start: float,
        end: float,
        dest_dir: Path,
        progress: ProgressCallback | None = None,
        reuse_existing: bool = True,
    ) -> VideoInfo:
        """Download only ``[start, end]`` (padded by ``clip_padding_seconds``)
        of ``source`` at full quality, for frame extraction around a verified
        match. Far cheaper than :meth:`fetch` on long videos.

        The cut is re-encoded to start exactly at the padded start, so the
        returned ``VideoInfo.clip_start`` maps source timestamps precisely.
        A local file is already on disk in full, so it is returned whole
        (``clip_start`` 0.0).
        """
        if end <= start:
            raise DownloadError(
                f"Invalid clip range: start={start:.3f} must be < end={end:.3f}",
                details={"start": start, "end": end},
            )
        source = (source or "").strip()
        if not source:
            logger.warning("Rejected input: empty video source")
            raise InvalidURLError("Video source is empty.")
        if not is_url(source):
            path = self._resolve_local(source)
            return self._describe(path, source, path.stem)

        pad = self._config.clip_padding_seconds
        clip_start = max(0.0, float(start) - pad)
        clip_end = float(end) + pad
        stem = f"clip_{int(clip_start * 1000)}_{int(clip_end * 1000)}"
        path = self._download(
            validate_url(source), dest_dir, progress, reuse_existing,
            fmt=self._video_format(), stem=stem, label="video clip",
            section=(clip_start, clip_end),
        )
        return self._describe(path, source, self._last_title, clip_start=clip_start)

    def _obtain(
        self,
        source: str,
        dest_dir: Path,
        progress: ProgressCallback | None,
        reuse_existing: bool,
        *,
        fmt: str,
        stem: str,
        label: str,
    ) -> tuple[Path, str | None]:
        source = (source or "").strip()
        if not source:
            logger.warning("Rejected input: empty video source")
            raise InvalidURLError("Video source is empty.")

        if is_url(source):
            path = self._download(validate_url(source), dest_dir, progress, reuse_existing,
                                  fmt=fmt, stem=stem, label=label)
            return path, self._last_title
        path = self._resolve_local(source)
        return path, path.stem

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
    def _video_format(self) -> str:
        h = self._config.max_height
        ext = self._config.container
        # Prefer a single progressive stream (no merge needed), else merge best
        # video+audio under the height cap, else anything under the cap, else best.
        return (
            f"best[height<={h}][ext={ext}]/"
            f"bestvideo[height<={h}][ext={ext}]+bestaudio[ext=m4a]/"
            f"bestvideo[height<={h}]+bestaudio/"
            f"best[height<={h}]/best"
        )

    def _search_format(self) -> str:
        h = self._config.search_max_height
        # Audio-only if the host offers it; otherwise the smallest combined
        # (video+audio) rendition; 'worst*[acodec!=none]' as a last resort
        # covers hosts whose only small formats are not combined.
        return (
            "bestaudio[ext=m4a]/"
            "bestaudio/"
            f"worst[height<={h}]/"
            "worst/"
            "worst*[acodec!=none]"
        )

    def _existing_download(self, dest_dir: Path, stem: str) -> Path | None:
        candidates = sorted(dest_dir.glob(f"{stem}.*"))
        # yt-dlp leaves .part / .ytdl files behind on interrupted downloads.
        candidates = [
            p for p in candidates if p.suffix not in (".part", ".ytdl") and p.stat().st_size > 0
        ]
        return candidates[0] if candidates else None

    def _ydl_options(
        self,
        dest_dir: Path,
        progress: ProgressCallback | None,
        fmt: str,
        stem: str,
        section: tuple[float, float] | None = None,
    ) -> dict[str, Any]:
        opts: dict[str, Any] = {
            "format": fmt,
            "merge_output_format": self._config.container,
            "outtmpl": str(dest_dir / f"{stem}.%(ext)s"),
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
        if section is not None:
            opts["download_ranges"] = download_range_func(None, [section])
            # Re-encode the cut so the clip begins exactly at section[0]; without
            # this the clip starts at the previous keyframe and clip_start would
            # be wrong, breaking the timestamp -> frame mapping.
            opts["force_keyframes_at_cuts"] = True
        return opts

    def _download(
        self,
        url: str,
        dest_dir: Path,
        progress: ProgressCallback | None,
        reuse_existing: bool,
        *,
        fmt: str,
        stem: str,
        label: str,
        section: tuple[float, float] | None = None,
    ) -> Path:
        dest_dir.mkdir(parents=True, exist_ok=True)
        self._last_title = None

        if reuse_existing and (existing := self._existing_download(dest_dir, stem)):
            logger.info("Reusing previously downloaded %s: %s", label, existing)
            return existing

        logger.info("Downloading %s (%s, format %r) -> %s", url, label, fmt, dest_dir)
        self._emit(progress, f"Starting {label} download", 0.0, {"url": url, "kind": label})

        try:
            with yt_dlp.YoutubeDL(self._ydl_options(dest_dir, progress, fmt, stem, section)) as ydl:
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
    def _describe(self, path: Path, source: str, title: str | None, clip_start: float = 0.0) -> VideoInfo:
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
            clip_start=clip_start,
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

    def _describe_media(self, path: Path, source: str, title: str | None) -> MediaInfo:
        probe = probe_media(path, self._ffprobe)
        if not probe.has_audio:
            logger.error("No audio stream in %s; nothing to search", path)
            raise UnsupportedVideoError(
                f"File has no audio stream, so the dialogue cannot be searched: {path.name}",
                details={"path": str(path)},
            )
        info = MediaInfo(
            path=path,
            source_url=source,
            title=title,
            duration=probe.duration,
            has_video=probe.has_video,
        )
        logger.info(
            "Search media ready: %s | %.1fs | video=%s",
            path.name,
            info.duration or 0.0,
            info.has_video,
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
