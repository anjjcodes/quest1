from pathlib import Path

import pytest
from yt_dlp.utils import DownloadError as YtDlpDownloadError
from yt_dlp.utils import UnsupportedError

from dialogue_locator.config import DownloadConfig
from dialogue_locator.exceptions import DownloadError, InvalidURLError, UnsupportedVideoError
from dialogue_locator.media import downloader as dl_module
from dialogue_locator.media.downloader import VideoDownloader, is_url, validate_url
from dialogue_locator.models import PipelineStage, ProgressEvent
from tests.conftest import expect_error, requires_ffmpeg


# --------------------------------------------------------------------------- #
# URL validation (pure, no I/O)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "url",
    [
        "https://ok.ru/video/248244667877",
        "https://www.youtube.com/watch?v=jNQXAC9IVRw",
        "http://example.com/clip.mp4",
        "  https://youtu.be/abc  ",
    ],
)
def test_validate_url_accepts(url):
    assert validate_url(url) == url.strip()
    assert is_url(url)


@pytest.mark.parametrize("url", ["", "   ", "ftp://x.com/v", "youtube.com/watch", "http://nohost", "/tmp/x.mp4"])
def test_validate_url_rejects(url):
    with expect_error(InvalidURLError):
        validate_url(url)


def test_is_url_false_for_paths():
    assert not is_url("/tmp/video.mp4")
    assert not is_url("video.mp4")


# --------------------------------------------------------------------------- #
# Local file path
# --------------------------------------------------------------------------- #
@requires_ffmpeg
def test_fetch_local_file(sample_video: Path, tmp_path: Path):
    info = VideoDownloader(DownloadConfig()).fetch(str(sample_video), tmp_path)
    assert info.path == sample_video.resolve()
    assert info.source_url == str(sample_video)
    assert info.title == "sample"
    assert info.fps == 25.0 and info.frame_count == 75
    assert (info.width, info.height) == (320, 240)


@requires_ffmpeg
def test_fetch_local_missing_file(tmp_path: Path):
    with expect_error(InvalidURLError):
        VideoDownloader(DownloadConfig()).fetch(str(tmp_path / "missing.mp4"), tmp_path)


@requires_ffmpeg
def test_fetch_audio_only_file_is_unsupported(sample_wav: Path, tmp_path: Path):
    with expect_error(UnsupportedVideoError):
        VideoDownloader(DownloadConfig()).fetch(str(sample_wav), tmp_path)


def test_fetch_empty_source(tmp_path: Path):
    with expect_error(InvalidURLError):
        VideoDownloader(DownloadConfig()).fetch("", tmp_path)


# --------------------------------------------------------------------------- #
# yt-dlp branch, with yt_dlp.YoutubeDL replaced by fakes (no network)
# --------------------------------------------------------------------------- #
class _FakeYDL:
    """Minimal stand-in for yt_dlp.YoutubeDL used as a context manager."""

    behaviour = "ok"
    produced: Path | None = None
    last_opts: dict | None = None

    def __init__(self, opts):
        _FakeYDL.last_opts = opts

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def extract_info(self, url, download=True):
        if self.behaviour == "unsupported":
            raise UnsupportedError(url)
        if self.behaviour == "ytdlp_error":
            raise YtDlpDownloadError("ERROR: [youtube] abc: Video unavailable")
        if self.behaviour == "crash":
            raise RuntimeError("weird")
        if self.behaviour == "nofile":
            return {"title": "ghost", "requested_downloads": [{"filepath": "/nonexistent/x.mp4"}]}
        for hook in self.last_opts["progress_hooks"]:
            hook({"status": "downloading", "downloaded_bytes": 50, "total_bytes": 100})
            hook({"status": "finished"})
        return {"title": "Fake Title", "requested_downloads": [{"filepath": str(self.produced)}]}


@pytest.fixture
def fake_ydl(monkeypatch: pytest.MonkeyPatch, sample_video: Path, tmp_path: Path):
    """Patch yt_dlp.YoutubeDL; 'downloads' are simulated by copying the sample video."""
    produced = tmp_path / "video.mp4"
    produced.write_bytes(sample_video.read_bytes())
    _FakeYDL.produced = produced
    _FakeYDL.behaviour = "ok"
    monkeypatch.setattr(dl_module.yt_dlp, "YoutubeDL", _FakeYDL)
    return _FakeYDL


@requires_ffmpeg
def test_fetch_url_success(fake_ydl, tmp_path: Path):
    events: list[ProgressEvent] = []
    info = VideoDownloader(DownloadConfig(max_height=360)).fetch(
        "https://example.com/v", tmp_path / "job", progress=events.append, reuse_existing=False
    )
    assert info.title == "Fake Title"
    assert info.path == fake_ydl.produced
    assert info.fps == 25.0
    assert all(e.stage is PipelineStage.DOWNLOAD for e in events)
    assert events[0].fraction == 0.0 and events[-1].fraction == 1.0
    # config reaches yt-dlp
    assert "height<=360" in fake_ydl.last_opts["format"]
    assert fake_ydl.last_opts["noplaylist"] is True


@requires_ffmpeg
@pytest.mark.parametrize(
    "behaviour, expected",
    [
        ("unsupported", UnsupportedVideoError),
        ("ytdlp_error", DownloadError),
        ("crash", DownloadError),
        ("nofile", DownloadError),
    ],
)
def test_fetch_url_error_mapping(fake_ydl, tmp_path: Path, behaviour, expected):
    fake_ydl.behaviour = behaviour
    with expect_error(expected) as exc:
        VideoDownloader(DownloadConfig()).fetch("https://example.com/v", tmp_path / "job", reuse_existing=False)
    assert exc.value.stage == "download"
    if behaviour == "ytdlp_error":
        assert "ERROR:" not in exc.value.message  # prefix stripped for users


@requires_ffmpeg
def test_fetch_url_reuses_existing_download(monkeypatch: pytest.MonkeyPatch, sample_video: Path, tmp_path: Path):
    job = tmp_path / "job"
    job.mkdir()
    (job / "video.mp4").write_bytes(sample_video.read_bytes())
    (job / "video.mp4.part").write_bytes(b"")  # leftover partial must be ignored

    def boom(*a, **k):
        raise AssertionError("network must not be touched when a download exists")

    monkeypatch.setattr(dl_module.yt_dlp, "YoutubeDL", boom)
    info = VideoDownloader(DownloadConfig()).fetch("https://example.com/v", job)
    assert info.path == job / "video.mp4"
    assert info.source_url == "https://example.com/v"


# --------------------------------------------------------------------------- #
# Real network (opt-in)
# --------------------------------------------------------------------------- #
@pytest.mark.network
@requires_ffmpeg
def test_real_youtube_download(tmp_path: Path):
    info = VideoDownloader(DownloadConfig(max_height=240)).fetch(
        "https://www.youtube.com/watch?v=jNQXAC9IVRw", tmp_path / "yt"
    )
    assert info.path.is_file() and info.path.suffix == ".mp4"
    assert info.duration and 18 < info.duration < 21
    assert info.fps and info.frame_count


@pytest.mark.network
def test_real_invalid_video(tmp_path: Path):
    with expect_error(DownloadError):
        VideoDownloader(DownloadConfig()).fetch("https://www.youtube.com/watch?v=xxxxxxxxxxx", tmp_path / "bad")
