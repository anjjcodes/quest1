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
def test_fetch_video_clip_local_file(sample_video: Path, tmp_path: Path):
    info = VideoDownloader(DownloadConfig()).fetch_video_clip(str(sample_video), 0.5, 1.5, tmp_path)
    assert info.path == sample_video.resolve()
    assert info.source_url == str(sample_video)
    assert info.title == "sample"
    assert info.fps == 25.0 and info.frame_count == 75
    assert (info.width, info.height) == (320, 240)


@requires_ffmpeg
def test_fetch_local_missing_file(tmp_path: Path):
    with expect_error(InvalidURLError):
        VideoDownloader(DownloadConfig()).fetch_search_media(str(tmp_path / "missing.mp4"), tmp_path)


@requires_ffmpeg
def test_fetch_video_clip_audio_only_file_is_unsupported(sample_wav: Path, tmp_path: Path):
    with expect_error(UnsupportedVideoError):
        VideoDownloader(DownloadConfig()).fetch_video_clip(str(sample_wav), 1.0, 2.0, tmp_path)


@requires_ffmpeg
def test_fetch_search_media_local_video(sample_video: Path, tmp_path: Path):
    info = VideoDownloader(DownloadConfig()).fetch_search_media(str(sample_video), tmp_path)
    assert info.path == sample_video.resolve()
    assert info.source_url == str(sample_video)
    assert info.has_video is True
    assert info.duration and info.duration > 0


@requires_ffmpeg
def test_fetch_search_media_local_wav_is_searchable(sample_wav: Path, tmp_path: Path):
    # An audio-only file cannot serve as the video, but it can be searched.
    info = VideoDownloader(DownloadConfig()).fetch_search_media(str(sample_wav), tmp_path)
    assert info.path == sample_wav.resolve()
    assert info.has_video is False


def test_fetch_empty_source(tmp_path: Path):
    with expect_error(InvalidURLError):
        VideoDownloader(DownloadConfig()).fetch_search_media("", tmp_path)


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
    _FakeYDL.last_opts = None
    monkeypatch.setattr(dl_module.yt_dlp, "YoutubeDL", _FakeYDL)
    return _FakeYDL


@requires_ffmpeg
def test_fetch_video_clip_url_success(fake_ydl, tmp_path: Path):
    events: list[ProgressEvent] = []
    info = VideoDownloader(DownloadConfig(max_height=360, clip_padding_seconds=2.0)).fetch_video_clip(
        "https://example.com/v", 10.0, 13.0, tmp_path / "job", progress=events.append, reuse_existing=False
    )
    assert info.title == "Fake Title"
    assert info.path == fake_ydl.produced
    assert info.fps == 25.0
    assert all(e.stage is PipelineStage.DOWNLOAD for e in events)
    assert events[0].fraction == 0.0 and events[-1].fraction == 1.0
    # config reaches yt-dlp
    assert "height<=360" in fake_ydl.last_opts["format"]
    assert fake_ydl.last_opts["noplaylist"] is True
    assert "clip_8000_15000.%(ext)s" in fake_ydl.last_opts["outtmpl"]


@requires_ffmpeg
def test_fetch_search_media_url(fake_ydl, tmp_path: Path):
    info = VideoDownloader(DownloadConfig(search_max_height=240)).fetch_search_media(
        "https://example.com/v", tmp_path / "job", reuse_existing=False
    )
    assert info.title == "Fake Title"
    assert info.path == fake_ydl.produced
    # audio-first format with a low-video fallback, cached under its own stem
    fmt = fake_ydl.last_opts["format"]
    assert fmt.startswith("bestaudio")
    assert "worst[height<=240]" in fmt
    assert "media.%(ext)s" in fake_ydl.last_opts["outtmpl"]


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
        VideoDownloader(DownloadConfig()).fetch_search_media(
            "https://example.com/v", tmp_path / "job", reuse_existing=False
        )
    assert exc.value.stage == "download"
    if behaviour == "ytdlp_error":
        assert "ERROR:" not in exc.value.message  # prefix stripped for users


@requires_ffmpeg
def test_fetch_search_media_reuses_existing_download(
    monkeypatch: pytest.MonkeyPatch, sample_video: Path, tmp_path: Path
):
    job = tmp_path / "job"
    job.mkdir()
    (job / "media.mp4").write_bytes(sample_video.read_bytes())
    (job / "media.mp4.part").write_bytes(b"")  # leftover partial must be ignored

    def boom(*a, **k):
        raise AssertionError("network must not be touched when a search download exists")

    monkeypatch.setattr(dl_module.yt_dlp, "YoutubeDL", boom)
    info = VideoDownloader(DownloadConfig()).fetch_search_media("https://example.com/v", job)
    assert info.path == job / "media.mp4"
    assert info.source_url == "https://example.com/v"


@requires_ffmpeg
def test_fetch_video_clip_url(fake_ydl, tmp_path: Path):
    info = VideoDownloader(DownloadConfig(clip_padding_seconds=2.0)).fetch_video_clip(
        "https://example.com/v", 10.0, 13.0, tmp_path / "job", reuse_existing=False
    )
    assert info.clip_start == 8.0  # 10 - 2s padding
    # a padded time range is requested and cut exactly at the boundaries
    assert fake_ydl.last_opts["download_ranges"] is not None
    assert fake_ydl.last_opts["force_keyframes_at_cuts"] is True
    assert "clip_8000_15000.%(ext)s" in fake_ydl.last_opts["outtmpl"]


@requires_ffmpeg
def test_fetch_video_clip_padding_clamped_at_zero(fake_ydl, tmp_path: Path):
    info = VideoDownloader(DownloadConfig(clip_padding_seconds=5.0)).fetch_video_clip(
        "https://example.com/v", 1.0, 2.0, tmp_path / "job", reuse_existing=False
    )
    assert info.clip_start == 0.0
    assert "clip_0_7000.%(ext)s" in fake_ydl.last_opts["outtmpl"]


@requires_ffmpeg
def test_fetch_video_clip_local_file_returned_whole(sample_video: Path, tmp_path: Path):
    info = VideoDownloader(DownloadConfig()).fetch_video_clip(str(sample_video), 1.0, 2.0, tmp_path)
    assert info.path == sample_video.resolve()
    assert info.clip_start == 0.0  # whole local file, no clipping


def test_fetch_video_clip_invalid_range(tmp_path: Path):
    with expect_error(DownloadError):
        VideoDownloader(DownloadConfig()).fetch_video_clip("https://example.com/v", 5.0, 5.0, tmp_path)


@requires_ffmpeg
def test_fetch_video_clip_reuses_cached_range(monkeypatch: pytest.MonkeyPatch, sample_video: Path, tmp_path: Path):
    job = tmp_path / "job"
    job.mkdir()
    (job / "clip_8000_15000.mp4").write_bytes(sample_video.read_bytes())

    def boom(*a, **k):
        raise AssertionError("network must not be touched when this clip range exists")

    monkeypatch.setattr(dl_module.yt_dlp, "YoutubeDL", boom)
    info = VideoDownloader(DownloadConfig(clip_padding_seconds=2.0)).fetch_video_clip(
        "https://example.com/v", 10.0, 13.0, job
    )
    assert info.path == job / "clip_8000_15000.mp4"
    assert info.clip_start == 8.0


@requires_ffmpeg
def test_search_and_clip_caches_are_independent(fake_ydl, sample_video: Path, tmp_path: Path):
    # A cached video clip must not satisfy the search fetch (and vice versa).
    job = tmp_path / "job"
    job.mkdir()
    (job / "clip_8000_15000.mp4").write_bytes(sample_video.read_bytes())

    VideoDownloader(DownloadConfig()).fetch_search_media("https://example.com/v", job)
    assert fake_ydl.last_opts is not None
    assert "media.%(ext)s" in fake_ydl.last_opts["outtmpl"]


# --------------------------------------------------------------------------- #
# Real network (opt-in)
# --------------------------------------------------------------------------- #
@pytest.mark.network
@requires_ffmpeg
def test_real_youtube_search_media(tmp_path: Path):
    info = VideoDownloader(DownloadConfig()).fetch_search_media(
        "https://www.youtube.com/watch?v=jNQXAC9IVRw", tmp_path / "yt"
    )
    assert info.path.is_file()
    assert info.has_video is False  # YouTube offers audio-only streams
    assert info.duration and 18 < info.duration < 21


@pytest.mark.network
@requires_ffmpeg
def test_real_youtube_video_clip(tmp_path: Path):
    # 19s source video; ask for 4-6s with 1s padding -> ~5s clip starting at 3s
    info = VideoDownloader(DownloadConfig(clip_padding_seconds=1.0)).fetch_video_clip(
        "https://www.youtube.com/watch?v=jNQXAC9IVRw", 4.0, 6.0, tmp_path / "yt"
    )
    assert info.path.is_file()
    assert info.clip_start == 3.0
    assert info.duration and 3 < info.duration < 8  # a section, not the whole 19s video
    assert info.fps and info.width


@pytest.mark.network
def test_real_invalid_video(tmp_path: Path):
    with expect_error(DownloadError):
        VideoDownloader(DownloadConfig()).fetch_search_media(
            "https://www.youtube.com/watch?v=xxxxxxxxxxx", tmp_path / "bad"
        )


# --------------------------------------------------------------------------- #
# time-throttled progress
# --------------------------------------------------------------------------- #
def test_progress_hook_time_throttled_with_speed_and_eta(monkeypatch):
    clock = {"t": 100.0}
    monkeypatch.setattr(dl_module.time, "monotonic", lambda: clock["t"])
    events: list[ProgressEvent] = []
    hook = VideoDownloader(DownloadConfig(progress_interval_seconds=2.0))._make_progress_hook(events.append)

    hook({"status": "downloading", "downloaded_bytes": 1_000_000, "total_bytes": 10_000_000,
          "speed": 195_000.0, "eta": 46})
    hook({"status": "downloading", "downloaded_bytes": 2_000_000, "total_bytes": 10_000_000})  # too soon
    clock["t"] += 2.1
    hook({"status": "downloading", "downloaded_bytes": 3_000_000, "total_bytes_estimate": 10_000_000,
          "speed": 2_400_000.0, "eta": 3665})
    assert len(events) == 2  # the middle call was throttled away
    assert events[0].message == "Downloading 10% \N{MIDDLE DOT} 1.0/10.0 MB \N{MIDDLE DOT} 195 KB/s \N{MIDDLE DOT} ETA 0:46"
    assert events[0].fraction == pytest.approx(0.1)
    assert events[0].details["speed_bps"] == 195_000 and events[0].details["eta_seconds"] == 46
    assert events[1].message == "Downloading 30% \N{MIDDLE DOT} 3.0/10.0 MB \N{MIDDLE DOT} 2.4 MB/s \N{MIDDLE DOT} ETA 1:01:05"

    # 'finished' resets the throttle so the next file (e.g. audio after video)
    # reports at once; without a total only the size is shown.
    hook({"status": "finished"})
    assert events[-1].fraction == 1.0
    hook({"status": "downloading", "downloaded_bytes": 500_000})
    assert events[-1].message == "Downloading 0.5 MB"
    assert events[-1].fraction is None
