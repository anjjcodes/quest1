"""Shared pytest fixtures.

Media fixtures are generated with FFmpeg's synthetic sources (``testsrc`` /
``sine``) so the suite runs fully offline and needs no binary files in git.
Tests that hit the network are marked ``network`` and only run when
``DL_RUN_NETWORK_TESTS=1`` is set.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

FFMPEG = shutil.which("ffmpeg")
FFPROBE = shutil.which("ffprobe")

requires_ffmpeg = pytest.mark.skipif(
    FFMPEG is None or FFPROBE is None, reason="ffmpeg/ffprobe not installed"
)


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "network: test downloads from the internet")


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if os.environ.get("DL_RUN_NETWORK_TESTS") == "1":
        return
    skip = pytest.mark.skip(reason="set DL_RUN_NETWORK_TESTS=1 to run network tests")
    for item in items:
        if "network" in item.keywords:
            item.add_marker(skip)


def _ffmpeg(*args: str) -> None:
    assert FFMPEG is not None
    subprocess.run([FFMPEG, "-v", "error", "-y", *args], check=True)


@pytest.fixture(scope="session")
def media_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tmp_path_factory.mktemp("media")


@pytest.fixture(scope="session")
def sample_video(media_dir: Path) -> Path:
    """3 s, 320x240 @ 25 fps H.264 with a 440 Hz sine audio track."""
    if FFMPEG is None:
        pytest.skip("ffmpeg not installed")
    path = media_dir / "sample.mp4"
    _ffmpeg(
        "-f", "lavfi", "-i", "testsrc=size=320x240:rate=25",
        "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=44100",
        "-t", "3", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", str(path),
    )  # fmt: skip
    return path


@pytest.fixture(scope="session")
def silent_video(media_dir: Path) -> Path:
    """2 s video with no audio stream at all."""
    if FFMPEG is None:
        pytest.skip("ffmpeg not installed")
    path = media_dir / "silent.mp4"
    _ffmpeg(
        "-f", "lavfi", "-i", "testsrc=size=160x120:rate=25",
        "-t", "2", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-an", str(path),
    )  # fmt: skip
    return path


@pytest.fixture(scope="session")
def sample_wav(media_dir: Path) -> Path:
    """4 s 16 kHz mono WAV (audio-only media)."""
    if FFMPEG is None:
        pytest.skip("ffmpeg not installed")
    path = media_dir / "sample.wav"
    _ffmpeg(
        "-f", "lavfi", "-i", "sine=frequency=300:sample_rate=16000",
        "-t", "4", "-ac", "1", "-c:a", "pcm_s16le", str(path),
    )  # fmt: skip
    return path


@pytest.fixture
def not_media(tmp_path: Path) -> Path:
    path = tmp_path / "notes.txt"
    path.write_text("this is not a video")
    return path
