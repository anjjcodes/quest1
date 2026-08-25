"""API tests with a fake pipeline (no models, no network) via FastAPI's TestClient."""

from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from dialogue_locator.api.app import create_app
from dialogue_locator.api.jobs import JobManager
from dialogue_locator.config import Settings
from dialogue_locator.exceptions import DownloadError, PipelineCancelledError
from dialogue_locator.models import (
    FrameInfo,
    LocalizationResult,
    MatchCandidate,
    PipelineStage,
    ProgressEvent,
    ResultStatus,
)
from dialogue_locator.pipeline import PipelineRequest

DIALOGUE = "My mind rebels at stagnation"


class FakePipeline:
    """Scriptable stand-in for DialoguePipeline."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.mode = "found"  # found | not_found | error | slow | crash
        self.release = threading.Event()  # 'slow' jobs wait on this
        self.started = threading.Event()
        self.warmed = False

    def warm_up(self):
        self.warmed = True

    def run(self, request: PipelineRequest, progress=None, should_cancel=None):
        self.started.set()
        progress(ProgressEvent(PipelineStage.DOWNLOAD, "Downloading 50%", 0.5))
        if self.mode == "slow":
            while not self.release.wait(0.02):
                if should_cancel and should_cancel():
                    raise PipelineCancelledError("Cancelled during download")
        if self.mode == "error":
            raise DownloadError("video unavailable", details={"url": request.source})
        if self.mode == "crash":
            raise RuntimeError("worker bug")
        progress(ProgressEvent(PipelineStage.DONE, "Done", 1.0))
        if self.mode == "not_found":
            return LocalizationResult(ResultStatus.NOT_FOUND, request.dialogue, request.source,
                                      near_misses=[MatchCandidate(score=41.0, start=1.0, end=2.0, matched_text="nope")])
        out = self.settings.storage.output_dir / request.job_id
        out.mkdir(parents=True, exist_ok=True)
        img = out / "frame.jpg"
        img.write_bytes(b"\xff\xd8\xff\xd9")  # minimal JPEG marker bytes
        m = MatchCandidate(score=97.0, start=312.48, end=314.0, matched_text="My mind rebels at stagnation.")
        return LocalizationResult(ResultStatus.FOUND, request.dialogue, request.source, match=m, first_pass=m,
                                  frame=FrameInfo(7812, 312.48, 25.0, img, 640, 360), transcribed_seconds=320.0)


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(storage={"work_dir": tmp_path / "w", "output_dir": tmp_path / "o"}, server={"max_concurrent_jobs": 1, "job_retention_seconds": 3600})


@pytest.fixture
def fake(settings) -> FakePipeline:
    return FakePipeline(settings)


@pytest.fixture
def client(settings, fake):
    with TestClient(create_app(settings, pipeline=fake, warm_up=True)) as c:
        yield c


def wait_finished(client: TestClient, job_id: str, timeout: float = 5.0) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        data = client.get(f"/api/jobs/{job_id}").json()
        if data["status"] in ("done", "failed", "cancelled"):
            return data
        time.sleep(0.02)
    raise AssertionError(f"job {job_id} did not finish: {data}")


# --------------------------------------------------------------------------- #
def test_health_and_warm_up(client, fake, settings):
    r = client.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert body["fast_model"] == settings.whisper.fast_model and body["active_jobs"] == 0
    time.sleep(0.05)
    assert fake.warmed


def test_index_without_web_ui_returns_docs_pointer(client):
    r = client.get("/")
    assert r.status_code == 200


def test_create_job_found_flow(client, fake):
    r = client.post("/api/jobs", json={"source": "https://ok.ru/video/1", "dialogue": DIALOGUE})
    assert r.status_code == 202
    created = r.json()
    assert created["status"] in ("queued", "running", "done")
    job_id = created["job_id"]

    data = wait_finished(client, job_id)
    assert data["status"] == "done"
    assert data["result"]["timestamp"] == "00:05:12.480"
    assert data["result"]["frame_number"] == 7812
    assert data["result"]["matched_text"] == "My mind rebels at stagnation."
    assert data["frame_url"] == f"/api/jobs/{job_id}/frame"
    assert data["progress"]["stage"] == "done"
    assert [p["stage"] for p in data["progress_log"]] == ["download", "done"]
    assert data["started_at"] and data["finished_at"]

    img = client.get(data["frame_url"])
    assert img.status_code == 200 and img.headers["content-type"] == "image/jpeg"
    assert img.content.startswith(b"\xff\xd8")
    # result persisted to disk next to the frame
    assert (fake.settings.storage.output_dir / job_id / "result.json").is_file()


def test_not_found_flow(client, fake):
    fake.mode = "not_found"
    job_id = client.post("/api/jobs", json={"source": "https://ok.ru/video/1", "dialogue": DIALOGUE}).json()["job_id"]
    data = wait_finished(client, job_id)
    assert data["status"] == "done" and data["result"]["status"] == "not_found"
    assert data["result"]["near_misses"][0]["score"] == 41.0
    assert data["frame_url"] is None
    assert client.get(f"/api/jobs/{job_id}/frame").status_code == 404


def test_pipeline_error_becomes_failed_job(client, fake):
    fake.mode = "error"
    job_id = client.post("/api/jobs", json={"source": "https://ok.ru/video/1", "dialogue": DIALOGUE}).json()["job_id"]
    data = wait_finished(client, job_id)
    assert data["status"] == "failed"
    assert data["error"]["stage"] == "download" and "unavailable" in data["error"]["message"]
    assert data["result"] is None


def test_unexpected_crash_becomes_failed_job(client, fake):
    fake.mode = "crash"
    job_id = client.post("/api/jobs", json={"source": "https://ok.ru/video/1", "dialogue": DIALOGUE}).json()["job_id"]
    data = wait_finished(client, job_id)
    assert data["status"] == "failed" and data["error"]["error"] == "RuntimeError"


@pytest.mark.parametrize(
    "body, stage",
    [
        ({"source": "https://ok.ru/video/1", "dialogue": "   "}, "input"),
        ({"source": "https://ok.ru/video/1", "dialogue": "one"}, "input"),
        ({"source": "ftp://x.com/v", "dialogue": DIALOGUE}, "input"),
        ({"source": "", "dialogue": DIALOGUE}, "input"),
        ({"source": "/no/such/file.mp4", "dialogue": DIALOGUE}, "input"),
    ],
)
def test_invalid_input_is_422_before_queueing(client, body, stage):
    r = client.post("/api/jobs", json=body)
    assert r.status_code == 422
    assert r.json()["stage"] == stage
    assert client.get("/api/jobs").json()["jobs"] == []


def test_missing_fields_is_422(client):
    assert client.post("/api/jobs", json={"source": "https://ok.ru/video/1"}).status_code == 422


def test_unknown_job_404(client):
    assert client.get("/api/jobs/nope").status_code == 404
    assert client.get("/api/jobs/nope/frame").status_code == 404
    assert client.delete("/api/jobs/nope").status_code == 404


def test_cancel_running_job(client, fake):
    fake.mode = "slow"
    job_id = client.post("/api/jobs", json={"source": "https://ok.ru/video/1", "dialogue": DIALOGUE}).json()["job_id"]
    assert fake.started.wait(2.0)
    assert client.get(f"/api/jobs/{job_id}").json()["status"] == "running"
    r = client.delete(f"/api/jobs/{job_id}")
    assert r.status_code == 200
    data = wait_finished(client, job_id)
    assert data["status"] == "cancelled"
    assert data["error"]["error"] == "PipelineCancelledError"


def test_queued_job_waits_for_slot_and_can_be_cancelled(client, fake):
    fake.mode = "slow"
    first = client.post("/api/jobs", json={"source": "https://ok.ru/video/1", "dialogue": DIALOGUE}).json()["job_id"]
    assert fake.started.wait(2.0)
    second = client.post("/api/jobs", json={"source": "https://ok.ru/video/2", "dialogue": DIALOGUE}).json()["job_id"]
    assert client.get(f"/api/jobs/{second}").json()["status"] == "queued"
    assert client.get("/api/health").json()["active_jobs"] == 2
    # cancel the queued one: immediate
    assert client.delete(f"/api/jobs/{second}").json()["status"] == "cancelled"
    # release the running one
    fake.release.set()
    assert wait_finished(client, first)["status"] == "done"
    listing = client.get("/api/jobs").json()["jobs"]
    assert {j["job_id"] for j in listing} == {first, second}


def test_delete_finished_job_removes_it(client):
    job_id = client.post("/api/jobs", json={"source": "https://ok.ru/video/1", "dialogue": DIALOGUE}).json()["job_id"]
    wait_finished(client, job_id)
    assert client.delete(f"/api/jobs/{job_id}").status_code == 200
    assert client.get(f"/api/jobs/{job_id}").status_code == 404


def test_retention_expires_finished_jobs(settings, fake):
    manager = JobManager(fake, max_concurrent=1, retention_seconds=0)  # 0 = keep forever
    job = manager.submit(PipelineRequest(source="https://ok.ru/video/1", dialogue=DIALOGUE))
    for _ in range(100):
        if job.status.finished:
            break
        time.sleep(0.02)
    assert manager.get(job.id) is not None

    manager2 = JobManager(fake, max_concurrent=1, retention_seconds=1)
    job2 = manager2.submit(PipelineRequest(source="https://ok.ru/video/1", dialogue=DIALOGUE))
    for _ in range(100):
        if job2.status.finished:
            break
        time.sleep(0.02)
    from datetime import UTC, datetime, timedelta
    job2.finished_at = datetime.now(UTC) - timedelta(seconds=5)
    assert manager2.get(job2.id) is None  # expired on access
    manager.shutdown(); manager2.shutdown()


def test_openapi_lists_routes(client):
    paths = client.get("/openapi.json").json()["paths"]
    assert {"/api/jobs", "/api/jobs/{job_id}", "/api/jobs/{job_id}/frame", "/api/health"} <= set(paths)


# --------------------------------------------------------------------------- #
# web UI is served
# --------------------------------------------------------------------------- #
def test_web_ui_served(client):
    r = client.get("/")
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/html")
    assert "Dialogue Locator" in r.text and '/static/app.js' in r.text
    assert client.get("/static/app.js").status_code == 200
    assert client.get("/static/app.css").status_code == 200


# --------------------------------------------------------------------------- #
# settings API
# --------------------------------------------------------------------------- #
def test_get_settings_defaults(client):
    s = client.get("/api/settings").json()
    assert s["stages"] == {"verification": True, "face_detection": True, "mouth_movement": True}
    assert s["match_threshold"] == 80.0
    assert s["face_min_confidence"] == 0.3
    assert s["mouth_movement_threshold"] == 0.02


def test_job_with_settings_overrides(client):
    body = client.get("/api/settings").json()
    body["match_threshold"] = 90.0
    body["stages"]["mouth_movement"] = False
    create = {"source": "https://ok.ru/video/1", "dialogue": DIALOGUE, "settings": body}
    job = client.post("/api/jobs", json=create).json()
    # the job echoes its effective settings and still completes (fake pipeline
    # type is preserved by the rebuild factory)
    assert job["settings"]["match_threshold"] == 90.0
    assert job["settings"]["stages"]["mouth_movement"] is False
    assert wait_finished(client, job["job_id"])["status"] == "done"
    # nothing was stored server-side: defaults are untouched
    assert client.get("/api/settings").json()["match_threshold"] == 80.0


def test_job_settings_cascade_downstream_only(client):
    body = client.get("/api/settings").json()

    def submit(stages):
        body["stages"] = stages
        create = {"source": "https://ok.ru/video/1", "dialogue": DIALOGUE, "settings": body}
        return client.post("/api/jobs", json=create).json()["settings"]["stages"]

    # face off pulls mouth off, leaves verification alone
    assert submit({"verification": True, "face_detection": False, "mouth_movement": True}) == {
        "verification": True, "face_detection": False, "mouth_movement": False,
    }
    # verification off pulls the whole visual chain off
    assert submit({"verification": False, "face_detection": True, "mouth_movement": True}) == {
        "verification": False, "face_detection": False, "mouth_movement": False,
    }
    # downstream-only change touches nothing upstream
    assert submit({"verification": True, "face_detection": True, "mouth_movement": False}) == {
        "verification": True, "face_detection": True, "mouth_movement": False,
    }


def test_job_without_settings_echoes_defaults(client):
    create = {"source": "https://ok.ru/video/1", "dialogue": DIALOGUE}
    job = client.post("/api/jobs", json=create).json()
    assert job["settings"]["match_threshold"] == 80.0
    assert job["settings"]["stages"] == {
        "verification": True, "face_detection": True, "mouth_movement": True,
    }


def test_job_settings_rejects_bad_values(client):
    body = client.get("/api/settings").json()
    body["match_threshold"] = 150
    create = {"source": "https://ok.ru/video/1", "dialogue": DIALOGUE, "settings": body}
    assert client.post("/api/jobs", json=create).status_code == 422
