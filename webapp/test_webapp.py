# webapp/test_webapp.py — run: pytest webapp/test_webapp.py -q
import io
import time

import pytest
from fastapi.testclient import TestClient

from webapp import main as webmain
from webapp import pipeline
from webapp.jobs import store


@pytest.fixture(autouse=True)
def _stub_engine():
    saved_pipeline = list(pipeline.PIPELINES["generate"])

    def fake_validate(job, **kwargs): job.log("INFO", "ok")
    def fake_load(job, **kwargs): return None
    def fake_segment(job, **kwargs):
        job.panels = [{"id": f"panel_{i:03d}", "panel_index": i,
                       "y_start": 0, "y_end": 10, "narration": "n",
                       "dialogue": "", "panel_type": "dialogue",
                       "confidence": 0.9,
                       "image_file": f"panel_{i:03d}.png"}
                      for i in range(1, 5)]
    def fake_gemini(job, **kwargs): pass
    def fake_rest(job, **kwargs): pass

    stubbed = list(saved_pipeline)
    stubbed[0] = ("validate_config", fake_validate)
    for i, name in enumerate(["load_images", "segment_panels",
                              "apply_order", "gemini_narration",
                              "build_script", "tts_audio",
                              "render_video", "save_outputs"]):
        if name == "segment_panels":
            fn = fake_segment
        elif name == "apply_order":
            fn = pipeline._apply_order
        else:
            fn = fake_rest
        stubbed[i] = (name, fn)
    pipeline.PIPELINES["generate"] = stubbed
    yield
    pipeline.PIPELINES["generate"] = saved_pipeline


def _wait_done(job_id, timeout=5.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        j = store.get(job_id)
        if j and j.status.value in ("completed", "failed", "cancelled"):
            return j
        time.sleep(0.05)
    raise AssertionError(f"job {job_id} never reached a terminal state")


def test_run_lifecycle_and_order(client=None):
    c = TestClient(webmain.app)
    up = c.post("/api/upload", files={"file": ("s.png", io.BytesIO(b"x"),
                                               "image/png")})
    assert up.status_code == 200 and "job_id" in up.json()
    session = up.json()["job_id"]

    order = ["panel_003", "panel_001", "panel_004", "panel_002"]
    r = c.post("/api/run", json={"session": session, "order": order})
    assert r.status_code == 200
    body = r.json()
    assert body["job_id"]
    assert body["status"] in ("queued", "running")

    job = _wait_done(body["job_id"])
    assert job.status.value == "completed"
    assert [p["id"] for p in job.panels] == order
    logtext = "\n".join(entry["msg"] for entry in job._logs)
    assert "1 = panel_003" in logtext
    assert job.error is None


def test_failure_reaches_status_and_message():
    c = TestClient(webmain.app)
    r = c.post("/api/run", json={"session": "nope"})
    assert r.status_code == 404


def test_credentials_passed_to_segment_stages():
    c = TestClient(webmain.app)
    up = c.post("/api/upload", files={"file": ("s.png", io.BytesIO(b"x"),
                                               "image/png")})
    assert up.status_code == 200
    session = up.json()["job_id"]

    captured = {}

    def fake_validate(job, **kwargs):
        captured["validate"] = kwargs
        job.log("INFO", "ok")

    def fake_segment(job, **kwargs):
        captured["segment"] = kwargs
        job.panels = [{"id": f"panel_{i:03d}", "panel_index": i,
                       "y_start": 0, "y_end": 10, "narration": "n",
                       "dialogue": "", "panel_type": "dialogue",
                       "confidence": 0.9,
                       "image_file": f"panel_{i:03d}.png"}
                      for i in range(1, 5)]

    saved = list(pipeline.PIPELINES["generate"])
    saved[0] = ("validate_config", fake_validate)
    new_pipeline = []
    for i, name in enumerate(["load_images", "segment_panels",
                              "apply_order", "gemini_narration",
                              "build_script", "tts_audio",
                              "render_video", "save_outputs"]):
        if name == "segment_panels":
            fn = fake_segment
        elif name == "apply_order":
            fn = pipeline._apply_order
        else:
            fn = lambda job, **kwargs: None
        new_pipeline.append((name, fn))
    pipeline.PIPELINES["generate"] = [("validate_config", fake_validate)] + new_pipeline
    r = c.post("/api/run", json={
        "session": session,
        "backend": "gemini",
        "api_key": "secret-key",
        "model": "gemini-2.5-flash",
        "endpoint": "https://custom.example.com",
        "cf_account_id": "cf-123",
    })
    assert r.status_code == 200
    job = _wait_done(r.json()["job_id"])
    assert job.status.value == "completed"
    assert captured["validate"]["api_key"] == "secret-key"
    assert captured["validate"]["model"] == "gemini-2.5-flash"
    assert captured["segment"]["api_key"] == "secret-key"
    assert captured["segment"]["model"] == "gemini-2.5-flash"
    assert captured["segment"]["base_url"] == "https://custom.example.com"
    assert captured["segment"]["cf_account_id"] == "cf-123"
    pipeline.PIPELINES["generate"] = saved
