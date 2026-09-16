# webapp/test_webapp.py — run: pytest webapp/test_webapp.py -q
import io
import time

import pytest
from fastapi.testclient import TestClient

from webapp import main as webmain
from webapp import pipeline
from webapp.jobs import store


def _png_bytes(w: int = 4, h: int = 8) -> bytes:
    """A real (tiny) PNG so upload validation and load_images pass."""
    from PIL import Image
    buf = io.BytesIO()
    Image.new("RGB", (w, h), (255, 255, 255)).save(buf, "PNG")
    return buf.getvalue()


def _stub_generate_pipeline(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace every heavy generate stage BY NAME (index-based stubbing
    broke whenever a stage was inserted) with fakes, keeping the real
    apply_confirmed/apply_order logic under test."""
    def fake_stage(job, **kwargs): job.log("INFO", "ok")
    def fake_segment(job, **kwargs):
        job.panels = [{"id": f"panel_{i:03d}", "panel_index": i,
                       "y_start": i, "y_end": i + 10, "narration": "n",
                       "dialogue": "", "panel_type": "dialogue",
                       "confidence": 0.9,
                       "image_file": f"panel_{i:03d}.png"}
                      for i in range(1, 5)]
    stubbed = []
    for name, _fn in pipeline.PIPELINES["generate"]:
        if name in ("apply_confirmed", "apply_order"):
            stubbed.append((name, _fn))
        elif name == "segment_panels":
            stubbed.append((name, fake_segment))
        else:
            stubbed.append((name, fake_stage))
    monkeypatch.setitem(pipeline.PIPELINES, "generate", stubbed)


@pytest.fixture(autouse=True)
def _stub_engine(monkeypatch):
    _stub_generate_pipeline(monkeypatch)
    yield


def _wait_done(job_id, timeout=5.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        j = store.get(job_id)
        if j and j.status.value in ("completed", "failed", "cancelled"):
            return j
        time.sleep(0.05)
    raise AssertionError(f"job {job_id} never reached a terminal state")


def test_run_lifecycle_and_order():
    c = TestClient(webmain.app)
    up = c.post("/api/upload", files={"file": ("s.png", io.BytesIO(_png_bytes()),
                                                "image/png")})
    assert up.status_code == 200 and "job_id" in up.json()
    session = up.json()["job_id"]

    order = ["panel_003", "panel_001", "panel_004", "panel_002"]
    r = c.post("/api/run", json={"session": session, "order": order, "backend": "none"})
    assert r.status_code == 200
    body = r.json()
    assert body["job_id"]
    assert body["status"] in ("queued", "running")

    job = _wait_done(body["job_id"])
    assert job.status.value == "completed", job.error
    assert [p["id"] for p in job.panels] == order
    logtext = "\n".join(entry["msg"] for entry in job._logs)
    assert "1 = panel_003" in logtext
    assert job.error is None


def test_upload_rejects_corrupt_image():
    c = TestClient(webmain.app)
    r = c.post("/api/upload", files={"file": ("s.png", io.BytesIO(b"x"),
                                               "image/png")})
    assert r.status_code == 400
    assert "not a valid" in r.json()["detail"]


def test_run_rejects_malformed_session_ids():
    """Session ids are directory names under webapp_output, so a malformed
    id (or a traversal attempt) must be rejected with 400 BEFORE any
    filesystem access — not 404 after probing the disk."""
    c = TestClient(webmain.app)
    for bad in ("nope", "../../sneaky", "../", "a" * 64, "A1B2C3D4E5F6"):
        r = c.post("/api/run", json={"session": bad})
        assert r.status_code == 400, bad


def test_run_unknown_wellformed_session_is_404():
    """A WELL-FORMED id that does not exist is still 404 (the regex gate must
    not swallow the 'upload first' case)."""
    c = TestClient(webmain.app)
    r = c.post("/api/run", json={"session": "a1b2c3d4e5f6"})
    assert r.status_code == 404


def test_confirmed_review_applies(tmp_path, monkeypatch):
    """Confirmed Panel Review state must drive the pipeline order even
    though _panel_validation stores its own working order."""
    c = TestClient(webmain.app)
    up = c.post("/api/upload", files={"file": ("s.png", io.BytesIO(_png_bytes()),
                                                "image/png")})
    session = up.json()["job_id"]
    from webapp import panel_api
    from webapp.main import OUTPUT_DIR
    d = OUTPUT_DIR / session
    d.mkdir(parents=True, exist_ok=True)
    import json as _json
    (d / "panels.json").write_text(_json.dumps({
        "source": "strip.png", "width": 4, "height": 40, "plan_hash": "x",
        "config": {},
        "panels": [
            {"id": "panel_001", "panel_index": 1, "y_start": 0, "y_end": 10,
             "narration": "A", "dialogue": "", "panel_type": "single",
             "confidence": 0.9, "image_file": "panel_001.png"},
            {"id": "panel_002", "panel_index": 2, "y_start": 10, "y_end": 20,
             "narration": "B", "dialogue": "", "panel_type": "single",
             "confidence": 0.9, "image_file": "panel_002.png"},
        ]}), "utf-8")
    # user confirms a reversed order in Panel Review
    # (PNGs must exist: get_panels filters panels whose image is missing)
    (d / "panel_001.png").write_bytes(_png_bytes())
    (d / "panel_002.png").write_bytes(_png_bytes())
    panel_api.set_order(session, ["panel_002", "panel_001"])
    panel_api.confirm(session, review_all=True)

    r = c.post("/api/run", json={"session": session, "order": None,
                                 "backend": "none"})
    job = _wait_done(r.json()["job_id"])
    assert job.status.value == "completed", job.error
    assert [p["id"] for p in job.panels] == ["panel_002", "panel_001"]
    assert (d / "panels_confirmed.json").is_file()


def test_credentials_passed_to_segment_stages(monkeypatch):
    c = TestClient(webmain.app)
    up = c.post("/api/upload", files={"file": ("s.png", io.BytesIO(_png_bytes()),
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

    names = [name for name, _fn in pipeline.PIPELINES["generate"]]
    stubbed = []
    for name in names:
        if name == "validate_config":
            fn = fake_validate
        elif name == "segment_panels":
            fn = fake_segment
        elif name == "apply_order":
            fn = pipeline._apply_order
        else:
            fn = fake_validate
        stubbed.append((name, fn))
    monkeypatch.setitem(pipeline.PIPELINES, "generate", stubbed)

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
    assert job.status.value == "completed", job.error
    assert captured["validate"]["api_key"] == "secret-key"
    assert captured["validate"]["model"] == "gemini-2.5-flash"
    assert captured["segment"]["api_key"] == "secret-key"
    assert captured["segment"]["model"] == "gemini-2.5-flash"
    assert captured["segment"]["base_url"] == "https://custom.example.com"
    assert captured["segment"]["cf_account_id"] == "cf-123"


def test_auto_crop_runs_segmentation_only(monkeypatch):
    """POST /api/manual-crop/{s}/auto-crop must run ONLY the segment
    pipeline (validate/load/segment/panel_validation) — never narration
    or render — and the job kind must be segment."""
    c = TestClient(webmain.app)
    up = c.post("/api/upload", files={"file": ("s.png",
                                                io.BytesIO(_png_bytes()),
                                                "image/png")})
    assert up.status_code == 200, up.text
    session = up.json()["job_id"]

    ran: list[str] = []

    def fake_seg_stage(job, **kwargs):
        ran.append(job.stage)

    # stub the segment pipeline: light fakes, but record every stage name
    stubbed = []
    for name, _fn in pipeline.PIPELINES["segment"]:
        stubbed.append((name, fake_seg_stage))
    monkeypatch.setitem(pipeline.PIPELINES, "segment", stubbed)

    r = c.post(f"/api/manual-crop/{session}/auto-crop",
               json={"backend": "none"})
    assert r.status_code == 200, r.text
    job_id = r.json()["job_id"]
    job = _wait_done(job_id)
    assert job.status.value == "completed", job.error
    assert job.kind == "segment"
    # exactly the 4 cropping stages ran — nothing else
    assert ran == ["validate_config", "load_images",
                   "segment_panels", "panel_validation"], ran


def test_auto_crop_rejects_bad_session():
    c = TestClient(webmain.app)
    r = c.post("/api/manual-crop/zzz-not-hex/auto-crop",
               json={"backend": "none"})
    assert r.status_code == 400
    r2 = c.post("/api/manual-crop/000000000000/auto-crop",
                json={"backend": "none"})
    assert r2.status_code == 404
