# webapp/test_pipeline_steps.py
"""Step-by-Step pipeline mode: checkpoints, validation, staleness, retry.

Covers the checkpoint architecture (webapp/checkpoint.py + the step
orchestration in webapp/pipeline.py):

1. Running ONE step produces a checkpoint ledger entry + validates the
   artifact (a stage that "finishes" with broken output must NOT be
   marked success).
2. Retry re-runs ONLY the failed step (no re-run of earlier stages).
3. Editing an artifact invalidates only dependent steps.
4. Reset-from marks the chosen step and later ones stale, keeping
   earlier checkpoints.
5. The debugger event log is structured and persisted.
"""
from __future__ import annotations

import io
import json
import time

import pytest
from fastapi.testclient import TestClient

from webapp import main as webmain
from webapp import pipeline
from webapp.jobs import store
from webapp.main import OUTPUT_DIR


def _png_strip() -> bytes:
    from PIL import Image, ImageDraw
    img = Image.new("RGB", (60, 120), (250, 250, 250))
    d = ImageDraw.Draw(img)
    d.rectangle([5, 8, 55, 50], outline=(0, 0, 0), width=2)
    d.rectangle([5, 68, 55, 112], outline=(0, 0, 0), width=2)
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()


def _wait_done(job_id, timeout=90):
    t0 = time.time()
    while time.time() - t0 < timeout:
        j = store.get(job_id)
        if j and j.status.value in ("completed", "failed", "cancelled"):
            return j
        time.sleep(0.2)
    raise AssertionError("job never reached terminal state")


@pytest.fixture()
def client():
    return TestClient(webmain.app)


@pytest.fixture()
def session(client):
    up = client.post("/api/upload", files={"file": ("s.png",
                                                    io.BytesIO(_png_strip()),
                                                    "image/png")})
    assert up.status_code == 200, up.text
    return up.json()["job_id"]


# ------------------------------------------------------------------ #
# Ledger + validation unit behavior (webapp.checkpoint)
# ------------------------------------------------------------------ #
def test_step_sequence_matches_pipeline():
    """Every user-facing step maps to a real generate-pipeline stage."""
    from webapp.checkpoint import STEP_SEQUENCE
    names = {n for n, _fn in pipeline.PIPELINES["generate"]}
    for s in STEP_SEQUENCE:
        assert s["pipeline_name"] in names, \
            f"{s['name']} ({s['pipeline_name']}) not in generate"


def test_validate_artifacts_missing_outputs(session):
    from webapp import checkpoint as cp
    # step 3 (segment) with no panels.json -> problems
    problems, _soft = cp.validate_artifacts(session, 3)
    assert any("panels.json" in p for p in problems)


def test_validate_artifacts_ok(session, client):
    from webapp import checkpoint as cp
    d = OUTPUT_DIR / session
    # fake a completed segmentation
    panels = {"panels": [{"id": "p1", "image_file": "panel_p1.png",
                          "y_start": 0, "y_end": 50}]}
    (d / "panels.json").write_text(json.dumps(panels), "utf-8")
    (d / "panel_p1.png").write_bytes(b"\x89PNG")   # exists = enough
    problems, _soft = cp.validate_artifacts(session, 3)
    assert problems == []


# ------------------------------------------------------------------ #
# Step execution via the API
# ------------------------------------------------------------------ #
def test_run_single_step_and_checkpoint(client, session):
    r = client.post(f"/api/pipeline/{session}/run-step",
                    json={"session": session, "stage": "validate_config",
                          "backend": "none"})
    assert r.status_code == 200, r.text
    job_id = r.json()["job_id"]
    job = _wait_done(job_id)
    assert job.status.value == "completed", job.error
    # checkpoint ledger: step 1 success, current now 2
    st = client.get(f"/api/pipeline/{session}").json()
    entry = next(e for e in st["state"]["steps"] if e["step"] == 1)
    assert entry["status"] == "success"
    assert st["state"]["completed_steps"] == [1]
    assert st["state"]["current_step"] == 2
    assert st["state"]["status"] == "paused"


def test_failed_step_recorded_with_debug_event(client, session, monkeypatch):
    # break segmentation on purpose
    def boom(job, **kwargs):
        raise RuntimeError("narration exploded")
    monkeypatch.setitem(pipeline.PIPELINES, "generate",
                        [("segment_panels", boom)] +
                        [(n, fn) for n, fn in
                         pipeline.PIPELINES["generate"]][1:])
    r = client.post(f"/api/pipeline/{session}/run-step",
                    json={"session": session, "stage": "segment_panels",
                          "backend": "none"})
    assert r.status_code == 200
    job = _wait_done(r.json()["job_id"])
    assert job.status.value == "failed"
    st = client.get(f"/api/pipeline/{session}").json()
    entry = next(e for e in st["state"]["steps"] if e["step"] == 3)
    assert entry["status"] == "failed"
    assert "narration exploded" in (entry["error"] or "")
    # debugger event exists with traceback
    evs = client.get(f"/api/pipeline/{session}/events?step=3").json()["events"]
    assert any(e["event_type"] == "step_error" and e.get("traceback")
               for e in evs)
    # retry target stays at step 3
    assert st["state"]["current_step"] == 3


def test_retry_only_failed_step(client, session, monkeypatch):
    """Retry re-runs the failed step WITHOUT re-running earlier ones."""
    calls: list[str] = []
    orig = pipeline.PIPELINES["generate"]
    wrapped = []
    for n, fn in orig:
        def mk(n=n, fn=fn):
            def w(job, **kw):
                calls.append(n)
                return fn(job, **kw)
            return w
        wrapped.append((n, mk()))
    monkeypatch.setitem(pipeline.PIPELINES, "generate", wrapped)
    # step 1 succeeds
    r = client.post(f"/api/pipeline/{session}/run-step",
                    json={"session": session, "stage": "validate_config",
                          "backend": "none"})
    _wait_done(r.json()["job_id"])
    # step 2 succeeds
    r = client.post(f"/api/pipeline/{session}/run-step",
                    json={"session": session, "stage": "load_images"})
    _wait_done(r.json()["job_id"])
    calls.clear()
    # retry step 2: only load_images runs again
    r = client.post(f"/api/pipeline/{session}/run-step",
                    json={"session": session, "stage": "load_images"})
    _wait_done(r.json()["job_id"])
    assert calls == ["load_images"], calls


def test_run_until_stops_at_target(client, session, monkeypatch):
    calls: list[str] = []
    orig = pipeline.PIPELINES["generate"]
    wrapped = []
    for n, fn in orig:
        def mk(n=n, fn=fn):
            def w(job, **kw):
                calls.append(n)
                return fn(job, **kw)
            return w
        wrapped.append((n, mk()))
    monkeypatch.setitem(pipeline.PIPELINES, "generate", wrapped)
    r = client.post(f"/api/pipeline/{session}/run-until",
                    json={"session": session, "from_stage": "validate_config",
                          "until_stage": "load_images", "backend": "none"})
    assert r.status_code == 200, r.text
    job = _wait_done(r.json()["job_id"])
    assert job.status.value == "completed"
    assert set(calls) <= {"validate_config", "load_images"}
    # ledger paused at apply-review (step 3 segment not run yet)
    st = client.get(f"/api/pipeline/{session}").json()
    assert st["state"]["current_step"] == 3


def test_edit_invalidation_and_reset_from(client, session):
    from webapp import checkpoint as cp
    # simulate steps 1-7 completed
    for step in range(1, 8):
        cp.record_step(session, step, status="success")
    # narration edit invalidates build_script (8) and later
    st = cp.apply_edit_invalidation(session, "narration_edit.json")
    entry7 = next(e for e in st["steps"] if e["step"] == 7)
    assert entry7["status"] == "success"     # gemini_narration survives
    assert st["current_step"] == 8           # resume at build_script
    # panel-review edit invalidates apply_review (5) and later
    st = cp.apply_edit_invalidation(session, "panels_edit.json")
    entry4 = next(e for e in st["steps"] if e["step"] == 4)
    assert entry4["status"] == "success"     # validation survives
    entry5 = next(e for e in st["steps"] if e["step"] == 5)
    assert entry5["status"] == "stale"
    assert st["completed_steps"] == [1, 2, 3, 4]
    assert st["current_step"] == 5
    # reset from apply_order (6) via the API
    st2 = client.post(f"/api/pipeline/{session}/reset-from",
                      json={"stage": "apply_order"}).json()["state"]
    done = {e["step"]: e["status"] for e in st2["steps"]}
    assert done[5] == "stale"
    assert done[6] == "stale"
    assert st2["completed_steps"] == [1, 2, 3, 4]
    assert st2["current_step"] == 6


def test_events_persist_and_rotate(session):
    from webapp import checkpoint as cp
    cp.log_event(session, step=1, event_type="test", message="hello",
                 severity="DEBUG")
    evs = cp.load_events(session)
    assert any(e["message"] == "hello" for e in evs)
    # structured fields present
    ev = next(e for e in evs if e["message"] == "hello")
    for key in ("t", "step", "event_type", "severity", "message"):
        assert key in ev


def test_unknown_stage_rejected(client, session):
    r = client.post(f"/api/pipeline/{session}/run-step",
                    json={"session": session, "stage": "nope"})
    assert r.status_code == 400


def test_bad_session_rejected(client):
    r = client.get("/api/pipeline/../etc")
    assert r.status_code in (400, 404)
    r2 = client.get("/api/pipeline/zzz-not-hex")
    assert r2.status_code == 400
