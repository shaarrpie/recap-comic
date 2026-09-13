# webapp/test_cinematic.py
"""Cinematic Studio API + semi-auto step routing tests."""
from __future__ import annotations

import io
import json

import pytest
from fastapi.testclient import TestClient

from webapp import main as webmain
from webapp import pipeline
from webapp.jobs import store


@pytest.fixture()
def client():
    return TestClient(webmain.app)


def _png_bytes() -> bytes:
    from PIL import Image
    buf = io.BytesIO()
    Image.new("RGB", (40, 20), (255, 255, 255)).save(buf, "PNG")
    return buf.getvalue()


def _wait_done(job_id, timeout=5.0):
    import time
    t0 = time.time()
    while time.time() - t0 < timeout:
        j = store.get(job_id)
        if j and j.status.value in ("completed", "failed", "cancelled"):
            return j
        time.sleep(0.05)
    raise AssertionError(f"job {job_id} never finished")


def _upload(client) -> str:
    r = client.post("/api/upload", files={
        "file": ("s.png", io.BytesIO(_png_bytes()), "image/png")})
    assert r.status_code == 200, r.text
    return r.json()["job_id"]


def _fake_panels(client, session: str) -> None:
    from webapp.main import OUTPUT_DIR
    d = OUTPUT_DIR / session
    d.mkdir(parents=True, exist_ok=True)
    (d / "panels.json").write_text(json.dumps({
        "source": "s.png", "width": 40, "height": 20,
        "plan_hash": "x", "config": {},
        "panels": [{"id": "panel_001", "panel_index": 1,
                    "y_start": 0, "y_end": 20, "narration": "boom",
                    "dialogue": "", "panel_type": "single",
                    "confidence": 0.9,
                    "image_file": "panel_001.png"}]}), "utf-8")


def test_cinematic_config_crud(client):  # noqa: ANN001
    s = _upload(client)
    _fake_panels(client, s)
    r = client.get(f"/api/cinematic/{s}/config")
    assert r.status_code == 200
    assert r.json()["style"] == "dynamic"

    r = client.post(f"/api/cinematic/{s}/config", json={
        "style": "subtle", "letterbox": True, "color_preset": "cold",
        "bgm_volume": 0.5})
    assert r.status_code == 200, r.text
    got = client.get(f"/api/cinematic/{s}/config").json()
    assert got["style"] == "subtle"
    assert got["letterbox"] is True
    assert got["color_preset"] == "cold"
    assert got["bgm_volume"] == 0.5

    # invalid values rejected
    r = client.post(f"/api/cinematic/{s}/config", json={
        "style": "turbo", "color_preset": "neon"})
    assert r.status_code == 400


def test_cinematic_session_validation(client):  # noqa: ANN001
    # The validation itself must reject traversal-style ids before any
    # disk lookup. (HTTP clients/Starlette may normalize `..` paths into
    # different routes before our handler runs, so we call the session
    # dir resolver directly — that is the security boundary.)
    from fastapi import HTTPException

    from webapp import cinematic_api as cin
    for bad in ("..", "../x", "deadbeef", "ZZZZZZZZZZZZ",
                "s../../etc", "aaaaaaaaaaaa"):
        # 12-hex "aaaaaaaaaaaa" is VALID format but nonexistent -> 404;
        # everything else must be 400 before touching the disk
        try:
            cin._session_dir(bad)
        except HTTPException as e:
            assert e.status_code in (400, 404), bad
        else:
            raise AssertionError(f"id {bad!r} passed validation")
    # and the routes built on it inherit the guard
    r = client.get("/api/cinematic/deadbeef/config")
    assert r.status_code == 400
    r = client.get("/api/cinematic/aaaaaaaaaaaa/config")
    assert r.status_code == 404


def test_panel_override_roundtrip(client):  # noqa: ANN001
    s = _upload(client)
    _fake_panels(client, s)
    r = client.post(f"/api/cinematic/{s}/panel-override", json={
        "panel_id": "panel_001",
        "overrides": {"effect_type": "punch_zoom", "zoom_peak": 1.3}})
    assert r.status_code == 200, r.text
    cfg = client.get(f"/api/cinematic/{s}/config").json()
    assert cfg["panel_overrides"]["panel_001"]["zoom_peak"] == 1.3
    # clear
    r = client.post(f"/api/cinematic/{s}/panel-override", json={
        "panel_id": "panel_001", "overrides": None})
    assert r.status_code == 200
    assert "panel_001" not in client.get(
        f"/api/cinematic/{s}/config").json()["panel_overrides"]


def test_effects_manifest(client):  # noqa: ANN001
    r = client.get("/api/cinematic/effects")
    assert r.status_code == 200
    man = r.json()
    ids = [e["id"] for e in man["effects"]]
    assert "punch_zoom" in ids and "color_grade" in ids
    assert "manhwa" in man["color_presets"] and "bw" in man["color_presets"]


def test_stages_checklist(client):  # noqa: ANN001
    s = _upload(client)
    _fake_panels(client, s)
    r = client.get(f"/api/pipeline/{s}/stages")
    assert r.status_code == 200, r.text
    st = r.json()["stages"]
    assert st["segment_panels"] is True
    assert st["render_video"] is False       # no recap.mp4 yet
    assert st["cinematic_render"] is False
    # bad session ids never produce a 200 (400 from validation, or 404
    # from the router when the path is unroutable)
    r = client.get("/api/pipeline/01234567890/stages")
    assert r.status_code in (400, 404)


def test_step_endpoint_routes_through_step_worker(client, monkeypatch):  # noqa: ANN001
    """POST /api/pipeline/step must use run_steps_job (checkpointed), NOT
    run_job (whose pipeline_step kind has an empty stage list)."""
    s = _upload(client)
    _fake_panels(client, s)
    calls = {}
    real_steps = pipeline.run_steps_job

    def spy_steps(job_id, first, last=None, **kw):
        calls["steps"] = (first, last)
        return real_steps(job_id, first, last, **kw)

    def spy_job(job_id, **kw):  # pragma: no cover - must not be called
        calls["job"] = job_id
        return real_steps(job_id, "segment_panels", "segment_panels", **kw)

    monkeypatch.setattr(pipeline, "run_steps_job", spy_steps)
    monkeypatch.setattr(pipeline, "run_job", spy_job)

    r = client.post("/api/pipeline/step", json={
        "session": s, "stage": "segment_panels", "backend": "none"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["stage"] == "segment_panels"
    assert "steps" in calls and "job" not in calls
    # worker thread may still be starting; wait for terminal state
    j = _wait_done(body["job_id"])
    assert j.status.value in ("completed", "failed"), j.error

    # legacy tts_audio alias maps to render_video, not a silent no-op
    r = client.post("/api/pipeline/step", json={
        "session": s, "stage": "tts_audio", "backend": "none"})
    assert r.status_code == 200, r.text
    assert r.json()["stage"] == "render_video"

    # unknown stage 400s
    r = client.post("/api/pipeline/step", json={
        "session": s, "stage": "warp_speed"})
    assert r.status_code == 400

    # invalid session 400s/404s before any worker starts
    r = client.post("/api/pipeline/step", json={
        "session": "..", "stage": "segment_panels"})
    assert r.status_code in (400, 404)


def test_editor_pan_and_speed(client):  # noqa: ANN001
    from webapp.editor_api import create_project_from_generation
    from webapp.main import OUTPUT_DIR
    s = _upload(client)
    _fake_panels(client, s)
    from PIL import Image
    d = OUTPUT_DIR / s
    Image.new("RGB", (40, 20), (240, 240, 240)).save(d / "panel_001.png")
    # minimal narration + timeline so create_project works
    (d / "narration.json").write_text(json.dumps({
        "meta": {"config_hash": "x", "input_hashes": {}},
        "mode": "recap", "voice": "none", "entries": []}), "utf-8")
    create_project_from_generation(s)

    r = client.post(f"/api/editor/{s}/pan", json={
        "panel_id": "panel_001", "direction": "pan_down",
        "zoom_factor": 1.2, "speed": 1.5})
    assert r.status_code == 200, r.text
    pan = r.json()["pan"]
    assert pan["kind"] == "pan_down"
    # invalid direction 400s
    r = client.post(f"/api/editor/{s}/pan", json={
        "panel_id": "panel_001", "direction": "teleport"})
    assert r.status_code == 400

    r = client.post(f"/api/editor/{s}/speed", json={
        "panel_id": "panel_001", "speed": 2.0, "freeze": False})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["speed"] == 2.0
    # bad type 400s, never 500
    r = client.post(f"/api/editor/{s}/speed", json={
        "panel_id": "panel_001", "speed": "fast"})
    assert r.status_code == 400
