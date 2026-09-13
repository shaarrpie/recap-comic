# webapp/test_review_chain.py
"""Panel Review across continuation chains.

Scenario from the user: run 002_1.png (panels appear in review), then run
003_1.png — its panels must be ADDED after the existing ones; the 002_1
panels must remain available, each strip keeping its own state.
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


@pytest.fixture()
def client():
    return TestClient(webmain.app)


def _wait_done(job_id, timeout=90):
    t0 = time.time()
    while time.time() - t0 < timeout:
        j = store.get(job_id)
        if j and j.status.value in ("completed", "failed", "cancelled"):
            return j
        time.sleep(0.2)
    raise AssertionError("job never reached terminal state")


def _make_session(client, name, n_panels=3):
    """Upload + write a panels.json with deterministic panels."""
    from PIL import Image
    r = client.post("/api/upload", files={"file": (name, io.BytesIO(b"x" * 8),
                                                   "image/png")})
    # corrupt upload rejected — use a real PNG
    buf = io.BytesIO()
    Image.new("RGB", (40, 20), (255, 255, 255)).save(buf, "PNG")
    r = client.post("/api/upload", files={"file": (name, buf, "image/png")})
    assert r.status_code == 200, r.text
    session = r.json()["job_id"]
    d = OUTPUT_DIR / session
    d.mkdir(parents=True, exist_ok=True)
    panels = []
    for i in range(1, n_panels + 1):
        Image.new("RGB", (40, 20), (250, 250, 250)).save(d / f"panel_{i:03d}.png")
        panels.append({"id": f"panel_{i:03d}", "panel_index": i,
                       "y_start": (i - 1) * 20, "y_end": i * 20,
                       "narration": f"{name} p{i}", "dialogue": "",
                       "panel_type": "single", "confidence": 0.9,
                       "image_file": f"panel_{i:03d}.png"})
    (d / "panels.json").write_text(json.dumps({
        "source": name, "width": 40, "height": n_panels * 20,
        "plan_hash": "t", "config": {}, "panels": panels}), "utf-8")
    return session


def _stub_heavy(monkeypatch, keep_real=("apply_confirmed", "merge_continuation",
                                        "apply_order", "save_outputs",
                                        "create_editor_project")):
    def fake_stage(job, **kwargs):
        job.log("INFO", "stubbed")

    def fake_segment(job, **kwargs):
        job.panels = [{"id": f"panel_{i:03d}", "panel_index": i,
                       "y_start": (i - 1) * 20, "y_end": i * 20,
                       "narration": "n", "dialogue": "",
                       "panel_type": "single", "confidence": 0.9,
                       "image_file": f"panel_{i:03d}.png"}
                      for i in range(1, 4)]

    def fake_render(job, **kwargs):
        d = OUTPUT_DIR / job.config["session"]
        (d / "recap.mp4").write_bytes(b"\x00\x00\x00\x18ftypmp42")

    stubbed = []
    for name, fn in pipeline.PIPELINES["generate"]:
        if name in keep_real:
            stubbed.append((name, fn))
        elif name == "segment_panels":
            stubbed.append((name, fake_segment))
        elif name == "render_video":
            stubbed.append((name, fake_render))
        else:
            stubbed.append((name, fake_stage))
    monkeypatch.setitem(pipeline.PIPELINES, "generate", stubbed)


def test_combined_review_adds_panels_not_replaces(client, monkeypatch):
    """002_1 run -> panels visible; 003_1 run continued -> combined review
    shows 002's panels STILL THERE followed by 003's panels."""
    _stub_heavy(monkeypatch)
    s1 = _make_session(client, "002_1.png")
    r = client.post("/api/run", json={"session": s1, "order": None,
                                      "backend": "none", "tts": "none"})
    j1 = _wait_done(r.json()["job_id"])
    assert j1.status.value == "completed", j1.error

    # Panel Review for strip 1 (combined view): only its 3 panels
    g = client.get(f"/api/panels/{s1}?combined=1").json()
    assert g["active"] == 3
    assert [p["narration"] for p in g["panels"] if not p["deleted"]] == \
        ["002_1.png p1", "002_1.png p2", "002_1.png p3"]

    s2 = _make_session(client, "003_1.png")
    r = client.post("/api/run", json={"session": s2, "order": None,
                                      "backend": "none", "tts": "none",
                                      "continue_from": s1})
    j2 = _wait_done(r.json()["job_id"])
    assert j2.status.value == "completed", j2.error

    # Combined review on strip 2: strip1's panels THEN strip2's panels
    g = client.get(f"/api/panels/{s2}?combined=1").json()
    assert g["active"] == 6, "3 + 3, not replaced"
    narrs = [p["narration"] for p in g["panels"] if not p["deleted"]]
    assert narrs == ["002_1.png p1", "002_1.png p2", "002_1.png p3",
                     "003_1.png p1", "003_1.png p2", "003_1.png p3"]
    # previous strip's review still intact from its own session
    g1 = client.get(f"/api/panels/{s1}?combined=1").json()
    assert g1["active"] == 3
    # per-strip state reported
    assert len(g["strips"]) == 2
    assert all(s["active"] == 3 for s in g["strips"])


def test_namespaced_ops_route_to_owner(client, monkeypatch):
    """Delete/review/reorder/confirm via namespaced ids touch ONLY the
    owning strip's edit layer; the other strip's state is untouched."""
    _stub_heavy(monkeypatch)
    s1 = _make_session(client, "002_1.png")
    r = client.post("/api/run", json={"session": s1, "order": None,
                                      "backend": "none", "tts": "none"})
    _wait_done(r.json()["job_id"])
    s2 = _make_session(client, "003_1.png")
    r = client.post("/api/run", json={"session": s2, "order": None,
                                      "backend": "none", "tts": "none",
                                      "continue_from": s1})
    _wait_done(r.json()["job_id"])

    g = client.get(f"/api/panels/{s2}?combined=1").json()
    by_narr = {p["narration"]: p for p in g["panels"]}
    victim = by_narr["002_1.png p2"]

    # delete a strip-1 panel via strip-2's combined review
    r = client.post(f"/api/panels/{s2}/delete", json={"ids": [victim["id"]]})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["active"] == 5

    # strip 1's own edit layer got the deletion; strip 2's did NOT
    e1 = json.loads((OUTPUT_DIR / s1 / "panels_edit.json").read_text("utf-8"))
    raw_id = victim["id"].split("_", 1)[1]
    assert raw_id in e1["deleted"]
    e2_path = OUTPUT_DIR / s2 / "panels_edit.json"
    if e2_path.is_file():
        e2 = json.loads(e2_path.read_text("utf-8"))
        assert not e2.get("deleted")

    # review another strip-1 panel (namespaced id)
    target = by_narr["002_1.png p1"]
    r = client.post(f"/api/panels/{s2}/review",
                    json={"panel_id": target["id"], "status": "reviewed"})
    assert r.status_code == 200
    e1 = json.loads((OUTPUT_DIR / s1 / "panels_edit.json").read_text("utf-8"))
    assert e1["review"].get(target["id"].split("_", 1)[1]) == "reviewed"

    # reorder strip-2's own panel via namespaced id (active-only validation)
    g = client.get(f"/api/panels/{s2}?combined=1").json()
    active_ids = [p["id"] for p in g["panels"] if not p["deleted"]]
    reordered = [active_ids[0], *active_ids[2:], active_ids[1]]
    r = client.post(f"/api/panels/{s2}/order", json={"ids": reordered})
    assert r.status_code == 200, r.text

    # restore the deleted strip-1 panel
    r = client.post(f"/api/panels/{s2}/restore", json={"ids": [victim["id"]]})
    assert r.status_code == 200
    assert r.json()["active"] == 6

    # confirm from the combined view locks BOTH strips
    r = client.post(f"/api/panels/{s2}/confirm", json={"review_all": True})
    assert r.status_code == 200
    for s in (s1, s2):
        e = json.loads((OUTPUT_DIR / s / "panels_edit.json").read_text("utf-8"))
        assert e["confirmed"] is True


def test_single_strip_view_unchanged(client, monkeypatch):
    """Without continuation, the combined view equals the single view (no
    regression for standalone strips)."""
    _stub_heavy(monkeypatch, keep_real=("apply_confirmed", "apply_order",
                                        "save_outputs",
                                        "create_editor_project"))
    s = _make_session(client, "solo.png")
    g = client.get(f"/api/panels/{s}?combined=1").json()
    assert g["active"] == 3
    assert g["strips"][0]["session"] == s
    # no source_session mismatch; ids namespaced with own session
    assert all(p["source_session"] == s for p in g["panels"])


def test_queue_run_continuation_live(client, monkeypatch):
    """The run-queued flow: second strip's run continues the first (the
    frontend picks the most recent session with panels automatically)."""
    _stub_heavy(monkeypatch)
    s1 = _make_session(client, "002_1.png")
    r = client.post("/api/run", json={"session": s1, "order": None,
                                      "backend": "none", "tts": "none"})
    _wait_done(r.json()["job_id"])
    # projects list now includes s1 with panels — the frontend picks the
    # most recent session WITH panels as continue_from; scope the check to
    # sessions created by THIS test (the workspace has many old ones)
    proj = client.get("/api/projects").json()["projects"]
    mine = [p for p in proj if p["id"] == s1]
    assert mine and mine[0]["panels"] > 0
    s2 = _make_session(client, "003_1.png")
    r = client.post("/api/run", json={"session": s2, "order": None,
                                      "backend": "none", "tts": "none",
                                      "continue_from": s1})
    j2 = _wait_done(r.json()["job_id"])
    assert j2.status.value == "completed", j2.error
    assert len(j2.panels) == 6
    # merged artifact preserves strip1's panels first
    merged = json.loads((OUTPUT_DIR / s2 / "panels_merged.json")
                        .read_text("utf-8"))
    assert [p["narration"] for p in merged["panels"]][:3] == \
        ["002_1.png p1", "002_1.png p2", "002_1.png p3"]
