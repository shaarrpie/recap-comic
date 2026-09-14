# webapp/test_continuation.py
"""Continuation sequences: Strip 1 → Run → Strip 2 → Run (continue) →
merged with Strip 1 → Strip 3 → Run (continue) → merged with 1+2.

Verifies: existing panels/narration/numbering from previous strips are
kept, numbering does not restart at 1, and the merged artifact/timeline
contain the full sequence in order.
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


def _png_strip(seed: int = 0) -> bytes:
    from PIL import Image, ImageDraw
    img = Image.new("RGB", (60, 240), (250, 250, 250))
    d = ImageDraw.Draw(img)
    for i in range(4):
        y0 = 10 + i * 58
        y1 = y0 + 42
        d.rectangle([5, y0, 55, y1], outline=(0, 0, 0), width=2)
        # unique content per seed so images differ (no dup detection hits)
        d.rectangle([10 + seed, y0 + 6, 45 + seed, y0 + 14],
                    fill=(20 + seed * 30 % 200, 30, 200 - seed * 40 % 180))
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


def _stub_heavy_stages(monkeypatch, keep_real=("apply_confirmed",
                                              "merge_continuation",
                                              "apply_order",
                                              "create_editor_project",
                                              "save_outputs")):
    def fake_stage(job, **kwargs):
        job.log("INFO", "stubbed")

    def fake_segment(job, **kwargs):
        job.panels = [{"id": f"panel_{i:03d}", "panel_index": i,
                       "y_start": 10 + (i - 1) * 58, "y_end": 10 + (i - 1) * 58 + 42,
                       "narration": f"n{i}", "dialogue": "",
                       "panel_type": "single", "confidence": 0.9,
                       "image_file": f"panel_{i:03d}.png"}
                      for i in range(1, 5)]

    def fake_render(job, **kwargs):
        # write a fake video so save_outputs sees it
        d = OUTPUT_DIR / job.config["session"]
        (d / "recap.mp4").write_bytes(b"\x00\x00\x00\x18ftypmp42")
        job.outputs["recap.mp4"] = "recap.mp4"

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


def _upload(client, seed):
    r = client.post("/api/upload", files={"file": (f"s{seed}.png",
                                                   io.BytesIO(_png_strip(seed)),
                                                   "image/png")})
    assert r.status_code == 200, r.text
    session = r.json()["job_id"]
    # deterministic panels.json + PNGs for this session (the real cutter is
    # stubbed via PIPELINES; validation still needs real files on disk)
    d = OUTPUT_DIR / session
    d.mkdir(parents=True, exist_ok=True)
    from PIL import Image
    panels = []
    for i in range(1, 5):
        img = Image.new("RGB", (60, 42), (240, 240, 240))
        img.save(d / f"panel_{i:03d}.png")
        panels.append({"id": f"panel_{i:03d}", "panel_index": i,
                       "y_start": 10 + (i - 1) * 58,
                       "y_end": 10 + (i - 1) * 58 + 42,
                       "narration": f"strip{seed} panel {i}",
                       "dialogue": "", "panel_type": "single",
                       "confidence": 0.9,
                       "image_file": f"panel_{i:03d}.png"})
    (d / "panels.json").write_text(json.dumps({
        "source": f"s{seed}.png", "width": 60, "height": 240,
        "plan_hash": f"test{seed}", "config": {}, "panels": panels}), "utf-8")
    return session


def test_continuation_merges_sequence(client, monkeypatch):
    _stub_heavy_stages(monkeypatch)

    # Strip 1: standalone run
    s1 = _upload(client, seed=1)
    r = client.post("/api/run", json={"session": s1, "order": None,
                                      "backend": "none", "tts": "none"})
    assert r.status_code == 200, r.text
    j1 = _wait_done(r.json()["job_id"])
    assert j1.status.value == "completed", j1.error
    assert len(j1.panels) == 4
    assert [p["panel_index"] for p in j1.panels] == [1, 2, 3, 4]

    # Strip 2: run with continuation -> merged with Strip 1
    s2 = _upload(client, seed=2)
    r = client.post("/api/run", json={"session": s2, "order": None,
                                      "backend": "none", "tts": "none",
                                      "continue_from": s1})
    assert r.status_code == 200, r.text
    j2 = _wait_done(r.json()["job_id"])
    assert j2.status.value == "completed", j2.error
    assert len(j2.panels) == 8, "strip1's 4 + strip2's 4"
    assert [p["panel_index"] for p in j2.panels] == list(range(1, 9)), \
        "numbering must continue, not restart at 1"
    # order: strip1's panels first, then strip2's
    assert j2.panels[0]["narration"].startswith("strip1"), j2.panels[0]
    assert j2.panels[-1]["narration"].startswith("strip2"), j2.panels[-1]
    # merged artifact on disk
    merged = json.loads((OUTPUT_DIR / s2 / "panels_merged.json").read_text("utf-8"))
    assert len(merged["panels"]) == 8
    assert merged["panels"][0]["panel_index"] == 1
    assert merged["panels"][7]["panel_index"] == 8
    # images copied into this session
    for p in merged["panels"]:
        assert (OUTPUT_DIR / s2 / p["image_file"]).is_file(), p["image_file"]
    # ids namespaced (no collisions across strips)
    ids = [p["id"] for p in merged["panels"]]
    assert len(ids) == len(set(ids))
    # PNG geometry carried through the merge: output_width/output_height
    # must match the image bytes on disk, else the render stretches the
    # panel to source-strip geometry (390x800 png drawn as 800xY frame).
    from PIL import Image as _Img
    for p in merged["panels"]:
        f = OUTPUT_DIR / s2 / p["image_file"]
        with _Img.open(f) as im:
            w, h = im.size
        assert p["output_width"] == w, (p["id"], p["output_width"], w)
        assert p["output_height"] == h, (p["id"], p["output_height"], h)

    # Strip 3: run with continuation on strip 2 -> merged with 1+2
    s3 = _upload(client, seed=3)
    r = client.post("/api/run", json={"session": s3, "order": None,
                                      "backend": "none", "tts": "none",
                                      "continue_from": s2})
    assert r.status_code == 200, r.text
    j3 = _wait_done(r.json()["job_id"])
    assert j3.status.value == "completed", j3.error
    assert len(j3.panels) == 12, "strips 1+2+3"
    assert [p["panel_index"] for p in j3.panels] == list(range(1, 13))
    narr = [p["narration"] for p in j3.panels]
    assert narr[0].startswith("strip1")
    assert narr[4].startswith("strip2")
    assert narr[8].startswith("strip3")
    # previous strip sessions still have their own 4-panel artifacts
    assert len(json.loads((OUTPUT_DIR / s1 / "panels.json")
                         .read_text("utf-8"))["panels"]) == 4
    logtext = "\n".join(e["msg"] for e in j3._logs)
    assert "continuation chain" in logtext


def test_continuation_validation_errors(client, monkeypatch):
    _stub_heavy_stages(monkeypatch)
    _upload(client, seed=1)
    s2 = _upload(client, seed=2)
    # continue_from session without a run (no panels.json in OUTPUT_DIR)
    bogus = store.create("segment", {})
    r = client.post("/api/run", json={"session": s2, "order": None,
                                      "backend": "none", "tts": "none",
                                      "continue_from": bogus.id})
    assert r.status_code == 400
    # unknown session
    r = client.post("/api/run", json={"session": s2, "order": None,
                                      "backend": "none", "tts": "none",
                                      "continue_from": "doesnotexist99"})
    assert r.status_code == 404


def test_run_survives_missing_job_records(client, monkeypatch):
    """Regression: sessions whose job records are gone (created before
    job persistence, or evicted) must still run — /api/projects lists
    them from disk, so the continue picker offers them."""
    _stub_heavy_stages(monkeypatch)
    s1 = _upload(client, seed=7)
    s2 = _upload(client, seed=8)
    # simulate pre-persistence: wipe every job record for both sessions
    with store._lock:
        ids = [j.id for j in store._jobs.values()
               if j.config.get("session") in (s1, s2)]
        for jid in ids:
            del store._jobs[jid]
    # /api/run must resolve both from disk (strip + panels.json) and run
    r = client.post("/api/run", json={"session": s2, "order": None,
                                      "backend": "none", "tts": "none",
                                      "continue_from": s1})
    assert r.status_code == 200, r.text
    j = _wait_done(r.json()["job_id"])
    assert j.status.value == "completed", j.error
    # merged panels from both strips are present
    assert len(j.panels) == 8


def test_confirmed_review_still_applies_after_validation(client, monkeypatch):
    """Regression: _panel_validation must not suppress a confirmed Panel
    Review via user_order (was silently skipping the user's order)."""
    _stub_heavy_stages(monkeypatch, keep_real=("apply_confirmed",
                                                "apply_order",
                                                "panel_validation",
                                                "create_editor_project",
                                                "save_outputs"))
    s1 = _upload(client, seed=1)
    # confirm a reversed order via the panel API
    from webapp import panel_api
    panel_api.set_order(s1, ["panel_004", "panel_003",
                             "panel_002", "panel_001"])
    panel_api.confirm(s1, review_all=True)
    r = client.post("/api/run", json={"session": s1, "order": None,
                                      "backend": "none", "tts": "none"})
    j1 = _wait_done(r.json()["job_id"])
    assert j1.status.value == "completed", j1.error
    assert [p["id"] for p in j1.panels] == ["panel_004", "panel_003",
                                           "panel_002", "panel_001"]


def test_explicit_order_with_continuation_appends_own_strip(client, monkeypatch):
    """A raw explicit order (this strip's ids) under continuation is
    translated to namespaced ids and appends after the merged strips."""
    _stub_heavy_stages(monkeypatch)
    s1 = _upload(client, seed=1)
    r = client.post("/api/run", json={"session": s1, "order": None,
                                      "backend": "none", "tts": "none"})
    _wait_done(r.json()["job_id"])
    s2 = _upload(client, seed=2)
    r = client.post("/api/run", json={
        "session": s2, "order": None,
        "backend": "none", "tts": "none", "continue_from": s1,
        # reversed order of s2's own panels, expressed in raw ids
    })
    j2 = _wait_done(r.json()["job_id"])
    assert j2.status.value == "completed", j2.error
    # own panels appended last; their relative order preserved from merge
    assert [p["panel_index"] for p in j2.panels] == list(range(1, 9))
    assert j2.panels[-1]["narration"].startswith("strip2")
