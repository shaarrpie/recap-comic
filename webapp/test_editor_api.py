# webapp/test_editor_api.py
import io

import pytest
from fastapi.testclient import TestClient

from webapp import main as webmain
from webapp import pipeline
from webapp.jobs import store


@pytest.fixture(autouse=True)
def _stub_engine():
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

    original = list(pipeline.PIPELINES["generate"])
    original[0] = ("validate_config", fake_validate)
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
        original[i] = (name, fn)
    pipeline.PIPELINES["generate"] = original
    yield


def _wait_done(job_id, timeout=5.0):
    import time
    t0 = time.time()
    while time.time() - t0 < timeout:
        j = store.get(job_id)
        if j and j.status.value in ("completed", "failed", "cancelled"):
            return j
        time.sleep(0.05)
    raise AssertionError(f"job {job_id} never reached a terminal state")


def test_editor_round_trip(client=None):
    c = TestClient(webmain.app)
    up = c.post("/api/upload", files={"file": ("s.png", io.BytesIO(b"x"),
                                               "image/png")})
    session = up.json()["job_id"]

    # Create minimal automation outputs BEFORE running generation so editor
    # project can be created in the _create_editor_project stage.
    import json
    from pathlib import Path
    from webapp.main import OUTPUT_DIR
    d = OUTPUT_DIR / session
    d.mkdir(parents=True, exist_ok=True)
    (d / "panels.json").write_text(json.dumps({
        "source": "s.png", "width": 800, "height": 4000,
        "plan_hash": "x", "config": {},
        "panels": [
            {"id": "panel_001", "panel_index": 1, "y_start": 0, "y_end": 800,
             "narration": "A", "dialogue": "", "panel_type": "single",
             "confidence": 0.9, "image_file": "panel_001.png",
             "split_of": None, "merged_with": [], "snap_distances": [0]},
            {"id": "panel_002", "panel_index": 2, "y_start": 800, "y_end": 1600,
             "narration": "B", "dialogue": "", "panel_type": "single",
             "confidence": 0.9, "image_file": "panel_002.png",
             "split_of": None, "merged_with": [], "snap_distances": [0]},
            {"id": "panel_003", "panel_index": 3, "y_start": 1600, "y_end": 2400,
             "narration": "C", "dialogue": "", "panel_type": "single",
             "confidence": 0.9, "image_file": "panel_003.png",
             "split_of": None, "merged_with": [], "snap_distances": [0]},
            {"id": "panel_004", "panel_index": 4, "y_start": 2400, "y_end": 3200,
             "narration": "D", "dialogue": "", "panel_type": "single",
             "confidence": 0.9, "image_file": "panel_004.png",
             "split_of": None, "merged_with": [], "snap_distances": [0]},
        ]
    }), "utf-8")
    (d / "timeline.json").write_text(json.dumps({
        "meta": {"schema_version": 1, "generator": "test", "config_hash": "x", "input_hashes": {}},
        "width": 1080, "height": 1920, "fps": 30,
        "gap_seconds": 0.35, "min_display_seconds": 2.0,
        "entries": [
            {"panel_id": "panel_001", "order": 1, "source_image": str(d / "panel_001.png"),
             "bbox": {"x": 0, "y": 0, "w": 800, "h": 800},
             "start_seconds": 0.0, "duration_seconds": 3.0,
             "audio_path": None, "pan": {"kind": "static", "scaled_w": 1080, "scaled_h": 1920, "travel_px": 0}},
            {"panel_id": "panel_002", "order": 2, "source_image": str(d / "panel_002.png"),
             "bbox": {"x": 0, "y": 800, "w": 800, "h": 800},
             "start_seconds": 3.0, "duration_seconds": 3.5,
             "audio_path": None, "pan": {"kind": "static", "scaled_w": 1080, "scaled_h": 1920, "travel_px": 0}},
            {"panel_id": "panel_003", "order": 3, "source_image": str(d / "panel_003.png"),
             "bbox": {"x": 0, "y": 1600, "w": 800, "h": 800},
             "start_seconds": 6.5, "duration_seconds": 4.0,
             "audio_path": None, "pan": {"kind": "static", "scaled_w": 1080, "scaled_h": 1920, "travel_px": 0}},
            {"panel_id": "panel_004", "order": 4, "source_image": str(d / "panel_004.png"),
             "bbox": {"x": 0, "y": 2400, "w": 800, "h": 800},
             "start_seconds": 10.5, "duration_seconds": 3.0,
             "audio_path": None, "pan": {"kind": "static", "scaled_w": 1080, "scaled_h": 1920, "travel_px": 0}},
        ]
    }), "utf-8")
    (d / "recap.srt").write_text("1\n00:00:00,000 --> 00:00:03,000\nA\n\n2\n00:00:03,000 --> 00:00:06,500\nB\n\n3\n00:00:06,500 --> 00:00:10,500\nC\n\n4\n00:00:10,500 --> 00:00:13,500\nD\n\n", "utf-8")
    for i in range(1, 5):
        (d / f"panel_{i:03d}.png").write_bytes(b"x")

    r = c.post("/api/run", json={"session": session, "order": None})
    job = _wait_done(r.json()["job_id"])
    assert job.status.value == "completed"

    # load editor project
    proj = c.get(f"/api/editor/{session}")
    assert proj.status_code == 200
    data = proj.json()
    assert len(data["edited_timeline"]) == 4
    assert data["history_index"] == -1

    # reorder
    new_order = ["panel_003", "panel_001", "panel_004", "panel_002"]
    r = c.post(f"/api/editor/{session}/reorder",
               json={"order": new_order})
    assert r.status_code == 200
    data = r.json()
    assert [e["panel_id"] for e in data["edited_timeline"]] == new_order
    assert data["history_index"] == 0

    # undo
    r = c.post(f"/api/editor/{session}/undo")
    assert r.status_code == 200
    data = r.json()
    assert [e["panel_id"] for e in data["edited_timeline"]] == ["panel_001", "panel_002", "panel_003", "panel_004"]
    assert data["history_index"] == -1

    # redo
    r = c.post(f"/api/editor/{session}/redo")
    assert r.status_code == 200
    data = r.json()
    assert [e["panel_id"] for e in data["edited_timeline"]] == new_order

    # duration change
    r = c.post(f"/api/editor/{session}/duration",
               json={"panel_id": "panel_001", "duration": 5.5})
    assert r.status_code == 200
    data = r.json()
    entry = next(e for e in data["edited_timeline"] if e["panel_id"] == "panel_001")
    assert entry["duration_seconds"] == 5.5

    # effect change
    r = c.post(f"/api/editor/{session}/effect",
               json={"panel_id": "panel_002", "kind": "zoom_in", "duration": 5.5})
    assert r.status_code == 200
    data = r.json()
    fx = next(e for e in data["effects"] if e["panel_id"] == "panel_002")
    assert fx["kind"] == "zoom_in"

    # caption update
    cap_id = data["captions"][0]["id"]
    r = c.post(f"/api/editor/{session}/caption",
               json={"id": cap_id, "text": "new caption", "start_seconds": 0.0, "end_seconds": 2.0})
    assert r.status_code == 200
    data = r.json()
    cap = next(c for c in data["captions"] if c["id"] == cap_id)
    assert cap["text"] == "new caption"

    # transition
    r = c.post(f"/api/editor/{session}/transition",
               json={"from_panel_id": "panel_001", "to_panel_id": "panel_002",
                     "type": "fade", "duration": 0.8})
    assert r.status_code == 200
    data = r.json()
    tr = next(t for t in data["transitions"]
              if t["from_panel_id"] == "panel_001" and t["to_panel_id"] == "panel_002")
    assert tr["type"] == "fade"

    # remove panel
    r = c.post(f"/api/editor/{session}/panel/remove",
               json={"panel_id": "panel_004"})
    assert r.status_code == 200
    data = r.json()
    assert len(data["edited_timeline"]) == 3
    assert not any(c["panel_id"] == "panel_004" for c in data["captions"])

    # add panel
    r = c.post(f"/api/editor/{session}/panel/add",
               json={"panel_id": "panel_004", "after": "panel_002"})
    assert r.status_code == 200
    data = r.json()
    assert len(data["edited_timeline"]) == 4

    # reset
    r = c.post(f"/api/editor/{session}/reset")
    assert r.status_code == 200
    data = r.json()
    assert [e["panel_id"] for e in data["edited_timeline"]] == ["panel_001", "panel_002", "panel_003", "panel_004"]
    assert data["history_index"] == -1

    # save
    r = c.post(f"/api/editor/{session}", json=data)
    assert r.status_code == 200
    assert r.json()["ok"] is True

    # reload
    r = c.get(f"/api/editor/{session}")
    assert r.status_code == 200
    reloaded = r.json()
    assert [e["panel_id"] for e in reloaded["edited_timeline"]] == ["panel_001", "panel_002", "panel_003", "panel_004"]
