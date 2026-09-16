# webapp/test_render_strategy.py
"""The webapp render must pick direct-vs-chunked from the REAL timeline
duration, not a hardcoded 120s.

webapp/pipeline._render_video used to call
pick_render_strategy(n_panels, 120), which made the total_seconds>150 branch
dead: a 10-panel / 400s video rendered DIRECT (one giant filter graph,
unbounded memory) while a 26-panel / 60s video went chunked.
"""
from __future__ import annotations

import json

import pytest

from webapp import pipeline
from webapp.jobs import Job, JobStatus

_TL_ENTRY = {
    "panel_id": "panel_001", "order": 1, "source_image": "panel_001.png",
    "bbox": {"x": 0, "y": 0, "w": 800, "h": 1200},
    "start_seconds": 0.0, "duration_seconds": 1.0,
    "audio_path": None,
    "pan": {"kind": "static", "scaled_w": 1080, "scaled_h": 1920, "travel_px": 0},
}


def _session(tmp_path, n_entries: int, per_panel_s: float, monkeypatch):
    sess = "abc123def456"
    d = tmp_path / sess
    d.mkdir(parents=True, exist_ok=True)
    entries = []
    t = 0.0
    for i in range(n_entries):
        e = dict(_TL_ENTRY, panel_id=f"panel_{i:03d}", order=i + 1,
                 start_seconds=t, duration_seconds=per_panel_s)
        entries.append(e)
        t += per_panel_s
    tl = {
        "meta": {"schema_version": 1, "generator": "test",
                 "config_hash": "x", "input_hashes": {}},
        "width": 1080, "height": 1920, "fps": 30,
        "gap_seconds": 0.0, "min_display_seconds": 1.0,
        "entries": entries,
    }
    (d / "timeline.json").write_text(json.dumps(tl), "utf-8")
    (d / "panels.json").write_text(
        json.dumps({"source": "x.png", "width": 800, "height": 2400,
                    "plan_hash": "x", "config": {}, "panels": []}), "utf-8")

    monkeypatch.setattr(pipeline, "OUTPUT_DIR", tmp_path)

    # No TTS, no ffmpeg: this test is about the strategy DECISION only.
    import recap_video
    monkeypatch.setattr(recap_video, "make_recap_video",
                        lambda *a, **k: None, raising=False)
    monkeypatch.setattr(pipeline, "_resolve_ffmpeg", lambda: "ffmpeg")
    # "Run" a chunk by producing the expected output file.
    def fake_sub(job_id, cmd):
        for arg in cmd:
            if arg.endswith(".partial.mp4"):
                with open(arg, "wb") as fh:
                    fh.write(b"fake")
                return
    monkeypatch.setattr(pipeline, "_run_render_subprocess", fake_sub)

    class _FakeWorker:
        def __init__(self, job_id, build_cmd, out_tmp, on_done=None, **kw):
            out_tmp.write_bytes(b"fake")
            if on_done:
                on_done(True, "")
        def start(self):
            pass
    import webapp.render_worker as rw
    monkeypatch.setattr(rw, "RenderWorker", _FakeWorker)

    job = Job(sess, "generate", {"session": sess, "strip_file": "x.png"})
    job.status = JobStatus.RUNNING
    return job


@pytest.mark.parametrize("n_entries,per_panel_s,expected", [
    (3, 150.0, "chunked"),   # long video, few panels: OLD code said "direct"
    (30, 1.0, "chunked"),    # many panels
    (5, 20.0, "direct"),     # genuinely small
])
def test_strategy_uses_real_duration(tmp_path, monkeypatch, n_entries,
                                     per_panel_s, expected):
    job = _session(tmp_path, n_entries, per_panel_s, monkeypatch)
    pipeline._render_video(job)
    logged = [ln["msg"] for ln in job.to_dict(include_logs=True)["logs"]]
    assert any(f"strategy={expected}" in m for m in logged), (
        f"expected strategy={expected} for {n_entries} panels x "
        f"{per_panel_s}s; logs: {[m for m in logged if 'strategy' in m]}")
