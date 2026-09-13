# webapp/test_render_resilience.py
"""End-to-end: render failures must not dead-end the automation.

Covers the real-world report: "render_video failed: ffmpeg not found on
PATH" — even though the imageio-ffmpeg wheel ships a bundled binary —
followed by a full job stop that forced manual checks between steps.

1. ffmpeg resolution falls back to the bundled binary (or soft-fails).
2. A render failure leaves timeline/SRT/editor project intact and the
   job COMPLETES with a visible render_error + retry affordance.
3. start_stage=render_video resumes ONLY the render stage.
"""
from __future__ import annotations

import io
import time
from pathlib import Path

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


def _wait_done(job_id, timeout=60):
    t0 = time.time()
    while time.time() - t0 < timeout:
        j = store.get(job_id)
        if j and j.status.value in ("completed", "failed", "cancelled"):
            return j
        time.sleep(0.2)
    raise AssertionError("job never reached terminal state")


@pytest.fixture()
def session_with_run():
    c = TestClient(webmain.app)
    up = c.post("/api/upload", files={"file": ("s.png",
                                                io.BytesIO(_png_strip()),
                                                "image/png")})
    assert up.status_code == 200, up.text
    session = up.json()["job_id"]
    return c, session


def test_ffmpeg_resolution_falls_back_to_bundled():
    """No ffmpeg on PATH must not raise if imageio-ffmpeg is installed."""
    import shutil
    if shutil.which("ffmpeg"):
        pytest.skip("ffmpeg on PATH; fallback path not exercised")
    resolved = pipeline._resolve_ffmpeg()
    assert resolved and Path(resolved).is_file(), resolved


def test_render_failure_completes_with_retry_state(session_with_run,
                                                   monkeypatch):
    """Simulated encoder crash => job completes, editor project exists,
    render_error visible, earlier outputs intact."""
    c, session = session_with_run

    # Stub everything except render_video; keep the real render stage but
    # force a soft failure by making ffmpeg resolution fail.
    def boom():
        raise pipeline._RenderSoftError("simulated ffmpeg missing")

    monkeypatch.setattr(pipeline, "_resolve_ffmpeg", boom)

    def fake_stage(job, **kwargs):
        job.log("INFO", "stubbed")

    def fake_segment(job, **kwargs):
        job.panels = [{"id": "panel_001", "panel_index": 1,
                       "y_start": 0, "y_end": 50, "narration": "A",
                       "dialogue": "", "panel_type": "single",
                       "confidence": 0.9, "image_file": "panel_001.png"}]

    stubbed = []
    for name, fn in pipeline.PIPELINES["generate"]:
        if name == "render_video" or name in ("apply_confirmed", "apply_order",
                      "create_editor_project", "save_outputs"):
            stubbed.append((name, fn))
        elif name == "segment_panels":
            stubbed.append((name, fake_segment))
        else:
            stubbed.append((name, fake_stage))
    monkeypatch.setitem(pipeline.PIPELINES, "generate", stubbed)

    d = OUTPUT_DIR / session
    d.mkdir(parents=True, exist_ok=True)
    (d / "panels.json").write_text(__import__("json").dumps({
        "source": "s.png", "width": 60, "height": 120, "plan_hash": "x",
        "config": {},
        "panels": [{"id": "panel_001", "panel_index": 1, "y_start": 0,
                    "y_end": 50, "narration": "A", "dialogue": "",
                    "panel_type": "single", "confidence": 0.9,
                    "image_file": "panel_001.png"}]}), "utf-8")
    (d / "panel_001.png").write_bytes(_png_strip())

    r = c.post("/api/run", json={"session": session, "order": None,
                                 "backend": "none", "tts": "none"})
    job = _wait_done(r.json()["job_id"])
    # The whole point: the job COMPLETES (not fails) so downstream stages
    # (save_outputs, create_editor_project) still run.
    assert job.status.value == "completed", job.error
    assert job.outputs.get("render_error"), job.outputs
    assert (d / "timeline.json").is_file()
    assert (d / "recap.srt").is_file()
    # editor project created despite no video
    assert (d / "editor.json").is_file()


def test_resume_from_render_stage_only(session_with_run, monkeypatch):
    """start_stage=render_video skips segmentation/narration and runs only
    the render stage onward."""
    c, session = session_with_run

    calls = {"segment": 0, "render": 0}

    def fake_segment(job, **kwargs):
        calls["segment"] += 1
        job.panels = [{"id": "panel_001", "panel_index": 1,
                       "y_start": 0, "y_end": 50, "narration": "A",
                       "dialogue": "", "panel_type": "single",
                       "confidence": 0.9, "image_file": "panel_001.png"}]

    def fake_render(job, **kwargs):
        calls["render"] += 1
        job.outputs["recap.mp4"] = "recap.mp4"

    def fake_stage(job, **kwargs):
        job.log("INFO", "stubbed")

    stubbed = []
    for name, _fn in pipeline.PIPELINES["generate"]:
        if name == "segment_panels":
            stubbed.append((name, fake_segment))
        elif name == "render_video":
            stubbed.append((name, fake_render))
        else:
            stubbed.append((name, fake_stage))
    monkeypatch.setitem(pipeline.PIPELINES, "generate", stubbed)

    d = OUTPUT_DIR / session
    d.mkdir(parents=True, exist_ok=True)
    (d / "panel_001.png").write_bytes(_png_strip())

    r = c.post("/api/run", json={"session": session, "order": None,
                                 "backend": "none", "tts": "none",
                                 "start_stage": "render_video"})
    assert r.status_code == 200
    job = _wait_done(r.json()["job_id"])
    assert job.status.value == "completed", job.error
    assert calls["segment"] == 0, "resume must not re-run segmentation"
    assert calls["render"] == 1, "resume must run the render stage"
    logtext = "\n".join(e["msg"] for e in job._logs)
    assert "resuming from stage 'render_video'" in logtext
    assert "skipping stage" in logtext
