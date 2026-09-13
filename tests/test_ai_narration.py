# tests/test_ai_narration.py
"""Post-crop AI narration: deterministic cut first, AI words after.

Geometry must NEVER change; only narration/dialogue are filled, via
Qwen -> Mistral with per-panel caching.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

import guided_pipeline as gp
from adapters import ai_narration as ain


def _session(tmp_path: Path, n: int = 2) -> Path:
    d = tmp_path / "sess"
    d.mkdir()
    panels = []
    for i in range(1, n + 1):
        Image.new("RGB", (100, 200), (i * 40, 10, 10)).save(d / f"panel_{i:03d}.png")
        panels.append({
            "id": f"{i:03d}", "panel_index": i,
            "y_start": (i - 1) * 200, "y_end": i * 200,
            "narration": "", "dialogue": "", "panel_type": "single",
            "confidence": 0.0, "image_file": f"panel_{i:03d}.png",
            "split_of": None, "merged_with": [], "snap_distances": [],
        })
    (d / "panels.json").write_text(json.dumps({
        "source": "strip.png", "width": 100, "height": n * 200,
        "plan_hash": "x", "config": {}, "panels": panels}), "utf-8")
    return d


def _fake_ok(model: str, prompt: str, b64: str) -> str:
    assert b64  # actual panel image bytes must be passed
    # narration-only contract: words requested, no coordinates requested
    assert "narration" in prompt and "y_start" not in prompt
    return json.dumps({"narration": "A hero stands.", "dialogue": "Go!"})


def _geo(d: Path) -> list[tuple]:
    data = json.loads((d / "panels.json").read_text("utf-8"))
    return [(p["id"], p["y_start"], p["y_end"], p["image_file"])
            for p in data["panels"]]


def test_fills_words_geometry_untouched(tmp_path: Path) -> None:
    d = _session(tmp_path)
    before = _geo(d)
    summary = ain.narrate_cropped_panels(
        d, api_key="test", gap_s=0, cache_dir=tmp_path / "cache",
        request_fn=_fake_ok)
    assert summary["panels"] == 2 and summary["narrated"] == 2
    assert summary["failed"] == []
    assert _geo(d) == before  # geometry byte-identical
    data = json.loads((d / "panels.json").read_text("utf-8"))
    assert all(p["narration"] == "A hero stands." for p in data["panels"])
    assert all(p["dialogue"] == "Go!" for p in data["panels"])
    assert (d / "ai_narration.json").is_file()
    assert (d / "narration.txt").is_file()


def test_cache_skips_api(tmp_path: Path) -> None:
    d = _session(tmp_path)
    ain.narrate_cropped_panels(d, api_key="test", gap_s=0,
                               cache_dir=tmp_path / "cache",
                               request_fn=_fake_ok)

    def _boom(model: str, prompt: str, b64: str) -> str:
        raise AssertionError("must not be called (cached)")
    summary = ain.narrate_cropped_panels(d, api_key="test", gap_s=0,
                                         cache_dir=tmp_path / "cache",
                                         request_fn=_boom)
    assert summary["cached"] == 2 and summary["narrated"] == 0


def test_primary_fail_uses_fallback(tmp_path: Path) -> None:
    d = _session(tmp_path, n=1)
    seen: list[str] = []

    def _fake(model: str, prompt: str, b64: str) -> str:
        seen.append(model)
        if "qwen" in model:
            raise TimeoutError("qwen down")
        return json.dumps({"narration": "Fallback words.", "dialogue": ""})
    summary = ain.narrate_cropped_panels(d, api_key="test", gap_s=0,
                                         cache_dir=tmp_path / "cache",
                                         request_fn=_fake)
    assert summary["narrated"] == 1
    assert seen[0] == "qwen/qwen3.5-397b-a17b:free"
    assert seen[-1] == "mistralai/mistral-medium-3.5"
    data = json.loads((d / "panels.json").read_text("utf-8"))
    assert data["panels"][0]["narration"] == "Fallback words."


def test_both_fail_keeps_old_text_and_raises(tmp_path: Path) -> None:
    d = _session(tmp_path, n=1)

    def _fake(model: str, prompt: str, b64: str) -> str:
        raise RuntimeError("all down")
    with pytest.raises(RuntimeError, match="every panel"):
        ain.narrate_cropped_panels(d, api_key="test", gap_s=0,
                                   cache_dir=tmp_path / "cache",
                                   request_fn=_fake)
    data = json.loads((d / "panels.json").read_text("utf-8"))
    assert data["panels"][0]["narration"] == ""  # nothing fabricated
    assert _geo(d) == [("001", 0, 200, "panel_001.png")]


def test_empty_ai_text_keeps_existing(tmp_path: Path) -> None:
    d = _session(tmp_path, n=1)
    data = json.loads((d / "panels.json").read_text("utf-8"))
    data["panels"][0]["narration"] = "Human words."
    (d / "panels.json").write_text(json.dumps(data), "utf-8")

    def _fake(model: str, prompt: str, b64: str) -> str:
        return json.dumps({"narration": "", "dialogue": ""})
    ain.narrate_cropped_panels(d, api_key="test", gap_s=0,
                               cache_dir=tmp_path / "cache",
                               request_fn=_fake)
    data = json.loads((d / "panels.json").read_text("utf-8"))
    assert data["panels"][0]["narration"] == "Human words."


def test_generate_endpoint_runs_job_and_fills_words(tmp_path: Path) -> None:
    import shutil
    import time

    from fastapi.testclient import TestClient

    import adapters.ai_narration as ain_module
    from webapp import main as webmain
    from webapp.jobs import store

    session = "abcdef123456"
    d = webmain.OUTPUT_DIR / session
    d.mkdir(parents=True, exist_ok=True)
    try:
        src = _session(tmp_path, n=1)
        shutil.copy(src / "panels.json", d / "panels.json")
        shutil.copy(src / "panel_001.png", d / "panel_001.png")

        real = ain_module.narrate_cropped_panels

        def _fast(session_dir, **kw):
            kw["request_fn"] = _fake_ok
            kw["gap_s"] = 0
            kw["cache_dir"] = tmp_path / "cache"
            return real(session_dir, **kw)

        ain_module.narrate_cropped_panels = _fast  # type: ignore[assignment]
        try:
            c = TestClient(webmain.app)
            r = c.post(f"/api/narration/{session}/generate", json={})
            # no key in test env -> 400 without monkeypatched key; use api_key
            if r.status_code == 400:
                r = c.post(f"/api/narration/{session}/generate",
                           json={"api_key": "test"})
            assert r.status_code == 200, r.text
            job_id = r.json()["job_id"]
            t0 = time.time()
            while time.time() - t0 < 20:
                j = store.get(job_id)
                if j and j.status.value in ("completed", "failed"):
                    break
                time.sleep(0.1)
            assert j is not None and j.status.value == "completed", \
                getattr(j, "error", None)
            data = json.loads((d / "panels.json").read_text("utf-8"))
            assert data["panels"][0]["narration"] == "A hero stands."
        finally:
            ain_module.narrate_cropped_panels = real  # type: ignore[assignment]
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_deterministic_backend_is_offline() -> None:
    for name in ("deterministic", "cv", "manual", "none"):
        assert gp.build_backend(name) is None


def test_deterministic_cut_needs_no_key(tmp_path: Path) -> None:
    import numpy as np

    rng = np.random.default_rng(3)
    arr = np.full((1200, 200, 3), 240, dtype=np.uint8)
    arr[40:590] = rng.integers(40, 215, (550, 200, 3), dtype=np.uint8)
    arr[590:606] = 255  # blank single-colour row band = the cut line
    arr[606:1160] = rng.integers(40, 215, (554, 200, 3), dtype=np.uint8)
    strip = tmp_path / "strip.png"
    Image.fromarray(arr).save(strip)
    plan, artifact, used = gp.run_guided(
        strip, tmp_path / "out", backend_name="deterministic")
    assert artifact is not None and len(artifact.panels) >= 2
    assert all(p.narration == "" for p in artifact.panels)  # no AI words
    assert used is True
