# webapp/pipeline.py
"""Staged background worker. One rule: between any two log lines
there is at most ONE bounded operation, and every operation has a
timeout that converts a hang into an explicit failure."""
from __future__ import annotations

import asyncio
import os
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from .jobs import CancelledError, Job, JobStatus, store

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "webapp_output"

STAGE_TIMEOUT_S = {
    "validate_config": 5,
    "load_images": 30,
    "segment_panels": 300,
    "apply_order": 5,
    "gemini_narration": 300,
    "build_script": 10,
    "tts_audio": 600,
    "render_video": 3600,
    "save_outputs": 10,
    "create_editor_project": 30,
}
JOB_TIMEOUT_S = 1800


class StageTimeoutError(RuntimeError):
    pass


def _with_timeout(fn: Callable[[], Any], seconds: float,
                  job: Job, stage: str) -> Any:
    with ThreadPoolExecutor(max_workers=1,
                            thread_name_prefix=stage) as ex:
        fut = ex.submit(fn)
        try:
            return fut.result(timeout=seconds)
        except TimeoutError as exc:
            raise StageTimeoutError(
                f"stage {stage} timed out after {int(seconds)}s") from exc


def _stage(job: Job, name: str, fn: Callable[..., Any]) -> Any:
    job.check_cancelled()
    job.stage = name
    job.touch()
    job.log("INFO", f"stage={name} started", name)
    t0 = time.time()
    try:
        result = _with_timeout(fn, STAGE_TIMEOUT_S[name], job, name)
    except StageTimeoutError as exc:
        job.fail(str(exc))
        raise
    except CancelledError:
        job.status = JobStatus.CANCELLED
        job.finished_at = time.time()
        job.touch()
        job.log("INFO", f"stage={name} cancelled", name)
        raise
    except Exception as exc:
        import traceback
        job.fail(f"stage {name} failed: {type(exc).__name__}: {exc}",
                 traceback.format_exc())
        raise
    dt = time.time() - t0
    job.touch()
    job.log("INFO", f"stage={name} completed duration={dt:.1f}s", name)
    return result


def _validate_config(job: Job, **kwargs: Any) -> None:
    backend = job.config.get("backend", "none")
    api_key = job.config.get("api_key", "") or os.environ.get("GEMINI_API_KEY", "")
    job.log("INFO",
            f"backend={backend} api_key={'set' if api_key else 'missing'}",
            "validate_config")
    if backend != "none" and not api_key:
        raise RuntimeError(
            f"{backend.upper()}_API_KEY not set; enter it in the webapp settings or .env")


def _load_images(job: Job, **kwargs: Any) -> Path:
    from PIL import Image
    strip = OUTPUT_DIR / job.config["session"] / job.config["strip_file"]
    if not strip.is_file():
        raise FileNotFoundError(f"strip not found: {strip.name}")
    with Image.open(strip) as img:
        w, h = img.size
    job.log("INFO", f"strip loaded {w}x{h}", "load_images")
    return strip


def _segment_panels(job: Job, **kwargs: Any) -> None:
    import guided_pipeline as gp
    session_dir = OUTPUT_DIR / job.config["session"]
    strip = session_dir / job.config["strip_file"]
    cache = BASE_DIR / ".cache" / "recap-comic"
    cache.mkdir(parents=True, exist_ok=True)
    backend_name = job.config.get("backend", "none")
    api_key = job.config.get("api_key", "")
    model = job.config.get("model", "")
    endpoint = job.config.get("endpoint", "")
    cf_account_id = job.config.get("cf_account_id", "")
    _plan, artifact, _used = gp.run_guided(
        strip, session_dir, backend_name=backend_name,
        cache_dir=cache, force=False, fallback=True,
        api_key=api_key or None, model=model or None,
        base_url=endpoint or None, cf_account_id=cf_account_id or None)
    assert artifact is not None
    job.panels = [{
        "id": p.id, "panel_index": p.panel_index,
        "y_start": p.y_start, "y_end": p.y_end,
        "narration": p.narration, "dialogue": p.dialogue,
        "panel_type": p.panel_type, "confidence": p.confidence,
        "image_file": p.image_file,
    } for p in artifact.panels]
    job.log("INFO", f"panels detected count={len(job.panels)}",
            "segment_panels")
    job.progress = 30


def _apply_order(job: Job, **kwargs: Any) -> list[str]:
    order = job.config.get("order") or [p["id"] for p in job.panels]
    have = {p["id"] for p in job.panels}
    if set(order) != have or len(order) != len(have):
        raise ValueError(
            f"panel order mismatch: got {len(order)} ids, "
            f"expected {len(have)} matching ids")
    by_id = {p["id"]: p for p in job.panels}
    job.panels = [by_id[i] for i in order]
    job.log("INFO", "panel order:", "apply_order")
    for n, pid in enumerate(order, 1):
        job.log("INFO", f"  {n} = {pid}", "apply_order")
    return order


def _gemini_narration(job: Job, **kwargs: Any) -> None:
    session_dir = OUTPUT_DIR / job.config["session"]
    import guided_pipeline as gp
    strip = session_dir / job.config["strip_file"]
    cache = BASE_DIR / ".cache" / "recap-comic"
    backend_name = job.config.get("backend", "gemini")
    api_key = kwargs.get("api_key") or job.config.get("api_key", "") or os.environ.get("GEMINI_API_KEY", "")
    model = kwargs.get("model") or job.config.get("model", "") or None
    endpoint = kwargs.get("base_url") or job.config.get("endpoint", "") or None
    cf_account_id = kwargs.get("cf_account_id") or job.config.get("cf_account_id", "") or None
    job.log("INFO", f"{backend_name} narration started", "gemini_narration")
    plan, _artifact, _used = gp.run_guided(
        strip, session_dir, backend_name=backend_name,
        cache_dir=cache, force=False, fallback=True,
        api_key=api_key or None, model=model,
        base_url=endpoint, cf_account_id=cf_account_id)
    job.log("INFO",
            f"narration completed panels={len(plan.entries)} "
            f"model={plan.model}", "gemini_narration")
    by_index = {e.panel_index: e for e in plan.entries}
    for p in job.panels:
        e = by_index.get(p["panel_index"])
        if e is not None:
            p["narration"] = e.narration
            p["dialogue"] = e.dialogue
            p["confidence"] = e.confidence
    job.progress = 60


def _build_script(job: Job, **kwargs: Any) -> None:
    from guided_cutter import CutArtifact
    from narrator import make_script_from_cut
    session_dir = OUTPUT_DIR / job.config["session"]
    artifact = CutArtifact.model_validate_json(
        (session_dir / "panels.json").read_text("utf-8"))
    script = make_script_from_cut(artifact,
                                  style=job.config.get("style", "recap"))
    (session_dir / "narration.txt").write_text(script, "utf-8")
    job.log("INFO", f"narration script chars={len(script)}", "build_script")


def _tts_audio(job: Job, **kwargs: Any) -> None:
    if job.config.get("tts", "edge") == "none":
        job.log("INFO", "tts skipped (tts=none)", "tts_audio")
        return
    from . import tts_helpers
    entries = [{"id": p["id"], "text": p["narration"]}
               for p in job.panels if p.get("narration", "").strip()]
    audio_dir = OUTPUT_DIR / job.config["session"] / "audio"
    audio_dir.mkdir(exist_ok=True)
    for i, e in enumerate(entries, 1):
        job.check_cancelled()
        asyncio.run(tts_helpers.synth_one(
            e["text"], job.config.get("voice", "en-US-AriaNeural"),
            audio_dir / f"{e['id']}.mp3", timeout_s=60))
        job.log("INFO", f"tts {i}/{len(entries)} {e['id']}", "tts_audio")
        job.progress = 60 + int(20 * i / max(1, len(entries)))


def _render_video(job: Job, **kwargs: Any) -> None:
    session_dir = OUTPUT_DIR / job.config["session"]
    panels_json = session_dir / "panels.json"
    out_mp4 = session_dir / "recap.mp4"
    from recap_video import VideoConfig, make_recap_video
    cfg = VideoConfig(tts=job.config.get("tts", "edge"),
                      voice=job.config.get("voice", "en-US-AriaNeural"))
    make_recap_video(panels_json, out_mp4, cfg, force=True,
                     dry_run=False)
    job.log("INFO", "render completed", "render_video")
    job.progress = 95


def _save_outputs(job: Job, **kwargs: Any) -> None:
    session_dir = OUTPUT_DIR / job.config["session"]
    for name in ("panels.json", "narration.txt", "recap.mp4",
                 "recap.srt", "timeline.json"):
        if (session_dir / name).exists():
            job.outputs[name] = name
    job.progress = 100


def _create_editor_project(job: Job, **kwargs: Any) -> None:
    from .editor_api import create_project_from_generation
    session = job.config["session"]
    try:
        proj = create_project_from_generation(session)
        job.outputs["editor.json"] = "editor.json"
        job.log("INFO", f"editor project created panels={len(proj['edited_timeline'])}", "create_editor_project")
    except Exception as exc:
        job.log("WARNING", f"editor project creation failed: {exc}", "create_editor_project")


PIPELINES: dict[str, list[tuple[str, Callable[[Job], Any]]]] = {
    "segment": [
        ("validate_config", _validate_config),
        ("load_images", _load_images),
        ("segment_panels", _segment_panels),
    ],
    "generate": [
        ("validate_config", _validate_config),
        ("load_images", _load_images),
        ("segment_panels", _segment_panels),
        ("apply_order", _apply_order),
        ("gemini_narration", _gemini_narration),
        ("build_script", _build_script),
        ("tts_audio", _tts_audio),
        ("render_video", _render_video),
        ("save_outputs", _save_outputs),
        ("create_editor_project", _create_editor_project),
    ],
}


def run_job(job_id: str, **kwargs: Any) -> None:
    job = store.get(job_id)
    if job is None:
        return
    job.status = JobStatus.RUNNING
    job.started_at = time.time()
    job.touch()
    job.log("INFO", f"worker started kind={job.kind}")
    t0 = time.time()
    try:
        for name, fn in PIPELINES[job.kind]:
            if time.time() - t0 > JOB_TIMEOUT_S:
                raise StageTimeoutError(
                    f"job exceeded {JOB_TIMEOUT_S}s total budget")
            _stage(job, name, lambda fn=fn: fn(job, **kwargs))
        job.status = JobStatus.COMPLETED
        job.stage = "done"
        job.finished_at = time.time()
        job.touch()
        job.log("INFO", "generation completed")
    except CancelledError:
        pass
    except Exception:
        if job.status not in (JobStatus.FAILED, JobStatus.CANCELLED):
            import traceback
            job.fail(f"worker crashed: {traceback.format_exc()[-300:]}")
