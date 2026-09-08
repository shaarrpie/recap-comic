# webapp/pipeline.py
"""Staged background worker. One rule: between any two log lines
there is at most ONE bounded operation, and every operation has a
timeout that converts a hang into an explicit failure."""
from __future__ import annotations

import asyncio
import json
import os
import threading
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
    "panel_validation": 120,
    "apply_confirmed": 5,
    "apply_order": 5,
    "gemini_narration": 300,
    "build_script": 10,
    "tts_audio": 600,
    "render_video": 3600,
    "save_outputs": 10,
    "create_editor_project": 30,
}
JOB_TIMEOUT_S = 5400


class StageTimeoutError(RuntimeError):
    pass


def _with_timeout(fn: Callable[[], Any], seconds: float,
                  job: Job, stage: str) -> Any:
    ex = ThreadPoolExecutor(max_workers=1,
                            thread_name_prefix=stage)
    try:
        fut = ex.submit(fn)
        return fut.result(timeout=seconds)
    except TimeoutError as exc:
        raise StageTimeoutError(
            f"stage {stage} timed out after {int(seconds)}s") from exc
    finally:
        ex.shutdown(wait=False, cancel_futures=True)


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
    api_key = kwargs.get("api_key") or job.config.get("api_key", "") or os.environ.get("GEMINI_API_KEY", "")
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
    api_key = kwargs.get("api_key") or job.config.get("api_key", "") or os.environ.get("GEMINI_API_KEY", "")
    model = kwargs.get("model") or job.config.get("model", "") or None
    base_url = kwargs.get("base_url") or job.config.get("endpoint", "") or None
    cf_account_id = kwargs.get("cf_account_id") or job.config.get("cf_account_id", "") or None
    _plan, artifact, _used = gp.run_guided(
        strip, session_dir, backend_name=backend_name,
        cache_dir=cache, force=False, fallback=True,
        api_key=api_key or None, model=model,
        base_url=base_url, cf_account_id=cf_account_id,
        validate=True)
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


def _apply_confirmed(job: Job, **kwargs: Any) -> None:
    """Apply the user's Panel Review decisions (order + soft-deletes).

    Tolerant no-op when the session has no panel data yet, so start_stage
    retries and stubbed pipelines keep working.  An explicit `order` passed
    to /api/run always takes precedence over the stored review state.
    """
    if job.config.get("order"):
        job.log("INFO", "explicit order provided; skipping confirmed review",
                "apply_confirmed")
        return
    from . import panel_api
    try:
        merged = panel_api.get_panels(job.config["session"])
    except Exception:
        job.log("INFO", "no panel review data; using AI segmentation as-is",
                "apply_confirmed")
        return
    if not merged.get("confirmed"):
        job.log("INFO", "panels not confirmed in review; using AI order",
                "apply_confirmed")
        return
    ids = [p["id"] for p in merged["panels"] if not p["deleted"]]
    by_id = {p["id"]: p for p in job.panels}
    missing = [i for i in ids if i not in by_id]
    if missing:
        raise ValueError(
            f"confirmed panel order references unknown panels: {missing}")
    job.panels = [by_id[i] for i in ids]
    job.config["order"] = ids
    job.log("INFO",
            f"confirmed review applied panels={len(ids)} "
            f"deleted={merged['deleted_count']}",
            "apply_confirmed")
    _write_confirmed_artifact(job)


def _write_confirmed_artifact(job: Job) -> Path:
    """Persist the confirmed panel set as panels_confirmed.json.

    Derives a CutArtifact from the AI baseline (panels.json, untouched) with
    the user's order/deletions applied and panel_index renumbered 1..n, so
    the script, TTS, timeline and captions all reflect the confirmed review.
    """
    from guided_cutter import CutArtifact
    session_dir = OUTPUT_DIR / job.config["session"]
    src = session_dir / "panels.json"
    artifact = CutArtifact.model_validate_json(src.read_text("utf-8"))
    by_id = {p.id: p for p in artifact.panels}
    ordered = []
    for p in job.panels:
        cp = by_id.get(p["id"])
        if cp is not None:
            ordered.append(cp)
    if not ordered:
        ordered = list(artifact.panels)
    for i, cp in enumerate(ordered, 1):
        cp.panel_index = i
    conf = artifact.model_copy(update={"panels": ordered})
    dst = session_dir / "panels_confirmed.json"
    tmp = dst.with_suffix(".tmp")
    tmp.write_text(conf.model_dump_json(indent=2) + "\n", "utf-8")
    tmp.replace(dst)
    job.log("INFO", f"confirmed artifact written panels={len(ordered)}",
            "build_script")
    return dst


def _gemini_narration(job: Job, **kwargs: Any) -> None:
    session_dir = OUTPUT_DIR / job.config["session"]
    from guided_cutter import CutArtifact
    artifact = CutArtifact.model_validate_json(
        _panels_source(session_dir).read_text("utf-8"))
    by_id = {p.id: p for p in artifact.panels}
    for p in job.panels:
        src = by_id.get(p["id"])
        if src is not None:
            p["narration"] = src.narration
            p["dialogue"] = src.dialogue
            p["confidence"] = src.confidence
    from .narration_api import apply_overrides_to_dicts
    n_ov = apply_overrides_to_dicts(job.config["session"], job.panels)
    if n_ov:
        job.log("INFO",
                f"narration overrides applied panels={n_ov}",
                "gemini_narration")
    job.progress = 60


def _panels_source(session_dir: Path) -> Path:
    """The renderer/script input: the user-confirmed artifact when Panel
    Review has confirmed it, otherwise the AI baseline (panels.json)."""
    conf = session_dir / "panels_confirmed.json"
    return conf if conf.is_file() else session_dir / "panels.json"


def _build_script(job: Job, **kwargs: Any) -> None:
    from guided_cutter import CutArtifact
    from narrator import make_script_from_cut
    session_dir = OUTPUT_DIR / job.config["session"]
    artifact = CutArtifact.model_validate_json(
        _panels_source(session_dir).read_text("utf-8"))
    from .narration_api import apply_overrides_to_cut
    n_ov = apply_overrides_to_cut(job.config["session"], artifact.panels)
    script = make_script_from_cut(artifact,
                                  style=job.config.get("style", "recap"))
    (session_dir / "narration.txt").write_text(script, "utf-8")
    if n_ov:
        job.log("INFO",
                f"narration overrides applied to script panels={n_ov}",
                "build_script")
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
    """Render video using the isolated RenderWorker for proper isolation,
    cancellation, and resource bounding."""
    session_dir = OUTPUT_DIR / job.config["session"]
    panels_json = _panels_source(session_dir)
    out_mp4 = session_dir / "recap.mp4"
    out_tmp = out_mp4.with_name(out_mp4.stem + ".partial.mp4")
    from adapters.render_ffmpeg import pick_render_strategy
    from recap_video import VideoConfig, make_recap_video

    from .render_worker import RenderWorker, render_profiles
    # Narrator Studio settings (voice.json) win; /api/run values are defaults.
    voice_cfg = {}
    vp = session_dir / "voice.json"
    if vp.is_file():
        try:
            voice_cfg = json.loads(vp.read_text("utf-8"))
        except Exception:
            voice_cfg = {}
    def _pct(key, default):
        v = voice_cfg.get(key, default)
        return f"{int(v):+d}%" if key == "rate" else f"{int(v):+d}Hz"
    cfg = VideoConfig(
        tts=voice_cfg.get("provider") or job.config.get("tts", "edge"),
        voice=voice_cfg.get("voice") or job.config.get("voice", "en-US-AriaNeural"),
        rate=_pct("rate", 0) if voice_cfg else "+0%",
        pitch=_pct("pitch", 0) if voice_cfg else "+0Hz",
        speed=float(voice_cfg.get("speed", 1.0) or 1.0))
    job.log("INFO",
            f"render input={panels_json.name} tts={cfg.tts} "
            f"voice={cfg.voice} rate={cfg.rate} pitch={cfg.pitch}",
            "render_video")
    # Determine render strategy
    render_profile = job.config.get("render_profile", "balanced")
    threads = job.config.get("ffmpeg_threads", "auto")
    cpu_count = os.cpu_count() or 4
    if threads == "auto":
        threads = str(max(2, cpu_count - 2))
    prof = render_profiles(cpu_count).get(render_profile, render_profiles(4)["balanced"])
    prof["threads"] = threads
    # TTS + timeline + captions (synchronous, cache-friendly)
    make_recap_video(panels_json, out_tmp, cfg, force=True, dry_run=True)
    # Build the render command
    strategy = pick_render_strategy(len(job.panels) if job.panels else 10, 120)
    if strategy == "chunked":
        from adapters.render_ffmpeg import render_chunked
        ta = _get_timeline_artifact(session_dir)
        try:
            render_chunked(ta, out_tmp, chunk_size=12, profile=prof)
            if out_tmp.is_file():
                out_tmp.replace(out_mp4)
            else:
                raise RuntimeError("chunked render did not produce output")
        except Exception:
            if out_tmp.is_file():
                out_tmp.unlink()
            raise
    else:
        from adapters.render_ffmpeg import build_command
        ta = _get_timeline_artifact(session_dir)
        ffmpeg_exe = getattr(cfg, "ffmpeg_exe", "ffmpeg") or "ffmpeg"
        def build_cmd(tmp):
            return build_command(ta, tmp, ffmpeg_exe=ffmpeg_exe)
        done = threading.Event()
        res = {}
        def on_done(ok, err):
            res["ok"] = ok
            res["err"] = err
            done.set()
        worker = RenderWorker(job.id, build_cmd, out_tmp, on_done=on_done)
        worker.start()
        try:
            done.wait()
        finally:
            pass
        if not res.get("ok"):
            raise RuntimeError(res.get("err", "render failed"))
        if out_tmp.is_file():
            out_tmp.replace(out_mp4)
        else:
            raise RuntimeError("render did not produce output")
    job.log("INFO", "render completed", "render_video")
    job.progress = 95


def _panel_validation(job: Job, **kwargs: Any) -> None:
    """Post-segmentation validation layer. Loads panels_validation.json
    (written by guided_cutter when validate=True), overlays user decisions
    from review.json, logs stats, and quarantines INVALID panels before
    downstream stages."""
    from panel_validator import apply_decisions, effective_ids, load_report, load_review
    session_dir = OUTPUT_DIR / job.config["session"]
    rep = load_report(session_dir)
    if rep is None:
        job.log("WARNING", "no validation report; run segmentation first",
                "panel_validation")
        return
    rep = apply_decisions(rep, load_review(session_dir))
    stats = rep["stats"]
    job.log("INFO", (f"Detected: {stats['detected']}  "
                     f"Accepted: {stats['accepted']}  "
                     f"Suspicious: {stats['suspicious']}  "
                     f"Rejected: {stats['rejected']}  "
                     f"Duplicates: {stats['duplicates']}"),
            "panel_validation")
    job.outputs["validation"] = rep
    ordered_ids = [p["id"] for p in job.panels]
    valid_ids = set(effective_ids(rep, ordered_ids))
    before = len(job.panels)
    job.panels = [p for p in job.panels if p["id"] in valid_ids]
    job.config["order"] = [p["id"] for p in job.panels]
    quarantined = before - len(job.panels)
    if quarantined:
        job.log("WARNING",
                f"quarantined {quarantined} invalid/suspicious panels; "
                f"remaining={len(job.panels)}",
                "panel_validation")


def _get_timeline_artifact(session_dir):
    """Load timeline.json as a TimelineArtifact for chunked rendering."""
    from adapters.schemas import TimelineArtifact
    tl_path = session_dir / "timeline.json"
    if not tl_path.is_file():
        raise FileNotFoundError(f"timeline.json not found in {session_dir}")
    return TimelineArtifact.model_validate_json(tl_path.read_text("utf-8"))


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
        ("panel_validation", _panel_validation),
    ],
    "generate": [
        ("validate_config", _validate_config),
        ("load_images", _load_images),
        ("segment_panels", _segment_panels),
        ("panel_validation", _panel_validation),
        ("apply_confirmed", _apply_confirmed),
        ("apply_order", _apply_order),
        ("gemini_narration", _gemini_narration),
        ("build_script", _build_script),
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
    start_stage = job.config.get("start_stage")
    try:
        for name, fn in PIPELINES[job.kind]:
            if start_stage and name != start_stage:
                if not hasattr(job, "_start_passed") or not job._start_passed:
                    job.log("INFO", f"skipping stage (before start_stage={start_stage}) {name}")
                    if name == start_stage:
                        job._start_passed = True
                    continue
                job._start_passed = True
            job._start_passed = True
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
