# webapp/pipeline.py
"""Staged background worker. One rule: between any two log lines
there is at most ONE bounded operation, and every operation has a
timeout that converts a hang into an explicit failure."""
from __future__ import annotations

import contextlib
import json
import os
import shutil
import subprocess
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from .jobs import CancelledError, Job, JobStatus, store

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "webapp_output"

# We run the whole pipeline inside the server process: switch the CLI
# progress bars off. Ten concurrent jobs each drawing a rich.Progress
# spinner into the shared uvicorn console garble the log output (the
# "Analyzing N strip chunks via LLM... 0%" frames interleaving with real
# log lines); the frontend tracks progress via job stages instead.
import recap_video as _recap_video  # noqa: E402
import strip_analyzer as _strip_analyzer  # noqa: E402

_strip_analyzer.EMBEDDED_MODE = True
_recap_video.EMBEDDED_MODE = True

STAGE_TIMEOUT_S = {
    "validate_config": 5,
    "load_images": 30,
    "segment_panels": 300,
    "panel_validation": 120,
    "apply_confirmed": 5,
    "merge_continuation": 60,
    "apply_order": 5,
    "gemini_narration": 300,
    "build_script": 30,
    "tts_audio": 600,
    "render_video": 3600,
    "save_outputs": 10,
    "create_editor_project": 30,
}
JOB_TIMEOUT_S = 5400
# Per-chunk ffmpeg budget: the whole render stage has 3600s, so a single
# hung chunk must never be allowed to eat the entire stage with no cancel
# path (/render-cancel cannot reach chunked subprocesses).
CHUNK_TIMEOUT_S = 1200


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


class _RenderSoftError(Exception):
    """Render failed but the session is still usable (see _render_video)."""


def _stage_continue_on_fail(job: Job, name: str, fn: Callable[..., Any]) -> Any:
    """Like _stage, but a _RenderSoftError marks a partial success and
    lets the pipeline continue (save_outputs, create_editor_project)."""
    job.check_cancelled()
    job.stage = name
    job.touch()
    job.log("INFO", f"stage={name} started", name)
    t0 = time.time()
    try:
        result = _with_timeout(fn, STAGE_TIMEOUT_S[name], job, name)
    except _RenderSoftError as exc:
        job.outputs["render_error"] = str(exc)[:300]
        job.log("WARNING", f"render failed ({exc}); continuing without "
                          "video — timeline/SRT/editor remain available. "
                          "Retry render from the Generate view when ffmpeg "
                          "is available.", "render_video")
        return None
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
    backend = job.config.get("backend", "xkiro")
    from adapters import ai_models as _ai
    api_key = (kwargs.get("api_key") or job.config.get("api_key", "")
               or _ai.api_key_from_env()
               or os.environ.get("GEMINI_API_KEY", ""))
    job.log("INFO",
            f"backend={backend} api_key={'set' if api_key else 'missing'}",
            "validate_config")
    if backend != "none" and not api_key:
        raise RuntimeError(
            f"{backend.upper()}_API_KEY not set; set XKIRO_API_KEY in .env "
            "or use --backend none for offline mode")


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
    backend_name = job.config.get("backend", "xkiro")
    from adapters import ai_models as _ai
    api_key = (kwargs.get("api_key") or job.config.get("api_key", "")
               or _ai.api_key_from_env()
               or os.environ.get("GEMINI_API_KEY", ""))
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
    # `order` may carry the explicit user-supplied order from /api/run;
    # promote it to user_order so retries and later stages see one key.
    if job.config.get("order"):
        job.config["user_order"] = job.config["order"]
        job.config["order"] = None
    order = job.config.get("user_order") or [p["id"] for p in job.panels]
    have = {p["id"] for p in job.panels}
    if job.config.get("continue_from"):
        # Translate raw own-strip ids to the namespaced merged ids first
        id_map = job.config.get("_own_id_map") or {}
        order = [id_map.get(i, i) for i in order]
        if set(order) != have:
            # With continuation, an explicit order can only cover THIS
            # strip's panels (previous strips' orders are frozen in their
            # own confirmed artifacts). Place the ordered own panels at
            # the end, keeping the merged sequence: prev strips -> own.
            prev = [i for i in order if i not in have]
            if prev:
                raise ValueError(
                    f"panel order references unknown panels: {prev}")
            own_ordered = [i for i in order if i in have]
            merged_first = [p["id"] for p in job.panels
                            if p["id"] not in set(order)]
            order = merged_first + own_ordered
            job.log("INFO", "continuation active: explicit order applies "
                            "to this strip's panels (appended after "
                            "previous strips)", "apply_order")
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
    if job.config.get("user_order"):
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
    active = [p for p in merged["panels"] if not p["deleted"]]
    by_id = {p["id"]: p for p in job.panels}
    missing = [p["id"] for p in active if p["id"] not in by_id]
    if missing:
        raise ValueError(
            f"confirmed panel order references unknown panels: {missing}")
    job.panels = [by_id[p["id"]] for p in active]
    job.config["order"] = [p["id"] for p in job.panels]
    job.log("INFO",
            f"confirmed review applied panels={len(job.panels)} "
            f"deleted={merged['deleted_count']}",
            "apply_confirmed")
    _write_confirmed_artifact(job)


def _write_confirmed_artifact(job: Job, panels: list[CutPanelLike] | None = None) -> Path:
    """Persist the confirmed panel set as panels_confirmed.json.

    Derives a CutArtifact from the AI baseline (panels.json, untouched) with
    the user's order/deletions applied and panel_index renumbered 1..n, so
    the script, TTS, timeline and captions all reflect the confirmed review.
    Narration overrides (Narration Studio) are baked in here too — the
    render reads this artifact, so without baking them the user's edited
    narration would never reach the video.
    """
    from guided_cutter import CutArtifact
    session_dir = OUTPUT_DIR / job.config["session"]
    if not job.config.get("continue_from"):
        # A fresh (non-continuation) run must not render a stale
        # panels_merged.json left by an earlier chained run — it would
        # silently ignore the fresh segmentation/confirmed panels.
        stale = session_dir / "panels_merged.json"
        if stale.is_file():
            stale.unlink()
            job.log("INFO", "removed stale panels_merged.json "
                            "(fresh run without continuation)",
                    "apply_confirmed")
    src = session_dir / "panels.json"
    artifact = CutArtifact.model_validate_json(src.read_text("utf-8"))
    by_id = {p.id: p for p in artifact.panels}
    ordered = []
    for p in (panels or job.panels):
        cp = by_id.get(p["id"])
        if cp is not None:
            ordered.append(cp)
    if not ordered:
        ordered = list(artifact.panels)
    for i, cp in enumerate(ordered, 1):
        cp.panel_index = i
    # bake narration overrides so the render sees them (panels.json keeps
    # the AI baseline; narration_edit.json stays the source of truth)
    from .narration_api import apply_overrides_to_cut
    n_ov = apply_overrides_to_cut(job.config["session"], ordered)
    if n_ov:
        job.log("INFO", f"narration overrides baked into confirmed artifact "
                        f"panels={n_ov}", "apply_confirmed")
    conf = artifact.model_copy(update={"panels": ordered})
    dst = session_dir / "panels_confirmed.json"
    tmp = dst.with_suffix(".tmp")
    tmp.write_text(conf.model_dump_json(indent=2) + "\n", "utf-8")
    tmp.replace(dst)
    job.log("INFO", f"confirmed artifact written panels={len(ordered)}",
            "build_script")
    return dst


def _continuation_chain(session: str, current_continue_from: str | None = None) -> list[str]:
    """Sessions to merge, oldest first, for a continuation run.

    Walks persisted continuation.json links backwards from the previous
    strip to the root. `current_continue_from` is THIS run's link (job
    config) — used as the walk's starting point since this session's own
    continuation.json may not exist yet. Cycle-guarded.
    """
    chain: list[str] = []
    cur: str | None = current_continue_from
    seen: set[str] = set()
    while cur:
        if cur in seen:
            break                      # cycle guard
        seen.add(cur)
        chain.append(cur)
        cur = _continuation_link(cur)
    chain.reverse()
    return chain


def _continuation_link(session: str) -> str | None:
    """Read the persisted continuation link for a session (None if absent)."""
    p = OUTPUT_DIR / session / "continuation.json"
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text("utf-8")).get("continue_from")
    except Exception:
        return None


def _panel_from_dict(p: dict, strip_width: int | None) -> Any:
    """Build a CutPanel from a live Panel-Review dict (AI or custom)."""
    from guided_cutter import CutPanel
    try:
        return CutPanel(
            id=p["id"], panel_index=p.get("panel_index", 0),
            y_start=p.get("y_start", 0), y_end=p.get("y_end", 0),
            narration=p.get("narration", ""), dialogue=p.get("dialogue", ""),
            panel_type=p.get("panel_type", "panel"),
            confidence=p.get("confidence", 1.0),
            image_file=p.get("image_file", ""),
            strip_width=strip_width)
    except Exception:
        return None


def _merge_continuation(job: Job, **kwargs: Any) -> None:
    """Merge previous strips' confirmed panels ahead of this strip's panels.

    Triggered by `continue_from` (set via /api/run when the user continues
    an existing sequence). The chain (strip1 <- strip2 <- strip3...) is
    walked so strip N's run merges ALL previous strips, in order, with:

    - ids namespaced per source session (s_<session4>_<panel_id>) so
      panels from different strips never collide;
    - panel numbering CONTINUING across strips (strip 2's first panel is
      n+1, not 1);
    - each source's user-confirmed order/deletions and narration edits
      preserved verbatim;
    - panel PNGs copied into this session so the merged render has all
      images locally, and TTS clips reuse previous strips' audio by id.
    """
    session = job.config["session"]
    cont_from = job.config.get("continue_from")
    if not cont_from:
        return
    from guided_cutter import CutArtifact, CutPanel
    session_dir = OUTPUT_DIR / session
    chain = _continuation_chain(session, current_continue_from=cont_from)
    job.log("INFO", f"continuation chain: {' -> '.join([*chain, session])}",
            "merge_continuation")
    # THIS strip's confirmed/AI panels (already ordered by apply_confirmed)
    own = [dict(p) for p in job.panels]
    if not own:
        job.log("WARNING", "no own panels; merging previous strips only",
                "merge_continuation")
    # OWN baseline: confirmed review or AI baseline ONLY — never
    # panels_merged.json. A stale merged artifact from a previous
    # continuation run would re-namespace already-namespaced ids
    # (sSSSSSSSS_sAAAAAAAA_panel_001) and double-count the old chain.
    own_src = None
    for name in ("panels_confirmed.json", "panels.json"):
        cand = session_dir / name
        if cand.is_file():
            own_src = cand
            break
    if own_src is None:
        job.log("WARNING", "own strip has no panels artifact; aborting merge",
                "merge_continuation")
        return
    own_art = CutArtifact.model_validate_json(own_src.read_text("utf-8"))
    panels_out: list[CutPanel] = []
    stats = {"strips": 0, "panels": 0}
    for src_session in chain:
        src_dir = OUTPUT_DIR / src_session
        # Live review state (latest edit layer, not a possibly-stale
        # confirmed artifact): deletions/order/review AND custom panels
        # (duplicate/split/merge) the user made in the combined Panel
        # Review must be reflected at merge time. Falls back to the
        # strip's own artifact when there is no live view.
        live_panels: list[dict] | None = None
        try:
            from .panel_api import _get_panels_one
            live = _get_panels_one(src_session)
            live_panels = [p for p in live["panels"] if not p["deleted"]]
            if not live_panels:
                job.log("WARNING",
                        f"strip {src_session}: no active panels; skipped",
                        "merge_continuation")
                continue
        except Exception:
            live_panels = None
        src_width: int | None = None
        by_oid: dict[str, CutPanel] = {}
        if live_panels is not None:
            # resolve custom panels against their edit-layer dicts and the
            # AI baseline for width/y-coordinates
            art = CutArtifact.model_validate_json(
                (src_dir / "panels.json").read_text("utf-8"))
            src_width = art.width
            for p in live_panels:
                if p.get("custom"):
                    continue        # handled below via dict conversion
                base = next((x for x in art.panels if x.id == p["id"]), None)
                if base is not None:
                    by_oid[p["id"]] = base
        else:
            # Read the strip's OWN artifact (confirmed review or AI
            # baseline), NEVER its panels_merged.json: that one contains
            # this same chain already merged, which would double-count and
            # double-namespace.
            src_panels_json = None
            for name in ("panels_confirmed.json", "panels.json"):
                cand = src_dir / name
                if cand.is_file():
                    src_panels_json = cand
                    break
            if src_panels_json is None:
                job.log("WARNING", f"strip {src_session} has no panels; skipped",
                        "merge_continuation")
                continue
            art = CutArtifact.model_validate_json(src_panels_json.read_text("utf-8"))
            src_width = art.width
            by_oid = {p.id: p for p in art.panels}
        merged_count = 0
        for p in (live_panels or [None] * 0) if live_panels is not None else []:
            # live dict panels (AI + custom), in the user's confirmed order
            q = _panel_from_dict(p, strip_width=src_width)
            if q is None:
                continue
            q.id = f"s{src_session[:8]}_{p['id']}"
            q.image_file = f"{q.id}.png"
            src_img = src_dir / p.get("image_file", "")
            dst_img = session_dir / q.image_file
            if src_img.is_file() and not dst_img.is_file():
                shutil.copyfile(src_img, dst_img)
            for ext in (".mp3", ".wav"):
                src_audio = src_dir / "audio" / f"{p['id']}{ext}"
                if src_audio.is_file():
                    (session_dir / "audio").mkdir(exist_ok=True)
                    shutil.copyfile(
                        src_audio, session_dir / "audio" / f"{q.id}{ext}")
                    break
            panels_out.append(q)
            stats["panels"] += 1
            merged_count += 1
        if live_panels is None:
            for _raw_id, p in by_oid.items():
                q = p.model_copy()
                q.id = f"s{src_session[:8]}_{p.id}"
                q.image_file = f"{q.id}.png"
                q.strip_width = src_width
                src_img = src_dir / p.image_file
                dst_img = session_dir / q.image_file
                if src_img.is_file() and not dst_img.is_file():
                    shutil.copyfile(src_img, dst_img)
                for ext in (".mp3", ".wav"):
                    src_audio = src_dir / "audio" / f"{p.id}{ext}"
                    if src_audio.is_file():
                        (session_dir / "audio").mkdir(exist_ok=True)
                        shutil.copyfile(
                            src_audio,
                            session_dir / "audio" / f"{q.id}{ext}")
                        break
                panels_out.append(q)
                stats["panels"] += 1
                merged_count += 1
        stats["strips"] += 1
        job.log("INFO", f"merged strip {src_session}: {merged_count} panels",
                "merge_continuation")
    # own panels: re-namespace too so ids stay uniformly unique
    own_ns: list[CutPanel] = []
    id_map: dict[str, str] = {}   # original id -> namespaced id (own strip)
    for p in own_art.panels:
        q = p.model_copy()
        q.id = f"s{session[:8]}_{p.id}"
        id_map[p.id] = q.id
        q.image_file = f"{q.id}.png"
        q.strip_width = own_art.width
        src_img = session_dir / p.image_file
        if src_img.is_file():
            dst_img = session_dir / q.image_file
            if not dst_img.is_file():
                shutil.copyfile(src_img, dst_img)
        for ext in (".mp3", ".wav"):
            src_audio = session_dir / "audio" / f"{p.id}{ext}"
            if src_audio.is_file():
                shutil.copyfile(src_audio,
                                session_dir / "audio" / f"{q.id}{ext}")
                break
        own_ns.append(q)
    panels_out.extend(own_ns)
    # Continue the numbering across strips (never restart at 1)
    for i, q in enumerate(panels_out, 1):
        q.panel_index = i
    merged = CutArtifact(
        source=own_art.source, width=own_art.width, height=own_art.height,
        plan_hash=f"merged:{cont_from}", config={"continuation": chain},
        panels=panels_out)
    dst = session_dir / "panels_merged.json"
    tmp = dst.with_suffix(".tmp")
    tmp.write_text(merged.model_dump_json(indent=2) + "\n", "utf-8")
    tmp.replace(dst)
    # Link the chain only AFTER the merged artifact landed: a failed merge
    # must not leave continuation.json pointing at strips whose merge
    # never completed (the combined Panel Review would show them anyway).
    (session_dir / "continuation.json").write_text(
        json.dumps({"continue_from": cont_from, "chain": chain}, indent=2),
        "utf-8")
    # expose the own-strip id translation so _apply_order can map a raw
    # user-supplied order onto the namespaced merged ids
    job.config["_own_id_map"] = id_map
    job.panels = [{
        "id": q.id, "panel_index": q.panel_index,
        "y_start": q.y_start, "y_end": q.y_end,
        "narration": q.narration, "dialogue": q.dialogue,
        "panel_type": q.panel_type, "confidence": q.confidence,
        "image_file": q.image_file,
    } for q in panels_out]
    job.log("INFO", f"merged artifact written: {stats['strips']} strips, "
                    f"{len(panels_out)} panels (numbering continues "
                    f"1..{len(panels_out)})", "merge_continuation")
    job.progress = 45


CutPanelLike = Any  # documentation alias


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
    """The renderer/script input, in precedence order:
    merged continuation artifact > user-confirmed artifact > AI baseline."""
    for name in ("panels_merged.json", "panels_confirmed.json"):
        p = session_dir / name
        if p.is_file():
            return p
    return session_dir / "panels.json"


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
    """Deprecated: never scheduled (TTS runs inside render via
    make_recap_video). Kept only so STAGE_TIMEOUT_S and older start_stage
    retries referencing it resolve."""
    job.log("WARNING", "tts_audio stage reached but not scheduled; "
                      "TTS runs inside render_video", "tts_audio")


def _resolve_ffmpeg() -> str:
    """ffmpeg resolution with bundled-binary fallback.

    shutil.which first; then the imageio-ffmpeg wheel (installed as a
    dependency) ships a full ffmpeg binary — use it instead of failing.
    """
    import shutil
    resolved = shutil.which("ffmpeg")
    if resolved:
        return resolved
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        raise _RenderSoftError(
            "ffmpeg not found on PATH or via imageio-ffmpeg; install it "
            "to render video (Windows: `winget install Gyan.FFmpeg`) or "
            "reinstall the app (`pip install -e .`) to pull the bundled "
            "binary. Panels/timeline/SRT on disk are unaffected.") from None


def _render_video(job: Job, **kwargs: Any) -> None:
    """Render video using the isolated RenderWorker for proper isolation,
    cancellation, and resource bounding.

    Failures AFTER the inputs are proven good (missing ffmpeg, encoder
    crash, cancelled slot) raise _RenderSoftError: panels, narration,
    timeline and SRT are already on disk, so the job continues to
    save_outputs/create_editor_project instead of dead-ending. Missing
    upstream inputs (panels/timeline) still hard-fail the job.
    """
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
    # TTS + timeline + captions FIRST (synchronous, cache-friendly): a
    # missing ffmpeg must not prevent timeline.json / recap.srt from
    # landing on disk — everything except the video survives a render
    # outage. If TTS itself fails (network/edge outage), rebuild the
    # timeline silently; the user can retry the render later.
    # NOTE: dry-run against the FINAL out path — make_recap_video derives
    # the SRT sidecar from out_path (recap.mp4 -> recap.srt). Passing the
    # .partial.mp4 tmp name used to write recap.partial.srt, so recap.srt
    # never existed for webapp sessions. Dry-run renders no video, so no
    # partial file is produced at this path.
    try:
        make_recap_video(panels_json, out_mp4, cfg, force=False, dry_run=True)
    except Exception as exc:  # noqa: BLE001 - degrade to silent timeline
        job.log("WARNING", f"TTS/timeline build failed ({exc}); retrying "
                          "with tts=none so the session stays usable",
                "render_video")
        cfg = cfg.model_copy(update={"tts": "none"})
        # tts=none changes the config hash, so the cached clips from the
        # failed attempt cannot be reused anyway — force is irrelevant
        # here, but keep it False for consistency: the cache is hash-keyed.
        make_recap_video(panels_json, out_mp4, cfg, force=False, dry_run=True)
    # Resolve ffmpeg ONCE (PATH, then the bundled imageio-ffmpeg binary)
    # and reuse the resolved absolute path for every subprocess. Without
    # this, Popen("ffmpeg") on Windows dies with WinError 2 after all
    # upstream stages succeeded.
    ffmpeg_exe = _resolve_ffmpeg()
    job.log("INFO", f"ffmpeg resolved: {ffmpeg_exe}", "render_video")
    # Determine render strategy
    render_profile = job.config.get("render_profile", "balanced")
    threads = job.config.get("ffmpeg_threads", "auto")
    cpu_count = os.cpu_count() or 4
    if threads == "auto":
        threads = str(max(2, cpu_count - 2))
    prof = render_profiles(cpu_count).get(render_profile, render_profiles(4)["balanced"])
    prof["threads"] = threads
    # Build the render command
    strategy = pick_render_strategy(len(job.panels) if job.panels else 10, 120)
    if strategy == "chunked":
        ta = _get_timeline_artifact(session_dir)
        # Chunked renders are heavy too: serialize them behind the same
        # one-render semaphore as direct renders, and make the job
        # cancellable while chunking.
        try:
            with render_slot() as cancel_event:
                for chunk_cmd in _chunk_commands(ta, out_tmp, ffmpeg_exe, prof):
                    job.check_cancelled()
                    if cancel_event is not None and cancel_event.is_set():
                        raise CancelledError("render cancelled")
                    _run_render_subprocess(job.id, chunk_cmd)
                if out_tmp.is_file():
                    out_tmp.replace(out_mp4)
                else:
                    raise _RenderSoftError(
                        "chunked render did not produce output")
        except _RenderSoftError:
            _cleanup_partial_render(out_tmp)
            raise
        except CancelledError:
            _cleanup_partial_render(out_tmp)
            raise
        except Exception as exc:
            _cleanup_partial_render(out_tmp)
            raise _RenderSoftError(f"chunked render failed: {exc}") from exc
    else:
        from adapters.render_ffmpeg import build_command
        ta = _get_timeline_artifact(session_dir)
        with render_slot() as cancel_event:
            def build_cmd(tmp):
                return build_command(ta, tmp, ffmpeg_exe=ffmpeg_exe)
            done = threading.Event()
            res = {}
            def on_done(ok, err):
                res["ok"] = ok
                res["err"] = err
                done.set()
            worker = RenderWorker(job.id, build_cmd, out_tmp, on_done=on_done,
                                  cancel_event=cancel_event,
                                  owns_render_slot=False)
            worker.start()
            try:
                while not done.wait(0.5):
                    job.check_cancelled()
            except CancelledError:
                worker.cancel(reason="cancelled by user")
                raise
        if not res.get("ok"):
            raise _RenderSoftError(res.get("err", "render failed"))
        if out_tmp.is_file():
            out_tmp.replace(out_mp4)
        else:
            raise _RenderSoftError("render did not produce output")
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
    # NOTE: intentionally NOT stored in config — a working order here is
    # not an explicit user order (see /api/run's `order`); storing it used
    # to make _apply_confirmed skip the user's confirmed Panel Review.
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


@contextlib.contextmanager
def render_slot():
    """Acquire the global one-render semaphore with cancellation support.

    Yields a threading.Event that render workers poll; setting it makes
    RenderWorker terminate ffmpeg. Released when the with-block exits.
    """
    from .render_worker import _RENDER_SEM
    _RENDER_SEM.acquire()
    cancel_event = threading.Event()
    try:
        yield cancel_event
    finally:
        cancel_event.set()          # best-effort: signal any straggling worker
        _RENDER_SEM.release()


def _chunk_commands(ta, out_tmp, ffmpeg_exe, prof):
    """Chunked-render commands without executing them (for cancellable runs)."""
    from adapters.render_ffmpeg import build_command_chunked
    segs, _concat, _tmp = build_command_chunked(ta, out_tmp, ffmpeg_exe=ffmpeg_exe,
                                                chunk_size=12, profile=prof)
    for cmd, _seg in segs:
        yield cmd
    yield _concat


def _cleanup_partial_render(out_tmp: Path) -> None:
    """Best-effort removal of failed/cancelled chunked-render leftovers:
    the .partial.mp4 and its recap.partial_parts/ segment directory
    (chunked renders regenerate all of them on retry)."""
    import shutil as _shutil
    with contextlib.suppress(OSError):
        if out_tmp.is_file():
            out_tmp.unlink()
    parts = out_tmp.parent / (out_tmp.stem + "_parts")
    with contextlib.suppress(OSError):
        if parts.is_dir():
            _shutil.rmtree(parts)


def _run_render_subprocess(job_id: str, cmd: list[str]) -> None:
    """Run one ffmpeg chunk command; raise with stderr tail on failure.

    Bounded by a per-chunk timeout: an unbounded subprocess.run blocks
    the stage thread forever on a hung ffmpeg (the /render-cancel path
    cannot reach chunked renders — _ACTIVE only tracks RenderWorkers).
    """
    proc = subprocess.run(cmd, capture_output=True, text=True,
                           shell=False, check=False,
                           timeout=CHUNK_TIMEOUT_S)
    if proc.returncode != 0:
        tail = "\n".join((proc.stderr or "").splitlines()[-20:])
        raise RuntimeError(f"ffmpeg chunk failed (exit {proc.returncode}):\n{tail}")


def _save_outputs(job: Job, **kwargs: Any) -> None:
    session_dir = OUTPUT_DIR / job.config["session"]
    for name in ("panels.json", "panels_merged.json", "panels_confirmed.json",
                 "narration.txt", "recap.mp4",
                 "recap.srt", "timeline.json"):
        if (session_dir / name).exists():
            job.outputs[name] = name
    if "recap.mp4" not in job.outputs and not job.outputs.get("render_error"):
        job.outputs["render_error"] = "recap.mp4 missing (render skipped)"
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
        ("merge_continuation", _merge_continuation),
        ("apply_order", _apply_order),
        ("gemini_narration", _gemini_narration),
        ("build_script", _build_script),
        ("render_video", _render_video),
        ("save_outputs", _save_outputs),
        ("create_editor_project", _create_editor_project),
    ],
    # One stage at a time (Step-by-Step debugger mode). The worker runs
    # the single requested stage, checkpoints it, and exits — "paused"
    # is simply the absence of a running worker.
    "pipeline_step": [],
    # Run stages continuously up to and including `until_stage`.
    "pipeline_until": [],
}


# --------------------------------------------------------------------------- #
# Step-by-Step / checkpoint orchestration (see webapp/checkpoint.py)
# --------------------------------------------------------------------------- #
def _run_steps(job: Job, first_stage: str, last_stage: str | None,
               **kwargs: Any) -> None:
    """Shared engine for step / until modes: runs a CONTIGUOUS range of
    real pipeline stages with checkpointing + artifact validation.

    Reuses the stage functions from PIPELINES["generate"]; never a
    duplicate implementation. `job.panels` is rehydrated from the
    session's artifacts so a retry of step 7 does not need steps 1-6.
    """
    import traceback as _tb

    from . import checkpoint as cp
    session = job.config["session"]

    stages = list(PIPELINES["generate"])
    names = [n for n, _fn in stages]
    if first_stage not in names:
        job.fail(f"unknown stage {first_stage!r}")
        return
    start_idx = names.index(first_stage)
    if last_stage is None:
        end_idx = start_idx            # single step
    elif last_stage in names:
        end_idx = names.index(last_stage)
    else:
        job.fail(f"unknown last stage {last_stage!r}")
        return
    if end_idx < start_idx:
        job.fail(f"stage {last_stage!r} precedes {first_stage!r}")
        return

    cp.log_event(session, step=None, event_type="run_begin",
                 message=f"step execution: {first_stage}"
                         + (f"..{last_stage}" if last_stage else ""),
                 severity="INFO")

    # Rehydrate panels from the newest artifact so a mid-pipeline retry
    # works without re-running segmentation (the classic resume gap).
    if start_idx > names.index("segment_panels"):
        _rehydrate_panels(job, session)

    for name, fn in stages[start_idx:end_idx + 1]:
        step_no = cp.BY_PIPELINE_NAME.get(name, {}).get("step")
        job.check_cancelled()
        # Non-checkpointed side-effect stage (merge_continuation): run
        # inline when in range, but never ledger it as a user step.
        if step_no is None:
            fn(job, **kwargs)
            continue
        t0 = time.time()
        cp.log_event(session, step=step_no, event_type="step_start",
                     message=f"Step {step_no} started", severity="INFO",
                     function=name)
        job.stage = name
        job.touch()
        try:
            _with_timeout(lambda fn=fn: fn(job, **kwargs),
                          STAGE_TIMEOUT_S[name], job, name)
        except StageTimeoutError as exc:
            _record_step_failure(job, cp, session, step_no, name, exc,
                                 time.time() - t0, _tb)
            raise
        except CancelledError:
            cp.log_event(session, step=step_no, event_type="step_cancel",
                         message=f"Step {step_no} cancelled", severity="INFO",
                         function=name, duration_s=time.time() - t0)
            job.status = JobStatus.CANCELLED
            job.finished_at = time.time()
            job.touch()
            raise
        except Exception as exc:
            duration = time.time() - t0
            if isinstance(exc, _RenderSoftError):
                # render soft-fail: panels/timeline/SRT are on disk; the
                # step is checkpointed with a warning, run continues.
                job.outputs["render_error"] = str(exc)[:300]
                cp.record_step(session, step_no, status="success",
                               artifacts=cp.BY_STEP[step_no]["produces"],
                               duration_s=duration,
                               warnings=[f"render failed: {exc} — outputs "
                                         "remain; retry from Pipeline view"])
                cp.log_event(session, step=step_no,
                             event_type="step_warning",
                             message=f"render failed softly: {exc}",
                             severity="WARNING", function=name,
                             duration_s=duration)
                job.log("WARNING", f"render failed softly ({exc}); "
                                    "continuing", name)
                continue
            _record_step_failure(job, cp, session, step_no, name, exc,
                                 duration, _tb)
            raise
        duration = time.time() - t0
        # Artifact validation gate: a stage that "finished" but produced
        # broken output is NOT marked successful.
        problems, soft = cp.validate_artifacts(session, step_no)
        artifacts = cp.BY_STEP[step_no]["produces"]
        if problems:
            err = "artifact validation failed: " + "; ".join(problems)
            exc = RuntimeError(err)
            _record_step_failure(job, cp, session, step_no, name, exc,
                                duration, _tb)
            raise exc
        cp.record_step(session, step_no, status="success",
                       artifacts=artifacts, duration_s=duration,
                       warnings=soft)
        cp.log_event(session, step=step_no, event_type="step_success",
                     message=f"Step {step_no} completed", severity="INFO",
                     function=name, duration_s=duration,
                     artifact=", ".join(artifacts) or None)
        if name == "render_video":
            job.progress = 95
        if name == "save_outputs":
            job.progress = 100
        job.touch()
    job.stage = "done"


def _record_step_failure(job: Job, cp: Any, session: str, step_no: int,
                         name: str, exc: Exception, duration: float,
                         tb_mod: Any) -> None:
    """Mark the step failed in the ledger + structured debugger event."""
    tb = "".join(tb_mod.format_exception(type(exc), exc, exc.__traceback__))
    frame_file = None
    for ln in tb.splitlines():
        s = ln.strip()
        if s.startswith('File "') and ("recap-comic" in s
                                       or "manhwa-recap" in s):
            frame_file = s
            break
    cp.record_step(session, step_no, status="failed",
                   error=f"{type(exc).__name__}: {exc}",
                   duration_s=duration)
    cp.log_event(session, step=step_no, event_type="step_error",
                 message=str(exc) or type(exc).__name__,
                 severity="ERROR", function=name,
                 duration_s=duration,
                 exception=f"{type(exc).__name__}: {exc}",
                 traceback=tb[-4000:], file=frame_file)
    job.log("ERROR", f"step {step_no} ({name}) failed: {exc}", name)
    job.stage = name
    if job.status != JobStatus.FAILED:
        job.fail(f"step {step_no} ({name}): {type(exc).__name__}: {exc}")


def _rehydrate_panels(job: Job, session: str) -> None:
    """Reload job.panels from the session's newest panels artifact.

    Retrying narration/script/render must not require re-running
    segmentation: panels.json (or panels_confirmed/merged) is the durable
    source of truth.
    """
    p = _panels_source(OUTPUT_DIR / session)
    if not p.is_file():
        return
    try:
        from guided_cutter import CutArtifact
        artifact = CutArtifact.model_validate_json(p.read_text("utf-8"))
        job.panels = [{
            "id": q.id, "panel_index": q.panel_index,
            "y_start": q.y_start, "y_end": q.y_end,
            "narration": q.narration, "dialogue": q.dialogue,
            "panel_type": q.panel_type, "confidence": q.confidence,
            "image_file": q.image_file,
        } for q in artifact.panels]
        job.log("INFO", f"panels rehydrated from {p.name} "
                        f"count={len(job.panels)}", "checkpoint")
    except Exception as exc:
        job.log("WARNING", f"panel rehydration failed: {exc}", "checkpoint")


def run_steps_job(job_id: str, first_stage: str,
                  last_stage: str | None = None, **kwargs: Any) -> None:
    """Worker entry for step / until modes. One bounded run, no loop."""
    job = store.get(job_id)
    if job is None:
        return
    job.status = JobStatus.RUNNING
    job.started_at = time.time()
    job.touch()
    job.log("INFO", f"step worker started first={first_stage} "
                    f"last={last_stage}")
    try:
        _run_steps(job, first_stage, last_stage, **kwargs)
        # _run_steps early-returns (job.fail already recorded) on bad
        # stage names / inverted ranges: only mark COMPLETED when the
        # steps actually ran to their end without failing.
        if job.status not in (JobStatus.FAILED, JobStatus.CANCELLED):
            job.status = JobStatus.COMPLETED
            job.finished_at = time.time()
            job.stage = "done"
            job.touch()
            job.log("INFO", "step run completed (paused after checkpoint)")
    except CancelledError:
        pass
    except Exception:
        if job.status not in (JobStatus.FAILED, JobStatus.CANCELLED):
            job.fail(f"worker crashed: {_tb_tail()}")


def _tb_tail() -> str:
    import traceback
    return traceback.format_exc()[-300:]


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
    if start_stage:
        names = [name for name, _fn in PIPELINES[job.kind]]
        if start_stage not in names:
            job.fail(f"unknown start_stage {start_stage!r}; valid: {names}")
            return
        job.log("INFO", f"resuming from stage {start_stage!r} "
                        "(earlier stages skipped; cached outputs on disk "
                        "are reused)", start_stage)
    try:
        started = start_stage is None
        for name, fn in PIPELINES[job.kind]:
            if not started:
                if name == start_stage:
                    started = True     # run this stage and everything after
                else:
                    job.log("INFO", f"skipping stage (before start_stage="
                                    f"{start_stage}) {name}")
                    continue
            if time.time() - t0 > JOB_TIMEOUT_S:
                raise StageTimeoutError(
                    f"job exceeded {JOB_TIMEOUT_S}s total budget")
            if name == "render_video":
                _stage_continue_on_fail(job, name,
                                       lambda fn=fn: fn(job, **kwargs))
            else:
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
