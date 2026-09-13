# webapp/checkpoint.py
"""Checkpoint-based pipeline state for Step-by-Step mode.

Layering (per session, one file):

    <session>/pipeline_state.json      progress ledger (small, hot)
    <session>/debug_events.jsonl      append-only structured debugger log

pipeline_state.json schema:

    {
      "version": 1,
      "steps": [ {"step": 3, "name": "segment_panels",
                  "status": "success|failed|stale",
                  "artifacts": ["panels.json"], "error": null,
                  "duration_s": 12.4, "finished_at": 1694...,
                  "input_hashes": {...}, "warnings": [...]} ],
      "current_step": 4,
      "completed_steps": [1,2,3],
      "status": "paused|running|done|failed",
      "last_successful_step": 3,
      "updated_at": ...
    }

Design rules:
- NEVER blocks or sleeps: the webapp worker thread runs ONE step then
  returns; "paused" is just the absence of a running worker.
- Reuses webapp.pipeline stage functions; no duplicate implementations.
- Staleness is hash-based: a step marked stale when an artifact it
  CONSUMED has changed since it ran (dependency-aware invalidation).
- Debug events are structured (timestamp/step/severity/message/...) and
  persisted so they survive refresh/restart; the file is rotated by size.
"""
from __future__ import annotations

import hashlib
import json
import threading
import time
from pathlib import Path
from typing import Any

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "webapp_output"

STATE_VERSION = 1
STATE_FILENAME = "pipeline_state.json"
EVENTS_FILENAME = "debug_events.jsonl"
MAX_EVENTS_BYTES = 512 * 1024        # rotate the debug log at 512 KB
MAX_EVENTS_KEEP = 40                 # keep last N lines after rotation

# The canonical step list mirrors webapp.pipeline.PIPELINES["generate"],
# minus validate_config/load_images which are trivial preflight steps that
# the step engine folds into "segment_panels" for the user (they are still
# individually logged in the debugger). Order matters.
STEP_SEQUENCE: list[dict[str, Any]] = [
    {"step": 1, "name": "check_config",       "pipeline_name": "validate_config",
     "label": "Check config",     "consumes": [], "produces": []},
    {"step": 2, "name": "load_images",         "pipeline_name": "load_images",
     "label": "Load images",      "consumes": ["strip"], "produces": []},
    {"step": 3, "name": "segment_panels",      "pipeline_name": "segment_panels",
     "label": "Segment panels",
     "consumes": ["strip"], "produces": ["panels.json", "panel images"]},
    {"step": 4, "name": "panel_validation",   "pipeline_name": "panel_validation",
     "label": "Validate panels",
     "consumes": ["panels.json", "panels_validation.json"],
     "produces": ["panels_validation.json"]},
    {"step": 5, "name": "apply_review",        "pipeline_name": "apply_confirmed",
     "label": "Apply review",
     "consumes": ["panels.json", "panels_edit.json", "panels_validation.json"],
     "produces": ["panels_confirmed.json"]},
    {"step": 6, "name": "apply_order",         "pipeline_name": "apply_order",
     "label": "Apply order",
     "consumes": ["panels.json", "panels_edit.json"],
     "produces": []},
    {"step": 7, "name": "gemini_narration",    "pipeline_name": "gemini_narration",
     "label": "Gemini narration",
     "consumes": ["panels.json", "panels_confirmed.json"],
     "produces": []},
    {"step": 8, "name": "build_script",        "pipeline_name": "build_script",
     "label": "Build script",
     "consumes": ["panels.json", "panels_confirmed.json", "narration_edit.json"],
     "produces": ["narration.txt"]},
    {"step": 9, "name": "render_video",        "pipeline_name": "render_video",
     "label": "Render video",
     "consumes": ["panels.json", "panels_confirmed.json", "narration_edit.json",
                  "voice.json"],
     "produces": ["recap.mp4", "recap.srt", "timeline.json"]},
    {"step": 10, "name": "save_outputs",       "pipeline_name": "save_outputs",
     "label": "Save outputs",
     "consumes": [], "produces": []},
    {"step": 11, "name": "editor_project",     "pipeline_name": "create_editor_project",
     "label": "Editor project",
     "consumes": ["timeline.json"], "produces": ["editor.json"]},
]

BY_STEP = {s["step"]: s for s in STEP_SEQUENCE}
BY_NAME = {s["name"]: s for s in STEP_SEQUENCE}
BY_PIPELINE_NAME = {s["pipeline_name"]: s for s in STEP_SEQUENCE}

# Artifacts a step consumes/produces -> the later steps they invalidate.
# Editing panel order invalidates apply_order+; editing narration
# invalidates build_script+; changing segmentation invalidates
# panel_validation+ (the review/order/narration/render chain).
EDIT_INVALIDATION: dict[str, int] = {
    # what changed                    -> first step to invalidate
    "panels.json": 4,                  # re-segmentation / manual crop
    "panels_edit.json": 5,             # Panel Review edits
    "panels_validation.json": 5,       # validation decision changes
    "panels_confirmed.json": 7,        # confirmed-set changes
    "narration_edit.json": 8,         # Narration Studio edits
    "voice.json": 9,                   # voice changes -> rerun render
    "order": 6,                        # explicit reorder
}

_STATE_LOCK = threading.Lock()


def session_dir(session: str) -> Path:
    d = OUTPUT_DIR / session
    return d


def _state_path(session: str) -> Path:
    return session_dir(session) / STATE_FILENAME


def _events_path(session: str) -> Path:
    return session_dir(session) / EVENTS_FILENAME


# --------------------------------------------------------------------------- #
# State file CRUD
# --------------------------------------------------------------------------- #
def _new_state(session: str) -> dict[str, Any]:
    return {
        "version": STATE_VERSION,
        "session": session,
        "steps": [],
        "current_step": 1,
        "completed_steps": [],
        "status": "idle",
        "last_successful_step": 0,
        "mode": None,             # "automation" | "step" (last used)
        "updated_at": time.time(),
    }


def load_state(session: str) -> dict[str, Any]:
    p = _state_path(session)
    if not p.is_file():
        return _new_state(session)
    try:
        data = json.loads(p.read_text("utf-8"))
        if not isinstance(data, dict) or "steps" not in data:
            return _new_state(session)
        return data
    except (OSError, ValueError):
        return _new_state(session)


def save_state(session: str, state: dict[str, Any]) -> None:
    state["updated_at"] = time.time()
    p = _state_path(session)
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(state, indent=2), "utf-8")
    tmp.replace(p)


def record_step(session: str, step: int, *, status: str,
               artifacts: list[str] | None = None,
               error: str | None = None,
               duration_s: float = 0.0,
               warnings: list[str] | None = None) -> dict[str, Any]:
    """Append/refresh one step entry in the ledger and return the state.

    status: success | failed | stale | skipped
    """
    with _STATE_LOCK:
        state = load_state(session)
        entry = {
            "step": step,
            "name": BY_STEP[step]["name"],
            "label": BY_STEP[step]["label"],
            "status": status,
            "artifacts": artifacts or [],
            "error": error,
            "duration_s": round(duration_s, 2),
            "warnings": warnings or [],
            "input_hashes": _step_input_hashes(session, step),
            "finished_at": time.time(),
        }
        # replace any prior entry for this step
        state["steps"] = [e for e in state["steps"] if e["step"] != step]
        state["steps"].append(entry)
        state["steps"].sort(key=lambda e: e["step"])
        if status == "success":
            if step not in state["completed_steps"]:
                state["completed_steps"].append(step)
                state["completed_steps"].sort()
            state["last_successful_step"] = step
            state["current_step"] = _next_step(step)
        elif status == "failed":
            state["current_step"] = step     # retry target
        state["status"] = ("failed" if status == "failed" else
                           "done" if step == len(STEP_SEQUENCE)
                           and status == "success" else "paused")
        save_state(session, state)
    return state


def _next_step(step: int) -> int:
    return min(step + 1, len(STEP_SEQUENCE))


def step_entry(session: str, step: int) -> dict[str, Any] | None:
    for e in load_state(session)["steps"]:
        if e["step"] == step:
            return e
    return None


# --------------------------------------------------------------------------- #
# Artifact validation (runs after every stage)
# --------------------------------------------------------------------------- #
def _artifact_paths(session: str, kind: str) -> list[Path]:
    d = session_dir(session)
    if kind == "strip":
        # whatever strip file the session actually uses
        for cand in d.glob("strip.*"):
            if cand.is_file() and cand.suffix.lower() in (".png", ".jpg",
                                                          ".jpeg", ".webp"):
                return [cand]
        return []
    p = d / kind
    return [p] if p.is_file() else []


def _hash_file(p: Path) -> str | None:
    try:
        h = hashlib.sha256()
        h.update(p.read_bytes())
        return h.hexdigest()[:16]
    except OSError:
        return None


def _step_input_hashes(session: str, step: int) -> dict[str, str]:
    out: dict[str, str] = {}
    for kind in BY_STEP[step]["consumes"]:
        for p in _artifact_paths(session, kind):
            out[kind] = _hash_file(p) or "?"
    return out


def validate_artifacts(session: str, step: int) -> tuple[list[str], list[str]]:
    """Verify expected outputs exist and are structurally valid.

    Returns (problems, soft). `problems` blocks marking the step
    successful; `soft` is surfaced as warnings only. A step whose outputs
    are missing/broken is NOT marked successful.
    """
    d = session_dir(session)
    problems: list[str] = []
    soft: list[str] = []

    def _json_loads_ok(name: str) -> dict | None:
        p = d / name
        if not p.is_file():
            return None
        try:
            return json.loads(p.read_text("utf-8"))
        except (OSError, ValueError):
            return {}

    def _panel_count(data: dict | None) -> int:
        if not isinstance(data, dict):
            return -1
        panels = data.get("panels", [])
        return len(panels) if isinstance(panels, list) else -1

    if step == 3:      # segment_panels
        data = _json_loads_ok("panels.json")
        if data is None:
            problems.append("panels.json missing or unreadable")
        elif _panel_count(data) == -1:
            problems.append("panels.json has no 'panels' list")
        else:
            n = _panel_count(data)
            if n == 0:
                problems.append("segmentation produced 0 panels")
            else:
                # verify a sample of panel images exist
                import random
                sample = random.sample(data["panels"],
                                        min(3, len(data["panels"])))
                for pp in sample:
                    img = pp.get("image_file") or ""
                    if img and not (d / img).is_file():
                        problems.append(
                            f"panel image missing: {img}")
    elif step == 4:    # panel_validation
        data = _json_loads_ok("panels_validation.json")
        if data is None:
            problems.append("panels_validation.json missing or unreadable")
        elif not isinstance(data.get("verdicts"), list):
            problems.append("panels_validation.json has no verdicts")
    elif step == 5:    # apply_review
        if not (d / "panels.json").is_file():
            problems.append("panels.json missing")
    elif step == 8:    # build_script
        p = d / "narration.txt"
        if not p.is_file():
            problems.append("narration.txt missing")
        elif p.stat().st_size == 0:
            # empty script is legitimate in offline fallback mode but
            # worth surfacing — a warning, not a hard failure.
            soft.append("narration.txt is empty (offline fallback?)")
    elif step == 9:    # render_video
        tl = _json_loads_ok("timeline.json")
        if tl is None:
            problems.append("timeline.json missing or unreadable")
        srt = d / "recap.srt"
        if not srt.is_file():
            problems.append("recap.srt missing")
    elif step == 11:   # editor_project
        if not (d / "editor.json").is_file():
            problems.append("editor.json missing")
    return problems, soft


# --------------------------------------------------------------------------- #
# Staleness / dependency-aware invalidation
# --------------------------------------------------------------------------- #
def mark_stale_from(session: str, first_step: int, reason: str) -> dict[str, Any]:
    """Invalidate `first_step` and every step after it; keep earlier ones.

    After invalidation the pipeline resumes AT first_step (the earliest
    stale step), never before it — "edit panel order -> rerun apply_order
    and later stages".
    """
    with _STATE_LOCK:
        state = load_state(session)
        for e in state["steps"]:
            if e["step"] >= first_step and e["status"] == "success":
                e["status"] = "stale"
                e.setdefault("warnings", []).append(reason)
        state["completed_steps"] = [s for s in state["completed_steps"]
                                    if s < first_step]
        # last successful strictly before the invalidation point
        earlier = state["completed_steps"]
        state["last_successful_step"] = max(earlier) if earlier else 0
        state["current_step"] = first_step
        save_state(session, state)
    return state


def detect_stale_steps(session: str) -> dict[str, list[int]]:
    """Compare stored input_hashes with current hashes; report drift.

    Returns {"stale": [step numbers]} for steps whose consumed artifacts
    changed since they last ran successfully.
    """
    state = load_state(session)
    stale: list[int] = []
    for e in state["steps"]:
        if e["status"] != "success":
            continue
        now = _step_input_hashes(session, e["step"])
        if now != e.get("input_hashes", {}):
            stale.append(e["step"])
    return {"stale": stale}


def apply_edit_invalidation(session: str, changed: str) -> dict[str, Any]:
    """Public hook for the edit APIs (panel/narration/voice/manual-crop):
    invalidate dependent steps when the user edits an artifact."""
    first = EDIT_INVALIDATION.get(changed)
    if first is None:
        return load_state(session)
    return mark_stale_from(
        session, first,
        f"user edited {changed}; downstream steps invalidated")


# --------------------------------------------------------------------------- #
# Structured debug events (persistent debugger)
# --------------------------------------------------------------------------- #
def log_event(session: str, *, step: int | None, event_type: str,
              message: str, severity: str = "INFO",
              function: str | None = None, file: str | None = None,
              duration_s: float | None = None,
              exception: str | None = None, traceback: str | None = None,
              artifact: str | None = None) -> None:
    """Append one structured event to debug_events.jsonl (rotated)."""
    ev = {
        "t": round(time.time(), 3),
        "step": step,
        "event_type": event_type,
        "function": function,
        "message": message,
        "severity": severity.upper(),
        "duration_s": duration_s,
        "file": file or artifact,
        "exception": exception,
        "traceback": traceback,
    }
    p = _events_path(session)
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        with _STATE_LOCK:
            with p.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(ev, ensure_ascii=False) + "\n")
            _rotate_events(p)
    except OSError:
        pass    # debugger logging must never break the pipeline


def _rotate_events(p: Path) -> None:
    try:
        if p.stat().st_size > MAX_EVENTS_BYTES:
            lines = p.read_text("utf-8").splitlines()
            tail = lines[-MAX_EVENTS_KEEP:]
            p.write_text("\n".join(tail) + "\n", "utf-8")
    except OSError:
        pass


def load_events(session: str, *, step: int | None = None,
                severity: str | None = None, limit: int = 400) -> list[dict]:
    """Read events (newest last). Optional step/severity filters."""
    p = _events_path(session)
    if not p.is_file():
        return []
    try:
        lines = p.read_text("utf-8").splitlines()
    except OSError:
        return []
    out: list[dict] = []
    for ln in lines:
        try:
            ev = json.loads(ln)
        except ValueError:
            continue
        if step is not None and ev.get("step") != step:
            continue
        if severity and ev.get("severity") != severity.upper():
            continue
        out.append(ev)
    return out[-limit:]


def delete_state(session: str) -> None:
    """Full reset: remove the ledger (artifacts stay on disk)."""
    with contextlib_suppress():
        _state_path(session).unlink(missing_ok=True)
    with contextlib_suppress():
        _events_path(session).unlink(missing_ok=True)


def contextlib_suppress():
    import contextlib
    return contextlib.suppress(OSError)
